"""The user's personal spelling conventions, learned from their corrections.

exp5 measured that romanization variation is not arbitrary: 1,200 letter-level
axes exist across 30,000 Hindi words, but the top 10 explain 57% of all
variation and the top 20 explain 70%. So conventions are learned at the LETTER
level, not the word level:

    "mujhe" -> "muze" is not a fact about that word.
    It is a fact about how this person spells ज, and it generalises to the
    316 other words containing it.

Learning at the word level would need ~17,000 corrections. At this level it
needs about ten.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field

from .languages import WORD

log = logging.getLogger(__name__)

MAX_SPAN = 3          # letter habits, not wholesale rewrites
PROMOTE_AT = 2        # distinct words before a rule generalises


@dataclass
class Conventions:
    """Per-user spelling rules plus exact per-word overrides."""

    # "jh" -> "z" style rewrites, applied to romanizer output
    rules: dict[str, str] = field(default_factory=dict)
    # evidence backing each candidate rule: (from, to) -> set of source words
    evidence: dict[str, set] = field(default_factory=dict)
    # exact fixes that beat every rule
    overrides: dict[str, str] = field(default_factory=dict)

    # ---------------------------------------------------------------- apply

    def apply(self, roman: str) -> str:
        """Rewrite one romanized word according to the learned rules.

        Longest source first, so "jh"->"z" wins over "j"->"z" on the same span.
        Only applied to text the romanizer produced from Devanagari, so English
        words in the transcript are never touched.
        """
        if not self.rules:
            return roman
        out, i = [], 0
        keys = sorted(self.rules, key=len, reverse=True)
        while i < len(roman):
            for k in keys:
                if roman.startswith(k, i):
                    out.append(self.rules[k])
                    i += len(k)
                    break
            else:
                out.append(roman[i])
                i += 1
        return "".join(out)

    # --------------------------------------------------------------- learn

    def learn(self, source_word: str, default: str, corrected: str) -> list[str]:
        """Record one correction. Returns human-readable notes about what changed.

        A single correction fixes that word immediately but does NOT generalise -
        you might simply have typo'd. A substitution seen in two different words
        is promoted to a rule.
        """
        notes: list[str] = []
        default, corrected = default.strip(), corrected.strip()
        if not corrected or default == corrected:
            return notes

        # The exact fix always applies, immediately.
        self.overrides[source_word] = corrected
        notes.append(f"{source_word}: {default} -> {corrected}")

        for a, b in _substitutions(default, corrected):
            key = f"{a}>{b}"
            self.evidence.setdefault(key, set()).add(source_word)
            seen = len(self.evidence[key])
            if seen < PROMOTE_AT or self.rules.get(a) == b:
                continue
            ok, collateral = validate_rule(a, b)
            if not ok:
                # Keep the per-word override, refuse the generalisation. This
                # is what stops "too" for तू from turning aur into aoor.
                log.info("rejected rule %s -> %s (would affect %.0f%% of words)",
                         a, b, collateral * 100)
                notes.append(f"kept {source_word} as {corrected} "
                             f"(too risky to apply everywhere)")
                continue
            self.rules[a] = b
            notes.append(f"learned convention: {a} -> {b} "
                         f"(seen in {seen} words)")
        return notes

    def seed(self, pairs: list[tuple[str, str, str]]) -> list[str]:
        """Bootstrap from onboarding: (source_word, default, user_spelling)."""
        notes = []
        for word, default, user in pairs:
            notes += self.learn(word, default, user)
        return notes

    # ---------------------------------------------------------- serialise

    def to_dict(self) -> dict:
        return {
            "rules": dict(self.rules),
            "overrides": dict(self.overrides),
            "evidence": {k: sorted(v) for k, v in self.evidence.items()},
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "Conventions":
        d = d or {}
        return cls(
            rules=dict(d.get("rules") or {}),
            overrides=dict(d.get("overrides") or {}),
            evidence={k: set(v) for k, v in (d.get("evidence") or {}).items()},
        )


VOWELS = set("aeiou")
MAX_COLLATERAL = 0.22     # share of the whole vocabulary a rule may rewrite


def validate_rule(pattern: str, replacement: str) -> tuple[bool, float]:
    """Is it safe to apply this substitution to every word, or only this one?

    Two guards, learned from a bug this caught in testing.

    1. Single vowels are unsafe. Someone typing "too" for तू teaches u -> oo,
       which then rewrites aur as aoor and bahut as bahoot. A single Latin
       vowel stands for several different Devanagari vowels depending on
       context, so a blind replace destroys the distinction. Consonants do not
       have this problem: jh and v map reliably back to specific letters, so
       jh -> z and v -> w are safe. Multi-character patterns are specific
       enough either way.

    2. Collateral. Whatever the pattern, a rule that rewrites a fifth of the
       entire vocabulary is over-reaching for the two examples behind it.

    Returns (safe, share_of_vocabulary_affected).
    """
    if len(pattern) == 1 and pattern in VOWELS:
        return False, 1.0

    try:
        from .lexicon import get_lexicon
        lex = get_lexicon("hi")
        lex.load()
        entries = lex._entries
    except Exception:
        return True, 0.0            # cannot check, do not block the user

    if not entries:
        return True, 0.0

    changed = total = 0
    for _word, forms in entries.items():
        if not forms:
            continue
        total += 1
        if pattern in forms[0] and forms[0].replace(pattern, replacement) != forms[0]:
            changed += 1

    share = changed / total if total else 0.0
    return share <= MAX_COLLATERAL, share


def _substitutions(default: str, corrected: str) -> list[tuple[str, str]]:
    """Letter-level differences between the default spelling and the user's."""
    out = []
    sm = difflib.SequenceMatcher(a=default, b=corrected, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        a, b = default[i1:i2], corrected[j1:j2]
        if a and b and len(a) <= MAX_SPAN and len(b) <= MAX_SPAN:
            out.append((a, b))
    return out


def align_words(native: list[str], roman: list[str], score) -> list[tuple[str, str]]:
    """Pair Devanagari words with the user's Latin words during onboarding.

    Transliteration preserves word order, so position alignment works when the
    counts match. When they do not - a dropped or merged word - fall back to a
    plausibility-scored alignment so one mismatch does not corrupt everything
    after it.
    """
    if len(native) == len(roman):
        return list(zip(native, roman))

    pairs, i, j = [], 0, 0
    while i < len(native) and j < len(roman):
        if score(native[i], roman[j]) > 0.4:
            pairs.append((native[i], roman[j]))
            i += 1
            j += 1
        elif i + 1 < len(native) and score(native[i + 1], roman[j]) > 0.4:
            i += 1          # a native word the user dropped
        elif j + 1 < len(roman) and score(native[i], roman[j + 1]) > 0.4:
            j += 1          # a Latin word with no native counterpart
        else:
            i += 1
            j += 1
    return pairs


def similarity(a: str, b: str) -> float:
    """Cheap plausibility score for the alignment fallback."""
    return difflib.SequenceMatcher(a=a.lower(), b=b.lower()).ratio()


# Tuned against real failures rather than guessed. Similarity alone is not
# enough: "aa" vs "araha" scores 0.57, but so does "mujhe" vs "muze", and the
# first is nonsense while the second is exactly what we want to learn. So the
# length guard does the work that the ratio cannot.
#
# Both the onboarding aligner and correction learning use these. They were
# separate numbers once, and the correction path's looser 0.45 let a markdown
# link in the field teach मैं -> "link": SequenceMatcher scores those 0.50,
# because two four-letter words sharing "in" is enough. One override like that
# rewrites every future "main".
MIN_SIMILARITY = 0.60
LEN_RATIO = (0.5, 1.8)          # transliteration roughly preserves length


def plausible_correction(default: str, candidate: str) -> bool:
    """Could `candidate` be this user's spelling of the same word?

    A correction is a respelling, not a replacement. Anything that fails this is
    a different word that happens to sit nearby - the field had other text in
    it, or the user rewrote the sentence - and learning from it corrupts the
    profile permanently.
    """
    if not default or not candidate:
        return False
    ratio = len(candidate) / len(default)
    if not (LEN_RATIO[0] <= ratio <= LEN_RATIO[1]):
        return False
    return similarity(default, candidate) >= MIN_SIMILARITY


def tokenize(text: str) -> list[str]:
    """Split into words, keeping native script words whole.

    Uses the shared mark-aware pattern rather than `[^\\W\\d_]+`. This path sees
    native script whenever a word failed to romanize or the user forced the
    original script, and the naive pattern shreds those into fragments - which
    then get aligned against Latin words and learned from.
    """
    return WORD.findall(text)
