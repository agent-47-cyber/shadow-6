"""Character model for words the lexicon does not contain (~8% of tokens).

A 4.5M-parameter encoder-decoder (2.56M in the hindi-only build) trained on 44k Dakshina word pairs. It learns
letter patterns rather than words, so it can spell forms it has never seen -
लेंगी -> "lengi" without that word being in any dictionary (exp4).

It is trained with torch but runs here in plain numpy, from data/translit.npz
(written by tools/export_translit.py). PyTorch would add over a gigabyte to the
packaged exe to run a model this small, and numpy is already a dependency. The
forward pass below is a line-for-line port of nn.Transformer in eval mode, and
tests/test_oov_numpy.py holds it to the torch output word for word.

If only the .pt is present and torch is installed, the torch path still works,
so a freshly trained checkpoint is usable before it is exported. Without either
the lexicon still covers 92% of tokens; unknown words simply pass through in
native script rather than crashing.
"""

from __future__ import annotations

import json
import importlib
import logging
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import paths

log = logging.getLogger(__name__)

DATA = paths.DATA
NPZ_CHECKPOINT = DATA / "translit.npz"           # what the app runs, no torch
MULTI_CHECKPOINT = DATA / "translit.pt"          # all languages, one model
LEGACY_CHECKPOINT = DATA / "translit_hi.pt"      # hindi-only, older builds

PAD, BOS, EOS = 0, 1, 2
MAX_OUT = 24
LN_EPS = 1e-5                                    # nn.LayerNorm's default

# The positional embedding has 64 slots, so a longer input indexes off the end
# and raises IndexError deep inside the encoder. Real words never come close -
# the longest in any shipped lexicon is 39 characters - but a transcript can
# arrive with no spaces at all, and then the whole utterance is one "word".
# Whisper does exactly this on run-on speech in Indic languages.
#
# 48 leaves headroom under 64 for the language tag and EOS. Anything longer is
# not a word the model was trained to spell anyway, so it is left in the native
# script: visible and correctable, which is the better failure.
MAX_IN = 48


def _architecture(state: dict) -> dict:
    """Recover the model's shape from its own weights.

    Hardcoding the architecture here as well as in the trainer meant the two
    could drift apart - and they did: the trainer moved to d=256 while this
    still said 192, so the checkpoint failed to load and the app silently fell
    back to dictionary-only. Reading it off the tensors cannot drift.
    """
    d = state["src_emb.weight"].shape[1]
    ff = state["core.encoder.layers.0.linear1.weight"].shape[0]
    layers = len({k.split(".")[3] for k in state
                  if k.startswith("core.encoder.layers.")})
    heads = 4 if d % 4 == 0 else 8         # not recoverable; 4 is what we train
    return {"d": d, "ff": ff, "layers": layers, "heads": heads}


# --- numpy forward pass ------------------------------------------------------

def _layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = np.square(x - mu).mean(-1, keepdims=True)   # biased, as torch does
    return (x - mu) / np.sqrt(var + LN_EPS) * w + b


def _lin(x, w, b):
    """x @ w + b over the last axis, done as one 2-D GEMM.

    numpy dispatches (B, T, d) @ (d, k) as a stack of B small products, which
    measured 7x slower than flattening to (B*T, d) first - the difference
    between ~200 ms and ~40 ms for a batch of 64 words.
    """
    return (x.reshape(-1, x.shape[-1]) @ w + b).reshape(*x.shape[:-1], w.shape[1])


def _softmax(s):
    s = s - s.max(-1, keepdims=True)
    np.exp(s, out=s)
    return s / s.sum(-1, keepdims=True)


class _NumpyTransliterator:
    """nn.Transformer (post-LN, relu, batch_first, eval) re-expressed in numpy.

    Weights are pre-transposed once at load so every projection is a plain
    2-D `x @ W` (see _lin).
    """

    def __init__(self, z, arch: dict):
        self.d, self.h = arch["d"], arch["heads"]
        self.dh = self.d // self.h
        self.n_layers = arch["layers"]
        f = lambda k: np.ascontiguousarray(z[k], dtype=np.float32)  # noqa: E731
        self.src_emb, self.tgt_emb, self.pos = (
            f("src_emb.weight"), f("tgt_emb.weight"), f("pos.weight"))
        self.out_w, self.out_b = f("out.weight").T.copy(), f("out.bias")

        def attn(p):
            # in_proj packs q, k and v row-wise; split so cross-attention can
            # project the memory once and reuse it for every decoding step.
            w, b = f(p + ".in_proj_weight"), f(p + ".in_proj_bias")
            d = self.d
            return {"wq": w[:d].T.copy(), "bq": b[:d],
                    "wkv": w[d:].T.copy(), "bkv": b[d:],
                    "wo": f(p + ".out_proj.weight").T.copy(),
                    "bo": f(p + ".out_proj.bias")}

        def block(p, n_norms):
            return {"ff1": f(p + ".linear1.weight").T.copy(), "fb1": f(p + ".linear1.bias"),
                    "ff2": f(p + ".linear2.weight").T.copy(), "fb2": f(p + ".linear2.bias"),
                    "norms": [(f(f"{p}.norm{i}.weight"), f(f"{p}.norm{i}.bias"))
                              for i in range(1, n_norms + 1)]}

        self.enc = []
        for i in range(self.n_layers):
            p = f"core.encoder.layers.{i}"
            self.enc.append({**block(p, 2), "sa": attn(p + ".self_attn")})
        self.dec = []
        for i in range(self.n_layers):
            p = f"core.decoder.layers.{i}"
            self.dec.append({**block(p, 3), "sa": attn(p + ".self_attn"),
                             "ca": attn(p + ".multihead_attn")})
        # nn.Transformer adds a final LayerNorm after each stack.
        self.enc_norm = (f("core.encoder.norm.weight"), f("core.encoder.norm.bias"))
        self.dec_norm = (f("core.decoder.norm.weight"), f("core.decoder.norm.bias"))

    def _split(self, x):                   # (B, T, d) -> (B, h, T, dh)
        B, T, _ = x.shape
        return x.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)

    def _merge(self, x):                   # (B, h, T, dh) -> (B, T, d)
        B, _, T, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.d)

    def _kv(self, a, x):
        kv = _lin(x, a["wkv"], a["bkv"])
        return self._split(kv[..., :self.d]), self._split(kv[..., self.d:])

    def _attend(self, a, x, k, v, bias):
        """Scaled dot-product attention of x's queries over given keys/values.

        `bias` is additive, -inf where a key must not be seen; it broadcasts
        over heads and query positions. Torch scales q before the product, so
        this does too - same arithmetic order, same rounding.
        """
        q = self._split(_lin(x, a["wq"], a["bq"])) * np.float32(1 / math.sqrt(self.dh))
        s = q @ k.transpose(0, 1, 3, 2)
        if bias is not None:
            s = s + bias
        return _lin(self._merge(_softmax(s) @ v), a["wo"], a["bo"])

    @staticmethod
    def _ff(L, x):
        return _lin(np.maximum(_lin(x, L["ff1"], L["fb1"]), 0), L["ff2"], L["fb2"])

    def _embed(self, ids, emb, start=0):
        n = ids.shape[1]
        return emb[ids] * np.float32(math.sqrt(self.d)) + self.pos[start:start + n]

    def encode(self, src):
        # src_key_padding_mask: padded source positions are never attended to,
        # neither in encoder self-attention nor later from the decoder.
        bias = np.where(src == PAD, -np.inf, 0).astype(np.float32)[:, None, None, :]
        x = self._embed(src, self.src_emb)
        for L in self.enc:
            (n1w, n1b), (n2w, n2b) = L["norms"]
            k, v = self._kv(L["sa"], x)
            x = _layer_norm(x + self._attend(L["sa"], x, k, v, bias), n1w, n1b)
            x = _layer_norm(x + self._ff(L, x), n2w, n2b)
        return _layer_norm(x, *self.enc_norm), bias

    def greedy(self, src, max_len=MAX_OUT):
        """Batched greedy decoding, token for token what the torch model emits.

        The torch version re-runs the whole decoder over the prefix each step
        under a causal mask. Here each layer caches its self-attention keys and
        values instead, and only the newest position is computed: with a causal
        mask a position's output depends on nothing after it, so the earlier
        rows would come out the same anyway. Causality is thus structural -
        the cache only ever holds the past - and a step costs O(1) positions
        rather than O(t).

        Rows also leave the batch once they emit EOS. Nothing after EOS is ever
        read, rows never interact, and a batch of mixed-length words otherwise
        spends most of its late steps on words that finished long ago. Finished
        rows are EOS-filled to the end, where torch would have kept generating
        tokens that the caller discards anyway.
        """
        B = src.shape[0]
        mem, mem_bias = self.encode(src)
        cross = [self._kv(L["ca"], mem) for L in self.dec]
        cache: list[tuple] = [(None, None)] * self.n_layers
        ys = np.full((B, max_len + 1), EOS, dtype=np.int64)
        ys[:, 0] = BOS
        live = np.arange(B)                # original row of each batch slot
        last = ys[:, :1]
        steps = 0
        for t in range(max_len):
            x = self._embed(last, self.tgt_emb, start=t)
            for i, L in enumerate(self.dec):
                (n1w, n1b), (n2w, n2b), (n3w, n3b) = L["norms"]
                k, v = self._kv(L["sa"], x)
                pk, pv = cache[i]
                if pk is not None:
                    k, v = np.concatenate([pk, k], 2), np.concatenate([pv, v], 2)
                cache[i] = (k, v)
                x = _layer_norm(x + self._attend(L["sa"], x, k, v, None), n1w, n1b)
                ck, cv = cross[i]
                x = _layer_norm(x + self._attend(L["ca"], x, ck, cv, mem_bias), n2w, n2b)
                x = _layer_norm(x + self._ff(L, x), n3w, n3b)
            h = _layer_norm(x[:, -1], *self.dec_norm)
            nxt = (h @ self.out_w + self.out_b).argmax(-1)
            ys[live, t + 1] = nxt
            steps = t + 1
            keep = nxt != EOS
            if not keep.any():
                break
            if not keep.all():
                live, nxt, mem_bias = live[keep], nxt[keep], mem_bias[keep]
                cross = [(ck[keep], cv[keep]) for ck, cv in cross]
                cache = [(ck[keep], cv[keep]) for ck, cv in cache]
            last = nxt[:, None]
        return ys[:, :steps + 1]


# --- torch fallback ------------------------------------------------------------

def _build():
    """The training-time model, for a .pt that has not been exported yet."""
    # Dynamic import keeps optional training-time PyTorch out of app bundles;
    # the distributed app uses the reviewed NumPy checkpoint instead.
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")

    class Transliterator(nn.Module):
        def __init__(self, n_src, n_tgt, d=192, heads=4, layers=3, ff=512):
            super().__init__()
            self.d = d
            self.src_emb = nn.Embedding(n_src, d, padding_idx=PAD)
            self.tgt_emb = nn.Embedding(n_tgt, d, padding_idx=PAD)
            self.pos = nn.Embedding(64, d)
            self.core = nn.Transformer(d_model=d, nhead=heads,
                                       num_encoder_layers=layers,
                                       num_decoder_layers=layers,
                                       dim_feedforward=ff, dropout=0.0,
                                       batch_first=True)
            self.out = nn.Linear(d, n_tgt)

        def _embed(self, x, emb):
            pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            return emb(x) * math.sqrt(self.d) + self.pos(pos)

        @torch.no_grad()
        def greedy(self, src, max_len=MAX_OUT):
            src = torch.as_tensor(src, dtype=torch.long)
            pad = (src == PAD)
            mem = self.core.encoder(self._embed(src, self.src_emb),
                                    src_key_padding_mask=pad)
            ys = torch.full((src.size(0), 1), BOS, dtype=torch.long)
            done = torch.zeros(src.size(0), dtype=torch.bool)
            for _ in range(max_len):
                m = nn.Transformer.generate_square_subsequent_mask(ys.size(1))
                h = self.core.decoder(self._embed(ys, self.tgt_emb), mem,
                                      tgt_mask=m, memory_key_padding_mask=pad)
                nxt = self.out(h[:, -1]).argmax(-1, keepdim=True)
                ys = torch.cat([ys, nxt], dim=1)
                done |= nxt.squeeze(1) == EOS
                if done.all():
                    break
            return ys.numpy()

    return Transliterator


class OOVModel:
    """Lazily loaded; absence is not an error.

    `backend` is "auto" (numpy archive if present, else torch on the .pt),
    "numpy" or "torch". Only the equivalence test asks for one explicitly.
    """

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self._model = None
        self._src = self._tgt = None
        self._tried = False
        self.available = False
        self.multilingual = False
        self.languages: set = set()

    def load(self) -> bool:
        if self._tried:
            return self.available
        self._tried = True
        if self.backend != "torch" and NPZ_CHECKPOINT.exists():
            return self._load_numpy()
        if self.backend == "numpy":
            log.info("no %s - unknown words will pass through", NPZ_CHECKPOINT.name)
            return False
        return self._load_torch()

    def _vocab(self, src_itos, tgt_itos, languages) -> None:
        self.languages = set(languages or ["hi"])
        self.multilingual = bool(languages)
        self._src = {c: i for i, c in enumerate(src_itos)}
        self._tgt = tgt_itos

    def _load_numpy(self) -> bool:
        try:
            # allow_pickle=False: the archive holds only arrays and JSON text,
            # so loading it cannot execute code the way unpickling a .pt can.
            with np.load(NPZ_CHECKPOINT, allow_pickle=False) as z:
                meta = {k: json.loads(str(z[f"meta.{k}"]))
                        for k in ("src_itos", "tgt_itos", "languages", "arch")}
                self._vocab(meta["src_itos"], meta["tgt_itos"], meta["languages"])
                self._model = _NumpyTransliterator(z, meta["arch"])
            self.available = True
            log.info("transliteration model ready (numpy): d=%d, %d chars, "
                     "languages %s", meta["arch"]["d"], len(self._src),
                     ",".join(sorted(self.languages)))
        except Exception:
            log.warning("could not load transliteration model", exc_info=True)
        return self.available

    def _load_torch(self) -> bool:
        checkpoint = (MULTI_CHECKPOINT if MULTI_CHECKPOINT.exists()
                      else LEGACY_CHECKPOINT)
        if not checkpoint.exists():
            log.info("no transliteration checkpoint in %s", DATA)
            return False
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            log.info("torch not installed - unknown words will pass through")
            return False
        try:
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self._vocab(ck["src_itos"], ck["tgt_itos"], ck.get("languages"))
            cls = _build()
            arch = ck.get("arch") or _architecture(ck["model"])
            self._model = cls(len(ck["src_itos"]), len(ck["tgt_itos"]), **arch)
            self._model.load_state_dict(ck["model"])
            self._model.eval()
            self.available = True
            log.info("transliteration model ready (torch): d=%d, %d chars, "
                     "languages %s", arch["d"], len(self._src),
                     ",".join(sorted(self.languages)))
        except Exception:
            log.warning("could not load transliteration model", exc_info=True)
        return self.available

    def spell(self, words: list[str], lang: str = "hi") -> dict[str, str]:
        """Romanize a batch of unknown native-script words.

        The multilingual model is told which language it is reading via a tag
        character prepended to the input, because the same letter can romanize
        differently across scripts.
        """
        if not words or not self.load():
            return {}
        if self.multilingual and lang not in self.languages:
            return {}

        # Over-long inputs would index past the positional embedding. Drop them
        # here rather than truncating: a spelling derived from the first 48
        # characters of a 600-character blob would be confidently wrong, and
        # leaving the native script is the honest outcome.
        words = [w for w in words if len(w) <= MAX_IN]
        if not words:
            return {}

        tag = [self._src[f"<{lang}>"]] if (self.multilingual
                                           and f"<{lang}>" in self._src) else []
        enc = [tag + [self._src[c] for c in w if c in self._src] + [EOS]
               for w in words]
        enc = [e if len(e) > 1 else [EOS] for e in enc]
        width = max(len(e) for e in enc)
        S = np.full((len(enc), width), PAD, dtype=np.int64)
        for k, e in enumerate(enc):
            S[k, :len(e)] = e

        out = {}
        for w, row in zip(words, self._model.greedy(S)):
            chars = []
            for i in row.tolist()[1:]:
                if i == EOS:
                    break
                if i > EOS:
                    chars.append(self._tgt[i])
            guess = "".join(chars)
            # Guard against the degenerate repetition seen on very short inputs.
            if guess and not _degenerate(guess):
                out[w] = guess
        return out


def _degenerate(s: str) -> bool:
    """Reject the model's known failure mode: looping output.

    exp4 saw "aaaaaaaa" for आ and "nanan" for न. A milder version appeared in
    testing as जाऊंगा -> "jaaungaanga", where a trigram recurs. Rejecting is
    safe: the word falls back to Devanagari, which is visible and correctable,
    rather than a plausible-looking wrong spelling that slips past unnoticed.
    """
    if len(s) >= 4 and len(set(s)) <= 2:
        return True
    if len(s) > MAX_OUT - 2:
        return True
    if len(s) >= 9:
        grams = [s[i:i + 3] for i in range(len(s) - 2)]
        if len(grams) - len(set(grams)) >= 2:
            return True
    return False


@lru_cache(maxsize=1)
def get_model() -> OOVModel:
    return OOVModel()
