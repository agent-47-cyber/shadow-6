"""Romanize Hindi-script spans while preserving English and punctuation.

The lexicon and numpy character model are from LiveWhisper's MIT-licensed
romanization pipeline. Their Dakshina-derived data remains CC BY-SA 4.0.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _romanizer():
    from .romanization import Romanizer
    return Romanizer("hi")


def romanize_hinglish(text: str) -> str:
    """Return Roman Hinglish for Hindi script; leave existing Latin text intact."""
    if not text:
        return text
    return _romanizer().text(text)
