"""Script conversion: Devanagari to the Latin spelling the user actually types.

Whisper transcribes Hindi and Marathi in Devanagari, but almost nobody types
Devanagari in chat. This package converts those spans to Latin, in three stages:

    lexicon    ~92% of tokens, a dictionary lookup      (measured, exp2)
    model      the remaining ~8%, a 2.5M char model     (measured, exp4)
    conventions the user's own spelling habits          (measured, exp5)

The conventions stage runs after BOTH of the first two. That matters: if only
the lexicon were personalised, the 8% coming from the model would come out in
someone else's spelling and the text would read as internally inconsistent.
"""

from .romanize import Romanizer, romanize_text

__all__ = ["Romanizer", "romanize_text"]
