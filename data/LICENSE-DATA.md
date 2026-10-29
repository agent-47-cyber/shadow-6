# Data licences

## Dakshina lexicons — `hi.lexicon.tsv`, `mr.lexicon.tsv`

Derived from the **Dakshina dataset** by Google Research.

- Source: https://github.com/google-research-datasets/dakshina
- Licence: **Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)**
- https://creativecommons.org/licenses/by-sa/4.0/

These files are a reformatting of Dakshina's romanization lexicons: each line is
a Devanagari word followed by its attested Latin spellings and their attestation
counts.

Because the licence is ShareAlike, **these data files remain CC BY-SA 4.0** and
are not covered by this project's MIT licence. The application source code in
`livewhisper/` is MIT.

## `translit_hi.pt` — character transliteration model

A 2.56M-parameter model trained by this project on the Dakshina Hindi lexicon
(see `experiments/exp4_oov_seq2seq.py`).

Trained weights are distributed under the same CC BY-SA 4.0 terms out of
caution, since the training data carries ShareAlike.
