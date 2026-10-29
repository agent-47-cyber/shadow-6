# Source and environment audit

## Foundation choice

Eqho was selected as the application foundation. Its existing Windows implementation already provides a system tray, global hotkey handling, audio capture/VAD, provisional transcription, a non-activating floating overlay, focus-aware insertion, settings, and an isolated model-host subprocess. Replacing those pieces with a second desktop app would add risk without improving the starting point.

LiveWhisper was inspected as the Roman Hinglish reference. Its source separates speech recognition from romanization and preserves Latin-script English while converting native-script spans. SHADOW-6 uses its Hindi lexicon/romanizer rather than a universal character substitution. Lexicon and transliteration model data carry separate CC BY-SA 4.0 terms, recorded in the repository notices.

The Eqho and LiveWhisper checkouts used for audit are kept under `../work/upstream-audit/`; their files are not mixed into the application source except for the reviewed LiveWhisper romanization modules/data in `src/romanization/` and `data/`. No LiveWhisper desktop UI or speech-recognition code was imported.

The Apex model card currently describes an Apache-2.0, 0.8B-parameter Transformers checkpoint. Its documented `AutoProcessor` / `AutoModelForSpeechSeq2Seq` path is not the same interface as faster-whisper/CTranslate2, so it is recorded as a candidate rather than incorrectly loaded through the existing backend. See <https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex>.

## Environment observed on 2026-10-29

- Python 3.14.3.
- PyTorch 2.11.0+cu128; `torch.cuda.is_available()` returned true.
- NVIDIA GeForce RTX 4070 SUPER; CTranslate2 enumerated one CUDA device.
- faster-whisper 1.2.1; CTranslate2 4.8.3.
- Hugging Face cache contains `Systran/faster-whisper-large-v3` and `Systran/faster-whisper-medium` snapshots.
- The application searches the existing standard Hugging Face hub cache for large-v3 and uses it without copying weights.

The project `.venv` was created with `--system-site-packages` to preserve the already working CUDA stack. Desktop-specific packages were installed into that venv. This is a deliberate reuse of the user's current inference runtime, rather than a fully isolated copy of PyTorch/CUDA.

## Not yet verified

No microphone recording or text insertion acceptance run was performed. The Apex model was not downloaded or run. No human-checked speech recordings were available for an accuracy benchmark. GPU model loading and the application's offline smoke check are verified only if the smoke command succeeds in the current setup. Windows installer packaging has not been run.
