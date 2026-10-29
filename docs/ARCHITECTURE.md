# Architecture

SHADOW-6 retains Eqho's process layout:

- `src/main.py` coordinates settings, audio/transcription, overlay, tray, hotkeys, and insertion.
- `src/audio.py` and `src/transcriber.py` manage live microphone input, VAD, partial/final chunks, and session cleanup.
- `src/model_host.py` isolates faster-whisper/CTranslate2 model loading and inference in a child process.
- `src/overlay.py` displays recording/transcription state without taking over the user's destination window.
- `src/hotkey.py`, `src/injector.py`, and `src/oskit/windows.py` handle shortcut and Windows insertion behavior.
- `src/hinglish.py` applies the local Hindi romanizer to Hindi-script spans on partial and final transcripts. English text is not passed through a translator.
- `src/settings.py` stores SHADOW-6 settings separately from Eqho's original settings and can point at the existing Hugging Face cache.

The recognizer returns provisional and complete chunks. The inherited application reconciles chunks before final insertion. Insertion does not send Enter. Model inference happens outside the UI thread and the overlay is already configured not to activate itself.

Large-v3 is the default baseline. Its CTranslate2 backend prefers CUDA when available; it can fall back to CPU if CUDA initialization fails. The existing model-host interface supports faster-whisper and whisper.cpp. Apex/Transformers is not yet an integrated backend.
