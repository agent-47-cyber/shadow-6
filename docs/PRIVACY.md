# Privacy

- Recognition and romanization run locally after model setup.
- Audio is captured only during an active dictation session and is not persisted by default.
- Transcript history is disabled by default; the existing history feature can be enabled by the user.
- Log messages record status, lengths, and timings; raw audio and full transcript text are not intentionally logged.
- Model weights are stored in the local Hugging Face or SHADOW-6 model cache, not in this repository.
- The app does not automatically send messages, press Enter, submit forms, or execute commands.
- Automatic text insertion depends on the target application's focus behavior. If target validation fails, use the app's available retry/copy recovery rather than trusting a changed foreground window.

Initial model setup may require a network connection to Hugging Face. Normal inference is offline once the selected model and its required files are available locally.
