# SHADOW-6 development notes

- Keep all development and generated test data inside this repository or its ignored `.venv`, `build`, and `dist` folders.
- Preserve Eqho's GNU AGPL-3.0 license and attribution. Preserve LiveWhisper's MIT notice and the Dakshina romanization data's CC BY-SA 4.0 notice.
- Do not commit model weights, local settings, transcripts, secrets, or private audio.
- Run `.venv\Scripts\python.exe run.py --smoke` and `tests\test_hinglish.py` after code changes. The smoke command must remain offline and must not download models.
- Never send Enter or submit a destination form/message. Keep focus validation and transcript recovery intact.
- Do not push or publish without the user's explicit approval.
