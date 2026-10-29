# SHADOW-6

**Speak naturally. Stay on target.**

SHADOW-6 is an offline-first Windows voice typing companion. Press a global shortcut, speak Hindi and English naturally, and review the transcription in the app you were already using. SHADOW-6 does not send messages, press Enter, submit forms, or execute commands.

<img src="assets/shadow6-logo.png" width="192" alt="SHADOW-6 shadow mask logo">

## Current state

This is an early SHADOW-6 derivative of [Eqho](https://github.com/DanielMevit/Eqho), licensed under GNU AGPL-3.0. It retains Eqho's Windows tray app, microphone/audio worker, model-host subprocess, floating overlay, hotkeys, settings dashboard, and text injection. Hindi-script spans are converted to Roman Hinglish with LiveWhisper's Hindi romanization pipeline. English spans are kept intact.

The initial local baseline is faster-whisper large-v3, with the medium model still selectable. SHADOW-6 prefers the user's existing Hugging Face model cache when large-v3 is present. The optional Oriserve Apex Transformers backend has **not** been integrated or benchmarked yet. Speech-to-text speed still depends on the model, GPU runtime, and microphone; no accuracy or latency benchmark is claimed.

## Download and run on Windows

Download the latest **SHADOW-6-portable** ZIP from [GitHub Releases](https://github.com/agent-47-cyber/shadow-6/releases/latest), extract it, and run `SHADOW-6.exe` from the extracted `SHADOW-6` folder. The portable package contains the application runtime and license notices, but not speech-model weights. On first use, connect to the internet so the selected Whisper model can be downloaded (large-v3 is about 3 GB); later transcription runs locally. CUDA is used when a compatible NVIDIA runtime is available; otherwise faster-whisper runs on the CPU. The Windows build is unsigned, so SmartScreen may show a reputation prompt.

Your selected language, model, shortcut, and setup completion are saved in `%APPDATA%\SHADOW-6\settings.json`; setup should only appear on first run. A cached model is loaded once and kept ready while SHADOW-6 runs. Live preview starts after about 0.4 seconds of captured speech and refreshes roughly every 0.35 seconds; the text may change as the sentence continues. A final phrase is committed after about 0.6 seconds of silence. Actual timing varies by hardware and model.

## Run from source (Windows)

Python 3.14.3, CUDA PyTorch 2.11.0+cu128, CTranslate2 4.8.3, faster-whisper 1.2.1, and an NVIDIA RTX 4070 SUPER were detected during setup. CUDA was available to both PyTorch and CTranslate2. The project venv reuses those existing inference packages and installs the desktop dependencies locally; this avoids replacing the working CUDA stack.

```powershell
cd "path\to\shadow-6"
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

The default shortcut is **Ctrl+Shift+Space**. Click into a text field first. When dictation ends, SHADOW-6 inserts recognized text into the prior target only when its focus checks allow it. Review the text and send it yourself. Roman Hinglish can be changed in **General → Speech → Roman Hinglish**.

While dictating, you can also say **“SHADOW-6 stop”** or **“S-6 stop”** at the end of a phrase to stop listening; the command is removed from the text. Spoken-word recognition is model-based and may need a short pause for the model to recognize the command.

The default recognition model is `large-v3`, and Hindi is the default recognition language. Roman Hinglish conversion is enabled. The medium model is available in **Settings → Models**. Model downloads happen through the existing background model setup flow; the app needs network access only when a selected model is not cached.

## Logo

`assets/shadow6-logo.png` is the user-provided SHADOW-6 artwork used as the app, tray, and onboarding icon. `assets/shadow6.ico` provides Windows icon sizes.

## Development checks

```powershell
.\.venv\Scripts\python.exe run.py --smoke
```

The smoke check uses the cached large-v3 model and stays offline. If that model is absent, the check reports that condition without downloading weights. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/PRIVACY.md](docs/PRIVACY.md), [docs/SOURCE_AUDIT.md](docs/SOURCE_AUDIT.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Windows packaging

The Windows portable application is built as an onedir package; it does not include speech model weights. The Inno Setup installer is optional and requires Inno Setup 6. This unsigned build may show a SmartScreen reputation prompt.

```powershell
.\packaging\windows\build.ps1
```

The Inno Setup installer is optional and requires Inno Setup 6. The installer/build workflow should be verified before distributing a binary.

## License

The application is a derivative of Eqho and is distributed under GNU AGPL-3.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and bundled romanization-data terms. The SHADOW-6 name and icon do not imply endorsement by Eqho's author.
