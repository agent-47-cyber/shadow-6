# Third-party notices

## Eqho

The Windows desktop application is derived from Eqho by Daniel Mevit. Eqho is licensed under GNU Affero General Public License version 3. Its source notices and license are retained in this repository. SHADOW-6 changes the product identity, defaults, branding, Hindi romanization, cache selection, and model selection behavior. No endorsement by Daniel Mevit is implied.

- Project: <https://github.com/DanielMevit/Eqho>
- License: [GNU AGPL-3.0](LICENSE)

## LiveWhisper romanization

The Hindi romanization source modules in `src/romanization/` are adapted from LiveWhisper by Sarthak Karandikar and are MIT licensed. The MIT notice is retained at [licenses/LiveWhisper-MIT.txt](licenses/LiveWhisper-MIT.txt).

- Project: <https://github.com/itsskofficial/LiveWhisper>
- Source code license: MIT
- The included Hindi lexicon and `data/translit.npz` derive from the Dakshina dataset and are separately licensed under CC BY-SA 4.0. Their data notice is at [data/LICENSE-DATA.md](data/LICENSE-DATA.md); the applicable license is <https://creativecommons.org/licenses/by-sa/4.0/>.

## Speech models

The cached faster-whisper large-v3 and medium weights are not copied into this repository. They are downloaded from Systran/Hugging Face when needed and remain subject to their upstream model and OpenAI Whisper terms. SHADOW-6 does not ship model weights.

Oriserve/Whisper-Hindi2Hinglish-Apex is an Apache-2.0 model candidate. It is not bundled or currently supported by the app. If added, it must be downloaded separately and its license/attribution must accompany the model files.

## Dependencies

Application dependencies have their own licenses. See each package's metadata and upstream project before redistributing a packaged build. In particular, CUDA/PyTorch distribution terms and included native libraries must be reviewed when preparing a public Windows binary.
