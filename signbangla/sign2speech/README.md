# Sign → Bengali speech

Run from the project root with the signbangla Python environment active:

```bash
python scripts/11_live_webcam_demo.py --test-speech
python scripts/11_live_webcam_demo.py --camera 0
```

The checkpoint belongs at `work/results/bigru/best_model.pt`; use `--checkpoint` for a different location. A local copy of the supplied checkpoint has been placed there on this computer.

The optional dependency is `edge-tts>=7.3.1,<8`. It is installed locally on this computer under `work/speech_dependencies`. On another machine, run `python -m pip install -r requirements-speech.txt` after installing the base dependencies. macOS playback uses its built-in `afplay`; Windows/Linux need `ffplay` on PATH.

## Flow and files

- `predictor.py`: loads the existing BiGRU/BiLSTM checkpoint.
- Existing MediaPipe and preprocessing code: produces normalized 64-frame landmark sequences.
- `stability.py`: requires three consecutive predictions at confidence >= 0.80.
- `speech.py`: uses only `bn-BD-NabanitaNeural` (Bangladesh, female), caches MP3s, and plays audio in a background worker. It retries empty service responses twice within a 25-second overall deadline and reports failures without stopping the camera.
- `scripts/11_live_webcam_demo.py`: camera, Bengali text display, and integration.

One confirmed sign is spoken per gesture attempt. Move both hands out of view for 0.8 seconds or press **c** to start another attempt. A visible resting hand is not a reliable neutral detector, so it does not rearm speech. Press **m** to change input mirroring (clears the buffer), **q** to quit. Input starts unmirrored to match dataset extraction. Audio in progress is not interrupted by **c**; new utterances are ignored while speech is busy.

Use `--confidence 0.85 --confirmations 4` for stricter confirmation, or `--no-speech` for recognition only. The checkpoint has no unknown/neutral class: stable predictions can still be wrong. This is isolated-word recognition, not sentence translation.

## Internet and cached audio

Edge-tts sends the accepted Bengali text to Microsoft's online speech service; webcam frames remain local. No API key is required. Internet is needed for uncached words. Generate all 60 word files in advance with:

```bash
python scripts/11_live_webcam_demo.py --cache-speech
```

Files are stored in `work/speech_cache`, keyed by voice and text. Cached words play without a network request. Service availability is external; a failed request shows a speech error. Press **c** to retry the sign. The voice is never silently replaced.

## Validation

```bash
python -m pytest -q tests/test_sign2speech.py
```

Tests cover uncertain predictions, repeat suppression, rearming, cache reuse, failed download cleanup, and nonblocking worker behavior. Actual camera accuracy and sound output must also be checked on the demo machine.
