# Team Notes

## Who owns what

Teammates work on Module 1 and the repo owner works on Module 2.

| Belongs to | Paths |
|---|---|
| **Module 1** (Sign → Speech) | `scripts/01_*` to `scripts/11_*`, `signbangla/annotations.py`, `dataset.py`, `metrics.py`, `models.py`, `pipeline.py`, `quality.py`, `tests/test_pipeline.py`, `work/results/`, `work/eval_demo/`, `work/mediapipe_test/`, `work/quality_plots/` |
| **Module 2** (Speech → Sign) | `signbangla/speech2sign/`, `scripts/20_*`, `tests/test_speech2sign.py`, `assets/fonts/`, `work/sign_dictionary/` |
| **Shared** | `signbangla/keypoints.py`, `labels.py`, `preprocess.py`, `utils.py`, `viz.py`, `signbangla/__init__.py`, `configs/config.yaml`, `requirements.txt`, `README.md`, `TEAM_NOTES.md`, `.gitignore`, `data/`, `work/raw_keypoints/`, `work/trials/`, `work/processed/`, `work/*.csv` |

## Where `work/` comes from

**Nothing in `work/` or `data/` is in git.**

| Source | Paths |
|---|---|
| `signbangla_keypoints.zip` on the team's Google Drive | `work/raw_keypoints/`, `work/processed/`, `work/trials/`, `work/sign_dictionary/`, `work/*.csv`, `work/split_summary.txt` |
| Scripts, on your own machine | `work/results/` (script 10), `work/quality_plots/` (script 05), `work/eval_demo/` (script 09), `work/mediapipe_test/` (script 02) |

Anyone who regenerates the data in the zip must tell the team and share a new zip.

## Rule for changing shared files

**Change a shared file only after telling the team, and run all the tests before pushing the change:**

```powershell
python -m pytest -q tests
```

Module 2 imports the shared code and reads the keypoint files, so a change there can break the other module. This includes the keypoint format, the label order and regenerating `work/raw_keypoints/`.

## Shared files that changed

18 videos are stored upside down, so their keypoints were re-extracted with the frames turned upright.

| File | What changed |
|---|---|
| `signbangla/keypoints.py` | `extract_video` takes `rotate_180` and records `rotated_180` in the video's `.json` |
| `signbangla/viz.py` | `write_preview_video` takes `rotate_180` |
| `signbangla/pipeline.py` | New `rotated_stems` and `needs_extraction` |
| `configs/config.yaml` | New `video.rotate_180` list of the 18 videos |
| `requirements.txt` | Added `imageio-ffmpeg`, which Module 2 uses to write H.264 video |
| `scripts/02_test_mediapipe.py` | Turns the listed videos upright |
| `scripts/03_extract_keypoints.py` | Turns the listed videos upright, and redoes any listed video not yet marked `rotated_180` |
| `scripts/05b_preview_trial.py` | Turns the listed videos upright |
| `.gitignore` | Ignores all of `work/` and the old presentation PDF |
| `README.md` | Added teammate setup. Removed the old results, which are pending retraining. Corrected the validation signers to U1 and U11, and rewrote the download section |

**Use the current zip.** The coordinates of the 18 videos changed, and the zip now holds all the data from `work/`. Get the current `signbangla_keypoints.zip` from the team's Google Drive and do not mix it with an older copy.

## What you need to do

- **Python 3.11 is required.** Install it before creating the environment. `mediapipe==0.10.14` does not install on Python 3.13 or newer. Follow "Setup for Teammates" in `README.md`.
- **Rerun script 01.** `work/manifest.csv` holds absolute paths from the repo owner's PC, and scripts 02 and 05b read them. Once the videos are in `data/BdSLW60/`, run the command below. It rewrites `work/manifest.csv` with your paths.

  ```powershell
  python scripts/01_inspect_dataset.py
  ```

- **Train before using the live demo.** `work/results/` is neither in git nor in the zip, so there is no `best_model.pt` until you run `scripts/10_train_baseline.py`.
