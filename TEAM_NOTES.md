# Team Notes

## Who owns what

Teammates work on Module 1 and the repo owner works on Module 2.

| Belongs to | Paths |
|---|---|
| **Module 1** (Sign → Speech) | `scripts/01_*` to `scripts/11_*`, `signbangla/annotations.py`, `dataset.py`, `metrics.py`, `models.py`, `pipeline.py`, `quality.py`, `tests/test_pipeline.py`, `work/results/`, `work/eval_demo/`, `work/mediapipe_test/`, `work/quality_plots/` |
| **Module 2** (Speech → Sign) | `signbangla/speech2sign/`, `scripts/20_*`, `scripts/21_*`, `scripts/40_*`–`scripts/45_*`, `requirements-m2.txt`, `tests/test_speech2sign.py`, `tests/test_text_processing.py`, `tests/test_core_vocab.py`, `tests/test_studio3d.py`, `tests/test_bdsl300.py`, `docs/core_vocabulary.md`, `work/core_vocab/`, `work/sign_dictionary_core/`, `assets/fonts/`, `work/sign_dictionary/` |
| **Shared** | `signbangla/keypoints.py`, `labels.py`, `labels401.py`, `preprocess.py`, `utils.py`, `viz.py`, `signbangla/__init__.py`, `signbangla/bdslw401.py`, `scripts/30_*` to `scripts/33_*`, `tests/test_labels401.py`, `tests/test_bdslw401.py`, `configs/config.yaml`, `requirements.txt`, `README.md`, `TEAM_NOTES.md`, `.gitignore`, `data/`, `work/raw_keypoints/`, `work/trials/`, `work/processed/`, `work/*.csv` |

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
| `configs/config.yaml` | New `video.rotate_180` list of the 18 videos. New Module 2 settings under `speech2sign`: `gif`, `audio`, `asr`, `matching`, `demo`, `motion`, `sentence`, `core_vocab`, `nonmanual`, `use_core_dictionary`, and `render.style` / `render.speed` / `render.fps`, and `studio3d` (the 3D browser style). `render.size` is now 640 and the caption 96 px (frames 640 x 736). New `bdslw401` section for the second dataset, which nothing in the current pipeline reads |
| `requirements.txt` | Added `imageio-ffmpeg`, which Module 2 uses to write H.264 video |
| `scripts/02_test_mediapipe.py` | Turns the listed videos upright |
| `scripts/03_extract_keypoints.py` | Turns the listed videos upright, and redoes any listed video not yet marked `rotated_180` |
| `scripts/05b_preview_trial.py` | Turns the listed videos upright |
| `.gitignore` | Ignores all of `work/` and the old presentation PDF |
| `README.md` | Added teammate setup. Removed the old results, which are pending retraining. Corrected the validation signers to U1 and U11, and rewrote the download section |

**Use the current zip.** The coordinates of the 18 videos changed, and the zip now holds all the data from `work/`. Get the current `signbangla_keypoints.zip` from the team's Google Drive and do not mix it with an older copy.

## Core vocabulary (Module 2, 9 Nov)

Module 2 now also uses BdSLW401 and a core vocabulary (pronouns, verbs, question words, numbers, greetings) taken from
BdSL40, BDSL49 and, if switched on, INCLUDE. No existing keypoint file was changed: BdSLW60 and BdSLW401 keypoints are
only read. New files and folders:

- `data/BdSL40/`, `data/INCLUDE/`, `data/BDSL49/`, `data/core_vocab/word_list.csv` (downloaded by `scripts/40`; about 2.5 GB, not in git)
- `work/core_vocab/` (keypoints) and `work/sign_dictionary_core/` (merged dictionary)

These are not in `signbangla_keypoints.zip` yet. Module 1 does not use them. Details are in `docs/core_vocabulary.md`.

## 3D studio style (Module 2, 15 Nov)

`scripts/21_speech_to_sign_demo.py --style studio3d` shows the signs as a 3D mannequin in the browser. It works with
`--text` and with the microphone; a new sentence appears on the page by itself.

```powershell
python scripts_speech_to_sign_demo.py --text "বাবা ভাই" --style studio3d
```

- New Module 2 files: `signbangla/speech2sign/lift3d.py` (2D -> 3D on a copy), `studio3d.py` (local server on
  127.0.0.1, standard library only), `web3d/` (the page, plus three.js 0.160.0 and its MIT licence in `web3d/vendor/`),
  `tests/test_studio3d.py`.
- Shared file changed: `configs/config.yaml` gets a new `speech2sign.studio3d` section (port 8765, camera, colours,
  lifting). Nothing else shared changed; Module 1 is untouched, and so are the keypoint files and the other styles.
- MP4/GIF saving is not available for this style yet.
- Face (17 Nov): only two grey eye dots on the head surface at the real eye points (the nose and mouth lines are off:
  `speech2sign.studio3d.face.show_nose_mouth: false`); a hand near the face is moved forward so it never sinks into the head. Settings `speech2sign.studio3d.face`, `hand_clearance`, `hand_clearance_frames`.
- The reference experiment stays in `scripts/experiments/studio3d/`.

## BdSL-300 (Module 2, 16 Nov)

**Source:** BdSL-300 on Kaggle (`hasibturjo/bdsl-300`), built on the SignBD-Word dataset. 300 word folders, 8,000
clips, 26 GB: 100 English folders (signers A-D, 5 takes each) and 200 romanised Bangla folders from SignBD-Word
(signers p1-p16, each signing filmed by a close and a full-body camera).

**Licence: CC BY-SA 4.0.** Anything we share that uses it (keypoints, signs, videos, the dictionary) must credit both
BdSL-300 and SignBD-Word, and be shared under the same licence. The data stays out of git: everything is under `work/`,
which is git-ignored.

- `scripts/43_bdsl300_download.py` lists the dataset and compares every folder with the words we already have
  (`work/core_vocab/BdSL300/overlap.csv`: new / duplicate / ambiguous; the Bangla of romanised folders is a guess).
  It downloads only new and ambiguous words, never duplicates.
- `scripts/44_bdsl300_extract.py` extracts keypoints with the same MediaPipe settings as BdSLW60/BdSLW401, then runs
  the quality check per clip and per word. `scripts/45_bdsl300_previews.py` makes one skeleton preview per word and
  contact sheets in `work/core_vocab_check/BdSL300/`.
- New and ambiguous words are in `data/core_vocab/word_list.csv` with `source = BdSL300, verified = false`. **A word is
  used in sentences only after someone sets its `verified` to `true`** and rebuilds the core dictionary
  (`python scripts/20_build_sign_dictionary.py --core`). Nothing is switched on now.
- Shared file changed: `configs/config.yaml` gets `speech2sign.bdsl300`. No existing keypoint file, BdSLW401 work or
  sign dictionary was changed.

## Ban-Sign-Sent-9K and DU dynamic signs (Module 2, 18 Nov)

Two more sources, for words our sentences need (negation, tense markers, greetings). Code in
`signbangla/extra_sources/`, scripts 46-49, tests `tests/test_extra_sources.py`.

**Sources and credit:**
- **Ban-Sign-Sent-9K** (`banglagov/Ban-Sign-Sent-9K-V1` on Hugging Face): 1,922 sign sentences with natural Bangla,
  9,610 videos by 5 signers (U4, U5, U6, U7, U19).
- **DU dynamic BdSL** (Kaggle `hassan0008jhh/dataset-bangla-sign-language`, folder `dynamic_dataset`): 40 word and
  phrase classes. We use 14 of them.

**Licence: neither source states one.** Treat both as research use only and never redistribute their videos,
keypoints or anything made from them. Credit both if they are used in a report. Everything stays under `work/` and
`data/`, which are not in git.

- `scripts/46_bansign9k_sentences.py`: the xlsx -> `data/bansign9k/sentences.csv`, vocabulary and typos (the source is
  never changed), and a Module 3 evaluation.
  - The xlsx's "Test Data Set" sheet has signers' names and e-mails in `metas`. That column is never read or copied.
- `scripts/47_dynamic_du.py`: downloads only the 14 classes in `dynamic_du.classes` by byte range, extracts keypoints
  and writes `work/dynamic_du/class_table.csv` (`needs_check = true`). Its `dataset_numpy` folder is hands-only
  arrays, not our 67-point format, so it is not used.
- `scripts/47b_dynamic_du_previews.py`: one skeleton video (6 clips) and one contact sheet (24 clips) per class in
  `work/core_vocab_check/dynamic_du/`, from the existing keypoints only. New config keys `dynamic_du.preview_*`.
- `scripts/48_bansign9k_videos.py`: the split zip (z01-z09 + .zip, about 20.5 GB), resumable with SHA-256 checks.
  - Stops if less than 60 GB is free.
  - Not needed for the current results: `scripts/49` fetches its subset by byte range.
- `scripts/49_bansign9k_last_sign.py`: about 800 videos with the signers balanced.
  - Cuts the final sign (না, নেই, ছিল, হবে, হয়েছে) by DTW against the same signer's video differing only in the
    last word, falling back to the motion cut. Clips under 0.3 s or over 2.5 s are dropped.
  - Writes `cuts.csv`, `word_quality.csv` and previews in `work/core_vocab_check/bansign9k/`.
  - 335 of the 800 videos say 1000 fps in the file (2 say 62.5) but have 30 fps worth of frames. The cut uses
    `bansign9k.fallback_fps` (30) for any rate outside `bansign9k.fps_range`.
  - Result: 794 of 800 clips kept, balanced over the 5 signers.
- New words are added to `data/core_vocab/word_list.csv` with `source BanSign9K` / `DynamicDU` and
  `verified = false`; না and নেই have category `negation`. Nothing is switched on.
- Shared files changed: `configs/config.yaml` (`bansign9k` and `dynamic_du` sections) and
  `data/core_vocab/word_list.csv` (rows added only).

## Module 3: sentence builder (18 Nov)

New package `signbangla/sign2text/` turns Module 1's sign words (dictionary forms, BdSL order) into a Bangla sentence
with rules: `build_sentence(words, is_question=False)` and `build_from_candidates(top5_per_sign)`.

- Scripts: `scripts/50_sentence_builder_eval.py` (50 test sentences -> `work/module3/eval/`) and
  `scripts/51_sentence_demo.py --words "আমি পানি চাওয়া"` (`--speak` uses `signbangla/sign2speech/speech.py` from the
  texttovoice branch once it is merged).
- Data: `data/module3/` (lexicon, verbs, predicates, test sentences), all rows `verified = false`. **Please check
  `verbs.csv` and `test_sentences.csv`** and set `verified` to `true` for rows that are right. `data/` is not in git,
  so these files are written from `signbangla/sign2text/tables.py` when missing; corrections that should be shared
  must also go into `tables.py`.
- Shared file changed: `configs/config.yaml` gets a `module3` section. No Module 1 or Module 2 code changed.
- On-screen demo (18 Nov): `python scripts\52_sentence_screen.py` (Words mode) or `--camera 0` (Camera mode with Module
  1's 401-word BiGRU, `work/bdslw401/results/results/bigru/best_model.pt`; this branch's `signbangla/models.py` loads it
  and gives the same predictions as the module-1 code, so no model code was copied). Sign segmentation thresholds are
  in `module3.screen.segment`, tuned on 60 recorded BdSLW60 videos (sign count exact on 73%, within one on 90%). Speak
  needs `signbangla/sign2speech` from the texttovoice branch.
- Rules from Ban-Sign-Sent-9K (18 Nov): `scripts/53_bansign9k_module3.py` splits the 1,922 sentence pairs (train /
  dev / test 70/15/15 by subject + food, `data/bansign9k/splits.csv`) and counts rules from train only
  (`data/module3/corpus_rules.json`). The rules cover:
  - kin compounds
  - the dropped verb খাওয়া
  - tense / negation from the final marker
  - the kin-subject person form
  - time-of-day locatives

  Dev 94.2% exact (ceiling 94.6%), test 90.9% (ceiling 91.2%, run once); the old 50 sentences stay at 47/50. New
  data files, all `verified = false`:
  - `data/module3/lexicon_extra.csv` (new words and spellings)
  - `data/module3/verb_tenses.csv` (present perfect খেয়েছে, habitual past খেতেন)

  **Please check both files.** The honorific choice (খান না but খায়নি for the same subject) is this dataset's style,
  not a grammar rule. Ban-Sign-Sent-9K (banglagov, Hugging Face) states no licence: research use only, never
  redistribute.
- Shared file changed: `configs/config.yaml` gets `module3.lexicon_extra`, `module3.verb_tenses` and
  `module3.corpus`. Tests: `tests/test_sentence_corpus.py`. All 202 tests pass.

## What you need to do

- **Python 3.11 is required.** Install it before creating the environment. `mediapipe==0.10.14` does not install on Python 3.13 or newer. Follow "Setup for Teammates" in `README.md`.
- **Rerun script 01.** `work/manifest.csv` holds absolute paths from the repo owner's PC, and scripts 02 and 05b read them. Once the videos are in `data/BdSLW60/`, run the command below. It rewrites `work/manifest.csv` with your paths.

  ```powershell
  python scripts/01_inspect_dataset.py
  ```

- **Train before using the live demo.** `work/results/` is neither in git nor in the zip, so there is no `best_model.pt` until you run `scripts/10_train_baseline.py`.
