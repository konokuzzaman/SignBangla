# SignBangla data handover

`signbangla_handover.zip` holds the generated data of the project (keypoints, reports, dictionaries, word lists). It
contains no videos, no model checkpoints, and not the keypoints you already have (`work/raw_keypoints/`,
`work/bdslw401/raw_keypoints/`). It was made by `scripts/tools/make_handover_zip.py`, where the full list of exclusions
is kept.

**Keep this zip private.** Do not upload it anywhere public and do not pass it on (see the licences below).

## What is in it

| Folder | Contents |
|---|---|
| `work/core_vocab/BdSL300/` | BdSL-300 words not already in BdSLW60/BdSLW401: `keypoints/raw/` (whole clip) and `keypoints/trials/` (signing part), `overlap.csv` (which folders are new, duplicate or ambiguous), `quality_report*.csv`, `word_quality*.csv`, logs. |
| `work/core_vocab/BdSL40/`, `INCLUDE/`, `BDSL49/` | Core-vocabulary keypoints: BdSL40 words, INCLUDE (Indian SL, unverified) and the BDSL49 digit handshapes (`digit_০.npy` ... `digit_৯.npy`, Bangla file names). |
| `work/core_vocab_check/` | Contact sheets (`.png`) to check the chosen signs by eye. The preview videos are not included; rebuild them with scripts 45, 47b and 49. |
| `work/bansign9k/` | Ban-Sign-Sent-9K: `keypoints/raw/` (794 + partner sentence videos), `keypoints/last_sign/` (the last sign of each sentence cut out: না, নেই, ছিল, হবে, হয়েছে), `cuts.csv` (cut and method per clip), `word_quality.csv`, `subset.csv`, video index and logs. |
| `work/dynamic_du/` | DU dynamic signs, 14 classes (greetings and short phrases): `keypoints/raw/` and `keypoints/trials/`, `quality_report.csv`, `class_table.csv` (Bangla per class, `needs_check = true`), file list and logs. |
| `work/bdslw401/eval_full/` | BdSLW401 full held-out set: `processed/` (15,770 clips of signers 1 and 11 = val, 4 and 8 = test), `manifest.csv` (split, quality filter and drop reason per clip), `predictions.csv` and the evaluation scripts. |
| `work/bdslw401/` (other) | Orientation check and extraction logs. |
| `work/processed/`, `work/trials/`, `work/*.csv` | BdSLW60 trials, processed trials, splits and quality reports (the same as in `signbangla_keypoints.zip`, possibly newer). |
| `work/sign_dictionary/`, `work/sign_dictionary_core/` | Module 2 sign dictionaries (60 words and 428-word core dictionary). |
| `work/module3/` | Module 3 evaluations: `eval/` (50 test sentences), `eval_bansign9k/` and `eval_bansign9k_v2/` (Ban-Sign-Sent-9K with the corpus rules; train/dev/test). |
| `data/module3/` | Module 3 tables: lexicon, verbs, predicates, verb tenses, corpus rules, test sentences. |
| `data/bansign9k/` | Ban-Sign-Sent-9K sentence pairs (`sentences.csv`), vocabulary, word timings, typos and the Module 3 train/dev/test `splits.csv`. The original `.xlsx` is not included because it contains personal data. |
| `data/core_vocab/word_list.csv` | Core word list with the source and `verified` flag of every sign. |
| `data/BdSLW401/word_list.csv` | The BdSLW401 word list (word number, Bangla, English). |

Also not included: `work/results/`, `work/bdslw401/results/` and `results.zip` (model checkpoints), all videos, the
Ban-Sign-Sent-9K video archive, and caches.

## Verified and unverified words

Most new words have **not** been checked by someone who knows BdSL. In `data/core_vocab/word_list.csv`:

| Source | Verified |
|---|---|
| BdSL40, BDSL49 digits | yes (24 rows) |
| BdSL-300 | false (204 rows) |
| Ban-Sign-Sent-9K (S001-S009) | false |
| DU dynamic (D001-D014) | false |
| INCLUDE, BDSL49 digit by digit | no |

**Only rows with verified = yes/true may be used for training or in the sign dictionary.** Do not train on the
unverified rows until a BdSL signer has checked them. The DU class names and the Ban-Sign-Sent-9K cut-outs also need
that check.

## Licences

- **BdSLW60, BdSLW401:** non-commercial, not to be redistributed (BdSLW401 is CC BY-NC-ND 4.0).
- **BdSL-300:** CC BY-SA 4.0. Credit both BdSL-300 and SignBD-Word; anything made from it must be shared under the same
  licence.
- **Ban-Sign-Sent-9K** and the **DU dynamic dataset:** no licence stated, so research use only. Credit the authors.
- Keep the zip private.

## How to install

Unzip into a **temporary** folder, never directly into the project, then copy only files you do not have yet, so no
existing file is overwritten.

Windows (replace `<temp>` and `<project>`):

```
robocopy <temp>\work <project>\work /E /XC /XN /XO
robocopy <temp>\data <project>\data /E /XC /XN /XO
```

Mac / Linux:

```
rsync -a --ignore-existing <temp>/work/ <project>/work/
rsync -a --ignore-existing <temp>/data/ <project>/data/
```

Then delete the temporary folder. Use a zip tool that reads UTF-8 names (Windows Explorer, 7-Zip, `unzip` 6.0 or
Python's `zipfile`) so the Bangla folder and file names come out correctly.
