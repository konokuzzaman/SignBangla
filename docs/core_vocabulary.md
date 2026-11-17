# Core vocabulary for Module 2 (Speech → Sign)

The 60 BdSLW60 words have no pronouns, verbs, question words, numbers or greetings, so sentences such as
"আমি ৩টা ট্যাবলেট খাব" could not be signed. The team has no BdSL signer, so the core vocabulary is built **only from
existing datasets**. Words with no sign are still recognised and shown as text.

## Word list

`signbangla/speech2sign/core_vocab.py` holds the list: 70 words (C001–C070) in 9 categories, plus 10 extra BdSL40 words
(C071–C080). `scripts/40_core_vocab_download.py` writes it to `data/core_vocab/word_list.csv` with the columns id,
bangla, english, category, source and verified. The IDs start with C, so they never clash with BdSLW word numbers. The
table is kept in code because `data/` is not in git.

## Sources

| Source | What it gives | Verified for BdSL? | Notes |
|---|---|---|---|
| BdSLW60 | 60 words | yes | unchanged; copied from `work/sign_dictionary` |
| BdSLW401 | 341 more words (অসুস্থ, জ্বর, ট্যাবলেট, মাথা ব্যাথা, ...) | yes | 3 trials per word chosen from 12 clips spread across signers |
| BdSL40 | আমি, তুমি, গতকাল, বেশি, + 10 extra words | yes, by the BdSL40 authors | see below |
| BDSL49 | digits ০–৯ | yes (BdSL dataset) | still photos turned into short signs |
| BDSL49, digit by digit | ১০, ২০, ১০০ and any other number | **no** | signing numbers digit by digit is our assumption |
| INCLUDE | সে, আমরা, তারা, ঠিক আছে, ধন্যবাদ, আজ, আগামীকাল, সকাল, রাত | **no** (Indian SL) | used only with `use_unverified_signs: true` |

**BdSL40.** BdSL40 is a set of INCLUDE (Indian Sign Language) videos whose signs its authors found to be the same in
BdSL, using the BdSL dictionary. Its video link on Google Drive is dead (404). The README says the dataset can be
rebuilt from INCLUDE using the authors' ISL-to-BdSL mapping, so we did that.
- Some BdSL40 labels are a different INCLUDE word. The video counts confirm the matches: "more" (বেশি) is INCLUDE
  "wide", "noon" is "Afternoon", and "glad" is "expensive".
- The mapping says 3 videos in INCLUDE's I folder are really "you", and 3 in the you folder are really "I". This is
  corrected (`INCLUDE_LABEL_FIXES`).
- The mapping lists "she" as different in BdSL (Daughter + pointing). সে therefore comes only from INCLUDE "he" and is
  unverified.

**Downloads.** Only the needed files are read out of the zips by byte range (`speech2sign/remote_zip.py`):
- INCLUDE / BdSL40: 8 videos per word, spread over the recording sessions (`videos_per_word`).
- BDSL49: 150 photos per digit.
- Total: about 2.5 GB, instead of 8+ GB of zips.
- Zenodo is slow per connection and answers "429 Too Many Requests" when there are too many requests. The download
  therefore fetches each file in one request and waits when it is told to.

**BDSL49 digits.** The recognition set's folders are numbered from 0, while the paper's Table 1 numbers labels from 1.
So ০–৯ are folders 36–45; the image counts per folder confirm this.

## Pipeline

```powershell
python scripts/40_core_vocab_download.py          # word list + downloads (safe to stop and rerun)
python scripts/41_core_vocab_extract.py           # keypoints for BdSL40 and INCLUDE videos
python scripts/42_digits_from_images.py           # digit signs from photos
python scripts/20_build_sign_dictionary.py --core # work/sign_dictionary_core + coverage report
python scripts/21_speech_to_sign_demo.py --text "আমি ৩টা ট্যাবলেট খাব"
```

- **Keypoints (41).** MediaPipe Holistic with exactly the settings of `scripts/03` (`keypoints.make_holistic`,
  `extract_video`), so the format matches BdSLW60. Each video is trimmed to its signing part: the frames where a wrist
  is above 70% of the way from the shoulders to the hips, plus 3 frames on each side. The pose model estimates the
  wrists even when the hand model loses a lowered hand. The checks of step 5 (`quality.trial_quality`) and an upright
  check run on the trimmed part.
- **Digits (42).** MediaPipe Hands runs in static-image mode on every photo.
  - Per digit, the most typical detection is kept: the medoid of all handshapes, which ignores odd photos. It is held at
    the median angle of all photos.
  - Left hands are mirrored to right hands.
  - The hand is sized to the median right-hand size in BdSLW60 and placed in front of the right shoulder.
  - The elbow is the median elbow of BdSLW60 frames with the wrist at that spot. A flat two-bone arm put the elbow
    inside the chest, because a raised forearm points at the camera.
  - Sign: rest pose → handshape → hold 0.5 s → rest. `work/core_vocab/BDSL49/best_hands.png` shows the chosen photos.
- **Dictionary (20 --core).** Preference order: BdSLW60, then the core sources, then BdSLW401. Each word keeps one
  signing hand: right-hand trials if there are any, never a mix. `dictionary.json` records source and verified per word.
  `digit_poses.npy` lets `SignDictionary` sign any number digit by digit.

## Text processing (`text_processing.WordMatcher.full()`)

- **Verb forms → base verb** for all 19 verbs (about 690 forms): খাব/খাবে/খাচ্ছি/খেয়েছি → খাওয়া, যাব/যাচ্ছি/গিয়েছি →
  যাওয়া, সাহায্য করব → সাহায্য করা. Forms that are other words are left out (দিন = day, নেই, চাবি).
- **Numbers.** Bangla or ASCII digits with or without a counter (৩টা, 25, ১০জন), and number words (তিন, দশ, একশো).
  নয় is read as negation, not nine.
- **Case endings** when a word is not found as written (বাবার → বাবা, বাড়িতে → বাড়ি).
- **Question words, negation and synonyms** (কি → কী, নাই → নেই, মাথা ব্যথা → মাথা ব্যাথা).
- **Dropped words.** The copula (আছ, আছে, হয়) is dropped quietly; BdSL usually leaves it out.
- **Words without a sign** stay in the caption, coloured by category: pronoun blue, verb purple, question orange,
  negation magenta, answer green, number teal, time brown, greeting olive. Unknown words are red. The window also lists
  them under "চিহ্ন নেই".

## Non-manual markers — an approximation

`signbangla/speech2sign/nonmanual.py` adds head and eyebrow movement:
- **Question** (ends with "?" or has a question word): a slight forward head tilt and raised eyebrows over the last 2
  signs.
- **Negation** (না / নেই / নয়): a side-to-side head shake over the last sign.

**This is an approximation made without a BdSL signer.** The movements, their size (0.05 / 0.06 shoulder widths) and
their timing (2.5 Hz shake, 0.25 s fades) are our estimate, not measured from BdSL recordings.
- It changes a copy of the face keypoints at render time only. Dictionary files are never changed.
- Only the figure and neon styles show it, because the skeleton style has no head or eyebrows.
- Settings: `speech2sign.nonmanual`.

## Limitations

- The BdSL40 and INCLUDE signs come from Indian signers. BdSL40's "same sign" judgement is by its authors from the BdSL
  dictionary, and no BdSL signer has checked our choice of videos or trimming.
- Signing numbers digit by digit is an assumption.
- The INCLUDE videos are filmed differently from BdSLW60 (1920×1080 at 25 fps, standing signers). They go through the
  same normalisation, but signing space and speed may differ.
- 53 of the 70 words still have no sign (44 with INCLUDE): no verbs, question words, negation, most time words or
  greetings. See `work/sign_dictionary_core/core_coverage.csv`.

## Result (9 Nov 2026)

| | Words with a sign | Verified for BdSL |
|---|---|---|
| Core list, default (INCLUDE off) | 17 of 70: আমি, তুমি, গতকাল, বেশি (BdSL40), ০–৯ (BDSL49), ১০, ২০, ১০০ (digit by digit) | 14 |
| Core list, `use_unverified_signs: true` | 26 of 70: + সে, আমরা, তারা, ঠিক আছে, ধন্যবাদ, আজ, আগামীকাল, সকাল, রাত | 14 |
| Extra BdSL40 words | 10: সময়, দুপুর, বন্ধু, শিক্ষক, ছাত্র, টাকা, বধির, নতুন, খারাপ, খুশি | 10 |
| Whole dictionary (default) | 428 words: BdSLW60 60, BdSLW401 341, BdSL40 14, BDSL49 10 + 3 | |

By category (default): pronoun 2/8, verb 0/19, question 0/7, negation 0/2, answer 0/2, number 13/13, time 1/8,
greeting 0/4, extra 1/7.

- **Downloads:** BdSL40 112 videos (1.3 GB), INCLUDE 72 videos (0.9 GB), BDSL49 1,500 photos (190 MB).
- **Quality checks:** BdSL40 111/112 videos pass and INCLUDE 70/72; the failures were `no_hand`.
- **Digits:** detection worked in 49–141 of 150 photos per digit.
- **BdSLW401:** all 341 words not in BdSLW60 got signs, 20 of them with fewer than 3 trials.
