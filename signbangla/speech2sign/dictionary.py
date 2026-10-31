"""The sign dictionary: for each of the 60 words, the best few trials as clean 30 fps keypoint sequences.

Layout of work/sign_dictionary/ (folder name in configs/config.yaml -> speech2sign):
    W<number>.npy     the chosen trials of one word joined in time, shape (sum of frames, 67, 3), absent parts = 0.
                      Trials differ in length, so they are joined rather than stacked; dictionary.json gives the
                      frame count of each, and SignDictionary.sequence() cuts them apart again.
    rest_pose.npy     (67, 3) hands-down pose, from the frames before the first sign in each video.
    dictionary.json   Bangla (spelling variants split), romanized, English, chosen trial ids and frame counts.
Words are W-numbers in file names because Bangla and romanized labels can contain "/" and spaces.
"""
import json
import unicodedata

import numpy as np
import pandas as pd

from ..keypoints import GROUPS, L_SHOULDER, N_KP, POSE, R_SHOULDER
from ..labels import LABELS
from ..preprocess import PreprocessError, preprocess_trial
from ..utils import atomic_save_npy, atomic_write_json, progress, subdir, work_dir

L_WRIST, R_WRIST, L_HIP, R_HIP = 15, 16, 23, 24
HAND_WRIST = {"lh": L_WRIST, "rh": R_WRIST}  # hand group -> its wrist in the pose group
REST_SKIP_LAST = 5  # frames right before the first sign: the hands are already rising
REST_MAX_FRAMES = 120
REST_MIN_FRAMES = 10


def split_variants(label):
    """'বাবা/আব্বা' -> ['বাবা', 'আব্বা']."""
    return [p.strip() for p in label.split("/") if p.strip()]


def normalize_word(text):
    """Bangla letters like য় have two Unicode spellings; NFC makes typed or ASR text compare equal to our labels."""
    return unicodedata.normalize("NFC", text).strip()


def group_present(seq, group):
    """(T, 67, 3) -> bool (T,): this body part has data in that frame (absent parts are exactly 0)."""
    return np.abs(seq[:, GROUPS[group]]).sum((1, 2)) > 0


def is_upright(seq):
    """False when hips are above the shoulders or the sides are swapped: a safety check, in case a video is stored
    upside down and is missing from config video.rotate_180."""
    have = group_present(seq, "pose")
    if not have.any():
        return False
    m = np.median(seq[have][:, POSE], axis=0)
    return bool(m[[L_HIP, R_HIP], 1].mean() > 0 and m[L_SHOULDER, 0] > m[R_SHOULDER, 0])


def trial_to_fps(raw, fps, width, height, pcfg, target_fps):
    """Raw trial -> normalized sequence at target_fps, keeping its real duration. Same centring, scaling and gap
    filling as training (preprocess_trial); only the output length differs."""
    n = max(2, int(round((len(raw) - 1) / fps * target_fps)) + 1)
    seq, _ = preprocess_trial(raw, width, height, dict(pcfg, target_frames=n))
    return seq


def rank_trials(rows, hand_weight=0.5):
    """Best first. RightHand trials whose detected dominant hand agrees come first, then other RightHand trials, then
    LeftHand. Inside a tier: duration near the word's median (seconds, because frame counts depend on the video's
    fps), plus a penalty for frames where the signing hand was not detected."""
    d = rows.copy()
    d["duration_s"] = d["n_frames"] / d["fps"]
    median = d["duration_s"].median()
    right = d["hand_annot"] == "RightHand"
    d["tier"] = np.where(right & (d["dominant_hand"] == "R"), 0, np.where(right, 1, 2))
    hand_frac = np.where(right, d["rh_frac"], d["lh_frac"])
    d["score"] = (d["duration_s"] - median).abs() / median + hand_weight * (1 - hand_frac)
    return d.sort_values(["tier", "score", "trial_id"])


def choose_word_trials(rows, tdir, pcfg, s2s):
    """Walk the ranked trials of one word and keep the first `n_trials` that preprocess cleanly and are upright."""
    chosen = []
    for r in rank_trials(rows, s2s["hand_weight"]).itertuples():
        try:
            seq = trial_to_fps(np.load(tdir / f"{r.trial_id}.npy"), r.fps, r.width, r.height, pcfg, s2s["fps"])
        except PreprocessError:
            continue
        if is_upright(seq):
            chosen.append((r, seq))
        if len(chosen) == s2s["n_trials"]:
            break
    return chosen


def build_rest_pose(q, raw_dir, pcfg):
    """Median pose before the first sign of every upright video. MediaPipe often loses lowered hands, so the hand
    shape comes only from frames with a detected hand and a lowered wrist, re-attached to the rest wrist."""
    poses, shapes = [], {"lh": [], "rh": []}
    for stem, d in progress(list(q.groupby("video_stem")), desc="rest pose"):
        stop = int(d["start"].min()) - REST_SKIP_LAST
        if stop < REST_MIN_FRAMES:
            continue
        r = d.iloc[0]
        raw = np.array(np.load(raw_dir / f"{stem}.npy", mmap_mode="r")[max(0, stop - REST_MAX_FRAMES):stop])
        try:
            seq, _ = preprocess_trial(raw, r["width"], r["height"], dict(pcfg, target_frames=len(raw)))
        except PreprocessError:
            continue
        if not is_upright(seq):
            continue
        have = group_present(seq, "pose")
        pose = np.median(seq[have], axis=0)
        poses.append(pose[POSE])
        for name, wrist in HAND_WRIST.items():
            low = group_present(seq, name) & have & (seq[:, wrist, 1] > 0.75 * pose[[L_HIP, R_HIP], 1].mean())
            if low.any():
                hand = seq[low][:, GROUPS[name]]
                shapes[name].append(np.median(hand - hand[:, :1] * [1, 1, 0], axis=0))
    if not poses:
        raise ValueError("No rest frames found: every video starts signing right away, or none is upright.")
    rest = np.zeros((N_KP, 3), np.float32)
    rest[POSE] = np.median(poses, axis=0)
    for name, wrist in HAND_WRIST.items():
        if shapes[name]:
            hand = np.median(shapes[name], axis=0)
            hand[:, :2] += rest[wrist, :2]
            rest[GROUPS[name]] = hand
    return rest, dict(videos=len(poses), lh_shapes=len(shapes["lh"]), rh_shapes=len(shapes["rh"]))


def build_dictionary(cfg):
    """quality_report.csv + work/trials -> work/sign_dictionary/. Returns the dictionary.json content."""
    s2s, pcfg = cfg["speech2sign"], cfg["preprocess"]
    q = pd.read_csv(work_dir(cfg) / "quality_report.csv")
    tdir, raw_dir, out = subdir(cfg, "trials"), subdir(cfg, "raw_keypoints"), subdir(cfg, s2s["dictionary_dir"])
    ok = q[q["ok"].astype(bool)]
    rest, rest_info = build_rest_pose(q, raw_dir, pcfg)
    atomic_save_npy(out / "rest_pose.npy", rest)

    words = {}
    for i, (number, bangla, roman, english) in enumerate(progress(LABELS, desc="choosing trials")):
        word = f"W{number}"
        rows = ok[ok["word"] == word]
        chosen = choose_word_trials(rows, tdir, pcfg, s2s) if len(rows) else []
        if not chosen:
            continue
        atomic_save_npy(out / f"{word}.npy", np.concatenate([seq for _, seq in chosen]))
        words[word] = dict(
            class_idx=i, bangla=split_variants(bangla), romanized=split_variants(roman), english=english,
            file=f"{word}.npy", median_duration_s=round(float((rows["n_frames"] / rows["fps"]).median()), 3),
            trials=[dict(trial_id=r.trial_id, signer=int(r.signer), hand=r.hand_annot, source_fps=round(float(r.fps), 2),
                         source_frames=int(r.n_frames), duration_s=round(float(r.duration_s), 3), frames=len(seq))
                    for r, seq in chosen])
    info = dict(fps=s2s["fps"], n_keypoints=N_KP, rest_pose="rest_pose.npy", rest_info=rest_info, words=words)
    atomic_write_json(out / "dictionary.json", info)
    return info


class SignDictionary:
    """Look words up by Bangla (any spelling variant), romanized, English or W-number, and load their sequences."""

    def __init__(self, folder):
        self.folder = folder
        self.info = json.loads((folder / "dictionary.json").read_text(encoding="utf-8"))
        self.words = self.info["words"]
        self.fps = self.info["fps"]
        self.rest_pose = np.load(folder / self.info["rest_pose"])
        self._index = {}
        for word, e in self.words.items():
            for name in (word, e["english"], *e["romanized"], *e["bangla"]):
                self._index[normalize_word(name).lower()] = word

    @classmethod
    def from_config(cls, cfg):
        return cls(work_dir(cfg) / cfg["speech2sign"]["dictionary_dir"])

    def find(self, name):
        """Any known name -> 'W20'. KeyError if the word is not in the vocabulary."""
        key = normalize_word(name).lower()
        if key not in self._index:
            raise KeyError(f"{name!r} is not in the {len(self.words)}-word sign dictionary")
        return self._index[key]

    def tokenize(self, text):
        """Split text into dictionary words, longest match first, because two labels contain a space (অডিও ক্যাসেট)."""
        parts, out, i = normalize_word(text).split(), [], 0
        while i < len(parts):
            for j in range(len(parts), i, -1):
                if " ".join(parts[i:j]).lower() in self._index:
                    out.append(" ".join(parts[i:j]))
                    i = j
                    break
            else:
                raise KeyError(f"{parts[i]!r} is not in the {len(self.words)}-word sign dictionary")
        return out

    def sequence(self, name, variant=0):
        """Trial number `variant` of a word as (T, 67, 3)."""
        e = self.words[self.find(name)]
        frames = [t["frames"] for t in e["trials"]]
        start = sum(frames[:variant])
        return np.load(self.folder / e["file"], mmap_mode="r")[start:start + frames[variant]].copy()
