"""BdSLW401 helpers: the clip list, which clips are stored upside down, and the extraction order.

Keypoints use signbangla/keypoints.py and the same config settings as BdSLW60, so both datasets have the same format.
Outputs go to work/<bdslw401.work_subdir>/, apart from the BdSLW60 outputs.
"""
import json
from collections import defaultdict
from pathlib import PurePosixPath

import numpy as np

from .keypoints import L_SHOULDER, POSE, R_SHOULDER
from .labels401 import parse_clip_name
from .utils import ROOT, subdir

INDEX_FILE = "front_index.json"  # written by scripts/30_bdslw401_download.py
L_HIP, R_HIP = 23, 24


def dataset_folder(cfg):
    return ROOT / cfg["bdslw401"]["root"]


def output_dir(cfg, name=None):
    sub = cfg["bdslw401"]["work_subdir"]
    return subdir(cfg, f"{sub}/{name}" if name else sub)


def list_clips(cfg):
    """All front-view clips as a dict: clip name without .mp4 -> full path. The zip's index is the list of what should
    exist, and it needs no slow folder scan."""
    folder = dataset_folder(cfg)
    path = folder / INDEX_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/30_bdslw401_download.py first.")
    files = json.loads(path.read_text(encoding="utf-8"))["files"]
    clips = {}
    for entry in files:
        name = PurePosixPath(entry["name"])
        if name.suffix.lower() != ".mp4":
            continue
        if name.stem in clips:
            raise SystemExit(f"Two clips are named {name.stem}: the outputs are keyed on the name.")
        clips[name.stem] = folder / entry["name"]
    return clips


def rotated_clips(stems, prefixes):
    """Clips to turn upright. A config entry is the start of a clip name: W351S17 is one signer's clips of one word, and
    W351S17F_03 is one clip."""
    prefixes = tuple(prefixes or ())
    return {s for s in stems if s.startswith(prefixes)} if prefixes else set()


def upright_vote(frame):
    """One raw frame (67, 3): True if the body is upright, False if upside down, None if the pose was not detected.
    Upside down means the hips are above the shoulders, or the signer's left shoulder is on the image's left."""
    body = frame[POSE]
    if np.isnan(body[:, 0]).any():
        return None
    shoulders_y, hips_y = body[[L_SHOULDER, R_SHOULDER], 1].mean(), body[[L_HIP, R_HIP], 1].mean()
    return bool(hips_y > shoulders_y and body[L_SHOULDER, 0] > body[R_SHOULDER, 0])


def priority_first(stems, per_word, exclude_signers):
    """All clip names, with up to `per_word` clips of each word first, so a dictionary can be built early.

    Those clips are spread across signers: signers take turns, each word starts at a different signer so the same ones
    are not always left out, and each signer gives the middle of their remaining clips. Excluded signers are never in
    this first group. The rest follow in name order. Every clip appears exactly once."""
    excluded = set(exclude_signers or ())
    by_word = defaultdict(lambda: defaultdict(list))
    for stem in sorted(stems):
        info = parse_clip_name(stem)
        if info is None:
            raise ValueError(f"not a BdSLW401 clip name: {stem}")
        if info["signer"] not in excluded:
            by_word[info["word_number"]][info["signer"]].append(stem)
    first = []
    for word in sorted(by_word):
        signers = by_word[word]
        order = sorted(signers)
        start = word % len(order)
        order = order[start:] + order[:start]
        chosen = []
        while len(chosen) < per_word and any(signers[s] for s in order):
            for signer in order:
                if signers[signer] and len(chosen) < per_word:
                    chosen.append(signers[signer].pop(len(signers[signer]) // 2))
        first += chosen
    picked = set(first)
    return first + [s for s in sorted(stems) if s not in picked]
