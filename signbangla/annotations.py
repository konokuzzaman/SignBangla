"""Read the BdSLW60 annotation JSON files and turn them into a flat list of trials.

BdSLW60 videos contain 10-15 repetitions of one sign; the JSON files mark where each repetition
starts and ends. We could not see the exact JSON layout when this was written, so the reader is
deliberately flexible: it walks the whole JSON, finds every object that has a start-frame key and an
end-frame key (matched by regex, see configs/config.yaml -> annotation), and remembers the video
name it belongs to. Run scripts/01_inspect_dataset.py to see what it found. If it finds 0 trials,
look at the printed JSON structure and adjust the regexes (or ask for help with the exact format).
"""
import json
import re
from pathlib import Path

from .labels import NAME_RE

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(key):
    return _NON_ALNUM.sub(" ", str(key).lower()).strip()


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num_list(v):
    return isinstance(v, list) and len(v) > 0 and all(_is_num(x) for x in v)


def _pick_key(obj, patterns, accept):
    """First key (in pattern priority order) whose normalised name matches and whose value passes `accept`."""
    keys = [(k, _norm(k)) for k in obj]
    for pat in patterns:
        rx = re.compile(pat)
        for k, nk in keys:
            if rx.search(nk) and accept(obj[k]):
                return k
    return None


def _video_stem(value):
    if isinstance(value, str):
        stem = Path(value.replace("\\", "/")).stem
        if NAME_RE.match(stem):
            return stem
    return None


_HAND_NAME = re.compile(r"^(left|right) ?hand$")


def _is_hand_name(v):
    """'LeftHand', 'right_hand', ... (BdSLW60 uses these as "Orientation" values and as dict keys)."""
    return isinstance(v, str) and bool(_HAND_NAME.match(_norm(v)))


def _hand_info(obj):
    for k, v in obj.items():
        nk = _norm(k)
        if isinstance(v, (str, int, float)) and ("hand" in nk or (nk == "orientation" and _is_hand_name(v))):
            return str(v)
    return None


def find_trials(obj, ann_cfg, video=None, out=None, hand=None):
    """Recursively collect trials -> list of dict(video, start, end, hand).

    The hand comes from the trial object itself if it has one, else from the nearest enclosing object
    (BdSLW60: {"Orientation": "LeftHand", ..., "trials": {...}}), else from a parent key like "LeftHand"."""
    out = [] if out is None else out
    sp, ep = ann_cfg["start_key_patterns"], ann_cfg["end_key_patterns"]
    if isinstance(obj, list):
        for item in obj:
            find_trials(item, ann_cfg, video, out, hand)
        return out
    if not isinstance(obj, dict):
        return out

    here = video
    for v in obj.values():  # a "video"/"file" field inside this object?
        s = _video_stem(v)
        if s:
            here = s
            break
    hand = _hand_info(obj) or hand

    # (a) one trial per object: {"start": 12, "end": 40}
    sk = _pick_key(obj, sp, _is_num)
    ek = _pick_key(obj, ep, _is_num)
    if sk is not None and ek is not None:
        out.append(dict(video=here, start=int(obj[sk]), end=int(obj[ek]), hand=hand))
        return out
    # (b) parallel lists: {"start_frames": [..], "end_frames": [..]}
    sk = _pick_key(obj, sp, _num_list)
    ek = _pick_key(obj, ep, _num_list)
    if sk is not None and ek is not None and len(obj[sk]) == len(obj[ek]):
        for s, e in zip(obj[sk], obj[ek]):
            out.append(dict(video=here, start=int(s), end=int(e), hand=hand))
        return out

    for k, v in obj.items():
        find_trials(v, ann_cfg, _video_stem(k) or here, out, k if _is_hand_name(k) else hand)
    return out


def load_annotation_file(path, ann_cfg):
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    return find_trials(data, ann_cfg)


def describe_json(obj, depth=3, indent=0, max_items=4):
    """Short text sketch of a JSON structure (for the inspect script)."""
    pad = "  " * indent
    lines = []
    if isinstance(obj, dict):
        lines.append(f"{pad}dict with {len(obj)} keys")
        if depth > 0:
            for k in list(obj)[:max_items]:
                lines.append(f"{pad}  key {k!r}:")
                lines += describe_json(obj[k], depth - 1, indent + 2, max_items)
            if len(obj) > max_items:
                lines.append(f"{pad}  ... ({len(obj) - max_items} more keys)")
    elif isinstance(obj, list):
        lines.append(f"{pad}list of {len(obj)}")
        if depth > 0 and obj:
            lines += describe_json(obj[0], depth - 1, indent + 1, max_items)
    else:
        lines.append(f"{pad}{type(obj).__name__}: {str(obj)[:60]!r}")
    return lines


def frames_slice(start, end, n_frames, ann_cfg):
    """Annotation frame numbers -> python slice bounds (lo, hi) on a clip with n_frames, or None if unusable."""
    base = int(ann_cfg.get("frame_base", 0))
    lo = start - base
    hi = end - base + (1 if ann_cfg.get("end_inclusive", True) else 0)
    lo, hi = max(lo, 0), min(hi, n_frames)
    if hi - lo < int(ann_cfg.get("min_trial_frames", 3)):
        return None
    return lo, hi
