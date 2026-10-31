"""Join sign sequences into one: rest pose -> sign -> eased transition -> sign ... -> rest pose.

The number of transition frames and the easing curve are plain parameters, so they can be compared in experiments.
Everything here works on normalized sequences (T, 67, 3) where absent parts are exactly 0.
"""
from dataclasses import dataclass

import numpy as np

from ..keypoints import GROUPS
from .dictionary import HAND_WRIST, group_present

EASINGS = {
    "linear": lambda t: t,
    "cosine": lambda t: 0.5 - 0.5 * np.cos(np.pi * t),
    "cubic": lambda t: 4 * t ** 3 if t < 0.5 else 1 - 0.5 * (2 - 2 * t) ** 3,
    "minimum_jerk": lambda t: t ** 3 * (10 - 15 * t + 6 * t ** 2),
}


def get_easing(easing):
    """A name from EASINGS, or any function mapping [0, 1] to [0, 1]."""
    if callable(easing):
        return easing
    if easing not in EASINGS:
        raise ValueError(f"unknown easing {easing!r}; choose from {sorted(EASINGS)} or pass a function")
    return EASINGS[easing]


def frame_motion(seq):
    """How much the arms and hands move at each frame, (T,). Smoothed a little so one jittery frame does not count."""
    T = len(seq)
    speed = np.zeros(T)
    if T < 2:
        return speed
    wrists = list(HAND_WRIST.values())
    speed[1:] = np.linalg.norm(np.diff(seq[:, wrists, :2], axis=0), axis=-1).max(-1)
    for name in HAND_WRIST:
        have = group_present(seq, name)
        step = np.linalg.norm(np.diff(seq[:, GROUPS[name], :2], axis=0), axis=-1).mean(-1)
        speed[1:] = np.maximum(speed[1:], np.where(have[1:] & have[:-1], step, 0.0))
    return np.convolve(speed, np.ones(3) / 3, mode="same")


def trim_idle(seq, rel_speed=0.10, min_speed=0.005, margin=1, min_frames=8, max_frac=0.15):
    """Cut the still frames at the start and end of a sign. A frame is idle below `rel_speed` x the sign's peak speed
    (or `min_speed`, in shoulder widths per frame). At most `max_frac` of the frames goes from each end, because a still
    hand can be the sign itself: মা rises to the chin and holds. Signs that would end up shorter than `min_frames` are
    kept whole."""
    speed = frame_motion(seq)
    moving = np.flatnonzero(speed > max(min_speed, rel_speed * speed.max()))
    if not len(moving):
        return seq
    T, most = len(seq), int(max_frac * len(seq))
    lo, hi = min(max(moving[0] - margin, 0), most), max(min(moving[-1] + 1 + margin, T), T - most)
    return seq[lo:hi] if hi - lo >= min_frames else seq


def fill_absent_hands(seq, rest):
    """Attach a hand to the wrist wherever none was detected, so hands do not pop in and out of the animation: the
    shape from the nearest frame of this sign that has one, else the rest pose's hand."""
    out = seq.copy()
    for name, wrist in HAND_WRIST.items():
        sl = GROUPS[name]
        have, pose_ok = group_present(seq, name), np.abs(seq[:, wrist]).sum(-1) > 0
        known = np.flatnonzero(have)
        if not len(known) and not np.abs(rest[sl]).sum():
            continue
        for t in np.flatnonzero(~have & pose_ok):
            src = seq[known[np.abs(known - t).argmin()]] if len(known) else rest
            out[t, sl] = src[sl]
            out[t, sl, :2] += seq[t, wrist, :2] - src[wrist, :2]
    return out


def _blend_frame(a, b, w):
    """One in-between frame, a -> b at weight w. A part present on only one side is never blended towards 0: the
    pose holds still, a hand keeps its shape and follows the blended wrist."""
    out = (1 - w) * a + w * b
    pose = GROUPS["pose"]
    a_pose, b_pose = np.abs(a[pose]).sum() > 0, np.abs(b[pose]).sum() > 0
    if a_pose != b_pose:
        out[pose] = a[pose] if a_pose else b[pose]
    for name, wrist in HAND_WRIST.items():
        sl = GROUPS[name]
        a_hand, b_hand = np.abs(a[sl]).sum() > 0, np.abs(b[sl]).sum() > 0
        if a_hand != b_hand:
            src = a if a_hand else b
            out[sl] = src[sl]
            out[sl, :2] += out[wrist, :2] - src[wrist, :2]
    return out


def transition(a, b, n, easing="cosine"):
    """`n` frames strictly between frames a and b, (n, 67, 3). n = 0 gives an empty array (a hard cut)."""
    ease = get_easing(easing)
    frames = [_blend_frame(a, b, float(ease((i + 1) / (n + 1)))) for i in range(n)]
    return np.stack(frames).astype(np.float32) if frames else np.zeros((0, *a.shape), np.float32)


@dataclass
class Stitched:
    seq: np.ndarray  # (T, 67, 3)
    segments: list  # dict(kind 'rest' | 'transition' | 'sign', label, start, end), end exclusive


def stitch(signs, rest, labels=None, n_transition=12, easing="cosine", n_rest=15, n_rest_transition=None,
           trim=True, fill_hands=True, idle_rel_speed=0.10, idle_min_speed=0.005, idle_margin=1, min_sign_frames=8,
           idle_max_frac=0.15):
    """Join a list of sign sequences. Starts and ends exactly in `rest`, held for `n_rest` frames.

    n_transition       frames between two signs (0 = hard cut)
    easing             name from EASINGS or a function
    n_rest_transition  frames between the rest pose and a sign (default: same as n_transition)
    """
    if not len(signs):
        raise ValueError("nothing to stitch: the word list is empty")
    labels = list(labels) if labels is not None else [str(i) for i in range(len(signs))]
    if len(labels) != len(signs):
        raise ValueError(f"{len(labels)} labels for {len(signs)} signs")
    if n_transition < 0 or n_rest < 1:
        raise ValueError("n_transition must be >= 0 and n_rest >= 1")
    n_edge = n_transition if n_rest_transition is None else n_rest_transition
    rest = np.asarray(rest, np.float32)

    parts, segments = [], []

    def add(kind, label, frames):
        start = segments[-1]["end"] if segments else 0
        segments.append(dict(kind=kind, label=label, start=start, end=start + len(frames)))
        parts.append(frames)

    add("rest", None, np.repeat(rest[None], n_rest, axis=0))
    prev = rest
    for i, (sign, label) in enumerate(zip(signs, labels)):
        s = np.asarray(sign, np.float32)
        if trim:
            s = trim_idle(s, idle_rel_speed, idle_min_speed, idle_margin, min_sign_frames, idle_max_frac)
        if fill_hands:
            s = fill_absent_hands(s, rest)
        add("transition", None, transition(prev, s[0], n_edge if i == 0 else n_transition, easing))
        add("sign", label, s)
        prev = s[-1]
    add("transition", None, transition(prev, rest, n_edge, easing))
    add("rest", None, np.repeat(rest[None], n_rest, axis=0))
    return Stitched(np.concatenate(parts), segments)


def stitch_words(words, dictionary, variant=0, **kw):
    """Stitch words by name (Bangla, romanized, English or W-number) from a SignDictionary."""
    signs = [dictionary.sequence(w, variant) for w in words]
    return stitch(signs, dictionary.rest_pose, labels=words, **kw)


def frame_captions(stitched, full_text):
    """One caption per frame: the full text in the rest pose, the current word during and right after its sign."""
    caps, current = [], full_text
    for seg in stitched.segments:
        if seg["kind"] == "sign":
            current = seg["label"]
        caps += [full_text if seg["kind"] == "rest" else current] * (seg["end"] - seg["start"])
    return caps
