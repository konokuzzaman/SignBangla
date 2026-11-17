"""Join sign sequences into one: rest pose -> sign -> eased transition -> sign ... -> rest pose.

sentence_motion is what the demos use for every drawing style: hands filled from the rest pose, Savitzky-Golay
smoothing, minimum-jerk transitions, a short hold after each sign, and 60 fps (settings: speech2sign.sentence).
stitch / stitch_words are the older joining with idle trimming and adjustable easing (scripts/20_m2_demo.py).
Everything here works on normalized sequences (T, 67, 3) where absent parts are exactly 0.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

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


# ---------------------------------------------------------------- sentence motion (every drawing style)
SENTENCE_KINDS = ("rest", "transition", "sign", "hold")


def minimum_jerk(a, b, n):
    """n frames from a to b, both included, with minimum-jerk easing (10s^3 - 15s^4 + 6s^5)."""
    s = np.linspace(0, 1, n)
    w = (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)[:, None, None]
    return a[None] * (1 - w) + b[None] * w


@dataclass
class Sentence:
    """A whole sentence at the output frame rate, with what happens in every frame."""
    seq: np.ndarray  # (T, 67, 3)
    kinds: np.ndarray  # (T,) index into SENTENCE_KINDS
    words: np.ndarray  # (T,) word being signed or moved to; -1 at rest and on the way back
    labels: list
    fps: float

    def captions(self, layout=None):
        """One caption per frame for render_frames: the sentence with the current word highlighted.

        layout  None: the caption is the signed words. Or (tokens, colours, positions): the caption is all `tokens`,
                including words without a sign, each with its own colour (None = usual); positions[k] is the token of
                sign k."""
        if layout is None:
            return [(tuple(self.labels), int(k)) for k in self.words]
        tokens, colors, positions = layout
        return [(tuple(tokens), positions[k] if k >= 0 else -1, tuple(colors)) for k in self.words]

    def retimed(self, speed):
        """The same sentence played at `speed` (0.5 = half speed, twice the frames), with smooth in-between frames."""
        if speed == 1.0 or len(self.seq) < 2:
            return self
        t = np.arange(len(self.seq))
        new_t = np.arange(0, len(self.seq) - 1 + 1e-9, speed)
        seq = CubicSpline(t, self.seq, axis=0)(new_t).astype(np.float32)
        idx = np.clip(np.round(new_t).astype(int), 0, len(self.seq) - 1)
        return Sentence(seq, self.kinds[idx], self.words[idx], self.labels, self.fps)


def sentence_motion(words, dictionary, variant=0, savgol_window=7, savgol_order=2, transition_frames=16, hold_frames=8,
                    rest_start_frames=10, rest_end_frames=12, upsample=2):
    """Words -> Sentence, the same motion for every style. Each sign (variant `variant`) gets its missing hands from the
    rest pose and Savitzky-Golay smoothing; the signs are joined rest -> transition -> sign -> hold -> transition ...
    -> rest with minimum-jerk transitions; a cubic spline then raises the 30 fps dictionary to `upsample` x the rate."""
    rest = np.asarray(dictionary.rest_pose, np.float32)
    parts, kinds, word_ids = [], [], []

    def add(frames, kind, word):
        parts.append(np.asarray(frames, np.float32))
        kinds.extend([SENTENCE_KINDS.index(kind)] * len(frames))
        word_ids.extend([word] * len(frames))

    add(np.repeat(rest[None], rest_start_frames, 0), "rest", -1)
    prev = rest
    for k, word in enumerate(words):
        sig = np.array(dictionary.sequence(word, variant), np.float32)
        for group in ("lh", "rh"):
            sl = GROUPS[group]
            missing = np.abs(sig[:, sl]).sum((1, 2)) == 0
            sig[missing, sl] = rest[sl]
        if len(sig) >= savgol_window:
            sig = savgol_filter(sig, savgol_window, savgol_order, axis=0).astype(np.float32)
        add(minimum_jerk(prev, sig[0], transition_frames)[1:], "transition", k)
        add(sig, "sign", k)
        add(np.repeat(sig[-1:], hold_frames, 0), "hold", k)
        prev = sig[-1]
    add(minimum_jerk(prev, rest, transition_frames)[1:], "transition", -1)
    add(np.repeat(rest[None], rest_end_frames, 0), "rest", -1)
    seq = np.concatenate(parts)
    kinds, word_ids = np.array(kinds), np.array(word_ids)
    if upsample > 1 and len(seq) > 1:
        t = np.arange(len(seq))
        fine = np.linspace(0, len(seq) - 1, len(seq) * upsample - (upsample - 1))
        seq = CubicSpline(t, seq, axis=0)(fine).astype(np.float32)
        idx = np.clip(np.round(fine).astype(int), 0, len(kinds) - 1)
        kinds, word_ids = kinds[idx], word_ids[idx]
    return Sentence(seq, kinds, word_ids, list(words), dictionary.fps * upsample)
