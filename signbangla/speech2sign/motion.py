"""Smooth motion for every drawing style: clean each sign, retarget it onto one fixed skeleton, join the signs in
joint-angle space, and sample the result at any frame rate and playback speed.

Steps (settings in configs/config.yaml -> speech2sign.motion):
1. Glitches: a joint that sits far from the median of its 5-frame neighbourhood (a one- or two-frame spike that comes
   back) is replaced by interpolation from the frames around it. Fast real changes, such as a handshape switching and
   staying, are not touched, because the median follows them.
2. Smoothing: Savitzky-Golay over time (zero-phase, so nothing lags). The hand shape is smoothed relative to its wrist
   with a short window, so quick handshape changes stay sharp while the wrist path loses its jitter.
3. Retargeting: every bone becomes a unit direction. The keypoints are rebuilt from those directions and bone lengths
   that never change (the medians over the whole sign dictionary), so limbs cannot stretch. Directions use depth
   scaled by pose_z_scale / hand_z_scale: those values keep the 3D lengths steadiest across frames, so a forearm
   pointing at the camera still looks short on screen.
4. Joining: rest pose -> transition -> sign -> hold -> transition -> ... -> rest. Transitions slerp every bone
   direction with minimum-jerk easing over 8-12 frames (more for longer moves) and bend the arms slightly toward the
   camera halfway, so the hands travel in front of the body.
5. Sampling: modified Akima ("makima") cubic curves through all values, so frames at 60 fps, or at 0.5x speed, are
   smooth. Unlike a plain cubic spline they do not overshoot: a held pose stays exactly still.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter

from ..keypoints import GROUPS, N_KP
from .stitch import EASINGS, fill_absent_hands, trim_idle

L_SH, R_SH, L_HIP, R_HIP, L_EAR, R_EAR = 11, 12, 23, 24, 7, 8
FACE = list(range(11))  # nose, eyes, ears, mouth
ARM_BONES = [(11, 13), (13, 15), (12, 14), (14, 16)]  # left upper, left fore, right upper, right fore
POSE_HAND_BONES = [(15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22)]
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
HANDS = ("lh", "rh")
DIRECTION_FIELDS = ("across", "torso", "neck", "arm", "phand", "hand")
KINDS = ("rest", "transition", "sign", "hold")


# ------------------------------------------------------------------ 1-2. cleaning
def _present(seq, group):
    return np.abs(seq[:, GROUPS[group]]).sum((1, 2)) > 0


def _spike_frames(x, thresh, k):
    """x (n, J, 3) -> bool (n,): frames where some joint is far from its 5-frame median (Hampel test)."""
    n = len(x)
    if n < 5:
        return np.zeros(n, bool)
    med = np.stack([np.median(x[max(0, i - 2):i + 3], axis=0) for i in range(n)])
    r = np.linalg.norm((x - med)[..., :2], axis=-1)  # (n, J), in the image plane where depth noise cannot add
    scale = 1.4826 * np.median(r, axis=0)
    return np.any(r > np.maximum(thresh, k * scale), axis=1)


def _fill_frames(x, bad):
    """Replace the bad frames of x (n, ...) by linear interpolation from the good frames on either side."""
    good = np.flatnonzero(~bad)
    if not len(good) or not bad.any():
        return x
    out = x.copy()
    flat = out.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        flat[bad, c] = np.interp(np.flatnonzero(bad), good, flat[good, c])
    return out


def remove_glitches(seq, pose_threshold=0.15, hand_threshold=0.1, k=4.0):
    """Returns (cleaned copy, number of frames fixed). Each group (pose, left hand, right hand) is tested on its own."""
    out = np.array(seq, np.float32)
    fixed = 0
    for group, thresh in (("pose", pose_threshold), ("lh", hand_threshold), ("rh", hand_threshold)):
        idx = np.flatnonzero(_present(out, group))
        part = out[idx][:, GROUPS[group]]
        bad = _spike_frames(part, thresh, k)
        if bad.any():
            out[idx[:, None], np.arange(N_KP)[GROUPS[group]][None, :]] = _fill_frames(part, bad)
            fixed += int(bad.sum())
    return out, fixed


def _savgol_runs(x, present, window, order):
    """Savitzky-Golay along time over each run of present frames. Runs shorter than the window get a shorter window."""
    out = x.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, present.astype(int), 0]))
    for start, end in zip(edges[::2], edges[1::2]):
        w = min(window, (end - start) if (end - start) % 2 else (end - start) - 1)
        if w > order + 1:
            out[start:end] = savgol_filter(x[start:end], w, order, axis=0, mode="interp")
    return out


def smooth(seq, pose_window=7, hand_window=5, order=3):
    """Zero-phase smoothing. Hands: wrist path with pose_window, finger shape relative to the wrist with hand_window."""
    out = np.array(seq, np.float32)
    pose = _present(out, "pose")
    out[:, GROUPS["pose"]] = _savgol_runs(out[:, GROUPS["pose"]], pose, pose_window, order)
    for group in HANDS:
        on = _present(out, group)
        hand = out[:, GROUPS[group]]
        wrist = _savgol_runs(hand[:, :1], on, pose_window, order)
        shape = _savgol_runs(hand - hand[:, :1], on, hand_window, order)
        out[:, GROUPS[group]] = np.where(on[:, None, None], wrist + shape, 0)
    return out


def clean_sign(seq, m):
    seq, fixed = remove_glitches(seq, m["glitch_pose_threshold"], m["glitch_hand_threshold"], m["glitch_k"])
    return smooth(seq, m["smooth_pose_window"], m["smooth_hand_window"], m["smooth_order"]), fixed


# ------------------------------------------------------------------ 3. retargeting
def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-9), 0)


def _scaled(p, zs):
    q = np.array(p, np.float64)
    q[..., 2] *= zs
    return q


def to_state(seq, pose_z=0.3, hand_z=1.0):
    """Keypoints (T, 67, 3) -> joint directions and a few positions. Directions are unit vectors in depth-scaled space."""
    P = _scaled(seq[:, :25], pose_z)
    root = (P[:, L_SH] + P[:, R_SH]) / 2
    head = (P[:, L_EAR] + P[:, R_EAR]) / 2
    state = dict(
        root=root,
        across=_unit(P[:, L_SH] - P[:, R_SH]),
        torso=_unit((P[:, L_HIP] + P[:, R_HIP]) / 2 - root),
        neck=_unit(head - root),
        face=P[:, FACE] - head[:, None],
        arm=np.stack([_unit(P[:, b] - P[:, a]) for a, b in ARM_BONES], axis=1),
        phand=np.stack([_unit(P[:, b] - P[:, a]) for a, b in POSE_HAND_BONES], axis=1),
    )
    hands, on = [], []
    for group in HANDS:
        H = _scaled(seq[:, GROUPS[group]], hand_z)
        hands.append(np.stack([_unit(H[:, b] - H[:, a]) for a, b in HAND_BONES], axis=1))
        on.append(_present(seq, group).astype(np.float64))
    state["hand"] = np.stack(hands, axis=1)  # (T, 2, 20, 3)
    state["hand_on"] = np.stack(on, axis=1)  # (T, 2)
    return state


def fit_lengths(seqs, pose_z=0.3, hand_z=1.0):
    """Median bone lengths (depth-scaled) over many sequences: the one skeleton every sign is drawn on."""
    parts = {k: [] for k in ("half", "hip_half", "torso", "neck", "arm", "phand", "hand")}
    for seq in seqs:
        on = np.abs(seq[:, :25]).sum((1, 2)) > 0
        P = _scaled(seq[on, :25], pose_z)
        if not len(P):
            continue
        root = (P[:, L_SH] + P[:, R_SH]) / 2
        parts["half"].append(np.linalg.norm(P[:, L_SH] - P[:, R_SH], axis=-1) / 2)
        parts["hip_half"].append(np.linalg.norm(P[:, L_HIP] - P[:, R_HIP], axis=-1) / 2)
        parts["torso"].append(np.linalg.norm((P[:, L_HIP] + P[:, R_HIP]) / 2 - root, axis=-1))
        parts["neck"].append(np.linalg.norm((P[:, L_EAR] + P[:, R_EAR]) / 2 - root, axis=-1))
        parts["arm"].append(np.stack([np.linalg.norm(P[:, b] - P[:, a], axis=-1) for a, b in ARM_BONES], 1))
        parts["phand"].append(np.stack([np.linalg.norm(P[:, b] - P[:, a], axis=-1) for a, b in POSE_HAND_BONES], 1))
        for group in HANDS:
            H = _scaled(seq[:, GROUPS[group]], hand_z)
            Hon = H[np.abs(seq[:, GROUPS[group]]).sum((1, 2)) > 0]
            if len(Hon):
                parts["hand"].append(np.stack([np.linalg.norm(Hon[:, b] - Hon[:, a], axis=-1) for a, b in HAND_BONES], 1))
    med = {k: np.median(np.concatenate(v), axis=0) for k, v in parts.items() if v}
    arm = med["arm"]
    phand = med["phand"]
    # one symmetric skeleton: the same length for both sides
    med["arm"] = np.array([(arm[0] + arm[2]) / 2, (arm[1] + arm[3]) / 2] * 2)
    med["phand"] = np.r_[(phand[:3] + phand[3:]) / 2, (phand[:3] + phand[3:]) / 2]
    return {k: np.asarray(v, np.float64) for k, v in med.items()}


def from_state(state, lengths, pose_z=0.3, hand_z=1.0):
    """Joint directions -> keypoints (T, 67, 3) on the fixed skeleton. Absent hands come back as zeros."""
    T = len(state["root"])
    out = np.zeros((T, N_KP, 3))
    root, across = state["root"], state["across"]
    P = np.zeros((T, 25, 3))
    P[:, L_SH] = root + lengths["half"] * across
    P[:, R_SH] = root - lengths["half"] * across
    hips = root + lengths["torso"] * state["torso"]
    P[:, L_HIP] = hips + lengths["hip_half"] * across
    P[:, R_HIP] = hips - lengths["hip_half"] * across
    head = root + lengths["neck"] * state["neck"]
    P[:, FACE] = head[:, None] + state["face"]
    for (a, b), d, length in zip(ARM_BONES, np.moveaxis(state["arm"], 1, 0), lengths["arm"]):
        P[:, b] = P[:, a] + length * d
    for (a, b), d, length in zip(POSE_HAND_BONES, np.moveaxis(state["phand"], 1, 0), lengths["phand"]):
        P[:, b] = P[:, a] + length * d
    P[..., 2] /= pose_z
    out[:, :25] = P
    for h, (group, wrist) in enumerate(zip(HANDS, (15, 16))):
        H = np.zeros((T, 21, 3))
        H[:, 0] = _scaled(P[:, wrist], hand_z)
        for (a, b), d, length in zip(HAND_BONES, np.moveaxis(state["hand"][:, h], 1, 0), lengths["hand"]):
            H[:, b] = H[:, a] + length * d
        H[..., 2] /= hand_z
        on = state["hand_on"][:, h] > 0.5
        out[:, GROUPS[group]] = np.where(on[:, None, None], H, 0)
    return out.astype(np.float32)


# ------------------------------------------------------------------ 4. joining
def slerp(a, b, w):
    """Unit vectors a -> b at weight w (broadcasts). Falls back to normalized lerp for tiny or opposite angles."""
    dot = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1, 1)
    theta = np.arccos(dot)
    s = np.sin(theta)
    safe = s > 1e-6
    wa = np.where(safe, np.sin((1 - w) * theta) / np.where(safe, s, 1), 1 - w)
    wb = np.where(safe, np.sin(w * theta) / np.where(safe, s, 1), w)
    return _unit(wa * a + wb * b)


def _frame(state, i):
    """Frame i of a state (negative i counts from the end), still with a time axis of length 1."""
    n = len(state["root"])
    i %= n
    return {k: v[i:i + 1] for k, v in state.items()}


def blend_states(a, b, w, front=0.0):
    """One in-between state. `front` bends the arm bones toward the camera (negative z) by that much."""
    out = {}
    for key in a:
        if key in DIRECTION_FIELDS:
            out[key] = slerp(a[key], b[key], w)
        else:
            out[key] = (1 - w) * a[key] + w * b[key]
    if front:
        arm = out["arm"].copy()
        arm[..., 2] -= front
        out["arm"] = _unit(arm)
    # a hand present on one side only keeps that side's shape instead of fading through nothing
    for h in range(2):
        on_a, on_b = a["hand_on"][:, h] > 0.5, b["hand_on"][:, h] > 0.5
        if on_a.all() != on_b.all():
            src = a if on_a.all() else b
            out["hand"][:, h] = src["hand"][:, h]
            out["hand_on"][:, h] = 1
    return out


def transition_states(a, b, n, easing="minimum_jerk", front=0.3):
    """n in-between frames from state a (one frame) to state b (one frame)."""
    ease = EASINGS[easing]
    frames = [blend_states(a, b, float(ease((i + 1) / (n + 1))), front * np.sin(np.pi * (i + 1) / (n + 1)))
              for i in range(n)]
    return {k: np.concatenate([f[k] for f in frames]) for k in a} if frames else {k: v[:0] for k, v in a.items()}


def wrist_distance(a, b, lengths, pose_z, hand_z):
    pa, pb = from_state(a, lengths, pose_z, hand_z)[0], from_state(b, lengths, pose_z, hand_z)[0]
    return float(np.linalg.norm((pa[[15, 16]] - pb[[15, 16]])[:, :2], axis=-1).max())


def transition_length(distance, m):
    """More frames for longer moves: min_frames for none, max_frames at transition_ref_distance or more."""
    frac = float(np.clip(distance / m["transition_ref_distance"], 0, 1))
    return int(round(m["transition_min_frames"] + frac * (m["transition_max_frames"] - m["transition_min_frames"])))


# ------------------------------------------------------------------ 5. sampling
@dataclass
class Motion:
    """A whole sentence as continuous curves. Times are in seconds of normal (1x) playback."""
    state: dict
    kinds: np.ndarray  # (T,) index into KINDS, at the source frame rate
    words: np.ndarray  # (T,) which word is being signed (or prepared for), -1 at rest
    labels: list  # the words, as given
    lengths: dict
    fps: float
    pose_z: float
    hand_z: float
    fixed_frames: int = 0  # glitch frames replaced

    def __post_init__(self):
        T = len(self.kinds)
        self.times = np.arange(T) / self.fps
        self.duration = float(self.times[-1]) if T > 1 else 0.0
        self._curves = {k: Akima1DInterpolator(self.times, v.reshape(T, -1), axis=0, method="makima") if T > 2 else None
                        for k, v in self.state.items()}

    def state_at(self, t):
        t = np.clip(np.atleast_1d(np.asarray(t, float)), 0, self.duration)
        out = {}
        for k, v in self.state.items():
            curve = self._curves[k]
            vals = curve(t) if curve is not None else np.repeat(v.reshape(1, -1), len(t), 0)
            out[k] = vals.reshape((len(t),) + v.shape[1:])
        for k in DIRECTION_FIELDS:
            out[k] = _unit(out[k])
        return out

    def keypoints(self, t):
        """Keypoints (n, 67, 3) on the fixed skeleton at times t (seconds)."""
        return from_state(self.state_at(t), self.lengths, self.pose_z, self.hand_z)

    def index_at(self, t):
        return np.clip(np.round(np.atleast_1d(t) * self.fps).astype(int), 0, len(self.kinds) - 1)

    def sample_times(self, fps=60, speed=1.0):
        """Frame times for a video at `fps` played at `speed` (0.5 = half speed, twice as many frames)."""
        n = int(np.floor(self.duration * fps / speed)) + 1
        return np.arange(n) * speed / fps

    def captions(self, t):
        """One (words, current word index) per time, for render_frames: the whole sentence, current word highlighted."""
        words = tuple(self.labels)
        return [(words, int(k)) for k in self.word_at(t)]

    def word_at(self, t):
        return self.words[self.index_at(t)]

    def kind_at(self, t):
        return np.array(KINDS)[self.kinds[self.index_at(t)]]


def _concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _repeat(state, n):
    return {k: np.repeat(v[-1:], n, axis=0) for k, v in state.items()}


_LENGTH_CACHE = {}


def dictionary_lengths(dictionary, pose_z, hand_z):
    key = (str(dictionary.folder), pose_z, hand_z)
    if key not in _LENGTH_CACHE:
        seqs = [dictionary.sequence(w, v) for w in dictionary.words for v in range(len(dictionary.words[w]["trials"]))]
        _LENGTH_CACHE[key] = fit_lengths(seqs, pose_z, hand_z)
    return _LENGTH_CACHE[key]


def build_motion(words, dictionary, s2s, variant=0):
    """Words (any names the dictionary knows) -> Motion, using speech2sign.motion and speech2sign.stitch settings."""
    m, st = s2s["motion"], s2s["stitch"]
    pz, hz, fps = m["pose_z_scale"], m["hand_z_scale"], dictionary.fps
    lengths = dictionary_lengths(dictionary, pz, hz)
    rest = to_state(np.asarray(dictionary.rest_pose, np.float32)[None], pz, hz)
    parts, kinds, word_ids, fixed = [], [], [], 0

    def add(state, kind, word):
        parts.append(state)
        n = len(state["root"])
        kinds.extend([KINDS.index(kind)] * n)
        word_ids.extend([word] * n)

    add(_repeat(rest, int(round(m["rest_start_s"] * fps))), "rest", -1)
    prev = rest
    for k, word in enumerate(words):
        seq = dictionary.sequence(word, variant)
        seq = trim_idle(seq, st["idle_rel_speed"], st["idle_min_speed"], st["idle_margin"], st["min_sign_frames"],
                        st["idle_max_frac"])
        seq, n_fixed = clean_sign(fill_absent_hands(seq, dictionary.rest_pose), m)
        fixed += n_fixed
        sign = to_state(seq, pz, hz)
        first = _frame(sign, 0)
        n = transition_length(wrist_distance(_frame(prev, -1), first, lengths, pz, hz), m)
        add(transition_states(_frame(prev, -1), first, n, m["easing"], m["front_bias"]), "transition", k)
        add(sign, "sign", k)
        add(_repeat(sign, int(round(m["hold_s"] * fps))), "hold", k)
        prev = sign
    last = _frame(prev, -1)
    n = transition_length(wrist_distance(last, rest, lengths, pz, hz), m)
    add(transition_states(last, rest, n, m["easing"], m["front_bias"]), "transition", -1)
    add(_repeat(rest, int(round(m["sentence_pause_s"] * fps))), "rest", -1)
    return Motion(_concat(parts), np.array(kinds), np.array(word_ids), list(words), lengths, fps, pz, hz, fixed)


# ------------------------------------------------------------------ measuring
JOINTS_MEASURED = list(range(11, 23)) + list(range(25, 67))  # shoulders, arms, pose hand points, both hands


def motion_metrics(seq, fps):
    """Largest per-frame joint jump (shoulder widths), the same as a speed (shoulder widths per second), and the mean
    jerk (third time derivative, shoulder widths per second cubed) over joints present in every frame."""
    x = np.asarray(seq, np.float64)[:, JOINTS_MEASURED, :2]
    present = np.abs(x).sum(-1) > 0
    keep = present.all(axis=0)
    x = x[:, keep]
    step = np.linalg.norm(np.diff(x, axis=0), axis=-1)
    jerk = np.linalg.norm(np.diff(x, n=3, axis=0), axis=-1) * fps ** 3
    return dict(max_jump=float(step.max()), max_speed=float(step.max() * fps), mean_jerk=float(jerk.mean()),
                frames=len(x), fps=fps)
