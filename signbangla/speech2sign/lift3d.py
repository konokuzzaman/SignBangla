"""Lift a stitched 2D sign sequence to 3D for the studio3d viewer.

The keypoints are front-view MediaPipe coordinates (x < 0 is the signer's right, y down, shoulder-width units). Their z
is too noisy to use directly, so arm depth comes from foreshortening instead: a bone that looks shorter than its full
length must point towards or away from the camera. MediaPipe's z only decides which of the two.

Output axes (three.js): x right, y up, z towards the camera. The body stays in the plane z = 0. This works on a copy;
the input is never changed.
"""
import numpy as np
from scipy.signal import savgol_filter

N_POINTS = 67
LEFT_HAND = slice(25, 46)
RIGHT_HAND = slice(46, 67)
# (shoulder, elbow, wrist, hand points) per arm
ARMS = ((11, 13, 15, LEFT_HAND), (12, 14, 16, RIGHT_HAND))

DEFAULTS = dict(
    smooth_window=21,  # Savitzky-Golay window for the depth curves (frames)
    smooth_order=2,
    bone_percentile=97,  # a bone's full length = this percentile of its 2D length in the sentence ...
    bone_margin=1.03,  # ... times this
    sign_gain=3.0,  # how quickly MediaPipe's z difference decides towards / away from the camera
    min_sign=0.3,  # the depth never fully vanishes when MediaPipe's z is undecided
    finger_depth_scale=1.6,  # finger depth relative to the wrist, from the hand landmarks' z
    wrist_front_min=0.12,  # the wrist stays at least this far in front of the body plane (shoulder widths)
    hand_front_min=0.05,  # every hand point stays at least this far in front of it
)


def _missing(points):
    """True per frame where a group of points is absent: all zeros or any NaN."""
    return np.isnan(points).any(axis=(-2, -1)) | (np.nan_to_num(points) == 0).all(axis=(-2, -1))


def fill_missing(seq, rest):
    """Copy of `seq` with absent hands replaced by the rest-pose hand and absent body points by the rest-pose point."""
    seq = np.array(seq, np.float32, copy=True)
    rest = np.asarray(rest, np.float32)
    for sl in (LEFT_HAND, RIGHT_HAND):
        gone = _missing(seq[:, sl])
        seq[gone, sl] = rest[sl]
    body = seq[:, :25]
    gone = np.isnan(body).any(-1) | (np.nan_to_num(body) == 0).all(-1)
    body[gone] = np.broadcast_to(rest[:25], body.shape)[gone]
    return np.nan_to_num(seq)


def _smooth(x, window, order):
    """Savitzky-Golay along time, with the window shrunk to fit short sequences."""
    n = len(x)
    window = min(window, n if n % 2 else n - 1)
    if window <= order:
        return x
    return savgol_filter(x, window, order, axis=0)


def _depth(parent_xy, child_xy, parent_z, child_z, o):
    """Depth of a child joint relative to its parent, from how short the bone looks."""
    d = np.linalg.norm(child_xy - parent_xy, axis=1)
    full = np.percentile(d, o["bone_percentile"]) * o["bone_margin"]
    size = np.sqrt(np.clip(full * full - d * d, 0, None))
    sign = np.tanh(-(child_z - parent_z) * o["sign_gain"])  # MediaPipe: negative z = towards the camera
    sign = np.where(np.abs(sign) < o["min_sign"], np.copysign(o["min_sign"], sign + 1e-9), sign)
    return _smooth(size * sign, o["smooth_window"], o["smooth_order"])


def lift_to_3d(seq, rest, **options):
    """(T, 67, 3) stitched sequence -> (T, 67, 3) in three.js axes (x right, y up, z towards the camera).

    - elbow and wrist depth from bone-length foreshortening, smoothed over time
    - the wrist is put at hand landmark 0, so the forearm meets the hand
    - finger depth from the hand landmarks, relative to the wrist
    - hands are kept in front of the body
    - an absent hand (all zeros or NaN) becomes the rest-pose hand
    `options` override DEFAULTS (speech2sign.studio3d.lift in the config)."""
    o = dict(DEFAULTS, **{k: v for k, v in options.items() if v is not None})
    unknown = set(o) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown lift options: {sorted(unknown)}")
    p = fill_missing(seq, rest)
    if p.ndim != 3 or p.shape[1:] != (N_POINTS, 3):
        raise ValueError(f"expected (T, {N_POINTS}, 3), got {p.shape}")
    out = np.zeros_like(p)
    out[..., 0] = p[..., 0]
    out[..., 1] = -p[..., 1]
    for shoulder, elbow, wrist, hand in ARMS:
        h = p[:, hand]
        elbow_z = _depth(p[:, shoulder, :2], p[:, elbow, :2], p[:, shoulder, 2], p[:, elbow, 2], o)
        out[:, elbow, 2] = elbow_z
        hand_xy = h[:, 0, :2]
        wrist_z = elbow_z + _depth(p[:, elbow, :2], hand_xy, p[:, elbow, 2], p[:, wrist, 2], o)
        wrist_z = np.maximum(wrist_z, o["wrist_front_min"])
        out[:, wrist, 0] = hand_xy[:, 0]
        out[:, wrist, 1] = -hand_xy[:, 1]
        out[:, wrist, 2] = wrist_z
        finger_z = wrist_z[:, None] - (h[..., 2] - h[:, :1, 2]) * o["finger_depth_scale"]
        out[:, hand, 2] = np.maximum(finger_z, o["hand_front_min"])
    return out.astype(np.float32)
