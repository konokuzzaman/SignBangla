"""Turn one raw trial (T, 67, 3, NaN = missing) into a fixed (64, 67, 3) training sequence.

Steps (in this order):
  1. aspect fix     x and z are multiplied by width/height so x and y share one unit.
  2. fill missing   short gaps of a missing hand / body are interpolated; long gaps stay absent.
  3. centre + scale origin = midpoint of the shoulders, unit = shoulder width (median over the trial).
  4. resample       linear interpolation in time to `target_frames` frames (works for any frame rate).
  5. absent = 0     body parts that are absent are exactly 0 (a hand that never appears stays all-zero).
"""
import numpy as np

from .keypoints import GROUPS, L_SHOULDER, N_KP, R_SHOULDER, group_valid


class PreprocessError(Exception):
    pass


def fill_gaps(x, valid, max_gap):
    """Interpolate runs of missing frames that are at most `max_gap` long (edge runs copy the nearest frame)."""
    T = len(valid)
    out, v2 = x.copy(), valid.copy()
    if not valid.any():
        return out, v2
    t = 0
    while t < T:
        if valid[t]:
            t += 1
            continue
        s = t
        while t < T and not valid[t]:
            t += 1
        e = t  # missing run is [s, e)
        if e - s <= max_gap:
            left = s - 1 if s > 0 else None
            right = e if e < T else None
            for tt in range(s, e):
                if left is None:
                    out[tt] = x[right]
                elif right is None:
                    out[tt] = x[left]
                else:
                    w = (tt - left) / (right - left)
                    out[tt] = (1 - w) * x[left] + w * x[right]
            v2[s:e] = True
    return out, v2


def resample_time(seq, pos):
    """Resample a sequence (T, K, 3) at fractional frame positions `pos`.

    A landmark counts as absent when it is exactly 0. Between two present frames we interpolate; next to an
    absent frame we copy the present one; an output frame is absent if its nearest input frame is absent
    (so absent hands never get blended towards 0).
    """
    T = seq.shape[0]
    pos = np.clip(np.asarray(pos, dtype=np.float64), 0, T - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (pos - i0)[:, None, None].astype(np.float32)
    present = np.abs(seq).sum(-1) > 0  # (T, K)
    a0, a1 = seq[i0], seq[i1]
    p0, p1 = present[i0][..., None], present[i1][..., None]
    lin = (1 - w) * a0 + w * a1
    val = np.where(p0 & p1, lin, np.where(p0, a0, a1))
    nearest_present = np.where(w < 0.5, p0, p1)
    return np.where(nearest_present, val, 0.0).astype(np.float32)


def preprocess_trial(raw, width, height, pcfg):
    """raw (T, 67, 3) with NaN -> (processed (N, 67, 3) float32, info dict). Raises PreprocessError if unusable."""
    if raw.ndim != 3 or raw.shape[1:] != (N_KP, 3):
        raise PreprocessError(f"bad shape {raw.shape}")
    T = raw.shape[0]
    if T < 2:
        raise PreprocessError("fewer than 2 frames")
    n_out = int(pcfg["target_frames"])
    x = raw.astype(np.float32).copy()

    # 1. aspect fix
    aspect = (width / height) if (width and height) else 1.0
    x[..., 0] *= aspect
    x[..., 2] *= aspect

    # 2. fill missing parts, group by group
    valid = group_valid(x)
    for name, sl in GROUPS.items():
        v = valid[name]
        if v.sum() < int(pcfg["min_valid_frames"]):
            valid[name] = np.zeros(T, bool)
            continue
        gap = pcfg["max_gap_pose"] if name == "pose" else pcfg["max_gap_hand"]
        x[:, sl], valid[name] = fill_gaps(x[:, sl], v, gap)

    # 3. centre on the shoulders, scale by shoulder width
    pv = valid["pose"]
    if not pv.any():
        raise PreprocessError("no body detected")
    pose = x[pv][:, :25]
    ls, rs = pose[:, L_SHOULDER], pose[:, R_SHOULDER]
    center = np.median((ls + rs) / 2, axis=0)
    scale = float(np.median(np.linalg.norm((ls - rs)[:, :2], axis=1)))
    if not np.isfinite(scale) or scale < 1e-3:
        raise PreprocessError(f"shoulder width too small ({scale:.5f})")
    x[:, GROUPS["pose"]] = (x[:, GROUPS["pose"]] - center) / scale  # pose z is relative to the hips -> re-centre it too
    for name in ("lh", "rh"):
        hand = x[:, GROUPS[name]]
        hand[..., :2] = (hand[..., :2] - center[:2]) / scale
        hand[..., 2] = hand[..., 2] / scale  # hand z is relative to the wrist: scale only
        x[:, GROUPS[name]] = hand

    # absent parts -> exactly 0 (so resample_time can tell present from absent)
    for name, sl in GROUPS.items():
        x[~valid[name], sl] = 0.0
    x = np.nan_to_num(x, nan=0.0)

    # 4. resample to a fixed number of frames
    out = resample_time(x, np.linspace(0, T - 1, n_out))

    info = dict(
        n_frames_raw=T,
        pose_frac=float((np.abs(out[:, GROUPS["pose"]]).sum((1, 2)) > 0).mean()),
        lh_frac=float((np.abs(out[:, GROUPS["lh"]]).sum((1, 2)) > 0).mean()),
        rh_frac=float((np.abs(out[:, GROUPS["rh"]]).sum((1, 2)) > 0).mean()),
        shoulder_width_norm=scale,
    )
    return out, info


# ----------------------------------------------------------------------------- hand dominance
def detect_dominant_hand(seq, min_frac=0.2):
    """'R', 'L' or 'none' from a sequence where absent parts are 0: the hand that is seen and moves more."""
    stats = {}
    for name in ("lh", "rh"):
        a = seq[:, GROUPS[name]]
        present = np.abs(a).sum((1, 2)) > 0
        pts = a[:, [0, 8, 12], :2]  # wrist, index tip, middle tip
        step = np.linalg.norm(np.diff(pts, axis=0), axis=-1).sum(-1)
        both = present[1:] & present[:-1]
        stats[name] = (present.mean(), float(step[both].sum()))
    cand = {n: m for n, (f, m) in stats.items() if f >= min_frac}
    if not cand:
        return "none"
    best = max(cand, key=cand.get)
    return "L" if best == "lh" else "R"


def _mirror_permutation():
    perm = np.arange(N_KP)
    for a, b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                 (17, 18), (19, 20), (21, 22), (23, 24)]:
        perm[a], perm[b] = b, a
    lh, rh = GROUPS["lh"], GROUPS["rh"]
    perm[lh] = np.arange(rh.start, rh.stop)
    perm[rh] = np.arange(lh.start, lh.stop)
    return perm


_MIRROR_PERM = _mirror_permutation()


def mirror_sequence(seq):
    """Mirror left <-> right (for a processed sequence centred at the shoulders). Optional experiment only."""
    out = seq[:, _MIRROR_PERM].copy()
    out[..., 0] *= -1
    return out
