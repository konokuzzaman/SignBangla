"""Cutting the last sign out of a whole-sentence video (raw (T, 67, 3) keypoints, image coordinates, NaN = missing).

Two methods:
- dtw: another video of the same signer whose sign sentence differs only in the last word. Both signing parts are
  aligned with dynamic time warping on the hand points; along the alignment the cost stays low while the two videos
  sign the same words and rises when they stop matching. The last sign starts where the final high-cost run begins.
  With several such partner videos the median start is used.
- motion: the final movement segment before the hands go back to rest. The signing part ends where the wrists drop
  below the rest line; the last sign starts at the pause (lowest hand speed) nearest to where an average-length last
  sign would start.
The signing part is where a wrist is raised above its own resting height in that video (signers in this data rest
their hands on the lap, well above the hips, so a fixed line between shoulders and hips does not work).
"""
import numpy as np

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16
LH, RH = slice(25, 46), slice(46, 67)


def signing_span(raw, raise_frac=0.1, margin=3, rest_quantile=0.9):
    """(start, end) frames of the signing part (end exclusive), or None. A wrist's resting height is the
    `rest_quantile` quantile of its y over the video (y grows downwards); a frame is signing when a wrist is more than
    raise_frac x the shoulder-to-hip distance above its resting height. The pose wrist is used, or the hand model's
    wrist where the pose has none. The span runs from the first to the last signing frame, plus `margin` frames."""
    raw = np.asarray(raw, np.float32)
    sh = np.nanmedian(raw[:, [L_SHOULDER, R_SHOULDER], 1]) if not np.isnan(raw[:, [L_SHOULDER, R_SHOULDER], 1]).all()         else np.nan
    hips = raw[:, [23, 24], 1]
    hip = np.nanmedian(hips) if not np.isnan(hips).all() else np.nan
    if not np.isfinite(sh) or not np.isfinite(hip) or hip <= sh:
        return None
    y = raw[:, [L_WRIST, R_WRIST], 1].copy()
    for k, h in enumerate((LH.start, RH.start)):
        miss = np.isnan(y[:, k])
        y[miss, k] = raw[miss, h, 1]
    active = np.zeros(len(raw), bool)
    for k in range(2):
        ok = ~np.isnan(y[:, k])
        if ok.sum() < 2:
            continue
        rest = np.quantile(y[ok, k], rest_quantile)
        active[ok] |= y[ok, k] < rest - raise_frac * (hip - sh)
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return None
    return max(0, int(idx[0]) - margin), min(len(raw), int(idx[-1]) + 1 + margin)


def _fill(x):
    """Fill NaN rows by linear interpolation in time (edges: nearest value). x: (T, D); all-NaN columns -> 0."""
    x = x.copy()
    t = np.arange(len(x))
    for d in range(x.shape[1]):
        ok = ~np.isnan(x[:, d])
        if ok.any():
            x[~ok, d] = np.interp(t[~ok], t[ok], x[ok, d])
        else:
            x[:, d] = 0.0
    return x


def hand_features(raw):
    """(T, 84): both hands' 21 points (x, y) relative to the shoulder centre, in shoulder widths. A hand the hand model
    missed is put at its pose wrist (all 21 points), so a lowered hand still has a position. Shoulders are taken as the
    median over the video, so body sway does not change the features."""
    raw = np.asarray(raw, np.float32)
    sh = raw[:, [L_SHOULDER, R_SHOULDER], :2]
    ok = ~np.isnan(sh).any((1, 2))
    if not ok.any():
        return None
    centre = np.median(sh[ok].mean(1), axis=0)
    width = float(np.median(np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=1)))
    if width < 1e-6:
        return None
    feats = []
    for hand, wrist in ((LH, L_WRIST), (RH, R_WRIST)):
        pts = raw[:, hand, :2].copy()
        miss = np.isnan(pts).any((1, 2))
        pts[miss] = raw[miss, wrist, None, :2]
        feats.append(((pts - centre) / width).reshape(len(raw), -1))
    return _fill(np.concatenate(feats, 1))


def dtw(a, b):
    """Dynamic time warping of (n, D) and (m, D). -> (path [(i, j)], local cost of each path step)."""
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).mean(-1))
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row, prev = acc[i], acc[i - 1]
        c = cost[i - 1]
        for j in range(1, m + 1):
            row[j] = c[j - 1] + min(prev[j - 1], prev[j], row[j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = int(np.argmin((acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])))
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    path.reverse()
    return path, np.array([cost[p] for p in path])


def _smooth(x, w):
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")


def dtw_start(a, b, factor=2.0, min_cost=0.15, smooth=5, end_tolerance=8):
    """Frame of `a` where it stops matching `b` (both: signing parts, (T, D) features), or None when there is no clear
    low-cost shared part followed by a high-cost end. The threshold is `factor` x the median cost over the first half
    of the alignment, at least `min_cost` (shoulder widths). The last `end_tolerance` steps may be low again (both
    videos lowering the hands)."""
    path, c = dtw(a, b)
    if len(path) < 4:
        return None
    c = _smooth(c, smooth)
    base = float(np.median(c[: max(2, len(c) // 2)]))
    high = c > max(min_cost, factor * base)
    idx = np.flatnonzero(high)
    if len(idx) == 0 or idx[-1] < len(c) - 1 - end_tolerance:
        return None  # the ends match: the last signs look the same
    k = int(idx[-1])
    while k > 0 and high[k - 1]:
        k -= 1
    if k == 0:
        return None  # nothing matches at all
    return path[k][0]


def wrist_speed(feats, fps, smooth=5):
    """Fastest hand's speed per frame (shoulder widths per second), smoothed. feats from hand_features."""
    w = feats.reshape(len(feats), 2, 21, 2)[:, :, 0]  # the two wrists
    v = np.zeros(len(feats))
    if len(feats) > 1:
        v[1:] = np.linalg.norm(np.diff(w, axis=0), axis=-1).max(1) * fps
        v[0] = v[1]
    return _smooth(v, smooth)


def motion_start(feats, fps, span, n_signs, min_s=0.3, max_s=2.5, weight=0.5, smooth=5):
    """Start frame of the last sign inside span (start, end): the local minimum of hand speed in the allowed window,
    scored by how slow it is (relative to the median speed of the signing part) and how close it is to where an
    average-length last sign would start ((end - start) / n_signs before the end). None if the window is empty."""
    s, e = span
    v = wrist_speed(feats, fps, smooth)
    lo, hi = max(s + 1, int(round(e - max_s * fps))), int(round(e - min_s * fps))
    if hi <= lo:
        return None
    prior = (e - s) / max(1, n_signs)
    med = float(np.median(v[s:e])) + 1e-6
    cand = [t for t in range(lo, hi + 1) if v[t] <= v[t - 1] and (t + 1 >= len(v) or v[t] <= v[t + 1])]
    if not cand:
        return int(round(min(hi, max(lo, e - prior))))
    score = [v[t] / med + weight * abs((e - t) - prior) / max(prior, 1.0) for t in cand]
    return int(cand[int(np.argmin(score))])


def cut_last_sign(raw, fps, n_signs, partners=(), settings=None):
    """-> dict(start, end, method, seconds, ok, reason, partners_used). start/end are frames of raw (end exclusive).
    partners: raw keypoints of the same signer's videos whose sign sentences differ only in the last word.
    settings: raise_frac, margin_frames, dtw_factor, dtw_min_cost, smooth, snap_back_s, min_s, max_s, motion_weight."""
    s = dict(raise_frac=0.1, margin_frames=3, dtw_factor=2.0, dtw_min_cost=0.15, smooth=5, snap_back_s=0.3, min_s=0.3,
             max_s=2.5, motion_weight=0.5)
    s.update(settings or {})
    out = dict(start=-1, end=-1, method="", seconds=0.0, ok=False, reason="", partners_used=0)
    span = signing_span(raw, s["raise_frac"], s["margin_frames"])
    feats = hand_features(raw)
    if span is None or feats is None:
        out["reason"] = "no signing found"
        return out
    start, end = span
    starts = []
    for p in partners:
        pspan, pf = signing_span(p, s["raise_frac"], s["margin_frames"]), hand_features(p)
        if pspan is None or pf is None:
            continue
        k = dtw_start(feats[start:end], pf[pspan[0]:pspan[1]], s["dtw_factor"], s["dtw_min_cost"], s["smooth"],
                      s["smooth"] + 2 * s["margin_frames"])
        if k is not None:
            starts.append(start + k)
    if starts:
        k = int(np.median(starts))
        back = int(round(s["snap_back_s"] * fps))
        if back > 0 and k > start:  # the alignment notices the change a little late: go back to the pause before it
            v = wrist_speed(feats, fps, s["smooth"])
            lo = max(start, k - back)
            k = lo + int(np.argmin(v[lo:k + 1]))
        out.update(start=k, method="dtw", partners_used=len(starts))
    else:
        k = motion_start(feats, fps, span, n_signs, s["min_s"], s["max_s"], s["motion_weight"], s["smooth"])
        if k is None:
            out["reason"] = "signing part too short"
            return out
        out.update(start=k, method="motion")
    out["end"] = end
    out["seconds"] = round((end - out["start"]) / fps, 3)
    if out["seconds"] < s["min_s"]:
        out["reason"] = f"shorter than {s['min_s']} s"
    elif out["seconds"] > s["max_s"]:
        out["reason"] = f"longer than {s['max_s']} s"
    else:
        out["ok"] = True
    return out
