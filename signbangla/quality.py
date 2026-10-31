"""Week 3: per-trial quality statistics on RAW trial keypoints (NaN = missing)."""
import numpy as np

from .keypoints import group_valid
from .preprocess import detect_dominant_hand


def trial_quality(raw, qcfg):
    """Return a dict of statistics, a list of flag strings, and ok (True when no flag is raised)."""
    T = raw.shape[0]
    v = group_valid(raw)
    pose_frac = float(v["pose"].mean()) if T else 0.0
    lh_frac = float(v["lh"].mean()) if T else 0.0
    rh_frac = float(v["rh"].mean()) if T else 0.0
    best = max(lh_frac, rh_frac)
    dom = detect_dominant_hand(np.nan_to_num(raw, nan=0.0)) if T >= 2 else "none"

    flags = []
    if T < qcfg["min_frames"]:
        flags.append("too_short")
    if pose_frac < qcfg["min_pose_frac"]:
        flags.append("low_pose")
    if best < qcfg["min_best_hand_frac"]:
        flags.append("no_hand")
    return dict(n_frames=T, pose_frac=pose_frac, lh_frac=lh_frac, rh_frac=rh_frac,
                best_hand_frac=best, dominant_hand=dom, flags="|".join(flags), ok=not flags)
