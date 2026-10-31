"""Optional (Week 3): write a short mp4 that draws the extracted keypoints on the ORIGINAL video frames of one trial.
Use it to confirm that the annotation frame numbers line up (frame_base / end_inclusive in the config).

    python scripts/05b_preview_trial.py U4W1F_t000
"""
import argparse

import _path  # noqa: F401
import numpy as np
import pandas as pd

from signbangla.utils import load_config, subdir, work_dir
from signbangla.viz import write_preview_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial_id")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    wd = work_dir(cfg)
    meta = pd.read_csv(wd / "trials_metadata.csv").set_index("trial_id").loc[args.trial_id]
    man = pd.read_csv(wd / "manifest.csv").set_index("stem").loc[meta["video_stem"]]
    raw = np.load(subdir(cfg, "trials") / f"{args.trial_id}.npy")
    out = subdir(cfg, "quality_plots") / f"preview_{args.trial_id}.mp4"
    write_preview_video(man["path"], raw, int(meta["lo"]), out,
                        rotate_180=meta["video_stem"] in (cfg["video"].get("rotate_180") or []))
    print("written", out)


if __name__ == "__main__":
    main()
