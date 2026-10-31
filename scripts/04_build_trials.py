"""WEEK 2 (Member 1): cut every extracted video into individual sign repetitions ("trials") with the annotation JSON.

    python scripts/04_build_trials.py
    python scripts/04_build_trials.py --whole-video      # only if each video file is already ONE sign (no annotations)

Output: work/trials/<video>_tNNN.npy and work/trials_metadata.csv. Compare the printed trial count and length
statistics with the paper (9307 trials, 9-164 frames, mean about 44).
"""
import argparse

import _path  # noqa: F401
import pandas as pd

from signbangla import pipeline
from signbangla.utils import load_config, work_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--whole-video", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)

    man = work_dir(cfg) / "manifest.csv"
    manifest = pd.read_csv(man) if man.exists() else pipeline.scan_videos(cfg)
    df, problems = pipeline.build_trials(cfg, manifest, whole_video=args.whole_video, limit=args.limit)
    print(f"\nTrials written: {len(df)}  (paper: 9307)")
    if len(df):
        print(f"Frames per trial: min {df['n_frames'].min()}, mean {df['n_frames'].mean():.1f}, max {df['n_frames'].max()}  (paper: 9 / 44.2 / 164)")
        print(f"Trials per signer:\n{df.groupby('signer').size().to_string()}")
        print(f"Words covered: {df['word'].nunique()} / 60")
    if problems:
        print(f"\n{len(problems)} problem(s), first 15:")
        print("\n".join("  " + p for p in problems[:15]))
    print("\nSaved work/trials_metadata.csv. Next: python scripts/05_check_quality.py")


if __name__ == "__main__":
    main()
