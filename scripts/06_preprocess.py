"""WEEK 2-3 (Member 2): centre on the body, scale by shoulder width, fill missing hands, resample to 64 frames.

    python scripts/06_preprocess.py
    python scripts/06_preprocess.py --include-bad     # also process trials flagged in step 5
    python scripts/06_preprocess.py --mirror-left     # optional experiment: mirror left-dominant trials (default: off)

Output: work/processed/<trial>.npy with shape (64, 67, 3) and work/processed_metadata.csv.
"""
import argparse

import _path  # noqa: F401

from signbangla import pipeline
from signbangla.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--include-bad", action="store_true")
    ap.add_argument("--mirror-left", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    try:
        df = pipeline.run_preprocess(cfg, include_bad=args.include_bad, mirror=True if args.mirror_left else None)
    except ValueError as e:
        raise SystemExit(str(e))
    ok = df["status"] == "ok"
    print(f"\nProcessed: {int(ok.sum())} ok, {int((~ok).sum())} failed")
    if (~ok).any():
        print(df[~ok][["trial_id", "error"]].head(10).to_string(index=False))
    if ok.any():
        print(f"Hand seen in processed sequences: left {df.loc[ok, 'lh_frac'].mean():.0%}, right {df.loc[ok, 'rh_frac'].mean():.0%} of frames")
        print(f"Dominant hand: {df.loc[ok, 'dominant_hand'].value_counts().to_dict()}")
    print("Next: python scripts/07_make_splits.py")


if __name__ == "__main__":
    main()
