"""WEEK 2 (Member 3): signer-wise split 14 train / 2 validation / 2 test -> work/splits.csv

    python scripts/07_make_splits.py

Signers never appear in two splits, so test accuracy means "works on a person the model has never seen".
Test signers default to U4 and U8 (the dataset paper's test signers); validation signers are drawn with a fixed seed.
"""
import argparse

import _path  # noqa: F401
import pandas as pd

from signbangla import pipeline
from signbangla.utils import load_config, work_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    try:
        df, mapping, problems = pipeline.make_splits(cfg)
    except ValueError as e:
        raise SystemExit(str(e))
    lines = ["Signer split:"]
    for s in ("train", "val", "test"):
        sg = sorted(k for k, v in mapping.items() if v == s)
        lines.append(f"  {s:5s}: {len(sg)} signers {sg} -> {int((df['split'] == s).sum())} trials")
    lines.append("\nTrials per class in each split (min / max):")
    for s in ("train", "val", "test"):
        c = df[df["split"] == s].groupby("class_idx").size()
        lines.append(f"  {s:5s}: {c.min()} / {c.max()}")
    lines.append("\nProblems: " + ("none" if not problems else "\n  " + "\n  ".join(problems)))
    text = "\n".join(lines)
    print(text)
    (work_dir(cfg) / "split_summary.txt").write_text(text, encoding="utf-8")
    print("\nSaved work/splits.csv and work/split_summary.txt. Next: python scripts/08_test_dataloader.py")


if __name__ == "__main__":
    main()
