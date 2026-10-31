"""WEEK 3 (Member 3): demo of the evaluation tools. Shows how every later training run should be evaluated.

    python scripts/09_eval_demo.py

Uses fake predictions (the true label, with ~30 % replaced by a random class) just to produce example outputs in
work/eval_demo/. In Week 4+ call:  from signbangla.metrics import evaluate_and_save
                                   evaluate_and_save(y_true, y_pred, "work/results/bigru", prefix="test")
"""
import argparse

import _path  # noqa: F401
import numpy as np
import pandas as pd

from signbangla.metrics import evaluate_and_save
from signbangla.utils import load_config, subdir, work_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sp = work_dir(cfg) / "splits.csv"
    if sp.exists():
        y_true = pd.read_csv(sp).query("split == 'test'")["class_idx"].to_numpy()
    else:
        y_true = np.random.default_rng(0).integers(0, 60, 1200)
        print("(no splits.csv yet - using random labels)")
    rng = np.random.default_rng(0)
    y_pred = np.where(rng.random(len(y_true)) < 0.3, rng.integers(0, 60, len(y_true)), y_true)
    r = evaluate_and_save(y_true, y_pred, subdir(cfg, "eval_demo"), prefix="demo")
    print(f"accuracy {r['accuracy']:.3f} | macro-F1 {r['macro_f1']:.3f} | weighted-F1 {r['weighted_f1']:.3f} | n = {r['n_samples']}")
    print("\nMost confused pairs:\n" + r["confused_pairs"].to_string(index=False))
    print(f"\nFiles written to {subdir(cfg, 'eval_demo')}")


if __name__ == "__main__":
    main()
