"""WEEK 3 (Member 1): check keypoint quality, plot skeletons, flag trials where detection failed.

    python scripts/05_check_quality.py

Writes work/quality_report.csv (stats + flags + ok), work/quality_plots/summary.png and skeleton snapshots of
random good trials and of the worst flagged trials. LOOK AT THE PICTURES - that is the actual check.
Trials with ok == False are dropped in step 6.
"""
import argparse

import _path  # noqa: F401
import numpy as np
import pandas as pd

from signbangla import pipeline
from signbangla.utils import load_config, subdir, work_dir
from signbangla.viz import plot_quality_summary, plot_trial_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--n-plots", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)

    df = pipeline.run_quality(cfg)
    plots = subdir(cfg, "quality_plots")
    bad = df[~df["ok"]]
    print(f"\nTrials checked: {len(df)} | OK: {int(df['ok'].sum())} | flagged: {len(bad)} ({len(bad) / max(len(df), 1):.1%})")
    if len(bad):
        print("Flag counts:", pd.Series("|".join(bad["flags"]).split("|")).value_counts().to_dict())
        bad.to_csv(work_dir(cfg) / "bad_trials.csv", index=False)
    print(f"Dominant hand (all trials): {df['dominant_hand'].value_counts().to_dict()}  (paper: about 7673 right / 1634 left)")
    print(f"Mean detection: pose {df['pose_frac'].mean():.1%}, best hand {df['best_hand_frac'].mean():.1%}")
    worst_words = df.groupby("word")["ok"].mean().sort_values().head(5)
    print("Words with the lowest share of OK trials:", {k: f"{v:.0%}" for k, v in worst_words.items()})

    plot_quality_summary(df, plots / "summary.png")
    rng = np.random.default_rng(args.seed)
    tdir = subdir(cfg, "trials")
    good = df[df["ok"]]
    picks = [("good", r) for r in good.sample(min(args.n_plots, len(good)), random_state=args.seed).itertuples()]
    picks += [("bad", r) for r in bad.sort_values("best_hand_frac").head(args.n_plots).itertuples()]
    for kind, r in picks:
        raw = np.load(tdir / f"{r.trial_id}.npy")
        plot_trial_grid(raw, plots / f"{kind}_{r.trial_id}.png", aspect=(r.width / r.height) if r.height else 1.0,
                        title=f"{r.trial_id} ({r.word}, signer {r.signer}) {kind} flags={r.flags or '-'}")
    print(f"\nPlots saved in {plots}\nSaved work/quality_report.csv. Next: python scripts/06_preprocess.py")


if __name__ == "__main__":
    main()
