"""WEEK 1 (Member 1): inspect BdSLW60 - count videos per word and per signer, check fps, look at the annotation files.

    python scripts/01_inspect_dataset.py

Writes work/manifest.csv and work/video_counts_signer_by_word.csv and prints everything you need to
confirm the dataset is complete. Read the "annotation files" section carefully: it shows the JSON structure and
how many trials (sign repetitions) were found.
"""
import argparse
import json

import _path  # noqa: F401
import pandas as pd

from signbangla import pipeline
from signbangla.annotations import describe_json
from signbangla.utils import dataset_root, load_config, work_dir

PAPER_TRIALS = 9307  # trials reported in the BdSLW60 paper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--no-probe", action="store_true", help="skip reading fps/frame counts")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = dataset_root(cfg)

    print(f"Dataset root: {root}")
    m = pipeline.scan_videos(cfg)
    print(f"\nVideos found: {len(m)}  (parsed names: {int(m['parsed'].sum())})")
    if not m["parsed"].all():
        print("  names that do not look like U<signer>W<word><view>:")
        print("   ", ", ".join(m.loc[~m["parsed"], "rel_path"].head(15)))
    if len(m) == 0:
        print("No videos found. Check paths.dataset_root and video.extensions in the config.")
        return
    if not args.no_probe:
        m = pipeline.probe_videos(m)
    m.to_csv(work_dir(cfg) / "manifest.csv", index=False)

    u = pipeline.usable_videos(cfg, m)
    print(f"Usable videos (60-word list, view '{cfg['video']['view']}'): {len(u)}")
    print(f"Signers: {u['signer'].nunique()} -> {sorted(u['signer'].unique())}")
    print(f"Words:   {u['word'].nunique()} (expected 60)")
    print(f"Views:   {m['view'].value_counts().to_dict()}")
    if "fps" in m:
        print(f"Frame rates: {m['fps'].round(1).value_counts().to_dict()}")
        print(f"Resolutions: {(m['width'].astype(str) + 'x' + m['height'].astype(str)).value_counts().head(5).to_dict()}")
        print(f"Frames per video: min {m['n_frames'].min()}, median {int(m['n_frames'].median())}, max {m['n_frames'].max()}")

    pd.set_option("display.width", 200)
    print("\nVideos per signer:")
    print(u.groupby("signer").size().to_string())
    print("\nVideos per word (should be roughly equal):")
    print(u.groupby("word").size().sort_values().to_string())
    pivot = u.pivot_table(index="signer", columns="word", values="stem", aggfunc="count", fill_value=0)
    pivot.to_csv(work_dir(cfg) / "video_counts_signer_by_word.csv")
    missing = (pivot == 0).sum().sum()
    print(f"\nsigner x word table saved (cells with 0 videos: {missing})")

    print("\n=== annotation files ===")
    ann, report = pipeline.collect_annotations(cfg)
    print(report.to_string(index=False) if len(report) else "no .json files found under the dataset root")
    n_trials = int(sum(len(v) for v in ann.values()))
    print(f"\nTrials found in annotations: {n_trials}  (paper reports {PAPER_TRIALS})  | videos covered: {len(ann)} of {len(u)}")
    first = next(iter(sorted(root.rglob("*.json"))), None)
    if first is not None:
        print(f"\nStructure of {first.relative_to(root)}:")
        with open(first, encoding="utf-8-sig") as f:
            print("\n".join(describe_json(json.load(f))))
        print("\nFirst 25 lines of that file:")
        print("\n".join(first.read_text(encoding="utf-8-sig").splitlines()[:25]))
    if n_trials == 0:
        print("\n!! No trials parsed. If every video is already one sign, use --whole-video in step 4.\n"
              "!! Otherwise adjust annotation.start_key_patterns / end_key_patterns in the config to the key names shown above.")


if __name__ == "__main__":
    main()
