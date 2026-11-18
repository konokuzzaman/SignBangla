"""Preprocess the full BdSLW401 dataset (51,087 trials, 401 classes).

Applies:
  1. Aspect ratio correction (x, z scaled by width/height)
  2. Temporal missing-hand gap interpolation (<= 6 frames)
  3. Shoulder-centered translation and shoulder-width distance scaling
  4. Uniform temporal resampling to 64 frames (64, 67, 3)
  5. Signer-independent splitting (Test: signers 4, 8; Val: signers 1, 11; Train: remaining 14)
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import _path  # noqa: F401
from signbangla.preprocess import PreprocessError, preprocess_trial
from signbangla.utils import load_config, work_dir

FILENAME_RE = re.compile(r"W(?P<word>\d+)S(?P<signer>\d+)F_(?P<rep>\d+)")


def process_one(args):
    npy_path, out_dir, pcfg, min_pose_frac, min_best_hand_frac = args
    stem = npy_path.stem
    m = FILENAME_RE.match(stem)
    if not m:
        return None, "regex_fail"

    word = int(m.group("word"))
    signer = int(m.group("signer"))
    class_idx = word - 1

    json_path = npy_path.with_suffix(".json")
    meta = {}
    if json_path.exists():
        try:
            with open(json_path) as f:
                meta = json.load(f)
        except Exception:
            pass

    width = meta.get("width", 1280)
    height = meta.get("height", 720)

    try:
        raw = np.load(npy_path)
        out, info = preprocess_trial(raw, width, height, pcfg)
    except Exception as e:
        return None, f"preprocess_error: {e}"

    # Quality filter
    best_hand_frac = max(info["lh_frac"], info["rh_frac"])
    if info["pose_frac"] < min_pose_frac:
        return None, "low_pose_quality"
    if best_hand_frac < min_best_hand_frac:
        return None, "low_hand_quality"

    # Save processed array
    out_path = out_dir / f"{stem}.npy"
    np.save(out_path, out.astype(np.float32))

    record = {
        "trial_id": stem,
        "word": word,
        "class_idx": class_idx,
        "signer": signer,
        "n_frames_raw": info["n_frames_raw"],
        "pose_frac": info["pose_frac"],
        "best_hand_frac": best_hand_frac,
        "lh_frac": info["lh_frac"],
        "rh_frac": info["rh_frac"],
        "file_name": f"{stem}.npy",
    }
    return record, "ok"


def assign_split(signer: int, test_signers=(4, 8), val_signers=(1, 11)):
    if signer in test_signers:
        return "test"
    elif signer in val_signers:
        return "val"
    else:
        return "train"


def main():
    parser = argparse.ArgumentParser(description="Preprocess BdSLW401 dataset")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4, help="Worker processes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pcfg = cfg["preprocess"]
    qcfg = cfg["quality"]

    raw_dir = work_dir(cfg) / "bdslw401" / "raw_keypoints"
    out_dir = work_dir(cfg) / "bdslw401" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(raw_dir.glob("*.npy"))
    n_total = len(npy_files)
    print(f"============================================================")
    print(f"  SignBangla - BdSLW401 Preprocessing Pipeline               ")
    print(f"============================================================")
    print(f"Source:      {raw_dir}")
    print(f"Destination: {out_dir}")
    print(f"Found:       {n_total:,} raw trial files")
    print(f"Workers:     {args.num_workers} processes")

    tasks = [
        (p, out_dir, pcfg, qcfg["min_pose_frac"], qcfg["min_best_hand_frac"])
        for p in npy_files
    ]

    t0 = time.time()
    records = []
    dropped = {}

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for rec, status in tqdm(executor.map(process_one, tasks, chunksize=100), total=n_total, desc="Normalizing"):
            if rec is not None:
                records.append(rec)
            else:
                dropped[status] = dropped.get(status, 0) + 1

    elapsed = time.time() - t0
    df = pd.DataFrame(records)

    # Assign signer-independent splits
    df["split"] = df["signer"].apply(assign_split)

    # Save splits.csv
    splits_csv = work_dir(cfg) / "bdslw401" / "splits.csv"
    df.to_csv(splits_csv, index=False)

    print(f"\n[+] Preprocessing Complete in {elapsed:.1f}s ({n_total/elapsed:.1f} clips/s)!")
    print(f"[+] Clean trials retained: {len(df):,} / {n_total:,} ({len(df)/n_total*100:.1f}%)")
    if dropped:
        print(f"[!] Filtered out:")
        for reason, count in dropped.items():
            print(f"     - {reason}: {count:,}")

    print(f"\n[+] Signer-Independent Split Summary:")
    split_counts = df["split"].value_counts()
    for s in ["train", "val", "test"]:
        cnt = split_counts.get(s, 0)
        print(f"     - {s.upper():<5}: {cnt:,} trials ({cnt/len(df)*100:.1f}%)")

    # Word coverage check
    n_words = df["word"].nunique()
    print(f"\n[+] Total unique classes in dataset: {n_words} / 401")
    print(f"[+] Splits saved to: {splits_csv}")


if __name__ == "__main__":
    main()

