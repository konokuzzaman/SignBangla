"""Check the downloaded BdSLW401 front-view clips: counts per word and signer, total size, and that each one opens.

    python scripts/31_bdslw401_verify.py

Writes these into the dataset folder, and nothing else:
    verify_report.csv     one row per clip: status, size, frames, fps, width, height
    clips_per_word.csv, clips_per_signer.csv
Status is ok, missing, wrong_size, bad_name or broken (OpenCV cannot open it or read a frame). Exits with an error if any
clip is not ok.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _path  # noqa: F401
import cv2

from signbangla.labels401 import N_WORDS_401, parse_clip_name
from signbangla.utils import ROOT, load_config, progress

INDEX_FILE = "front_index.json"
EXPECTED_SIGNERS = 18
MAX_LISTED = 40


def check_clip(item):
    """(path, expected size) -> a row for the report. Opens the clip and reads its first frame."""
    path, expected = item
    row = dict(path=path, status="ok", size=0, frames=0, fps=0.0, width=0, height=0)
    file = Path(path)
    if not file.exists():
        return dict(row, status="missing")
    row["size"] = file.stat().st_size
    if expected is not None and row["size"] != expected:
        return dict(row, status="wrong_size")
    if parse_clip_name(file.name) is None:
        return dict(row, status="bad_name")
    video = cv2.VideoCapture(path)
    try:
        ok = video.isOpened()
        if ok:
            ok, _ = video.read()
            row.update(frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT)), fps=round(video.get(cv2.CAP_PROP_FPS), 3),
                       width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        video.release()
    return row if ok and row["frames"] > 0 else dict(row, status="broken")


def write_counts(path, column, counts):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([column, "clips"])
        writer.writerows(sorted(counts.items()))


def spread(counts):
    values = sorted(counts.values())
    return f"min {values[0]}, median {values[len(values) // 2]}, max {values[-1]}" if values else "none"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = ROOT / load_config(args.config)["bdslw401"]["root"]
    index_path = root / INDEX_FILE
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found. Run scripts/30_bdslw401_download.py first.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected = {str(root / e["name"]): e["size"] for e in index["files"] if e["name"].lower().endswith(".mp4")}
    extra = [str(p) for p in root.glob("Front/**/*.mp4") if str(p) not in expected]
    items = [*expected.items(), *((p, None) for p in extra)]

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(progress(pool.map(check_clip, items), total=len(items), desc="checking clips"))
    with open(root / "verify_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(dict(r, path=str(Path(r["path"]).relative_to(root))) for r in rows)

    present = [r for r in rows if r["status"] != "missing"]
    parsed = [(r, parse_clip_name(Path(r["path"]).name)) for r in present]
    good = [(r, p) for r, p in parsed if p]
    words, signers = Counter(p["word_number"] for _, p in good), Counter(p["signer"] for _, p in good)
    views, splits = Counter(p["view"] for _, p in good), Counter(Path(r["path"]).parent.name for r, _ in good)
    write_counts(root / "clips_per_word.csv", "word", words)
    write_counts(root / "clips_per_signer.csv", "signer", signers)

    print(f"\nExpected {len(expected)} clips, {sum(expected.values()) / 1e9:.2f} GB (from the zip's index).")
    print(f"Found {len(present)} clips, {sum(r['size'] for r in present) / 1e9:.2f} GB. Extra clips not in the index: {len(extra)}.")
    print(f"Words: {len(words)} of {N_WORDS_401} ({spread(words)} clips each).")
    print(f"Signers: {len(signers)} of {EXPECTED_SIGNERS} ({spread(signers)} clips each).")
    print("Clips per signer: " + ", ".join(f"S{s:02d}={n}" for s, n in sorted(signers.items())))
    print(f"Views: {dict(views)}. Splits: {dict(splits)}.")
    absent = sorted(set(range(1, N_WORDS_401 + 1)) - set(words))
    print(f"Words with no clips: {absent if absent else 'none'}.")
    status = Counter(r["status"] for r in rows)
    print(f"Status: {dict(status)}.")
    bad = [r for r in rows if r["status"] != "ok"]
    for r in bad[:MAX_LISTED]:
        print(f"  {r['status']:10s} {Path(r['path']).relative_to(root)}")
    if len(bad) > MAX_LISTED:
        print(f"  ... and {len(bad) - MAX_LISTED} more: see {root / 'verify_report.csv'}")
    if bad:
        raise SystemExit(f"{len(bad)} of {len(rows)} clips are not ok.")
    print("All clips are ok.")


if __name__ == "__main__":
    main()
