"""DU dynamic signs: preview videos and contact sheets to check every class by eye, like the Ban-Sign-Sent-9K previews of
scripts/49. Only reads the keypoints of scripts/47 (keypoints/trials/ and quality_report.csv); nothing is downloaded or
extracted.

    python scripts/47b_dynamic_du_previews.py
    python scripts/47b_dynamic_du_previews.py --classes hello,Please

The videos have no signer ids, so clips are picked evenly over the recording time stamps (the recordings of one signer
are close together), from the good clips whose length is within dynamic_du.preview_length of the class median.
In work/core_vocab_check/dynamic_du/:
    <class>.mp4          preview_clips clips in a row, skeleton style, captioned with the Bangla
    sheet_<class>.png    up to 24 clips: middle frame, clip name, split, length, hand
<class> is the folder name with spaces as underscores. Research use only: never shared.
"""
import argparse
import csv

import _path  # noqa: F401
import cv2
import numpy as np

from signbangla.extra_sources import dynamic_du as D
from signbangla.speech2sign.core_dictionary import candidate
from signbangla.speech2sign.render import draw_frame, render_options, save_animation
from signbangla.utils import ROOT, load_config, subdir, work_dir

TILE, LABEL_H, COLUMNS, SHEET_CLIPS = 240, 44, 6, 24


def evenly(rows, n):
    if len(rows) <= n:
        return list(rows)
    return [rows[round(i * (len(rows) - 1) / (n - 1))] for i in range(n)] if n > 1 else [rows[len(rows) // 2]]


def good_clips(report, folder, length):
    rows = [r for r in report if r["folder"] == folder and r["ok"] == "True"]
    if not rows:
        return []
    median = float(np.median([int(r["n_frames"]) for r in rows]))
    lo, hi = median * (1 - length), median * (1 + length)
    near = [r for r in rows if lo <= int(r["n_frames"]) <= hi] or rows
    return sorted(near, key=lambda r: int(r["stem"].rsplit("__", 1)[1]))


def tile(seq, r, opts):
    img = draw_frame(seq[len(seq) // 2], "skeleton", TILE, opts["view"], supersample=2)
    bar = np.full((LABEL_H, TILE, 3), 255, np.uint8)
    name = r["stem"].rsplit("__", 2)
    cv2.putText(bar, f"{name[1]} {name[2]}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    seconds = int(r["n_frames"]) / float(r["fps"])
    cv2.putText(bar, f"{seconds:.2f} s, hand {r['dominant_hand']}", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (90, 90, 90), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def previews(folder, report, cfg, trial_dir, out_dir):
    s2s, c = cfg["speech2sign"], cfg["dynamic_du"]
    opts = dict(render_options(s2s["render"]), style="skeleton")
    bangla = D.CLASSES[folder][0]
    rows = good_clips(report, folder, c["preview_length"])
    shown = evenly(rows, SHEET_CLIPS)
    in_video = {r["stem"] for r in evenly(shown, c["preview_clips"])}
    seqs, tiles = [], []
    for r in shown:
        cand = candidate(np.load(trial_dir / f"{r['stem']}.npy"), float(r["fps"]), int(r["width"]), int(r["height"]),
                         cfg, r["stem"], trim=False)
        if cand is None:
            continue
        if r["stem"] in in_video:
            seqs.append(cand["seq"])
        tiles.append(tile(cand["seq"], r, opts))
    name, written = folder.replace(" ", "_"), []
    if seqs:
        path = out_dir / f"{name}.mp4"
        save_animation(np.concatenate(seqs), path, fps=c["preview_fps"], captions=bangla, font_path=ROOT / s2s["font"],
                       **opts)
        written.append((path, len(seqs)))
    if tiles:
        tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % COLUMNS)
        sheet = np.vstack([np.hstack(tiles[i:i + COLUMNS]) for i in range(0, len(tiles), COLUMNS)])
        path = out_dir / f"sheet_{name}.png"
        cv2.imwrite(str(path), sheet)
        written.append((path, len(tiles)))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--classes", help="comma-separated class folders (default: dynamic_du.classes)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["dynamic_du"]
    folders = args.classes.split(",") if args.classes else c["classes"]
    with open(work_dir(cfg) / c["dir"] / "quality_report.csv", encoding="utf-8-sig", newline="") as f:
        report = list(csv.DictReader(f))
    trial_dir = work_dir(cfg) / c["keypoints_dir"] / "trials"
    out_dir = subdir(cfg, c["preview_dir"])
    for folder in folders:
        for path, n in previews(folder, report, cfg, trial_dir, out_dir):
            print(f"{folder}: {path.relative_to(ROOT).as_posix()} ({n} clips)", flush=True)


if __name__ == "__main__":
    main()
