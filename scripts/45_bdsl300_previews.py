"""BdSL-300, step 6: one preview per word to check by eye, and contact sheets.

    python scripts/45_bdsl300_previews.py
    python scripts/45_bdsl300_previews.py --folders Eat,Go

For every word in quality_report.csv (scripts/44), the best good clip is chosen the way the core dictionary chooses
trials (one hand, right first; duration near the word's median; few frames without the signing hand), normalised and
resampled like the dictionary, and drawn in the skeleton style:
    <preview_dir>/<folder>.mp4        captioned with the folder name
    <preview_dir>/sheet_01.png ...    about 20 words per sheet: the clip's most raised-hand frame, folder, English,
                                      good clips, status
Only the files of this script are written; keypoint files are only read.
"""
import argparse

import _path  # noqa: F401
import cv2
import numpy as np
import pandas as pd

from signbangla.speech2sign.core_dictionary import candidate, choose
from signbangla.speech2sign.render import draw_frame, render_options, save_animation
from signbangla.utils import ROOT, load_config, progress, subdir

TILE = 320
LABEL_H = 64
COLUMNS = 5
STYLE = "skeleton"


def best_clip(rows, trial_dir, cfg):
    """The chosen clip of one word: (candidate dict, report row) or None."""
    s2s = cfg["speech2sign"]
    cands, by_stem = [], {}
    for r in rows.itertuples():
        path = trial_dir / f"{r.stem}.npy"
        if not path.exists():
            continue
        cand = candidate(np.load(path), r.fps, r.width, r.height, cfg, r.stem, trim=False)
        if cand:
            cands.append(cand)
            by_stem[r.stem] = r
    chosen = choose(cands, 1, s2s["hand_weight"])
    return (chosen[0], by_stem[chosen[0]["ref"]]) if chosen else None


def peak_frame(seq):
    """The frame where a wrist is highest (smallest y): usually the clearest moment of the sign."""
    wrists = seq[:, [15, 16], 1]
    wrists = np.where(np.abs(seq[:, [15, 16]]).sum(-1) == 0, np.inf, wrists)
    return int(np.argmin(wrists.min(1))) if np.isfinite(wrists).any() else len(seq) // 2


def tile(seq, label_lines, view):
    img = draw_frame(seq[peak_frame(seq)], STYLE, TILE, view, supersample=2)
    bar = np.full((LABEL_H, TILE, 3), 255, np.uint8)
    for i, (text, scale, color) in enumerate(label_lines):
        cv2.putText(bar, text, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    out = np.vstack([img, bar])
    cv2.rectangle(out, (0, 0), (TILE - 1, TILE + LABEL_H - 1), (200, 200, 200), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--folders", help="comma-separated folders (default: every word in the quality report)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s2s = cfg["speech2sign"]
    b = s2s["bdsl300"]
    base, out = subdir(cfg, b["dir"]), subdir(cfg, b["preview_dir"])
    trial_dir = subdir(cfg, f"{b['keypoints_dir']}/trials")
    report_path = base / "quality_report.csv"
    if not report_path.exists():
        report_path = base / "quality_report_test.csv"  # only the test words so far
    if not report_path.exists():
        raise SystemExit("No quality report yet. Run scripts/44_bdsl300_extract.py first.")
    report = pd.read_csv(report_path, keep_default_na=False)
    report = report[report["ok"].astype(str) == "True"]
    overlap = pd.read_csv(base / "overlap.csv", encoding="utf-8-sig", keep_default_na=False).set_index("folder")
    counts = report.groupby("folder").size()
    folders = sorted(counts.index, key=str.lower)
    if args.folders:
        folders = [f for f in folders if f in {x.strip() for x in args.folders.split(",")}]

    opts = dict(render_options(s2s["render"]), style=STYLE)
    tiles, written = [], []
    for folder in progress(folders, desc="previews"):
        best = best_clip(report[report["folder"] == folder], trial_dir, cfg)
        o = overlap.loc[folder]
        if best is None:
            print(f"  {folder}: no usable clip")
            continue
        cand, row = best
        path = out / f"{folder.replace(' ', '_')}.mp4"
        save_animation(cand["seq"], path, fps=s2s["fps"], captions=folder, font_path=ROOT / s2s["font"], **opts)
        written.append(path)
        good = int(counts[folder])
        tiles.append(tile(cand["seq"], [
            (folder, 0.6, (20, 20, 20)),
            (f"{o['english'][:28]} | {good} good | {o['status']}", 0.42,
             (20, 20, 160) if good < b["min_good_clips"] else (90, 90, 90)),
        ], opts["view"]))
    sheets = []
    per = b["sheet_words"]
    for k in range(0, len(tiles), per):
        group = tiles[k:k + per]
        blank = np.full_like(group[0], 255)
        group += [blank] * (-len(group) % COLUMNS)
        rows = [np.hstack(group[i:i + COLUMNS]) for i in range(0, len(group), COLUMNS)]
        path = out / f"sheet_{k // per + 1:02d}.png"
        cv2.imwrite(str(path), np.vstack(rows))
        sheets.append(path)
    print(f"{len(written)} previews and {len(sheets)} contact sheets in {out}")


if __name__ == "__main__":
    main()
