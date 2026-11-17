"""Find BdSLW401 clips stored upside down, before extracting keypoints.

    python scripts/32_bdslw401_check_orientation.py

Checks one clip of every signer-word pair: MediaPipe looks at 3 frames, and a body with its hips above its shoulders
counts as upside down. Writes to work/bdslw401/orientation/:
    orientation_check.csv   one row per clip checked: votes, verdict, frame size
    signers_*.png           sample frames of every signer, as MediaPipe sees them, to check by eye
    flagged.png             sample frames of clips not found upright
Use --all-clips-of W351S17 to check every clip of the pairs whose name starts with that text.
"""
import argparse
import random
from multiprocessing import Pool

import _path  # noqa: F401
import cv2
import numpy as np
import pandas as pd

from signbangla.bdslw401 import list_clips, output_dir, rotated_clips, upright_vote
from signbangla.keypoints import make_holistic, results_to_frame
from signbangla.labels401 import parse_clip_name
from signbangla.utils import load_config, progress

SAMPLE_POINTS = (0.2, 0.5, 0.8)  # where in the clip to look
SHEET_CLIPS_PER_SIGNER = 6
SHEET_SIGNERS_PER_FILE = 9
THUMB_WIDTH = 170
_HOL = None
_ROTATED = set()


def _init(cfg, rotated):
    global _HOL, _ROTATED
    _HOL, _ROTATED = make_holistic(cfg), rotated


def read_sample_frames(path):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for point in SAMPLE_POINTS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(point * max(total - 1, 0)))
        ok, bgr = cap.read()
        if ok:
            frames.append(bgr)
    cap.release()
    return frames, total


def _check(item):
    stem, path = item
    row = dict(stem=stem, upright=0, upside_down=0, no_pose=0, verdict="unreadable", width=0, height=0, frames=0, listed=stem in _ROTATED)
    frames, row["frames"] = read_sample_frames(path)
    if not frames:
        return row
    row["height"], row["width"] = frames[0].shape[:2]
    if hasattr(_HOL, "reset"):
        _HOL.reset()
    for bgr in frames:
        if row["listed"]:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        vote = upright_vote(results_to_frame(_HOL.process(rgb)))
        key = "no_pose" if vote is None else "upright" if vote else "upside_down"
        row[key] += 1
    if row["upside_down"] > row["upright"]:
        row["verdict"] = "upside_down"
    elif row["upright"]:
        row["verdict"] = "mixed" if row["upside_down"] else "upright"
    else:
        row["verdict"] = "no_pose"
    return row


def one_clip_per_pair(clips):
    """The middle clip of each signer-word pair."""
    pairs = {}
    for stem in sorted(clips):
        info = parse_clip_name(stem)
        pairs.setdefault((info["word_number"], info["signer"]), []).append(stem)
    return [names[len(names) // 2] for names in pairs.values()]


def thumbnail(path, label, rotate=False):
    frames, _ = read_sample_frames(path)
    h = int(THUMB_WIDTH * 3 / 4)
    img = np.full((h, THUMB_WIDTH, 3), 200, np.uint8)
    if frames:
        frame = cv2.rotate(frames[1 if len(frames) > 1 else 0], cv2.ROTATE_180) if rotate else frames[1 if len(frames) > 1 else 0]
        fh, fw = frame.shape[:2]
        tw, th = THUMB_WIDTH, int(fh * THUMB_WIDTH / fw)
        if th > h:
            tw, th = int(fw * h / fh), h
        img[:th, :tw] = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (THUMB_WIDTH, 16), (255, 255, 255), -1)
    cv2.putText(img, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def save_sheet(path, rows_of_thumbs):
    width = max(len(r) for r in rows_of_thumbs)
    blank = np.full_like(rows_of_thumbs[0][0], 255)
    cv2.imwrite(str(path), np.vstack([np.hstack(r + [blank] * (width - len(r))) for r in rows_of_thumbs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--all-clips-of", nargs="*", help="check every clip whose name starts with one of these")
    args = ap.parse_args()
    cfg = load_config(args.config)
    clips = list_clips(cfg)
    rotated = rotated_clips(clips, cfg["bdslw401"].get("rotate_180"))
    out = output_dir(cfg, "orientation")
    if args.all_clips_of:
        todo = [s for s in sorted(clips) if s.startswith(tuple(args.all_clips_of))]
    else:
        todo = one_clip_per_pair(clips)
    print(f"Checking {len(todo)} clips with {args.workers} workers...")
    with Pool(args.workers, initializer=_init, initargs=(cfg, rotated)) as pool:
        rows = list(progress(pool.imap_unordered(_check, [(s, clips[s]) for s in todo]), total=len(todo), desc="checking"))
    df = pd.DataFrame(rows).sort_values("stem")
    info = df["stem"].map(parse_clip_name)
    df["word"], df["signer"] = info.map(lambda i: i["word_number"]), info.map(lambda i: i["signer"])
    name = "orientation_clips.csv" if args.all_clips_of else "orientation_check.csv"
    df.to_csv(out / name, index=False)

    print("\nClips by verdict:", df["verdict"].value_counts().to_dict())
    print("Frame sizes:", df.groupby(["width", "height"]).size().to_dict())
    print(df.pivot_table(index="signer", columns="verdict", values="stem", aggfunc="count", fill_value=0).to_string())
    flagged = df[~df["verdict"].isin(["upright"])]
    print("\nNot upright:", flagged[["stem", "upright", "upside_down", "no_pose", "verdict"]].to_string(index=False) if len(flagged) else "none")
    if args.all_clips_of:
        return

    rng = random.Random(0)
    sheet, file_no = [], 1
    for signer, group in df.groupby("signer"):
        chosen = rng.sample(list(group["stem"]), min(SHEET_CLIPS_PER_SIGNER, len(group)))
        sheet.append([thumbnail(clips[s], s, s in rotated) for s in chosen])
        if len(sheet) == SHEET_SIGNERS_PER_FILE:
            save_sheet(out / f"signers_{file_no}.png", sheet)
            sheet, file_no = [], file_no + 1
    if sheet:
        save_sheet(out / f"signers_{file_no}.png", sheet)
    if len(flagged):
        stems = list(flagged["stem"])[:60]
        thumbs = [thumbnail(clips[s], s, s in rotated) for s in stems]
        save_sheet(out / "flagged.png", [thumbs[i:i + 6] for i in range(0, len(thumbs), 6)])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
