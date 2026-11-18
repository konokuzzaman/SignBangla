"""Dhaka University dynamic BdSL videos (Kaggle): list the files, download only the wanted class folders, extract
keypoints and check them.

    python scripts/47_dynamic_du.py --list-only     (zip index -> file_list.csv and dataset_numpy_info.md; no videos)
    python scripts/47_dynamic_du.py                 (download the classes in dynamic_du.classes, extract, check)
    python scripts/47_dynamic_du.py --check-only    (quality reports and class table only)

The Kaggle zip is 15.5 GB; only the .avi files of the wanted classes are fetched, each by its byte range (the code of
scripts/30 and 43). Uses KAGGLE_API_TOKEN or ~/.kaggle; the token is never printed. Under work/dynamic_du/:
    zip_index.json, file_list.csv (every file; dataset_numpy as one row per video), dataset_numpy_info.md
    videos/<path in the zip>, download_log.csv
    keypoints/raw/<class>__<split>__<stamp>.npy + .json, keypoints/trials/ (signing part), extract_log.csv
    quality_report.csv (per video), class_table.csv (per class: Bangla, good videos, needs_check)
Safe to stop and run again. Research use only: videos and keypoints are never shared.
"""
import argparse
import csv
import importlib.util
import sys
from collections import Counter, defaultdict

import _path  # noqa: F401
import numpy as np

from signbangla.extra_sources import dynamic_du as D
from signbangla.extra_sources.extract import extract_clips, is_done, load_raw
from signbangla.extra_sources.word_list import add_words
from signbangla.speech2sign.core_signs import check_clip
from signbangla.utils import ROOT, atomic_save_npy, load_config, progress, subdir

_spec = importlib.util.spec_from_file_location("bdsl300_download", ROOT / "scripts" / "43_bdsl300_download.py")
dl = importlib.util.module_from_spec(_spec)  # load_index and fetch_all of scripts/43 (Kaggle byte ranges)
_spec.loader.exec_module(dl)
NUMPY = "dynamic_dataset/dataset_numpy/"


def write_csv(path, rows, columns=None):
    columns = columns or (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(rows)


def file_list(files):
    """Every file of the zip, except that dataset_numpy (one small .npy per frame) gets one row per video."""
    rows, numpy = [], defaultdict(lambda: [0, 0])
    for e in files:
        parts = e["name"].split("/")
        if e["name"].startswith(NUMPY):
            key = "/".join(parts[:5])
            numpy[key][0] += 1
            numpy[key][1] += e["size"]
            continue
        v = D.parse_video(e["name"])
        if v:
            kind = "video"
        elif parts[0] == "dynamic_dataset" and e["name"].endswith(".jpg"):
            kind = "frame_image"
        else:
            kind = parts[0]
        rows.append(dict(name=e["name"], kind=kind, split=v["split"] if v else "", folder=v["folder"] if v else "",
                         files=1, bytes=e["size"]))
    for key, (n, size) in sorted(numpy.items()):
        p = key.split("/")
        rows.append(dict(name=key + "/", kind="numpy_frames", split=p[2], folder=p[3], files=n, bytes=size))
    return rows


def numpy_info(files, base):
    """What dataset_numpy holds, from the index and the frames of one video fetched by --list-only."""
    numpy = [e for e in files if e["name"].startswith(NUMPY)]
    per_video = Counter("/".join(e["name"].split("/")[:5]) for e in numpy)
    lines = ["# dynamic_dataset/dataset_numpy", "",
             f"- {len(numpy):,} files in {len(per_video):,} video folders (train/val/test, the same 40 classes as the "
             f"videos), {sum(e['size'] for e in numpy) / 1e6:.0f} MB",
             f"- one .npy per frame (frame_000.npy, ...); frames per video: {dict(Counter(per_video.values()))}",
             f"- file sizes: {dict(Counter(e['size'] for e in numpy))} bytes (128-byte .npy header + 128 float32)"]
    sample = sorted((base / "numpy_sample").rglob("frame_*.npy"))
    if sample:
        st = np.stack([np.load(p, allow_pickle=False) for p in sample])
        hand_a, hand_b = (st[:, :63] != 0).any(1), (st[:, 63:126] != 0).any(1)
        pts = st[hand_a, :63].reshape(-1, 21, 3)
        where = sample[0].parent.relative_to(base / "numpy_sample").as_posix()
        lines += [f"- sample {where}: {len(st)} frames, each an array of shape {st.shape[1:]}, {st.dtype}",
                  "- layout: values 0-62 = 21 hand points x (x, y, z) of one hand, 63-125 = the other hand, zeros when "
                  "a hand is not found (MediaPipe Hands style: x, y in 0..1 of the image, z small); 126-127 = two more "
                  f"numbers, always set (about {st[:, 126].mean():.2f} and {st[:, 127].mean():.2f}; perhaps a "
                  "reference point)",
                  f"- in the sample: first hand in {hand_a.sum()} of {len(st)} frames, second hand in {hand_b.sum()}; "
                  f"x {pts[..., 0].min():.2f} to {pts[..., 0].max():.2f}, z {pts[..., 2].min():.3f} to "
                  f"{pts[..., 2].max():.3f}"]
    lines += ["", "No pose or face points, so it is not the project's 67-point format and is not used: keypoints are "
                  "extracted again from the videos."]
    (base / "dataset_numpy_info.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def fetch_numpy_sample(index, base, url):
    """The frames of one test video of 'hello', to see what the arrays hold."""
    names = [e for e in index["files"] if e["name"].startswith(NUMPY + "test/hello/")]
    first = sorted({e["name"].split("/")[4] for e in names})[0]
    root = base / "numpy_sample"
    client = dl.kaggle_zip.RangeClient(url)
    try:
        for e in names:
            if e["name"].split("/")[4] == first and not dl.kaggle_zip.already_done(e, root):
                dl.kaggle_zip.with_retries(lambda: dl.kaggle_zip.fetch_file(e, root, client))
    finally:
        client.close()


def quality(clips, cfg, raw_dir, trial_dir):
    c = cfg["speech2sign"]["core_vocab"]
    rows = []
    for stem, (_p, info) in progress(clips.items(), total=len(clips), desc="checking"):
        if not is_done(raw_dir, stem):
            continue
        raw, meta = load_raw(raw_dir, stem)
        trial, checks = check_clip(raw, cfg["quality"], c["trim_raise_frac"], c["trim_margin_frames"])
        if trial is not None:
            atomic_save_npy(trial_dir / f"{stem}.npy", trial)
        rows.append(dict(stem=stem, folder=info["folder"], split=info["split"], fps=meta["fps"], width=meta["width"],
                         height=meta["height"], video_frames=len(raw), **checks))
    return rows


def class_table(index, report):
    videos = Counter(v["folder"] for e in index["files"] if (v := D.parse_video(e["name"])))
    by = defaultdict(list)
    for r in report:
        by[r["folder"]].append(r)
    rows = []
    for folder, (bangla, english, category, note) in D.CLASSES.items():
        rs = by.get(folder, [])
        good = [r for r in rs if r["ok"]]
        hands = Counter(r.get("dominant_hand", "") for r in good)
        flags = Counter(f for r in rs if not r["ok"] for f in str(r["flags"]).split("|") if f)
        rows.append(dict(folder=folder, bangla=bangla, english=english, category=category, videos=videos[folder],
                         extracted=len(rs), good=len(good), right_hand=hands.get("R", 0), left_hand=hands.get("L", 0),
                         flags=" ".join(f"{f}:{n}" for f, n in flags.items()), needs_check=True, note=note))
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--classes", help="comma-separated class folders (default: dynamic_du.classes)")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["dynamic_du"]
    base, videos = subdir(cfg, d["dir"]), subdir(cfg, d["videos_dir"])
    raw_dir, trial_dir = subdir(cfg, f"{d['keypoints_dir']}/raw"), subdir(cfg, f"{d['keypoints_dir']}/trials")
    have_index = (base / dl.INDEX_FILE).exists()
    url = None if args.check_only and have_index else dl.kaggle_zip.DownloadUrl(d["kaggle_dataset"])
    index = dl.load_index(base, url)
    write_csv(base / "file_list.csv", file_list(index["files"]))
    if args.list_only:
        fetch_numpy_sample(index, base, url)
        print("\n".join(numpy_info(index["files"], base)))
        print(f"Wrote {(base / 'file_list.csv').relative_to(ROOT)} and dataset_numpy_info.md")
        return

    wanted = [c.strip() for c in args.classes.split(",")] if args.classes else d["classes"]
    missing = [c for c in wanted if c not in D.CLASSES]
    if missing:
        raise SystemExit(f"no Bangla entry in dynamic_du.CLASSES for {missing}")
    entries = [e for e in index["files"] if (v := D.parse_video(e["name"])) and v["folder"] in wanted]
    if not args.check_only:
        jobs = [e for e in entries if not dl.kaggle_zip.already_done(e, videos)]
        print(f"{len(entries)} videos in {len(wanted)} classes ({sum(e['size'] for e in entries) / 1e9:.2f} GB), "
              f"{len(entries) - len(jobs)} already here", flush=True)
        if jobs:
            failed = dl.fetch_all(jobs, videos, url, d["download_workers"], base / "download_log.csv")
            if failed:
                print(f"{len(failed)} downloads failed (see download_log.csv); run again to retry them.")
    clips = {}
    for e in entries:
        v = D.parse_video(e["name"])
        path = videos / e["name"]
        if path.exists() and path.stat().st_size == e["size"]:
            clips[D.clip_stem(v)] = (path, dict(v, bangla=D.CLASSES[v["folder"]][0]))
    if not args.check_only:
        extract_clips(clips, cfg, raw_dir, base / "extract_log.csv", args.workers or d["extract_workers"])
    report = quality(clips, cfg, raw_dir, trial_dir)
    if report:
        write_csv(base / "quality_report.csv", report)
    table = class_table(index, report)
    write_csv(base / "class_table.csv", table)
    print(f"\n{sum(r['ok'] for r in report)} of {len(report)} extracted videos pass the quality threshold")
    print("| class | Bangla | videos | extracted | good | R | L |")
    for r in table:
        print(f"| {r['folder']} | {r['bangla']} | {r['videos']} | {r['extracted']} | {r['good']} | {r['right_hand']} | "
              f"{r['left_hand']} |")
    added = add_words(ROOT / cfg["speech2sign"]["core_vocab"]["word_list"], D.SOURCE, "D",
                      [dict(bangla=r["bangla"], english=r["english"], category=r["category"], folder=r["folder"],
                            has_clips=r["good"] > 0) for r in table if r["folder"] in wanted])
    print(f"Word list: {len(added)} rows added (source {D.SOURCE}, verified false)"
          + "".join(f"\n  {r['id']} {r['bangla']} ({r['category']}) folder={r['folder']} core_id={r['core_id']}"
                    for r in added))


if __name__ == "__main__":
    main()
