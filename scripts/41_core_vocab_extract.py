"""MODULE 2 core vocabulary, step 2: keypoints from the BdSL40 and INCLUDE videos, trimmed to the signing part.

    python scripts/41_core_vocab_extract.py
    python scripts/41_core_vocab_extract.py --source BdSL40 --workers 4

MediaPipe Holistic runs with exactly the settings of scripts/03 (signbangla.keypoints.make_holistic and extract_video with
configs/config.yaml -> keypoints), so the keypoints match BdSLW60. For each video in data/<source>/<id>/:
    work/core_vocab/<source>/raw_keypoints/<id>_<video>.npy + .json    whole video (NaN = missing, like step 3)
    work/core_vocab/<source>/trials/<id>_<video>.npy                     only the signing part (hands up)
    work/core_vocab/<source>/quality_report.csv                          the checks of step 5, plus upright and trimming
Videos already extracted are skipped. Next: python scripts/42_digits_from_images.py
"""
import argparse
import json
import time
from multiprocessing import Pool, cpu_count

import _path  # noqa: F401
import numpy as np
import pandas as pd

from signbangla.keypoints import extract_video, make_holistic
from signbangla.speech2sign.core_signs import check_clip
from signbangla.utils import ROOT, atomic_save_npy, atomic_write_json, load_config, progress, subdir

SOURCES = {"BdSL40": "bdsl40_dir", "INCLUDE": "include_dir"}
VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv"}
_HOL = None


def _init(cfg):
    global _HOL
    _HOL = make_holistic(cfg)


def _extract(job):
    stem, path, out = job
    t0 = time.time()
    try:
        arr, meta = extract_video(path, _HOL)
        if meta["n_frames"] == 0:
            return stem, "no frames could be read"
        meta["seconds"] = round(time.time() - t0, 1)
        meta["video"] = str(path.relative_to(ROOT).as_posix())
        atomic_save_npy(out / f"{stem}.npy", arr)
        atomic_write_json(out / f"{stem}.json", meta)  # written last: marks the video as done
        return stem, ""
    except Exception as e:  # noqa: BLE001 - one bad video must not stop the rest
        return stem, f"{type(e).__name__}: {e}"


def check(stem, raw_dir, trial_dir, cfg, c):
    raw = np.load(raw_dir / f"{stem}.npy")
    meta = json.loads((raw_dir / f"{stem}.json").read_text(encoding="utf-8"))
    row = dict(stem=stem, id=stem.split("_", 1)[0], video=meta.get("video"), fps=meta["fps"], width=meta["width"],
               height=meta["height"], video_frames=len(raw))
    trial, checks = check_clip(raw, cfg["quality"], c["trim_raise_frac"], c["trim_margin_frames"])
    if trial is not None:
        atomic_save_npy(trial_dir / f"{stem}.npy", trial)
    return dict(row, **checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--source", choices=list(SOURCES))
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["speech2sign"]["core_vocab"]
    for source in [args.source] if args.source else list(SOURCES):
        folder = ROOT / c[SOURCES[source]]
        videos = sorted(p for p in folder.glob("*/*") if p.suffix.lower() in VIDEO_EXT)
        if not videos:
            print(f"{source}: no videos in {folder}. Run scripts/40_core_vocab_download.py first.")
            continue
        base = subdir(cfg, f"{c['work_subdir']}/{source}")
        raw_dir, trial_dir = subdir(cfg, f"{c['work_subdir']}/{source}/raw_keypoints"), subdir(cfg, f"{c['work_subdir']}/{source}/trials")
        jobs = [(f"{p.parent.name}_{p.stem}", p, raw_dir) for p in videos]
        todo = [j for j in jobs if not (raw_dir / f"{j[0]}.json").exists()]
        print(f"{source}: {len(videos)} videos, {len(todo)} to extract, {args.workers} worker(s)")
        failed = []
        if todo:
            with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
                for stem, err in progress(pool.imap_unordered(_extract, todo), total=len(todo), desc=source):
                    if err:
                        failed.append((stem, err))
        for stem, err in failed:
            print(f"  failed {stem}: {err}")
        done = [j[0] for j in jobs if (raw_dir / f"{j[0]}.json").exists()]
        report = pd.DataFrame([check(s, raw_dir, trial_dir, cfg, c) for s in done])
        report.to_csv(base / "quality_report.csv", index=False)
        ok = report[report["ok"]]
        print(f"  OK {len(ok)} / {len(report)}; flags: "
              f"{report.loc[~report['ok'], 'flags'].str.split('|').explode().value_counts().to_dict()}")
        print(f"  signing part: median {np.median(report['n_frames']):.0f} of {np.median(report['video_frames']):.0f} "
              f"frames; dominant hand {ok['dominant_hand'].value_counts().to_dict()}")
        per_word = ok.groupby("id").size()
        print(f"  OK videos per word: {per_word.to_dict()}")
    print("Next: python scripts/42_digits_from_images.py")


if __name__ == "__main__":
    main()
