"""WEEK 2 (Member 1): MediaPipe Holistic over every video -> work/raw_keypoints/<video>.npy  (+ .json with fps/size).

    python scripts/03_extract_keypoints.py --limit 3          # smoke test on 3 videos first!
    python scripts/03_extract_keypoints.py --workers 4        # the real run (hours on a laptop; safe to stop and restart)

Videos listed in config video.rotate_180 are turned upright first, and are redone if their output was made without that.
Videos already extracted are skipped, so you can stop (Ctrl+C) and continue later. Frames are processed in order
so MediaPipe can track the hands across the whole video; trials are cut out later (step 4).
"""
import argparse
import time
import traceback
from multiprocessing import Pool, cpu_count

import _path  # noqa: F401
import pandas as pd

from signbangla import pipeline
from signbangla.keypoints import extract_video, make_holistic
from signbangla.utils import atomic_save_npy, atomic_write_json, load_config, progress, subdir, work_dir

_CFG = None
_HOL = None
_OUT = None
_ROTATED = set()


def _init(cfg, out):
    global _CFG, _HOL, _OUT, _ROTATED
    _CFG, _OUT = cfg, out
    _ROTATED = pipeline.rotated_stems(cfg)
    _HOL = make_holistic(cfg)


def _work(item):
    stem, path = item
    t0 = time.time()
    try:
        arr, meta = extract_video(path, _HOL, rotate_180=stem in _ROTATED)
        if meta["n_frames"] == 0:
            return stem, "failed", "no frames could be read", 0.0
        meta["seconds"] = round(time.time() - t0, 1)
        atomic_save_npy(_OUT / f"{stem}.npy", arr)
        atomic_write_json(_OUT / f"{stem}.json", meta)  # written last: its presence marks the video as finished
        return stem, "ok", "", meta["seconds"]
    except Exception as e:
        return stem, "failed", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    ap.add_argument("--limit", type=int, help="only the first N videos (smoke test)")
    ap.add_argument("--only", help="only videos whose name contains this text, e.g. U4W1")
    ap.add_argument("--force", action="store_true", help="redo videos that already have output")
    args = ap.parse_args()
    cfg = load_config(args.config)

    manifest = pipeline.scan_videos(cfg)
    vids = pipeline.usable_videos(cfg, manifest)
    if args.only:
        vids = vids[vids["stem"].str.contains(args.only, case=False)]
    if vids["stem"].duplicated().any():
        raise SystemExit("Two videos share the same file name in different folders - rename or tell me, the pipeline keys on the name.")
    out = subdir(cfg, "raw_keypoints")
    rotated = pipeline.rotated_stems(cfg)
    todo = [(r.stem, r.path) for r in vids.itertuples() if args.force or pipeline.needs_extraction(r.stem, out, rotated)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(vids)} usable videos, {len(todo)} to process, {args.workers} worker(s)")
    if not todo:
        return

    results = []
    if args.workers <= 1:
        _init(cfg, out)
        results = [_work(t) for t in progress(todo, desc="extracting")]
    else:
        with Pool(args.workers, initializer=_init, initargs=(cfg, out)) as pool:
            for r in progress(pool.imap_unordered(_work, todo), total=len(todo), desc="extracting"):
                results.append(r)
    log = pd.DataFrame(results, columns=["stem", "status", "error", "seconds"])
    log.to_csv(work_dir(cfg) / "extract_log.csv", index=False)
    print(f"done: {(log['status'] == 'ok').sum()} ok, {(log['status'] != 'ok').sum()} failed (see work/extract_log.csv)")
    if (log["status"] != "ok").any():
        print(log[log["status"] != "ok"][["stem", "error"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
