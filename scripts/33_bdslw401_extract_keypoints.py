"""MediaPipe Holistic over the BdSLW401 front-view clips -> work/bdslw401/raw_keypoints/<clip>.npy (+ .json with fps/size).

    python scripts/33_bdslw401_extract_keypoints.py --limit 30 --workers 6     (test run)
    python scripts/33_bdslw401_extract_keypoints.py --workers 6                (full run)

Same MediaPipe version, settings and 67-keypoint format as BdSLW60 (signbangla/keypoints.py). Up to 12 clips per word come
first (config bdslw401.priority_*), so the sign dictionary can be built before the full run ends.

Safe to stop with Ctrl+C or by closing the window: a clip counts as done only when its .json exists, and that is written
last. The same command continues, skips finished clips and retries failed ones. work/bdslw401/extract_log.csv gets one
line per clip as it finishes, and extract_failed.csv lists the failures of the latest run.
"""
import argparse
import csv
import os
import signal
import time
import traceback
from multiprocessing import Pool

os.environ.setdefault("GLOG_minloglevel", "2")  # MediaPipe's log lines would bury the progress bar

import _path  # noqa: F401
import numpy as np

from signbangla import pipeline
from signbangla.bdslw401 import list_clips, output_dir, priority_first, rotated_clips
from signbangla.keypoints import extract_video, make_holistic
from signbangla.utils import atomic_save_npy, atomic_write_json, load_config, progress

LOG_COLUMNS = ["clip", "status", "frames", "seconds", "error"]
RESTART_WORKER_AFTER = 1000  # clips, so that any slow memory growth inside MediaPipe cannot build up
_HOL = _OUT = None
_ROTATED = set()


def _init(cfg, out, rotated):
    global _HOL, _OUT, _ROTATED
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process
    _HOL, _OUT, _ROTATED = make_holistic(cfg), out, rotated


def _work(item):
    stem, path = item
    started = time.time()
    try:
        arr, meta = extract_video(path, _HOL, rotate_180=stem in _ROTATED)
        if meta["n_frames"] == 0:
            return stem, "failed", 0, 0.0, "no frames could be read"
        meta["seconds"] = round(time.time() - started, 2)
        atomic_save_npy(_OUT / f"{stem}.npy", arr)
        atomic_write_json(_OUT / f"{stem}.json", meta)  # written last: its presence marks the clip as finished
        return stem, "ok", meta["n_frames"], meta["seconds"], ""
    except Exception as e:
        return stem, "failed", 0, 0.0, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=1)}"


def summary(done, failed_count, frames, wall, cfg_total, finished, out, sample):
    print(f"\nThis run: {done} ok, {failed_count} failed, {wall:.0f} s.")
    print(f"Finished overall: {finished} of {cfg_total} clips.")
    if not done:
        return
    print(f"Seconds per clip: {wall / done:.2f} (all workers together). Frames per second: {frames / wall:.1f}. "
          f"Frames per clip: {frames / done:.1f}.")
    left = cfg_total - finished
    print(f"At this speed, the {left} clips left would take about {left * wall / done / 3600:.1f} hours.")
    if sample:
        arrays = [np.load(out / f"{s}.npy", mmap_mode="r") for s in sample]
        print("Sample shapes:", [tuple(a.shape) for a in arrays], "dtype", arrays[0].dtype)
        size = np.mean([(out / f"{s}.npy").stat().st_size for s in sample])
        print(f"Average .npy size: {size / 1024:.0f} KB, about {size * cfg_total / 1e9:.1f} GB for every clip.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="only the next N clips (test run)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    b = cfg["bdslw401"]
    clips = list_clips(cfg)
    rotated = rotated_clips(clips, b.get("rotate_180"))
    out, root = output_dir(cfg, "raw_keypoints"), output_dir(cfg)
    for leftover in out.glob("*.tmp"):  # from a stop in the middle of writing
        leftover.unlink()

    order = priority_first(clips, b["priority_clips_per_word"], b["priority_exclude_signers"])
    todo = [s for s in order if pipeline.needs_extraction(s, out, rotated)]
    finished = len(order) - len(todo)
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(order)} clips, {finished} already done, {len(todo)} to process now, {args.workers} workers, "
          f"{len(rotated)} clips turned upright.", flush=True)
    if not todo:
        return

    log_path = root / "extract_log.csv"
    new_log = not log_path.exists()
    failures, done, frames, sample = [], 0, 0, []
    started = time.time()
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        log = csv.writer(log_file)
        if new_log:
            log.writerow(LOG_COLUMNS)
        pool = Pool(args.workers, initializer=_init, initargs=(cfg, out, rotated), maxtasksperchild=RESTART_WORKER_AFTER)
        try:
            items = [(s, clips[s]) for s in todo]
            for row in progress(pool.imap_unordered(_work, items), total=len(items), desc="extracting"):
                log.writerow(row)
                log_file.flush()
                if row[1] == "ok":
                    done += 1
                    frames += row[2]
                    if len(sample) < 5:
                        sample.append(row[0])
                else:
                    failures.append(row)
            pool.close()
        except KeyboardInterrupt:
            print("\nStopping. Finished clips are saved; run the same command to continue.", flush=True)
            pool.terminate()
        pool.join()

    with open(root / "extract_failed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_COLUMNS)
        writer.writerows(failures)
    summary(done, len(failures), frames, time.time() - started, len(order), finished + done, out, sample)
    if failures:
        print(f"{len(failures)} clips failed, listed in {root / 'extract_failed.csv'}. Run the same command to retry them.")


if __name__ == "__main__":
    main()
