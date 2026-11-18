"""Keypoints for downloaded videos, with exactly the settings of scripts/03, 33 and 44 (signbangla.keypoints), so they
have the 67-point format of BdSLW60, BdSLW401 and BdSL-300. In raw_dir:
    <stem>.npy + <stem>.json    whole video, NaN = missing (the .json is written last: it marks the video done)
Safe to stop and run again: finished videos are skipped, failed ones retried. One log line per video.
A worker that cannot start (for example an import error) stops the whole run with an error; workers are never
restarted in a loop. The videos are done in batches of workers x RESTART_WORKER_AFTER, each with a new pool, so
MediaPipe's slow memory growth cannot build up (the pool's own max_tasks_per_child can hang on Windows when all
workers retire at once).
"""
import csv
import json
import os
import signal
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from ..keypoints import extract_video, make_holistic
from ..utils import ROOT, atomic_save_npy, atomic_write_json

LOG_COLUMNS = ["clip", "status", "frames", "seconds", "error"]
RESTART_WORKER_AFTER = 300  # videos, so slow memory growth inside MediaPipe cannot build up
_HOL = _OUT = None


def _init(cfg, out):
    global _HOL, _OUT
    os.environ.setdefault("GLOG_minloglevel", "2")
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process
    _HOL, _OUT = make_holistic(cfg), out


def _work(item):
    stem, path, info = item
    started = time.time()
    try:
        arr, meta = extract_video(path, _HOL)
        if meta["n_frames"] == 0:
            return stem, "failed", 0, 0.0, "no frames could be read"
        try:
            video = str(path.relative_to(ROOT).as_posix())
        except ValueError:
            video = str(path)
        meta.update(seconds=round(time.time() - started, 2), video=video, **info)
        atomic_save_npy(_OUT / f"{stem}.npy", arr)
        atomic_write_json(_OUT / f"{stem}.json", meta)
        return stem, "ok", meta["n_frames"], meta["seconds"], ""
    except Exception as e:  # noqa: BLE001 - one bad video must not stop the rest
        return stem, "failed", 0, 0.0, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=1)}"


class _NoBar:
    def update(self, n):
        pass

    def close(self):
        pass


def _bar(total):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc="extracting")
    except ImportError:
        return _NoBar()


def is_done(raw_dir, stem):
    return (raw_dir / f"{stem}.json").exists()


def extract_clips(clips, cfg, raw_dir, log_path, workers):
    """clips: {stem: (video path, info dict saved in the .json)}. -> dict(done, failed, seconds, frames)."""
    for leftover in raw_dir.glob("*.tmp"):
        leftover.unlink()
    todo = [(s, p, i) for s, (p, i) in clips.items() if not is_done(raw_dir, s)]
    print(f"{len(clips)} videos, {len(clips) - len(todo)} already extracted, {len(todo)} to extract, {workers} workers",
          flush=True)
    out = dict(done=0, failed=[], seconds=0.0, frames=0)
    if not todo:
        return out
    new_log = not log_path.exists()
    started = time.time()
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        log = csv.writer(log_file)
        if new_log:
            log.writerow(LOG_COLUMNS)
        batch = workers * RESTART_WORKER_AFTER
        bar = _bar(len(todo))
        try:
            for first in range(0, len(todo), batch):
                pool = ProcessPoolExecutor(workers, initializer=_init, initargs=(cfg, raw_dir))
                try:
                    jobs = [pool.submit(_work, item) for item in todo[first:first + batch]]
                    for job in as_completed(jobs):
                        row = job.result()
                        log.writerow(row)
                        log_file.flush()
                        bar.update(1)
                        if row[1] == "ok":
                            out["done"] += 1
                            out["frames"] += row[2]
                        else:
                            out["failed"].append(row)
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
        except KeyboardInterrupt:
            print("\nStopping. Finished videos are saved; run the same command to continue.", flush=True)
        except BrokenProcessPool as e:
            raise RuntimeError("a keypoint worker stopped unexpectedly; finished videos are saved") from e
        finally:
            bar.close()
    out["seconds"] = time.time() - started
    if out["done"]:
        print(f"  {out['done']} ok, {len(out['failed'])} failed in {out['seconds'] / 60:.1f} min: "
              f"{out['done'] / out['seconds'] * 60:.1f} videos per minute, {out['frames'] / out['seconds']:.0f} frames "
              f"per second", flush=True)
    return out


def load_raw(raw_dir, stem):
    return np.load(raw_dir / f"{stem}.npy"), json.loads((raw_dir / f"{stem}.json").read_text(encoding="utf-8"))
