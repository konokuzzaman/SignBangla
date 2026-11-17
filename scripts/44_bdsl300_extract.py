"""BdSL-300, steps 4-5: keypoints for the downloaded clips, then a quality check per clip and per word.

    python scripts/44_bdsl300_extract.py --folders Eat,Go,kothay,noy,apni     (test a few words; prints clips per minute)
    python scripts/44_bdsl300_extract.py                                      (every downloaded clip)
    python scripts/44_bdsl300_extract.py --check-only                         (quality reports only)

MediaPipe Holistic with exactly the settings of scripts/03 and 33 (signbangla.keypoints.make_holistic, extract_video),
so the keypoints have the 67-point format of BdSLW60 and BdSLW401; normalisation happens later, in the same
preprocessing as theirs (dictionary.trial_to_fps). Under speech2sign.bdsl300.keypoints_dir:
    raw/<folder>__<clip>.npy + .json      whole clip, NaN = missing (the .json is written last: it marks the clip done)
    trials/<folder>__<clip>.npy           the signing part (scripts/41: wrists raised), when there is one
Under speech2sign.bdsl300.dir:
    extract_log.csv       one line per clip as it finishes; extract_failed.csv lists the failures of the last run
    quality_report.csv    per clip: the checks of step 5 + upright, as for BdSL40 (ok = passes the project threshold)
    word_quality.csv      per word: good clips, clips with both hands in most frames, signer split, hands

Left-handed clips are not mirrored, the same as the core dictionary: each word later uses one hand only (right-hand
clips if there are any, else left-hand ones). The report counts them.
Safe to stop with Ctrl+C and run again: finished clips are skipped and failed ones retried.
Next: python scripts/45_bdsl300_previews.py
"""
import argparse
import csv
import json
import os
import signal
import time
import traceback
from collections import Counter
from multiprocessing import Pool

os.environ.setdefault("GLOG_minloglevel", "2")  # MediaPipe's log lines would bury the progress bar

import _path  # noqa: F401
import numpy as np
import pandas as pd

from signbangla.keypoints import extract_video, make_holistic
from signbangla.speech2sign import bdsl300 as b3
from signbangla.speech2sign.core_signs import check_clip
from signbangla.utils import ROOT, atomic_save_npy, atomic_write_json, load_config, progress, subdir

LOG_COLUMNS = ["clip", "status", "frames", "seconds", "error"]
RESTART_WORKER_AFTER = 500  # clips, so that slow memory growth inside MediaPipe cannot build up
_HOL = _OUT = None


def _init(cfg, out):
    global _HOL, _OUT
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process
    _HOL, _OUT = make_holistic(cfg), out


def _work(item):
    stem, path, info = item
    started = time.time()
    try:
        arr, meta = extract_video(path, _HOL)
        if meta["n_frames"] == 0:
            return stem, "failed", 0, 0.0, "no frames could be read"
        meta.update(seconds=round(time.time() - started, 2), video=str(path.relative_to(ROOT).as_posix()), **info)
        atomic_save_npy(_OUT / f"{stem}.npy", arr)
        atomic_write_json(_OUT / f"{stem}.json", meta)  # written last: its presence marks the clip as finished
        return stem, "ok", meta["n_frames"], meta["seconds"], ""
    except Exception as e:  # noqa: BLE001 - one bad clip must not stop the rest
        return stem, "failed", 0, 0.0, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=1)}"


def list_clips(videos, folders):
    """{stem: (path, info)} for the downloaded clips of `folders`."""
    out = {}
    for folder in sorted(folders):
        for path in sorted((videos / folder).glob("*")):
            info = b3.parse_clip(f"{folder}/{path.name}")
            if info:
                out[b3.clip_stem(f"{folder}/{path.name}")] = (path, info)
    return out


def extract(clips, cfg, raw_dir, base, workers):
    todo = [(s, p, i) for s, (p, i) in clips.items() if not (raw_dir / f"{s}.json").exists()]
    print(f"Step 4: {len(clips)} clips downloaded, {len(clips) - len(todo)} already extracted, {len(todo)} to extract, "
          f"{workers} workers", flush=True)
    if not todo:
        return
    log_path = base / "extract_log.csv"
    new_log = not log_path.exists()
    failures, done, frames = [], 0, 0
    started = time.time()
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        log = csv.writer(log_file)
        if new_log:
            log.writerow(LOG_COLUMNS)
        pool = Pool(workers, initializer=_init, initargs=(cfg, raw_dir), maxtasksperchild=RESTART_WORKER_AFTER)
        try:
            for row in progress(pool.imap_unordered(_work, todo), total=len(todo), desc="extracting"):
                log.writerow(row)
                log_file.flush()
                if row[1] == "ok":
                    done += 1
                    frames += row[2]
                else:
                    failures.append(row)
            pool.close()
        except KeyboardInterrupt:
            print("\nStopping. Finished clips are saved; run the same command to continue.", flush=True)
            pool.terminate()
        pool.join()
    wall = time.time() - started
    with open(base / "extract_failed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLUMNS)
        w.writerows(failures)
    if done:
        print(f"  {done} ok, {len(failures)} failed in {wall / 60:.1f} min: {done / wall * 60:.1f} clips per minute, "
              f"{frames / wall:.0f} frames per second, {frames / done:.0f} frames per clip", flush=True)
    if failures:
        print(f"  {len(failures)} clips failed, listed in {base / 'extract_failed.csv'}. Run again to retry them.")


def quality(clips, cfg, raw_dir, trial_dir):
    """Per-clip checks (the project threshold: step 5 + upright, on the signing part)."""
    c, b = cfg["speech2sign"]["core_vocab"], cfg["speech2sign"]["bdsl300"]
    rows = []
    for stem, (_path, info) in progress(clips.items(), total=len(clips), desc="checking"):
        if not (raw_dir / f"{stem}.json").exists():
            continue
        raw = np.load(raw_dir / f"{stem}.npy")
        meta = json.loads((raw_dir / f"{stem}.json").read_text(encoding="utf-8"))
        trial, checks = check_clip(raw, cfg["quality"], c["trim_raise_frac"], c["trim_margin_frames"])
        if trial is not None:
            atomic_save_npy(trial_dir / f"{stem}.npy", trial)
        both = bool(checks.get("lh_frac", 0) > b["both_hands_frac"] and checks.get("rh_frac", 0) > b["both_hands_frac"])
        rows.append(dict(stem=stem, folder=info["folder"], file=info["file"], signer=info["signer"], view=info["view"],
                         fps=meta["fps"], width=meta["width"], height=meta["height"], video_frames=len(raw),
                         both_hands=both, **checks))
    return pd.DataFrame(rows)


def word_report(report, overlap, min_good):
    rows = []
    for folder, g in report.groupby("folder"):
        good = g[g["ok"]]
        split = Counter(good["signer"])
        hands = Counter(good["dominant_hand"])
        o = overlap.loc[folder]
        rows.append(dict(folder=folder, english=o["english"], status=o["status"], clips=len(g), good_clips=len(good),
                         good_signers=len(split), both_hands_good=int(good["both_hands"].sum()),
                         right_hand=hands.get("R", 0), left_hand=hands.get("L", 0), mirrored=0,
                         signer_split=" ".join(f"{s}:{n}" for s, n in sorted(split.items(), key=lambda x: _signer_key(x[0]))),
                         views=" ".join(f"{v}:{n}" for v, n in sorted(Counter(good["view"].replace("", "single")).items())),
                         enough=len(good) >= min_good,
                         flags=" ".join(f"{f}:{n}" for f, n in Counter(
                             f for fl in g.loc[~g["ok"], "flags"] for f in str(fl).split("|") if f).items())))
    return pd.DataFrame(rows)


def _signer_key(s):
    return (0, s) if not s.startswith("p") else (1, f"{int(s[1:]):03d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--folders", help="comma-separated folders (default: every downloaded new or ambiguous word)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--check-only", action="store_true", help="no extraction, only the quality reports")
    args = ap.parse_args()
    cfg = load_config(args.config)
    b = cfg["speech2sign"]["bdsl300"]
    base, videos = subdir(cfg, b["dir"]), subdir(cfg, b["videos_dir"])
    raw_dir, trial_dir = subdir(cfg, f"{b['keypoints_dir']}/raw"), subdir(cfg, f"{b['keypoints_dir']}/trials")
    for leftover in raw_dir.glob("*.tmp"):  # from a stop in the middle of writing one of our own files
        leftover.unlink()

    overlap = pd.read_csv(base / "overlap.csv", encoding="utf-8-sig", keep_default_na=False).set_index("folder")
    wanted = set(overlap.index[overlap["status"].isin([b3.NEW, b3.AMBIGUOUS])])
    if args.folders:
        wanted &= {f.strip() for f in args.folders.split(",")}
    clips = list_clips(videos, wanted)
    if not args.check_only:
        extract(clips, cfg, raw_dir, base, args.workers or b["extract_workers"])

    report = quality(clips, cfg, raw_dir, trial_dir)
    if report.empty:
        print("No extracted clips yet.")
        return
    words = word_report(report, overlap, b["min_good_clips"])
    if args.folders:  # a test run: keep the full reports as they are
        report.to_csv(base / "quality_report_test.csv", index=False)
        words.to_csv(base / "word_quality_test.csv", index=False)
    else:
        report.to_csv(base / "quality_report.csv", index=False)
        words.to_csv(base / "word_quality.csv", index=False)
    ok = report[report["ok"]]
    print(f"Step 5: {len(ok)} of {len(report)} clips pass the quality threshold "
          f"(flags: {Counter(f for fl in report.loc[~report['ok'], 'flags'] for f in str(fl).split('|') if f)})")
    print(f"  good clips by hand: {Counter(ok['dominant_hand'])}; mirrored: 0 (one hand per word is used instead)")
    print(f"  good clips with both hands in most frames: {int(ok['both_hands'].sum())}")
    for view, g in report.groupby("view"):
        print(f"  view {view or 'single'}: {g['ok'].mean():.0%} pass, hand seen in {g['best_hand_frac'].mean():.0%} of "
              f"frames on average")
    few = words[~words["enough"]]
    print(f"  {len(words)} words; {len(few)} with fewer than {b['min_good_clips']} good clips"
          + (f": {', '.join(f'{r.folder} ({r.good_clips})' for r in few.itertuples())}" if len(few) else ""))


if __name__ == "__main__":
    main()
