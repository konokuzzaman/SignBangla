"""BdSL-300, steps 1-3: list the dataset, compare its words with the words we already have, download the ones we need.

    python scripts/43_bdsl300_download.py --only list                  (steps 1-2: folder_list.csv, overlap.csv)
    python scripts/43_bdsl300_download.py --folders Eat,Go,kothay,noy,apni   (download a few words first)
    python scripts/43_bdsl300_download.py                              (every new and ambiguous word)

Kaggle hasibturjo/bdsl-300 (CC BY-SA 4.0, built on SignBD-Word) is one 23.5 GB zip. Like scripts/30, this reads the zip's
index once (saved as zip_index.json) and then fetches single clips by byte range, so only the wanted clips are
downloaded. The Kaggle token comes from KAGGLE_API_TOKEN or ~/.kaggle and is never printed or saved.

Writes, under speech2sign.bdsl300.dir (work/core_vocab/BdSL300/):
    folder_list.csv     all 300 folders: layout, clips, signers, size
    overlap.csv         each folder compared by meaning with BdSLW60, BdSLW401 and the core words that have a sign
                        (status new / duplicate / ambiguous; the Bangla is a guess, needs_check)
    videos/<folder>/    clips of the new and ambiguous words only (duplicates are never downloaded)
    download_log.csv    one line per clip as it finishes; download_failed.csv lists the failures of the last run
New and ambiguous words are also added to data/core_vocab/word_list.csv with verified = false.

Safe to stop and run again: a clip gets its final name only after its size and CRC are checked, finished clips are
skipped and failed ones retried. Next: python scripts/44_bdsl300_extract.py
"""
import argparse
import csv
import http.client
import importlib.util
import io
import json
import queue
import shutil
import threading
import time
import zipfile
import zlib
from collections import Counter, defaultdict

import _path  # noqa: F401

from signbangla.speech2sign import bdsl300 as b3
from signbangla.utils import ROOT, atomic_write_json, load_config, subdir, work_dir

INDEX_FILE = "zip_index.json"
LOG_COLUMNS = ["clip", "status", "bytes", "seconds", "error"]
REPORT_SECONDS = 30
MAX_FAILED = 200

_spec = importlib.util.spec_from_file_location("bdslw401_download", ROOT / "scripts" / "30_bdslw401_download.py")
kaggle_zip = importlib.util.module_from_spec(_spec)  # the Kaggle byte-range download of scripts/30, reused
_spec.loader.exec_module(kaggle_zip)


def load_index(base, url):
    """Every file in the zip with its place, read once and kept in zip_index.json."""
    path = base / INDEX_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    client = kaggle_zip.RangeClient(url)
    try:
        size = kaggle_zip.remote_size(client)
        print(f"Reading the zip's index ({size / 1e9:.1f} GB zip, only the index is fetched)...", flush=True)
        with zipfile.ZipFile(io.BufferedReader(kaggle_zip.RemoteFile(client, size), buffer_size=1 << 20)) as z:
            infos = z.infolist()
    finally:
        client.close()
    files = [dict(name=i.filename, offset=i.header_offset, method=i.compress_type, packed=i.compress_size,
                  size=i.file_size, crc=i.CRC) for i in infos if not i.is_dir()]
    index = dict(dataset_bytes=size, files=files)
    atomic_write_json(path, index)
    return index


def write_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(rows)


def folder_list(files):
    by = defaultdict(list)
    for e in files:
        p = b3.parse_clip(e["name"])
        if p:
            by[p["folder"]].append((p, e["size"]))
    rows = []
    for folder in sorted(by):
        clips = by[folder]
        rows.append(dict(folder=folder, layout="signbd_word" if b3.is_romanised(folder) else "english",
                         clips=len(clips), signers=len({p["signer"] for p, _ in clips}),
                         views="|".join(sorted({p["view"] or "single" for p, _ in clips})),
                         size_mb=round(sum(s for _, s in clips) / 1e6, 1)))
    return rows


def fetch_all(jobs, root, url, workers, log_path):
    """Download jobs (zip entries) with `workers` threads; one log line per clip. Returns the failures."""
    q = queue.Queue()
    for e in jobs:
        q.put(e)
    lock, stop, failed = threading.Lock(), threading.Event(), []
    done = Counter()
    new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(LOG_COLUMNS)

    def worker():
        client = kaggle_zip.RangeClient(url)
        while not stop.is_set():
            try:
                entry = q.get_nowait()
            except queue.Empty:
                break
            started = time.time()
            try:
                kaggle_zip.with_retries(lambda: kaggle_zip.fetch_file(entry, root, client))
                row = [entry["name"], "ok", entry["size"], round(time.time() - started, 1), ""]
            except (OSError, http.client.HTTPException, zlib.error, ValueError, RuntimeError) as e:
                row = [entry["name"], "failed", 0, round(time.time() - started, 1), f"{type(e).__name__}: {e}"[:200]]
            with lock:
                log.writerow(row)
                log_file.flush()
                if row[1] == "ok":
                    done["files"] += 1
                    done["bytes"] += entry["size"]
                else:
                    failed.append(row)
                    if len(failed) >= MAX_FAILED:  # probably the connection: stop instead of failing every clip
                        stop.set()
        client.close()

    total = sum(e["size"] for e in jobs)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    started = time.time()
    for t in threads:
        t.start()

    def report():
        speed = done["bytes"] / max(time.time() - started, 1e-3)
        left = (total - done["bytes"]) / speed / 3600 if speed else 0
        print(f"{done['files']}/{len(jobs)} clips, {done['bytes'] / 1e9:.2f}/{total / 1e9:.2f} GB, "
              f"{speed / 1e6:.1f} MB/s, about {left:.1f} h left, {len(failed)} failed", flush=True)

    next_report = started + REPORT_SECONDS
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            if time.time() >= next_report:
                report()
                next_report += REPORT_SECONDS
    except KeyboardInterrupt:
        stop.set()
        print("Stopping after the current clips. Run the same command to continue.", flush=True)
        for t in threads:
            t.join(timeout=30)
    report()
    log_file.close()
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--only", choices=["list"], help="list: steps 1-2 only (no download)")
    ap.add_argument("--folders", help="comma-separated folders to download (default: every new and ambiguous one)")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    s2s = cfg["speech2sign"]
    c, cv_cfg = s2s["bdsl300"], s2s["core_vocab"]
    base, videos = subdir(cfg, c["dir"]), subdir(cfg, c["videos_dir"])

    url = kaggle_zip.DownloadUrl(c["kaggle_dataset"])
    index = load_index(base, url)
    files = index["files"]
    folders = folder_list(files)
    write_csv(base / "folder_list.csv", folders, list(folders[0]))
    print(f"Step 1: {len(folders)} folders, {len(files)} clips, {sum(e['size'] for e in files) / 1e9:.2f} GB "
          f"-> {base / 'folder_list.csv'}")

    existing = b3.existing_words(ROOT / cfg["bdslw401"]["root"] / "word_list.csv",
                                 work_dir(cfg) / cv_cfg["dictionary_dir"] / "core_coverage.csv", ROOT / cv_cfg["word_list"])
    rows = b3.overlap_rows([f["folder"] for f in folders], existing)
    write_csv(base / "overlap.csv", rows, ["folder", "english", "bangla_guess", "needs_check", "status",
                                            "matches_existing_word", "category", "core_id", "note", "id"])
    counts = Counter(r["status"] for r in rows)
    print(f"Step 2: compared with {len(existing)} existing words: {counts['new']} new, {counts['duplicate']} duplicate, "
          f"{counts['ambiguous']} ambiguous -> {base / 'overlap.csv'}")
    clashes = b3.spelling_clashes(existing)
    if clashes:
        print(f"  spelled like an existing word but marked new (check these): {clashes}")
    added = b3.add_to_word_list(ROOT / cv_cfg["word_list"], rows)
    print(f"  {cv_cfg['word_list']}: {added} BdSL-300 words added (verified = false)")
    if args.only == "list":
        return

    wanted = {r["folder"] for r in rows if r["status"] in (b3.NEW, b3.AMBIGUOUS)}
    if args.folders:
        asked = [f.strip() for f in args.folders.split(",") if f.strip()]
        skipped = [f for f in asked if f not in wanted]
        if skipped:
            print(f"  not downloaded (duplicate or unknown folder): {skipped}")
        wanted &= set(asked)
    jobs = [e for e in b3.select_clips(files, c["views"]) if e["name"].split("/")[0] in wanted]
    todo = [e for e in jobs if not kaggle_zip.already_done(e, videos)]
    size = sum(e["size"] for e in todo)
    print(f"Step 3: {len(wanted)} words, {len(jobs)} clips (views: {c['views']}), "
          f"{sum(e['size'] for e in jobs) / 1e9:.2f} GB; already here {len(jobs) - len(todo)}; to fetch {len(todo)} "
          f"({size / 1e9:.2f} GB)", flush=True)
    free = shutil.disk_usage(videos).free
    if size * 1.02 > free:
        raise SystemExit(f"Not enough disk space: {size / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free.")
    failed = fetch_all(todo, videos, url, args.workers or c["download_workers"], base / "download_log.csv")
    write_csv(base / "download_failed.csv", [dict(zip(LOG_COLUMNS, r)) for r in failed], LOG_COLUMNS)
    if failed:
        raise SystemExit(f"{len(failed)} clips failed, listed in {base / 'download_failed.csv'}. "
                         "Run the same command to retry them.")
    print("Done. Next: python scripts/44_bdsl300_extract.py")


if __name__ == "__main__":
    main()
