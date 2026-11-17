"""MODULE 2 core vocabulary, step 1: write the word list and download only the videos and photos we need.

    python scripts/40_core_vocab_download.py                  (everything)
    python scripts/40_core_vocab_download.py --only bdsl49    (bdsl40 | include | bdsl49 | list)

Writes data/core_vocab/word_list.csv from signbangla/speech2sign/core_vocab.py, then:
  BdSL40   the BdSL40 download link is dead, so its videos are taken from INCLUDE on Zenodo using the BdSL40 mapping
           (with its fix for 3 swapped I/you videos)      -> data/BdSL40/<id>/<video>
  INCLUDE  Indian SL videos of words on our list (unverified) -> data/INCLUDE/<id>/<video>
  BDSL49   photos of the digits 0-9 (recognition set)       -> data/BDSL49/<digit>/<photo>
Only the needed files are read out of the zips by byte range (a few GB instead of 2.8 + 8 GB of zips). Files already on
disk are skipped, so it is safe to stop and run it again. A manifest.csv in each folder lists what came from where.
Next: python scripts/41_core_vocab_extract.py
"""
import argparse
import csv
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

import _path  # noqa: F401

from signbangla.speech2sign import core_vocab as cv
from signbangla.speech2sign.remote_zip import USER_AGENT, fetch_member, list_members
from signbangla.utils import ROOT, atomic_write_json, load_config, progress

INDEX_FILE = "zip_index.json"  # the zips' indexes, saved so a restart does not read them again


def spread(items, n, key=None):
    """n items evenly spread over the sorted list (INCLUDE file numbers follow recording sessions, so this mixes
    signers). n = 0 or more than there are: all of them."""
    items = sorted(items, key=key)
    if not n or n >= len(items):
        return items
    if n == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def zip_indexes(urls, path):
    """{url: members}, read from the servers once and kept in `path`."""
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [u for u in urls if u not in index]
    for url in progress(missing, desc="reading zip indexes") if missing else []:
        index[url] = list_members(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, index)
    return {u: index[u] for u in urls}


def include_files(categories, index_path):
    """{INCLUDE word: [(zip url, member)]} for the given categories, with the BdSL40 label fixes applied."""
    urls = [u for c in categories for u in cv.include_zip_urls(c)]
    words = defaultdict(list)
    for url, members in zip_indexes(urls, index_path).items():
        for m in members:
            parts = PurePosixPath(m["name"]).parts
            if len(parts) < 3:
                continue
            word = parts[1].split(". ", 1)[1]
            word = cv.INCLUDE_LABEL_FIXES.get((word, PurePosixPath(m["name"]).stem), word)
            words[word].append((url, m))
    return words


def fetch_all(jobs, workers, desc):
    """jobs: (url, member, dest). Returns the failures; the rest are on disk."""
    def one(job):
        url, m, dest = job
        try:
            fetch_member(url, m, dest)
            return None
        except Exception as e:  # noqa: BLE001 - report and go on; running again retries it
            return f"{m['name']}: {type(e).__name__}: {e}"

    with ThreadPoolExecutor(workers) as pool:
        failed = [f for f in progress(pool.map(one, jobs), total=len(jobs), desc=desc) if f]
    for f in failed:
        print(f"  failed {f}")
    if failed:
        print(f"  {len(failed)} failed: run the same command again to retry them")
    return failed


def download_include(sources, out, per_word, workers):
    """sources: {id: (category, INCLUDE word, ...)} -> files under out/<id>/."""
    files = include_files(sorted({s[0] for s in sources.values()}), out / INDEX_FILE)
    jobs, rows = [], []
    for word_id, (category, word, *_rest) in sources.items():
        found = files.get(word, [])
        if not found:
            print(f"  {word_id}: INCLUDE word {word!r} not found")
        for url, m in spread(found, per_word, key=lambda f: f[1]["name"]):
            dest = out / word_id / PurePosixPath(m["name"]).name
            jobs.append((url, m, dest))
            rows.append(dict(id=word_id, include_category=category, include_word=word, member=m["name"],
                             file=dest.relative_to(out).as_posix(), bytes=m["file_size"]))
    todo = [j for j in jobs if not (j[2].exists() and j[2].stat().st_size == j[1]["file_size"])]
    print(f"  {len(jobs)} videos, {sum(r['bytes'] for r in rows) / 1e9:.2f} GB; {len(todo)} still to download")
    write_manifest(out / "manifest.csv", rows)
    fetch_all(todo, workers, out.name)


def download_bdsl49(out, per_digit, workers):
    url = urllib.request.urlopen(urllib.request.Request(cv.BDSL49_URL, method="HEAD",
                                                        headers={"User-Agent": USER_AGENT})).url  # signed storage link
    by_folder = defaultdict(list)
    for m in list_members(url):
        parts = PurePosixPath(m["name"]).parts
        if len(parts) == 4 and parts[2].isdigit() and int(parts[2]) in cv.BDSL49_DIGIT_FOLDERS:
            by_folder[int(parts[2])].append(m)
    jobs, rows = [], []
    for folder, members in sorted(by_folder.items()):
        digit = cv.BDSL49_DIGIT_FOLDERS[folder]
        for m in spread(members, per_digit, key=lambda x: x["name"]):
            split = PurePosixPath(m["name"]).parts[1]
            dest = out / digit / f"{split}_{PurePosixPath(m['name']).name}"
            jobs.append((url, m, dest))
            rows.append(dict(digit=digit, folder=folder, member=m["name"], file=dest.relative_to(out).as_posix(),
                             bytes=m["file_size"]))
    print(f"  {len(jobs)} photos, {sum(r['bytes'] for r in rows) / 1e6:.0f} MB")
    write_manifest(out / "manifest.csv", rows)
    fetch_all(jobs, workers, "BDSL49")


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--only", choices=["list", "bdsl40", "include", "bdsl49"])
    ap.add_argument("--workers", type=int, help="parallel downloads (default: speech2sign.core_vocab.download_workers)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["speech2sign"]["core_vocab"]
    if args.workers:
        c["download_workers"] = args.workers
    todo = [args.only] if args.only else ["list", "bdsl40", "include", "bdsl49"]

    if "list" in todo:
        cv.write_word_list(ROOT / c["word_list"])
        print(f"Wrote {c['word_list']} ({len(cv.CORE_WORDS)} words)")
    if "bdsl40" in todo:
        print("BdSL40 (from INCLUDE, verified for BdSL):")
        download_include(cv.BDSL40_SOURCES, ROOT / c["bdsl40_dir"], c["videos_per_word"], c["download_workers"])
    if "include" in todo:
        print("INCLUDE (Indian SL, not verified for BdSL):")
        download_include(cv.INCLUDE_SOURCES, ROOT / c["include_dir"], c["videos_per_word"], c["download_workers"])
    if "bdsl49" in todo:
        print("BDSL49 digits:")
        download_bdsl49(ROOT / c["bdsl49_dir"], c["digit_images_per_class"], c["download_workers"])
    print("Next: python scripts/41_core_vocab_extract.py")


if __name__ == "__main__":
    main()
