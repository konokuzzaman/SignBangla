"""Handover zip of our generated data for a teammate: signbangla_handover.zip in the project root.

    python scripts/tools/make_handover_zip.py --dry-run   (only the summary: files and size per folder)
    python scripts/tools/make_handover_zip.py             (summary, zip, then a check of the zip)

Only the paths in INCLUDE are packed, minus the exclusions below. Nothing in work/ or data/ is changed. The zip is
written as signbangla_handover.zip.part and renamed when it is complete, so a stopped run never leaves a zip that
looks finished. File names are stored as UTF-8 (Bangla folder names stay correct). Standard library only.
"""
import argparse
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "signbangla_handover.zip"
NOTES = "HANDOVER.md"  # next to the zip, also packed into it

INCLUDE = ["work", "data/core_vocab", "data/module3", "data/bansign9k", "data/BdSLW401/word_list.csv"]

# Exclusions (paths relative to the project root, with forward slashes)
EXCLUDE_PATHS = {
    # videos
    "work/core_vocab/BdSL300/videos",
    "work/dynamic_du/videos",
    "work/bansign9k/videos",
    "data/bansign9k/zip",                            # part of the Ban-Sign-Sent-9K video archive
    # data the teammate already has
    "work/bdslw401/raw_keypoints",
    "work/raw_keypoints",
    # model folders, so nobody's checkpoints get overwritten
    "work/results",
    "work/bdslw401/results",
    "work/bdslw401/results.zip",
    # caches
    "work/module3/speech_cache",
    "work/.matplotlib",
    # personal data: the original sheet has signers' names and emails (sentences.csv holds what we use)
    "data/bansign9k/Bangla_Sign_Sentence.xlsx",
}
EXCLUDE_DIR_NAMES = {"__MACOSX", "__pycache__", ".pytest_cache"}  # anywhere
EXCLUDE_FILE_NAMES = {".DS_Store"}
EXCLUDE_SUFFIXES = {".mp4", ".avi", ".mov"}               # any video file, case ignored
PROGRESS_EVERY = 5000


def collect():
    """-> sorted list of (archive name, file path, size)."""
    files = []
    for entry in INCLUDE:
        start = ROOT / entry
        if start.is_file():
            files.append((entry, start, start.stat().st_size))
            continue
        if not start.is_dir():
            print(f"missing, skipped: {entry}", flush=True)
            continue
        for folder, dirs, names in os.walk(start):
            rel_folder = Path(folder).relative_to(ROOT).as_posix()
            dirs[:] = sorted(d for d in dirs
                             if d not in EXCLUDE_DIR_NAMES and f"{rel_folder}/{d}" not in EXCLUDE_PATHS)
            for name in sorted(names):
                rel = f"{rel_folder}/{name}"
                if (name in EXCLUDE_FILE_NAMES or rel in EXCLUDE_PATHS
                        or Path(name).suffix.lower() in EXCLUDE_SUFFIXES):
                    continue
                path = Path(folder) / name
                files.append((rel, path, path.stat().st_size))
    notes = ROOT / NOTES
    if notes.exists():
        files.append((NOTES, notes, notes.stat().st_size))
    return sorted(files)


def group_of(name):
    parts = name.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else (parts[0] if len(parts) == 1 else name)


def size_text(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def summary(files):
    by = defaultdict(lambda: [0, 0])
    for name, _p, size in files:
        by[group_of(name)][0] += 1
        by[group_of(name)][1] += size
    width = max(len(g) for g in by)
    print(f"{'folder':<{width}}  {'files':>8}  {'size':>10}")
    for g in sorted(by):
        print(f"{g:<{width}}  {by[g][0]:>8,}  {size_text(by[g][1]):>10}")
    total = sum(s for _n, _p, s in files)
    print(f"{'total':<{width}}  {len(files):>8,}  {size_text(total):>10}", flush=True)


def write_zip(files):
    part = OUT.with_name(OUT.name + ".part")
    started = time.time()
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for i, (name, path, _size) in enumerate(files, 1):
            zf.write(path, name)
            if i % PROGRESS_EVERY == 0 or i == len(files):
                print(f"  {i:,} / {len(files):,} files ({time.time() - started:.0f} s)", flush=True)
    part.replace(OUT)


def verify(files):
    print("Checking the zip ...", flush=True)
    with zipfile.ZipFile(OUT) as zf:
        bad = zf.testzip()
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
    ok = True
    print(f"  testzip: {'no corrupt entries' if bad is None else 'first corrupt entry: ' + bad}")
    ok &= bad is None
    print(f"  files: {len(names):,} in the zip, {len(files):,} in the dry run")
    ok &= len(names) == len(files)
    wanted = {n for n, _p, _s in files if not n.isascii()}
    in_zip = {n for n in names if not n.isascii()}
    missing = sorted(wanted - in_zip)
    print(f"  names with Bangla (non-ASCII) characters: {len(wanted):,} expected, {len(wanted & in_zip):,} found"
          + (f", missing e.g. {missing[:3]}" if missing else ""))
    ok &= not missing and len(in_zip) == len(wanted)
    folders = sorted({n.rsplit("/", 1)[0] for n in wanted})
    for f in folders[:5]:
        print(f"    {f}/")
    if len(folders) > 5:
        print(f"    ... {len(folders) - 5} more folders")
    print(f"  size: {size_text(OUT.stat().st_size)} ({OUT.stat().st_size:,} bytes)")
    print("Zip OK" if ok else "Zip check FAILED", flush=True)
    return ok


def main():
    sys.stdout.reconfigure(errors="replace")  # Bangla names on a console without UTF-8
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only print what would be packed")
    args = ap.parse_args()
    files = collect()
    summary(files)
    if args.dry_run:
        return 0
    if OUT.exists():
        print(f"{OUT.name} already exists; move or delete it first.", flush=True)
        return 1
    if not (ROOT / NOTES).exists():
        print(f"{NOTES} is missing; it should go into the zip.", flush=True)
        return 1
    write_zip(files)
    return 0 if verify(files) else 1


if __name__ == "__main__":
    sys.exit(main())
