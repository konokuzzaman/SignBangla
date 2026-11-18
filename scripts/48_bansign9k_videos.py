"""Ban-Sign-Sent-9K videos: download the split zip (Sign_Videos.z01 ... z09 + Sign_Videos.zip, about 20.5 GB) from
Hugging Face and extract it, without joining the parts.

    python scripts/48_bansign9k_videos.py --index       (free space check, Sign_Videos.zip only, video_index.csv)
    python scripts/48_bansign9k_videos.py               (also the other parts, then every video into work/)
    python scripts/48_bansign9k_videos.py --no-extract  (download only)

Stops before downloading when the drive has less than bansign9k.min_free_gb free. Downloads resume where they
stopped (HTTP Range) and are checked against the size and SHA-256 that Hugging Face lists; one log line per part in
work/bansign9k/download_log.csv. Videos are written as work/bansign9k/videos/Sign_Videos/<sentence id>/<file>, each
checked against the zip's CRC; finished videos are skipped. Safe to stop and run again.
Research use only (no licence stated): the videos are never shared. Next: python scripts/49_bansign9k_last_sign.py
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import _path  # noqa: F401

from signbangla.extra_sources import bansign9k as B
from signbangla.extra_sources import split_zip as S
from signbangla.utils import ROOT, load_config, progress, subdir

USER_AGENT = "signbangla"
LAST = "Sign_Videos.zip"


def hf_files(dataset):
    """{file name: (size, sha256)} of the video parts, from the Hugging Face file listing."""
    url = f"https://huggingface.co/api/datasets/{dataset}/tree/main"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as r:
        items = json.loads(r.read().decode("utf-8"))
    return {i["path"]: (i["size"], i.get("lfs", {}).get("oid", "")) for i in items
            if i["type"] == "file" and i["path"].startswith("Sign_Videos.")}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def log_line(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "file", "status", "bytes", "seconds", "note"])
        w.writerow(row)


def download(dataset, name, size, digest, dest_dir, log_path):
    """Resumable download of one part; the final name appears only after the size and SHA-256 check."""
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size == size:
        return "present"
    part = dest.with_name(name + ".part")
    url = f"https://huggingface.co/datasets/{dataset}/resolve/main/{name}"
    started, have = time.time(), part.stat().st_size if part.exists() else 0
    first, last_report = have, time.time()
    while have < size:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as f:
                if have and r.status != 206:
                    raise IOError(f"server ignored the byte range (HTTP {r.status})")
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if time.time() - last_report > 30:
                        speed = (have - first) / (time.time() - started)
                        print(f"  {name}: {have / 1e9:.2f}/{size / 1e9:.2f} GB, {speed / 1e6:.1f} MB/s, "
                              f"about {(size - have) / max(speed, 1) / 60:.0f} min left", flush=True)
                        last_report = time.time()
        except (OSError, IOError) as e:
            log_line(log_path, [time.strftime("%Y-%m-%d %H:%M:%S"), name, "retry", have, round(time.time() - started),
                                f"{type(e).__name__}: {e}"[:200]])
            time.sleep(10)
            have = part.stat().st_size
    if have != size:
        raise IOError(f"{name}: {have} bytes, expected {size}")
    if digest and sha256(part) != digest:
        part.unlink()
        raise IOError(f"{name}: SHA-256 does not match; the file was removed, run again")
    part.replace(dest)
    log_line(log_path, [time.strftime("%Y-%m-%d %H:%M:%S"), name, "ok", size, round(time.time() - started),
                        f"resumed at {first}" if first else ""])
    return "downloaded"


def write_index(members, sentences, path):
    by_id = {s["sentence_id"]: s for s in sentences}
    rows = []
    for m in members:
        v = B.parse_video(m["name"])
        s = by_id.get(v["sentence_id"]) if v else None
        rows.append(dict(name=m["name"], sentence_id=v["sentence_id"] if v else "", signer=v["signer"] if v else "",
                         date=v["date"] if v else "", kind=v["kind"] if v else "", part=m["disk"] + 1,
                         bytes=m["size"], in_sentence_list=s is not None,
                         last_word=B.marker(s["sign"]) if s else "", sign=s["sign"] if s else ""))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--index", action="store_true", help="only Sign_Videos.zip and the video index")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["bansign9k"]
    zip_dir, base = ROOT / c["zip_dir"], subdir(cfg, c["dir"])
    zip_dir.mkdir(parents=True, exist_ok=True)
    log_path = base / "download_log.csv"

    files = hf_files(c["hf_dataset"])
    need = sum(size for name, (size, _) in files.items() if not (zip_dir / name).exists())
    free = shutil.disk_usage(zip_dir).free
    print(f"Free on the drive: {free / 1e9:.0f} GB; still to download {need / 1e9:.1f} GB"
          + ("" if args.index else " + about 20.6 GB of extracted videos"), flush=True)
    if free / 1e9 < c["min_free_gb"]:
        raise SystemExit(f"Less than {c['min_free_gb']} GB free: stopping before the download.")

    print(f"{LAST}: {download(c['hf_dataset'], LAST, *files[LAST], zip_dir, log_path)}", flush=True)
    members, info = S.read_index(zip_dir / LAST)
    sentences = B.read_sentences(ROOT / c["xlsx"])
    rows = write_index(members, sentences, base / "video_index.csv")
    parsed = [r for r in rows if r["sentence_id"] != ""]
    print(f"{len(members)} videos in {info['disk'] + 1} parts; {len(parsed)} named Sign_Videos/<id>/(REC|UL)_CS_S<id>_"
          f"U<signer>_<date>_<ms>.mp4; signers {dict(Counter(r['signer'] for r in parsed))}; "
          f"{len({r['sentence_id'] for r in parsed})} sentences, "
          f"{sum(1 for r in parsed if not r['in_sentence_list'])} videos of sentences not in the xlsx", flush=True)
    if args.index:
        return

    for i in range(info["disk"]):
        name = f"Sign_Videos.z{i + 1:02d}"
        print(f"{name}: {download(c['hf_dataset'], name, *files[name], zip_dir, log_path)}", flush=True)
    if args.no_extract:
        return
    out_dir = subdir(cfg, c["videos_dir"])
    reader = S.SplitReader(S.part_paths(zip_dir / LAST, info["disk"] + 1))
    failed = []
    try:
        for m in progress(members, total=len(members), desc="extracting videos"):
            try:
                S.extract_member(reader, m, out_dir / m["name"])
            except (ValueError, OSError) as e:
                failed.append((m["name"], str(e)))
    finally:
        reader.close()
    print(f"Extracted into {out_dir.relative_to(ROOT)}: {len(members) - len(failed)} ok, {len(failed)} failed")
    for name, err in failed[:20]:
        print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
