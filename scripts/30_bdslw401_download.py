"""Download the BdSLW401 front-view clips and word-list PDF without downloading the rest of the 52 GB zip.

    python scripts/30_bdslw401_download.py
    python scripts/30_bdslw401_download.py --limit 20       (try a few clips first)

The zip on Kaggle allows partial requests, so this reads its index, then fetches each wanted file by its byte range and
checks the file's size and CRC. Files appear under their final name only when complete, and files already on disk are
skipped, so it is safe to press Ctrl+C or lose the connection and run the same command again.

Also writes word_list.csv and new_words.csv (words not in BdSLW60). Next: python scripts/31_bdslw401_verify.py
"""
import argparse
import base64
import http.client
import io
import json
import os
import queue
import re
import shutil
import struct
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path
from urllib.parse import urlsplit

import _path  # noqa: F401

from signbangla.labels import LABELS
from signbangla.labels401 import LABELS_401, write_word_list
from signbangla.utils import ROOT, atomic_write_json, load_config

USER_AGENT = "signbangla-download-dataset"  # Kaggle answers 404 to some default Python User-Agents
INDEX_FILE = "front_index.json"
FAILED_FILE = "download_failed.txt"
LOCAL_HEADER_SIZE = 30
EXTRA_GUESS = 512  # asked for in the same request as the data, so most files need only one
MAX_ATTEMPTS = 6
REPORT_SECONDS = 30
MAX_FAILED = 200


def kaggle_auth_header():
    """The Authorization header. Never print its value."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        path = Path.home() / ".kaggle" / "access_token"
        token = path.read_text(encoding="utf-8").strip() if path.exists() else None
    if token:
        return f"Bearer {token}"
    path = Path.home() / ".kaggle" / "kaggle.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return "Basic " + base64.b64encode(f"{data['username']}:{data['key']}".encode()).decode()
    raise SystemExit("Kaggle token not found. Set KAGGLE_API_TOKEN or create ~/.kaggle/access_token.")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class DownloadUrl:
    """The zip's temporary storage URL, shared by all threads and asked for again when it expires. It holds a signature,
    so it is never printed."""

    def __init__(self, dataset):
        self.api_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"
        self.lock = threading.Lock()
        self.version = 0
        self.refresh(0)

    def refresh(self, seen_version):
        with self.lock:
            if self.version != seen_version:
                return  # another thread already did it
            request = urllib.request.Request(self.api_url, headers={"Authorization": kaggle_auth_header(),
                                                                    "User-Agent": USER_AGENT})
            try:
                urllib.request.build_opener(_NoRedirect).open(request, timeout=60)
            except urllib.error.HTTPError as e:
                if e.code != 302:
                    raise RuntimeError(f"Kaggle answered {e.code} when asked for the download") from None
                location = e.headers["Location"]
            else:
                raise RuntimeError("Kaggle did not redirect to the download")
            parts = urlsplit(location)
            self.host, self.target = parts.netloc, f"{parts.path}?{parts.query}"
            self.version += 1


class RangeClient:
    """Byte-range requests over one kept-open connection. One per thread."""

    def __init__(self, url):
        self.url, self.conn = url, None

    def get(self, start, end):
        """Bytes start..end inclusive. Raises OSError or http.client.HTTPException on failure, so callers can retry."""
        version = self.url.version
        if self.conn is None or self.conn.host != self.url.host.split(":")[0]:
            self.close()
            self.conn = http.client.HTTPSConnection(self.url.host, timeout=120)
        try:
            self.conn.request("GET", self.url.target, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
            response = self.conn.getresponse()
            data = response.read()
        except (OSError, http.client.HTTPException):
            self.close()
            raise
        if response.status == 206:
            if len(data) != end - start + 1:
                self.close()
                raise OSError(f"short read: {len(data)} of {end - start + 1} bytes")
            return data
        if response.status in (400, 401, 403):  # the temporary URL has expired
            self.url.refresh(version)
        raise OSError(f"storage answered {response.status}")

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


class RemoteFile(io.RawIOBase):
    """Lets zipfile read the zip's index over the network."""

    def __init__(self, client, size):
        self.client, self.size, self.pos = client, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, buffer):
        count = min(len(buffer), self.size - self.pos)
        if count <= 0:
            return 0
        data = with_retries(lambda: self.client.get(self.pos, self.pos + count - 1))
        buffer[:count] = data
        self.pos += count
        return count


def with_retries(call):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call()
        except (OSError, http.client.HTTPException, zlib.error, ValueError):
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)


def remote_size(client):
    """The zip's total size, from a request for its headers only."""
    conn =http.client.HTTPSConnection(client.url.host, timeout=60)
    try:
        conn.request("HEAD", client.url.target, headers={"User-Agent": USER_AGENT})
        response = conn.getresponse()
        response.read()
        if response.status != 200:
            raise OSError(f"storage answered {response.status}")
        return int(response.headers["Content-Length"])
    finally:
        conn.close()


def load_index(root, client, wanted):
    """The wanted files' places in the zip. Saved after the first run, so a restart does not read the index again."""
    path = root / INDEX_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    size = remote_size(client)
    print(f"Reading the zip's index ({size / 1e9:.1f} GB zip, only the index is fetched now)...", flush=True)
    with zipfile.ZipFile(io.BufferedReader(RemoteFile(client, size), buffer_size=1 << 20)) as archive:
        infos = archive.infolist()
    files = [dict(name=i.filename, offset=i.header_offset, method=i.compress_type, packed=i.compress_size,
                  size=i.file_size, crc=i.CRC) for i in infos if wanted.match(i.filename)]
    index = dict(zip_bytes=size, zip_entries=len(infos), files=files)
    atomic_write_json(path, index)
    return index


def fetch_file(entry, root, client):
    """Fetch one file, check it, and only then give it its final name."""
    name = entry["name"].encode("utf-8")
    start, packed = entry["offset"], entry["packed"]
    data = client.get(start, start + LOCAL_HEADER_SIZE + len(name) + EXTRA_GUESS + packed - 1)
    if data[:4] != b"PK\x03\x04":
        raise ValueError("not a zip file header")
    name_len, extra_len = struct.unpack("<HH", data[26:30])
    begin = LOCAL_HEADER_SIZE + name_len + extra_len
    if len(data) < begin + packed:
        data = client.get(start, start + begin + packed - 1)
    payload = data[begin:begin + packed]
    if entry["method"] == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(payload, -15)
    elif entry["method"] != zipfile.ZIP_STORED:
        raise ValueError(f"unknown compression {entry['method']}")
    if len(payload) != entry["size"] or zlib.crc32(payload) != entry["crc"]:
        raise ValueError("size or CRC does not match")
    out = root / entry["name"]
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    part.write_bytes(payload)
    os.replace(part, out)


def already_done(entry, root):
    path = root / entry["name"]
    return path.exists() and path.stat().st_size == entry["size"]


def write_word_lists(root):
    write_word_list(root / "word_list.csv")
    known = {number for number, *_ in LABELS}
    write_word_list(root / "new_words.csv", [row for row in LABELS_401 if row[0] not in known])


class Totals:
    def __init__(self, count, size):
        self.lock = threading.Lock()
        self.count, self.size = count, size
        self.done = self.bytes = 0
        self.failed = []

    def add(self, size):
        with self.lock:
            self.done += 1
            self.bytes += size


def worker(jobs, root, url, totals, stop):
    client = RangeClient(url)
    while not stop.is_set():
        try:
            entry = jobs.get_nowait()
        except queue.Empty:
            break
        try:
            with_retries(lambda: fetch_file(entry, root, client))
            totals.add(entry["size"])
        except (OSError, http.client.HTTPException, zlib.error, ValueError, RuntimeError) as e:
            with totals.lock:
                totals.failed.append((entry["name"], type(e).__name__, str(e)[:80]))
                if len(totals.failed) >= MAX_FAILED:  # probably the connection, so do not race through the whole list
                    stop.set()
    client.close()


def report(totals, started):
    elapsed = max(time.time() - started, 1e-3)
    speed = totals.bytes / elapsed
    left = (totals.size - totals.bytes) / speed if speed else 0
    print(f"{totals.done}/{totals.count} files, {totals.bytes / 1e9:.2f}/{totals.size / 1e9:.2f} GB, "
          f"{speed / 1e6:.1f} MB/s, about {left / 3600:.1f} h left, {len(totals.failed)} failed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--limit", type=int, help="fetch at most this many missing files, then stop")
    ap.add_argument("--workers", type=int, help="parallel downloads (default: bdslw401.download_workers in the config)")
    args = ap.parse_args()
    cfg = load_config(args.config)["bdslw401"]
    root = ROOT / cfg["root"]
    root.mkdir(parents=True, exist_ok=True)
    write_word_lists(root)

    url = DownloadUrl(cfg["kaggle_dataset"])
    client = RangeClient(url)
    index = load_index(root, client, re.compile(cfg["download_pattern"]))
    client.close()
    missing = [e for e in index["files"] if not already_done(e, root)]
    total = sum(e["size"] for e in index["files"])
    print(f"Wanted {len(index['files'])} files, {total / 1e9:.2f} GB. Already here: {len(index['files']) - len(missing)}. "
          f"To fetch: {len(missing)}.", flush=True)
    if args.limit is not None:
        missing = missing[:args.limit]
    size = sum(e["size"] for e in missing)
    free = shutil.disk_usage(root).free
    if size * 1.02 > free:
        raise SystemExit(f"Not enough disk space: {size / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free.")

    jobs = queue.Queue()
    for entry in missing:
        jobs.put(entry)
    totals, stop, started = Totals(len(missing), size), threading.Event(), time.time()
    threads = [threading.Thread(target=worker, args=(jobs, root, url, totals, stop), daemon=True)
               for _ in range(args.workers or cfg["download_workers"])]
    for t in threads:
        t.start()
    next_report = started + REPORT_SECONDS
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            if time.time() >= next_report:
                report(totals, started)
                next_report += REPORT_SECONDS
    except KeyboardInterrupt:
        stop.set()
        print("Stopping after the current files. Run the same command to continue.", flush=True)
        for t in threads:
            t.join(timeout=30)
    report(totals, started)

    failed_path = root / FAILED_FILE
    if totals.failed:
        failed_path.write_text("".join(f"{n}\t{k}\t{m}\n" for n, k, m in totals.failed), encoding="utf-8")
        raise SystemExit(f"{len(totals.failed)} files failed, listed in {failed_path}. Run the same command to retry them.")
    if failed_path.exists():
        failed_path.unlink()
    print("Stopped early. Run the same command to continue." if stop.is_set() else
          "Done. Next: python scripts/31_bdslw401_verify.py", flush=True)


if __name__ == "__main__":
    main()
