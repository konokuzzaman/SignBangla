"""Read single files out of a zip on a web server without downloading the whole zip.

The server must allow partial requests (HTTP Range); Zenodo and Mendeley do. The zip's index (at the end of the file) is
read once with list_members; then each wanted member is fetched with ONE request (fetch_member) and checked against
the CRC in the index. Servers limit how many requests a client makes (Zenodo: about 60 a minute, then "429 Too Many
Requests"), so requests are spaced out and a 429 waits as long as the server asks.

    members = list_members(url)                      # [{name, offset, compress_size, file_size, method, crc}, ...]
    fetch_member(url, members[0], Path("data/x.mp4"))
"""
import http.client
import io
import os
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

USER_AGENT = "signbangla-remote-zip"
BLOCK = 1 << 20  # read ahead 1 MiB per request while reading the index
MAX_ATTEMPTS = 10
RATE_LIMIT_WAIT = 60  # seconds to wait after "429 Too Many Requests" when the server does not say how long
MIN_INTERVAL = 1.2  # seconds between the starts of two requests from this process
LOCAL_HEADER = 30
EXTRA_GUESS = 1024  # a local header's name + extra field, fetched with the data so one request is enough
_pace_lock, _last_start = threading.Lock(), [0.0]


def _pace():
    with _pace_lock:
        wait = _last_start[0] + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_start[0] = time.monotonic()


def _request(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
    for attempt in range(MAX_ATTEMPTS):
        _pace()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                if r.status != 206:
                    raise IOError(f"server ignored the byte range (HTTP {r.status}): {url}")
                return r.read()
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or attempt == MAX_ATTEMPTS - 1:
                raise
            wait = e.headers.get("Retry-After", "") if e.code == 429 else ""
            time.sleep(int(wait) if wait.isdigit() else RATE_LIMIT_WAIT if e.code == 429 else 2 ** attempt)
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError):
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(min(60, 2 ** attempt))  # IncompleteRead, reset connections: try again


def remote_size(url):
    _pace()
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        content_range = r.headers.get("Content-Range", "")
        if r.status != 206 or "/" not in content_range:
            raise IOError(f"server does not allow partial downloads: {url}")
        return int(content_range.rsplit("/", 1)[1])


class RemoteFile(io.RawIOBase):
    """A seekable read-only file over HTTP Range requests, with a simple read-ahead buffer (used to read the index)."""

    def __init__(self, url, size=None):
        self.url, self.size, self.pos = url, size or remote_size(url), 0
        self._buf_start, self._buf = 0, b""
        self.bytes_fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        end = self.pos + n
        if not (self._buf_start <= self.pos and end <= self._buf_start + len(self._buf)):
            fetch_end = min(self.size, max(end, self.pos + BLOCK)) - 1
            self._buf = _request(self.url, self.pos, fetch_end)
            self._buf_start = self.pos
            self.bytes_fetched += len(self._buf)
        off = self.pos - self._buf_start
        out = self._buf[off:off + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


class RemoteZip:
    """A ZipFile over a RemoteFile: for looking around (namelist, small members). Downloads use fetch_member."""

    def __init__(self, url, size=None):
        self.file = RemoteFile(url, size)
        self.zip = zipfile.ZipFile(self.file)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.zip.close()

    def infolist(self):
        return self.zip.infolist()

    def namelist(self):
        return self.zip.namelist()


def list_members(url):
    """The zip's index: one dict per file (directories left out)."""
    with RemoteZip(url) as z:
        return [dict(name=i.filename, offset=i.header_offset, compress_size=i.compress_size, file_size=i.file_size,
                     method=i.compress_type, crc=i.CRC) for i in z.infolist() if not i.is_dir()]


def fetch_member(url, m, dest):
    """Save one member (a dict from list_members) as `dest` with a single request. Checks size and CRC, and writes to
    a temporary name first, so a file under its final name is complete. Returns bytes written (0 if already there)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size == m["file_size"]:
        return 0
    if m["method"] not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise IOError(f"{m['name']}: compression method {m['method']} is not supported")
    for attempt in range(3):
        raw = _request(url, m["offset"], m["offset"] + LOCAL_HEADER + EXTRA_GUESS + m["compress_size"] - 1)
        if raw[:4] != b"PK\x03\x04":
            raise IOError(f"{m['name']}: no local file header at offset {m['offset']}")
        start = LOCAL_HEADER + int.from_bytes(raw[26:28], "little") + int.from_bytes(raw[28:30], "little")
        if len(raw) < start + m["compress_size"]:  # unusually long extra field: fetch the rest
            raw += _request(url, m["offset"] + len(raw), m["offset"] + start + m["compress_size"] - 1)
        data = raw[start:start + m["compress_size"]]
        if m["method"] == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(data, -15)
        if len(data) == m["file_size"] and zlib.crc32(data) == m["crc"]:
            break
    else:
        raise IOError(f"{m['name']}: size or CRC wrong after 3 tries")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return len(data)
