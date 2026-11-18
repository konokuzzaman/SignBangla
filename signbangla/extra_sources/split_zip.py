"""Read a split zip (name.z01, name.z02, ..., name.zip) without joining the parts.

The index (central directory) is at the end of the last part (.zip), so the member list is known from that part alone:

    members = read_index(Path("Sign_Videos.zip"))     # [{name, disk, offset, method, packed, size, crc}, ...]
    parts = part_paths(Path("Sign_Videos.zip"), n_disks)
    extract_member(parts, members[0], Path("out/video.mp4"))

Disk i is name.z{i+1:02d}, the last disk is name.zip. A member's offset counts from the start of its disk; its data may
run on into the next part. Only stored and deflated members are supported (what Windows, zip and 7-Zip write).
"""
import os
import struct
import zlib
from pathlib import Path

EOCD_SIG, EOCD64_LOC_SIG, EOCD64_SIG = b"PK\x05\x06", b"PK\x06\x07", b"PK\x06\x06"
CD_SIG, LOCAL_SIG = b"PK\x01\x02", b"PK\x03\x04"
STORED, DEFLATED = 0, 8
CHUNK = 1 << 22


def _eocd(tail):
    i = tail.rfind(EOCD_SIG)
    if i < 0:
        raise ValueError("no end-of-zip record: not the last part of a zip")
    return i


def index_info(last_part):
    """-> dict(disk=this part's disk number, cd_disk, cd_offset, cd_size, entries), ZIP64 records followed."""
    size = last_part.stat().st_size
    with open(last_part, "rb") as f:
        f.seek(max(0, size - (1 << 16) - 22))
        tail = f.read()
        i = _eocd(tail)
        disk, cd_disk, _n_here, entries, cd_size, cd_offset = struct.unpack("<HHHHII", tail[i + 4:i + 20])
        loc = tail.rfind(EOCD64_LOC_SIG, 0, i)
        if loc >= 0 and (0xFFFF in (disk, cd_disk, entries) or 0xFFFFFFFF in (cd_size, cd_offset)):
            eocd64_disk, eocd64_offset, _total = struct.unpack("<IQI", tail[loc + 4:loc + 20])
            if eocd64_disk != disk:
                raise ValueError("ZIP64 record on another part is not supported")
            f.seek(eocd64_offset)
            rec = f.read(56)
            if rec[:4] != EOCD64_SIG:
                raise ValueError("bad ZIP64 end record")
            disk, cd_disk, _n_here, entries, cd_size, cd_offset = struct.unpack("<IIQQQQ", rec[16:56])
    return dict(disk=disk, cd_disk=cd_disk, cd_offset=cd_offset, cd_size=cd_size, entries=entries)


def _zip64_extra(extra, size, packed, offset, disk):
    """Fill in the 0xFFFFFFFF / 0xFFFF fields from the ZIP64 extra field (id 1)."""
    j = 0
    while j + 4 <= len(extra):
        hid, hlen = struct.unpack("<HH", extra[j:j + 4])
        data = extra[j + 4:j + 4 + hlen]
        if hid == 1:
            k = 0
            vals = []
            for field in ("size", "packed", "offset"):
                cur = {"size": size, "packed": packed, "offset": offset}[field]
                if cur == 0xFFFFFFFF:
                    vals.append((field, struct.unpack("<Q", data[k:k + 8])[0]))
                    k += 8
            got = dict(vals)
            size, packed, offset = got.get("size", size), got.get("packed", packed), got.get("offset", offset)
            if disk == 0xFFFF:
                disk = struct.unpack("<I", data[k:k + 4])[0]
        j += 4 + hlen
    return size, packed, offset, disk


def parse_central_directory(data):
    """Central directory bytes -> list of member dicts (directories left out)."""
    out, i = [], 0
    while i + 46 <= len(data) and data[i:i + 4] == CD_SIG:
        (flags, method, crc, packed, size, n_len, e_len, c_len, disk, offset) = struct.unpack(
            "<4xHH4xIIIHHHH6xI", data[i + 4:i + 46])
        raw = data[i + 46:i + 46 + n_len]
        name = raw.decode("utf-8" if flags & 0x800 else "cp437")
        extra = data[i + 46 + n_len:i + 46 + n_len + e_len]
        size, packed, offset, disk = _zip64_extra(extra, size, packed, offset, disk)
        if not name.endswith("/"):
            out.append(dict(name=name, disk=disk, offset=offset, method=method, packed=packed, size=size, crc=crc,
                            encrypted=bool(flags & 1)))
        i += 46 + n_len + e_len + c_len
    return out


def read_index(last_part):
    """Every member of the split zip, from its last part only (the central directory must start in that part)."""
    last_part = Path(last_part)
    info = index_info(last_part)
    if info["cd_disk"] != info["disk"]:
        raise ValueError(f"the index starts on part {info['cd_disk'] + 1}, not in {last_part.name}")
    with open(last_part, "rb") as f:
        f.seek(info["cd_offset"])
        data = f.read(info["cd_size"])
    members = parse_central_directory(data)
    if len(members) > info["entries"]:
        raise ValueError("more members than the end record says")
    return members, info


def part_paths(last_part, n_disks):
    """Parts in disk order: name.z01 ... name.z{n-1}, then name.zip."""
    last_part = Path(last_part)
    return [last_part.with_suffix(f".z{i + 1:02d}") for i in range(n_disks - 1)] + [last_part]


class SplitReader:
    """The parts read as one stream: (disk, offset) -> bytes, crossing into the next part where needed."""

    def __init__(self, parts):
        self.parts = [Path(p) for p in parts]
        self.sizes = [p.stat().st_size for p in self.parts]
        self.files = {}

    def _file(self, k):
        if k not in self.files:
            self.files[k] = open(self.parts[k], "rb")
        return self.files[k]

    def read(self, disk, offset, n):
        out = bytearray()
        while n > 0:
            if disk >= len(self.parts):
                raise ValueError("read past the last part")
            left = self.sizes[disk] - offset
            if left <= 0:
                disk, offset = disk + 1, offset - self.sizes[disk]
                continue
            f = self._file(disk)
            f.seek(offset)
            chunk = f.read(min(n, left))
            out += chunk
            n -= len(chunk)
            disk, offset = disk + 1, 0
        return bytes(out)

    def close(self):
        for f in self.files.values():
            f.close()
        self.files = {}


def data_start(reader, m):
    """(disk, offset) where the member's data begins, after its local header."""
    head = reader.read(m["disk"], m["offset"], 30)
    if head[:4] != LOCAL_SIG:
        raise ValueError(f"{m['name']}: no local header at part {m['disk'] + 1}, offset {m['offset']}")
    n_len, e_len = struct.unpack("<HH", head[26:30])
    disk, offset = m["disk"], m["offset"] + 30 + n_len + e_len
    while disk < len(reader.sizes) and offset >= reader.sizes[disk]:
        offset -= reader.sizes[disk]
        disk += 1
    return disk, offset


def extract_member(reader, m, dest):
    """Write one member to dest (via dest.part, checked against size and CRC). -> bytes written (0 if already there)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size == m["size"]:
        return 0
    if m.get("encrypted"):
        raise ValueError(f"{m['name']}: encrypted")
    if m["method"] not in (STORED, DEFLATED):
        raise ValueError(f"{m['name']}: compression method {m['method']} is not supported")
    disk, offset = data_start(reader, m)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    crc, written, left = 0, 0, m["packed"]
    inflater = zlib.decompressobj(-15) if m["method"] == DEFLATED else None
    with open(tmp, "wb") as out:
        while left > 0:
            n = min(CHUNK, left)
            chunk = reader.read(disk, offset, n)
            offset += n
            while disk < len(reader.sizes) and offset >= reader.sizes[disk]:
                offset -= reader.sizes[disk]
                disk += 1
            left -= n
            data = inflater.decompress(chunk) if inflater else chunk
            crc = zlib.crc32(data, crc)
            written += len(data)
            out.write(data)
        if inflater:
            data = inflater.flush()
            crc = zlib.crc32(data, crc)
            written += len(data)
            out.write(data)
    if written != m["size"] or crc != m["crc"]:
        tmp.unlink()
        raise ValueError(f"{m['name']}: size or CRC wrong")
    os.replace(tmp, dest)
    return written


class HttpSplitReader(SplitReader):
    """The same as SplitReader, but each part is a URL on a server that allows byte ranges (Hugging Face does), so
    single members can be fetched without downloading the parts. sizes: the parts' sizes in bytes."""

    def __init__(self, urls, sizes, user_agent="signbangla", attempts=6):
        self.urls, self.sizes, self.user_agent, self.attempts = list(urls), list(sizes), user_agent, attempts
        self.parts, self.files = self.urls, {}

    def _get(self, k, start, n):
        import time
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.urls[k], headers={"User-Agent": self.user_agent,
                                                            "Range": f"bytes={start}-{start + n - 1}"})
        for attempt in range(self.attempts):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise IOError(f"server ignored the byte range (HTTP {r.status})")
                    data = r.read()
                if len(data) != n:
                    raise IOError(f"short read: {len(data)} of {n} bytes")
                return data
            except (OSError, urllib.error.URLError):
                if attempt == self.attempts - 1:
                    raise
                time.sleep(2 ** attempt)

    def read(self, disk, offset, n):
        out = bytearray()
        while n > 0:
            if disk >= len(self.sizes):
                raise ValueError("read past the last part")
            left = self.sizes[disk] - offset
            if left <= 0:
                disk, offset = disk + 1, offset - self.sizes[disk]
                continue
            take = min(n, left)
            out += self._get(disk, offset, take)
            n -= take
            disk, offset = disk + 1, 0
        return bytes(out)

    def close(self):
        pass
