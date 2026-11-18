"""Local review page for the new sign words (BdSL-300, Ban-Sign-Sent-9K last signs, DU dynamic signs).

    python scripts\\tools\\review_words.py                    (opens the browser at the first unreviewed word)
    python scripts\\tools\\review_words.py --no-browser

Reviewed: every row of data/core_vocab/word_list.csv from those sources with verified = false, plus the rows already
answered on this page (review_log.csv). Words without clips are shown in a separate list and cannot be answered.
For each word the page shows the preview video and contact sheet made by scripts 45, 47b and 49, the numbers of the
quality reports and a consistency score (how close the word's clips are to each other compared with the clips of the
other words of the same source), computed once from the existing keypoints and kept in
work/core_vocab_check/consistency.csv.

Answers: Correct -> verified = true, Wrong -> verified = no (reason in the log), Not sure -> verified stays false
("unsure" in the log); the Bangla and English fields can be corrected at the same time. Saving changes only the line of
that word in word_list.csv (every other line stays byte for byte the same, UTF-8 with BOM kept). Before the first
change of a session the file is copied to data/core_vocab/backup/word_list_<time>.csv. Every answer is added to
data/core_vocab/review_log.csv.

The server uses the standard library only and is bound to 127.0.0.1.
"""
import argparse
import csv
import io
import json
import mimetypes
import os
import shutil
import sys
import threading
import time
import unicodedata
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REVIEW_SOURCES = ("BdSL300", "BanSign9K", "DynamicDU")
EDITABLE = ("bangla", "english", "verified")
LOG_COLUMNS = ["time", "id", "decision", "old", "new", "note"]
DECISIONS = {"correct": "true", "wrong": "no", "unsure": "false"}
BOM = "﻿"

FIRST = ("না", "নেই", "দয়া করে", "ঘুমানো", "হ্যালো")
MARKER_WORDS = ("ছিল", "হবে", "হয়েছে")
GROUP_NAMES = {1: "1. most used", 2: "2. verbs, questions, pronouns", 3: "3. tense markers and time words",
               4: "4. DU phrases", 5: "5. everything else"}
MONTHS_SEASONS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                  "november", "december", "spring", "summer", "rainy season", "autumn", "late autumn", "winter"}


def key(text):
    """Comparable form of a Bangla word (য় can be one or two code points)."""
    return unicodedata.normalize("NFD", str(text)).strip()


def is_false(value):
    return str(value).strip().lower() == "false"


# ----------------------------------------------------------------------------- word list file
class WordListFile:
    """Reads word_list.csv and changes single rows in place.

    Only the line of the changed row is rewritten; every other line, the BOM and the line endings stay as they are.
    The file is written to a temporary file first and then replaced, so a crash never leaves half a file."""

    def __init__(self, path, backup_dir, log_path, review_ids):
        self.path, self.backup_dir, self.log_path = Path(path), Path(backup_dir), Path(log_path)
        self.review_ids = set(review_ids)
        self.backup = None
        self.lock = threading.Lock()

    def _load(self):
        text = self.path.read_bytes().decode("utf-8")
        bom = text.startswith(BOM)
        lines = (text[1:] if bom else text).splitlines(keepends=True)
        header = next(csv.reader([lines[0]]))
        return bom, lines, header

    def rows(self):
        _bom, lines, header = self._load()
        return [dict(zip(header, next(csv.reader([line])))) for line in lines[1:] if line.strip()]

    def _backup_once(self):
        if self.backup is None:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            target = self.backup_dir / f"word_list_{datetime.now():%Y%m%d_%H%M%S}.csv"
            n = 1
            while target.exists():
                target = self.backup_dir / f"word_list_{datetime.now():%Y%m%d_%H%M%S}_{n}.csv"
                n += 1
            shutil.copy2(self.path, target)
            self.backup = target
        return self.backup

    def update(self, word_id, changes, decision, note=""):
        """changes: {column: new value} for columns in EDITABLE. -> (old row, new row)."""
        if word_id not in self.review_ids:
            raise ValueError(f"{word_id} is not under review")
        bad = set(changes) - set(EDITABLE)
        if bad:
            raise ValueError(f"cannot change {sorted(bad)}")
        for col, value in changes.items():
            if any(c in str(value) for c in "\r\n"):
                raise ValueError(f"{col} must be one line")
            if col == "bangla" and not str(value).strip():
                raise ValueError("bangla cannot be empty")
        with self.lock:
            bom, lines, header = self._load()
            id_col = header.index("id")
            found = None
            for i, line in enumerate(lines[1:], 1):
                fields = next(csv.reader([line]), [])
                if len(fields) != len(header):
                    if line.strip():
                        raise ValueError(f"line {i + 1} of {self.path.name} does not have {len(header)} columns")
                    continue
                if fields[id_col] == word_id:
                    if found is not None:
                        raise ValueError(f"{word_id} appears twice")
                    found = (i, fields)
            if found is None:
                raise ValueError(f"{word_id} not found")
            i, fields = found
            old = dict(zip(header, fields))
            new = dict(old, **{k: str(v).strip() for k, v in changes.items()})
            if new != old:
                line = lines[i]
                ending = line[len(line.rstrip("\r\n")):] or "\n"
                buf = io.StringIO()
                csv.writer(buf, lineterminator=ending).writerow([new[h] for h in header])
                lines[i] = buf.getvalue()
                self._backup_once()
                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_bytes(((BOM if bom else "") + "".join(lines)).encode("utf-8"))
                os.replace(tmp, self.path)
            self._log(word_id, decision, old, new, note)
            return old, new

    def _log(self, word_id, decision, old, new, note):
        changed = [k for k in EDITABLE if old.get(k) != new.get(k)] or ["verified"]
        new_file = not self.log_path.exists()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(LOG_COLUMNS)
            w.writerow([datetime.now().isoformat(timespec="seconds"), word_id, decision,
                        "; ".join(f"{k}={old.get(k, '')}" for k in changed),
                        "; ".join(f"{k}={new.get(k, '')}" for k in changed), note])


def read_log(path):
    """{id: last log row} of earlier answers."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def answer(store, word_id, decision, bangla=None, english=None, reason=""):
    """One answer from the page -> (old row, new row)."""
    if decision not in DECISIONS:
        raise ValueError(f"unknown decision {decision!r}")
    changes = {"verified": DECISIONS[decision]}
    if bangla is not None:
        changes["bangla"] = bangla
    if english is not None:
        changes["english"] = english
    note = reason.strip()
    if decision == "unsure":
        note = "unsure" + (f": {note}" if note else "")
    return store.update(word_id, changes, decision, note)


# ----------------------------------------------------------------------------- clips and reports per source
def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def source_paths(cfg):
    from signbangla.utils import work_dir
    w = work_dir(cfg)
    b3, b9, du = cfg["speech2sign"]["bdsl300"], cfg["bansign9k"], cfg["dynamic_du"]
    return dict(
        check=w / "core_vocab_check",
        bdsl300=dict(base=w / b3["dir"], trials=w / b3["keypoints_dir"] / "trials", preview=w / b3["preview_dir"],
                     sheet_words=int(b3["sheet_words"])),
        bansign9k=dict(base=w / b9["dir"], clips=w / b9["keypoints_dir"] / "last_sign", preview=w / b9["preview_dir"]),
        dynamic_du=dict(base=w / du["dir"], trials=w / du["keypoints_dir"] / "trials", preview=w / du["preview_dir"]),
    )


def word_sources(cfg):
    """Per source: {folder or word key: info} with good clips, signers, notes, preview files and the clip list
    [(npy path, fps, width, height, hand, signer)]."""
    p = source_paths(cfg)
    check = p["check"]

    def rel(path):
        return path.relative_to(check).as_posix() if path.exists() else ""

    out = {"BdSL300": {}, "BanSign9K": {}, "DynamicDU": {}}

    # BdSL-300
    b = p["bdsl300"]
    overlap = {r["folder"]: r for r in read_csv(b["base"] / "overlap.csv")}
    clips = defaultdict(list)
    for r in read_csv(b["base"] / "quality_report.csv"):
        if r["ok"] == "True":
            clips[r["folder"]].append((b["trials"] / f"{r['stem']}.npy", float(r["fps"]), int(r["width"]),
                                       int(r["height"]), r["dominant_hand"], r["signer"]))
    with_preview = sorted((f for f in clips if (b["preview"] / f"{f.replace(' ', '_')}.mp4").exists()), key=str.lower)
    sheet_of = {f: i // b["sheet_words"] + 1 for i, f in enumerate(with_preview)}
    for r in read_csv(b["base"] / "word_quality.csv"):
        f, o = r["folder"], overlap.get(r["folder"], {})
        n = sheet_of.get(f)
        out["BdSL300"][f] = dict(
            good_clips=int(r["good_clips"] or 0), clips=int(r["clips"] or 0), signers=r["signer_split"],
            unclear=o.get("note") or "ambiguous in overlap.csv" if o.get("status") == "ambiguous" else "",
            matches=o.get("matches_existing_word", ""),
            video=rel(b["preview"] / f"{f.replace(' ', '_')}.mp4"),
            sheet=rel(b["preview"] / f"sheet_{n:02d}.png") if n else "",
            sheet_note=f"tile {with_preview.index(f) % b['sheet_words'] + 1} of {b['sheet_words']}" if n else "",
            clip_list=clips.get(f, []))

    # Ban-Sign-Sent-9K last signs
    b = p["bansign9k"]
    word_ids = {}
    for r in read_csv(b["base"] / "word_quality.csv"):
        word_ids[key(r["word"])] = r["word_id"]
        wid = r["word_id"]
        out["BanSign9K"][key(r["word"])] = dict(
            good_clips=int(r["good_clips"] or 0), clips=int(r["videos"] or 0), signers=r["signers"], unclear="",
            matches="", video=rel(b["preview"] / f"{wid}.mp4"), sheet=rel(b["preview"] / f"sheet_{wid}.png"),
            sheet_note="", clip_list=[])
    for r in read_csv(b["base"] / "cuts.csv"):
        k = key(r["word"])
        if r.get("good") == "True" and k in out["BanSign9K"]:
            out["BanSign9K"][k]["clip_list"].append(
                (b["clips"] / f"{word_ids[k]}__{r['stem']}.npy", float(r["fps"]), int(r["width"]), int(r["height"]),
                 r["hand"], r["signer"]))

    # DU dynamic signs
    b = p["dynamic_du"]
    clips = defaultdict(list)
    for r in read_csv(b["base"] / "quality_report.csv"):
        if r["ok"] == "True":
            clips[r["folder"]].append((b["trials"] / f"{r['stem']}.npy", float(r["fps"]), int(r["width"]),
                                       int(r["height"]), r["dominant_hand"], ""))
    for r in read_csv(b["base"] / "class_table.csv"):
        f, name = r["folder"], r["folder"].replace(" ", "_")
        out["DynamicDU"][f] = dict(
            good_clips=int(r["good"] or 0), clips=int(r["videos"] or 0),
            signers="not recorded in the dataset", unclear=r.get("note", ""), matches="",
            video=rel(b["preview"] / f"{name}.mp4"), sheet=rel(b["preview"] / f"sheet_{name}.png"), sheet_note="",
            clip_list=clips.get(f, []))
    return out


def info_for(row, sources):
    table = sources.get(row["source"], {})
    if row["source"] == "BanSign9K":
        return table.get(key(row["bangla"]))
    return table.get(row["folder"])


# ----------------------------------------------------------------------------- consistency score
def clip_features(raw, fps, width, height, hand, pcfg, frames, shape_weight):
    """One clip -> a flat feature vector: arm path (shoulders, elbows, wrists) and the signing hand's shape relative to
    its wrist, both in shoulder widths, resampled to `frames`. Left-handed clips are mirrored first."""
    import numpy as np

    from signbangla.keypoints import GROUPS
    from signbangla.preprocess import mirror_sequence, preprocess_trial
    seq, _ = preprocess_trial(raw, width, height, dict(pcfg, target_frames=frames))
    if hand == "L":
        seq = mirror_sequence(seq)
    arm = seq[:, 11:17, :2].reshape(frames, -1)
    rh = seq[:, GROUPS["rh"], :2]
    present = np.abs(rh).sum((1, 2)) > 0
    shape = np.where(present[:, None, None], rh - rh[:, :1], 0).reshape(frames, -1)
    return np.concatenate([arm, shape_weight * shape], 1).ravel().astype(np.float64) / np.sqrt(frames)


def spread(clip_list, n):
    """Up to n clips, taken in turn from each signer."""
    by = defaultdict(list)
    for c in clip_list:
        by[c[5]].append(c)
    out = []
    while len(out) < n and any(by.values()):
        for s in sorted(by):
            if by[s] and len(out) < n:
                out.append(by[s].pop(0))
    return out


def consistency_scores(groups, warn_ratio, warn_nearest):
    """groups: {word: [feature vectors]} of one source. -> {word: dict(within, between, ratio, nearest_same, warn)}.

    within  = mean distance between the word's own clips; between = mean distance to the clips of the other words;
    ratio   = within / between (low = the clips agree and differ from other words);
    nearest_same = share of the word's clips whose nearest other clip belongs to the same word."""
    import numpy as np
    words = [w for w, f in groups.items() if len(f) >= 2]
    if len(words) < 2:
        return {}
    X = np.stack([f for w in words for f in groups[w]])
    labels = np.array([i for i, w in enumerate(words) for _ in groups[w]])
    sq = (X * X).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    np.fill_diagonal(D, np.inf)
    nearest = labels[np.argmin(D, 1)]
    out = {}
    for i, w in enumerate(words):
        mine = labels == i
        inner = D[np.ix_(mine, mine)]
        within = float(inner[np.isfinite(inner)].mean())
        between = float(D[np.ix_(mine, ~mine)].mean())
        ratio = within / between if between > 0 else float("nan")
        near = float((nearest[mine] == i).mean())
        reasons = []
        if ratio > warn_ratio:
            reasons.append(f"clips differ almost as much as different words (ratio {ratio:.2f})")
        if near <= warn_nearest:
            reasons.append(f"{near:.0%} of clips have a clip of this word as nearest")
        out[w] = dict(within=round(within, 3), between=round(between, 3), ratio=round(ratio, 3),
                      nearest_same=round(near, 3), warn=" and ".join(reasons))
    return out


CONSISTENCY_COLUMNS = ["id", "source", "word", "clips_used", "signers_used", "within", "between", "ratio",
                       "nearest_same", "chance", "warn"]


def compute_consistency(cfg, rows, sources, path):
    import numpy as np

    from signbangla.preprocess import PreprocessError
    r = cfg["review_words"]
    out = []
    for source in REVIEW_SOURCES:
        groups, meta = {}, {}
        ids = [w for w in rows if w["source"] == source]
        for w in ids:
            info = info_for(w, sources)
            if not info or not info["clip_list"]:
                continue
            feats, signers = [], set()
            for npy, fps, width, height, hand, signer in spread(info["clip_list"], r["consistency_clips"]):
                if not npy.exists():
                    continue
                try:
                    feats.append(clip_features(np.load(npy), fps, width, height, hand, cfg["preprocess"],
                                               r["consistency_frames"], r["hand_shape_weight"]))
                    signers.add(signer)
                except PreprocessError:
                    continue
            groups[w["id"]] = feats
            meta[w["id"]] = (w, len(feats), len(signers - {""}))
        scores = consistency_scores(groups, r["warn_ratio"], r["warn_nearest"])
        chance = round(1 / max(1, len(scores)), 3)
        print(f"  {source}: {len(scores)} words scored", flush=True)
        for wid, s in scores.items():
            w, n, ns = meta[wid]
            out.append(dict(id=wid, source=source, word=w["bangla"], clips_used=n, signers_used=ns, chance=chance, **s))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, CONSISTENCY_COLUMNS)
        w.writeheader()
        w.writerows(out)
    return {r["id"]: r for r in out}


def load_consistency(cfg, rows, sources):
    from signbangla.utils import work_dir
    path = work_dir(cfg) / cfg["review_words"]["consistency"]
    if path.exists():
        return {r["id"]: r for r in read_csv(path)}
    print(f"Computing the consistency scores once (kept in {path.relative_to(ROOT).as_posix()}) ...", flush=True)
    return {k: {c: str(v) for c, v in d.items()} for k, d in compute_consistency(cfg, rows, sources, path).items()}


# ----------------------------------------------------------------------------- order and page data
def group_of(row):
    k, source, cat = key(row["bangla"]), row["source"], row["category"]
    if k in {key(w) for w in FIRST}:
        return 1
    if source == "BdSL300" and cat in ("verb", "question", "pronoun"):
        return 2
    if k in {key(w) for w in MARKER_WORDS}:
        return 3
    if cat == "time" and row["english"].strip().lower() not in MONTHS_SEASONS:
        return 3
    if source == "DynamicDU":
        return 4
    return 5


def sort_key(row):
    g = group_of(row)
    k = key(row["bangla"])
    first = [key(w) for w in FIRST]
    markers = [key(w) for w in MARKER_WORDS]
    if g == 1:
        return g, first.index(k), row["id"]
    if g == 2:
        return g, ("verb", "question", "pronoun").index(row["category"]), row["id"]
    if g == 3:
        return g, markers.index(k) if k in markers else len(markers), row["id"]
    return g, row["category"], row["id"]


def status_of(row, last):
    if last:
        return last["decision"]
    v = row["verified"].strip().lower()
    return "correct" if v in ("true", "yes") else "wrong" if v == "no" else "unreviewed"


def review_rows(all_rows, log):
    return [r for r in all_rows if r.get("source") in REVIEW_SOURCES and (is_false(r["verified"]) or r["id"] in log)]


def build_words(cfg, store, sources, consistency, log_path):
    log = read_log(log_path)
    rows = review_rows(store.rows(), log)
    words, no_clips = [], []
    for r in sorted(rows, key=sort_key):
        info = info_for(r, sources)
        if not info or (not info["good_clips"] and not info["video"]):
            no_clips.append(dict(id=r["id"], bangla=r["bangla"], english=r["english"], source=r["source"],
                                 category=r["category"]))
            continue
        c = consistency.get(r["id"], {})
        last = log.get(r["id"])
        words.append(dict(
            id=r["id"], bangla=r["bangla"], english=r["english"], source=r["source"], folder=r["folder"],
            category=r["category"], verified=r["verified"], core_id=r.get("core_id", ""),
            group=group_of(r), group_name=GROUP_NAMES[group_of(r)], status=status_of(r, last),
            note=(last or {}).get("note", ""), good_clips=info["good_clips"], clips=info["clips"],
            signers=info["signers"], unclear=info["unclear"], matches=info["matches"], video=info["video"],
            sheet=info["sheet"], sheet_note=info["sheet_note"],
            consistency=dict(ratio=c.get("ratio", ""), nearest_same=c.get("nearest_same", ""),
                             chance=c.get("chance", ""), clips_used=c.get("clips_used", ""), warn=c.get("warn", ""))))
    return dict(words=words, no_clips=no_clips, groups=Counter(w["group_name"] for w in words))


# ----------------------------------------------------------------------------- server
class ReviewApp:
    def __init__(self, cfg, open_browser=True):
        r = cfg["review_words"]
        self.cfg = cfg
        self.web_dir = ROOT / r["web_dir"]
        self.font = ROOT / cfg["speech2sign"]["font"]
        self.check_dir = source_paths(cfg)["check"]
        self.log_path = ROOT / r["log"]
        path = ROOT / cfg["speech2sign"]["core_vocab"]["word_list"]
        first = WordListFile(path, ROOT / r["backup_dir"], self.log_path, [])
        rows = review_rows(first.rows(), read_log(self.log_path))
        self.store = WordListFile(path, ROOT / r["backup_dir"], self.log_path, [w["id"] for w in rows])
        self.sources = word_sources(cfg)
        self.consistency = load_consistency(cfg, rows, self.sources)
        self.open_browser = open_browser and r["open_browser"]

    def data(self):
        return build_words(self.cfg, self.store, self.sources, self.consistency, self.log_path)

    def serve(self):
        from signbangla.speech2sign.studio3d import _LocalServer
        r = self.cfg["review_words"]
        handler = _make_handler(self)
        last = None
        for port in range(r["port"], r["port"] + max(1, r["port_tries"])):
            try:
                httpd = _LocalServer((r["host"], port), handler)
                break
            except OSError as e:  # port busy: try the next one
                last = e
        else:
            raise OSError(f"no free port in {r['port']}-{r['port'] + r['port_tries'] - 1}: {last}")
        url = f"http://{r['host']}:{httpd.server_address[1]}/"
        d = self.data()
        todo = sum(w["status"] == "unreviewed" for w in d["words"])
        print(f"{len(d['words'])} words to review ({todo} not answered yet), {len(d['no_clips'])} without clips")
        for name, n in sorted(d["groups"].items()):
            print(f"  {name}: {n}")
        print(f"Review page: {url}  (Ctrl+C to stop)", flush=True)
        if self.open_browser:
            threading.Timer(0.5, webbrowser.open, [url]).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped. Every answer is already saved.")
        finally:
            httpd.server_close()


def _safe_file(base, rel):
    base = base.resolve()
    path = (base / unquote(rel)).resolve()
    return path if path.is_file() and base in path.parents else None


def _make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - name required by http.server
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._file(app.web_dir / "index.html")
            if path == "/api/words":
                return self._json(app.data())
            if path == "/font.ttf":
                return self._file(app.font)
            if path.startswith("/media/"):
                f = _safe_file(app.check_dir, path[len("/media/"):])
                return self._file(f) if f else self.send_error(HTTPStatus.NOT_FOUND)
            f = _safe_file(app.web_dir, path.lstrip("/"))
            return self._file(f) if f else self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/answer":
                return self.send_error(HTTPStatus.NOT_FOUND)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
                old, new = answer(app.store, body["id"], body["decision"], body.get("bangla"), body.get("english"),
                                  body.get("reason", ""))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            print(f"  {new['id']} {new['bangla']}: {body['decision']} (verified {old['verified']} -> {new['verified']})",
                  flush=True)
            return self._json({"ok": True, "row": new, "backup": str(app.store.backup or "")})

        def _file(self, path):
            size = path.stat().st_size
            kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix == ".js":
                kind = "text/javascript"
            start, end = 0, size - 1
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):  # videos are fetched in ranges
                a, _, b = rng[6:].split(",")[0].partition("-")
                if a:
                    start, end = int(a), int(b) if b else size - 1
                else:
                    start = max(0, size - int(b))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    return self.end_headers()
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", kind + ("; charset=utf-8" if kind.startswith("text/") else ""))
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        chunk = f.read(min(left, 1 << 16))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # the browser stopped the video request

        def _json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the console for the answers
            pass

    return Handler


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    from signbangla.utils import load_config
    cfg = load_config(args.config)
    started = time.time()
    app = ReviewApp(cfg, open_browser=not args.no_browser)
    print(f"Ready in {time.time() - started:.0f} s", flush=True)
    app.serve()


if __name__ == "__main__":
    main()
