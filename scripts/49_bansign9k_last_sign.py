"""Ban-Sign-Sent-9K: keypoints for a subset of the sentence videos, then the LAST sign of each video cut out as its own
trial (the marker words না, নেই, ছিল, হবে, হয়েছে).

    python scripts/49_bansign9k_last_sign.py --time-test 10   (choose the subset, extract 10 videos, print an estimate)
    python scripts/49_bansign9k_last_sign.py                  (all of the subset: videos, keypoints, cuts, reports)
    python scripts/49_bansign9k_last_sign.py --cut-only       (cuts, reports and previews from the keypoints there)

Subset (bansign9k.subset): up to per_marker videos per last word, spread evenly over the 5 signers, taken as groups of
sentences that differ only in the last word, so each video has another video of the same signer to align with.
Videos come from work/bansign9k/videos when scripts/48 has extracted them; otherwise each one is read from the split
zip (local parts if complete, else from Hugging Face by byte range) and checked against the zip's CRC.

Cutting (signbangla/extra_sources/last_sign.py, settings bansign9k.cut): DTW against the partner videos, else the
motion fallback; cuts shorter than min_s or longer than max_s are dropped. Under work/bansign9k/:
    subset.csv, video_index_remote.json (zip index), fetch_log.csv, keypoints/raw/S<id>_U<n>.npy + .json,
    keypoints/last_sign/<word id>__S<id>_U<n>.npy, cuts.csv (method per clip), word_quality.csv, extract_log.csv
Previews in work/core_vocab_check/bansign9k/: <word>.mp4 (several clips in a row) and sheet_<word>.png.
Safe to stop and run again. Research use only: videos and keypoints are never shared.
"""
import argparse
import csv
import json
import shutil
import struct
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import _path  # noqa: F401
import cv2
import numpy as np

from signbangla.extra_sources import bansign9k as B
from signbangla.extra_sources import split_zip as S
from signbangla.extra_sources.extract import extract_clips, is_done, load_raw
from signbangla.extra_sources.last_sign import cut_last_sign
from signbangla.extra_sources.word_list import add_words
from signbangla.quality import trial_quality
from signbangla.speech2sign.core_dictionary import candidate
from signbangla.speech2sign.render import draw_frame, render_options, save_animation
from signbangla.utils import ROOT, atomic_save_npy, atomic_write_json, load_config, progress, subdir

HF = "https://huggingface.co/datasets/{}/resolve/main/{}"
PART_SIZE = 2097152000  # every part but the last (from the Hugging Face listing)
WORD_IDS = {B.MARKERS[0]: "na", B.MARKERS[1]: "nei", B.MARKERS[2]: "chhilo", B.MARKERS[3]: "hobe",
            B.MARKERS[4]: "hoyeche"}
FETCH_WORKERS = 6
TILE, LABEL_H, COLUMNS = 240, 44, 6


def write_csv(path, rows, columns=None):
    columns = columns or (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(rows)


def zip_reader(cfg):
    """(reader, members): local parts when all are complete, else Hugging Face byte ranges."""
    c = cfg["bansign9k"]
    zip_dir = ROOT / c["zip_dir"]
    last = zip_dir / "Sign_Videos.zip"
    cache = subdir(cfg, c["dir"]) / "video_index_remote.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        members, sizes = saved["members"], saved["sizes"]
    else:
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location("bansign9k_videos", ROOT / "scripts" / "48_bansign9k_videos.py")
        v = module_from_spec(spec)
        spec.loader.exec_module(v)
        files = v.hf_files(c["hf_dataset"])
        names = sorted(n for n in files if n != "Sign_Videos.zip") + ["Sign_Videos.zip"]
        sizes = [files[n][0] for n in names]
        remote = S.HttpSplitReader([HF.format(c["hf_dataset"], n) for n in names], sizes)
        tail_len = min(sizes[-1], 1 << 16)
        tail = remote.read(len(sizes) - 1, sizes[-1] - tail_len, tail_len)
        i = tail.rfind(S.EOCD_SIG)
        _disk, _cd_disk, _n, _total, cd_size, cd_offset = struct.unpack("<HHHHII", tail[i + 4:i + 20])
        members = S.parse_central_directory(remote.read(len(sizes) - 1, cd_offset, cd_size))
        atomic_write_json(cache, dict(members=members, sizes=sizes))
    parts = S.part_paths(last, len(sizes))
    if all(p.exists() and p.stat().st_size == s for p, s in zip(parts, sizes)):
        return S.SplitReader(parts), members
    names = [p.name for p in parts]
    return S.HttpSplitReader([HF.format(c["hf_dataset"], n) for n in names], sizes), members


def fetch_videos(chosen, members, reader, videos_dir, log_path):
    by_name = {m["name"]: m for m in members}
    todo = [v for v in chosen if not ((videos_dir / v["name"]).exists()
                                      and (videos_dir / v["name"]).stat().st_size == by_name[v["name"]]["size"])]
    print(f"{len(chosen)} videos chosen, {len(chosen) - len(todo)} already on disk, {len(todo)} to fetch", flush=True)
    if not todo:
        return
    lock, new = threading.Lock(), not log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if new:
        log.writerow(["video", "status", "bytes", "seconds", "error"])

    def one(v):
        started = time.time()
        try:
            n = S.extract_member(reader, by_name[v["name"]], videos_dir / v["name"])
            return [v["name"], "ok", n, round(time.time() - started, 1), ""]
        except (OSError, ValueError) as e:
            return [v["name"], "failed", 0, round(time.time() - started, 1), f"{type(e).__name__}: {e}"[:200]]

    workers = FETCH_WORKERS if isinstance(reader, S.HttpSplitReader) else 1
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, v) for v in todo]
        for f in progress(as_completed(futures), total=len(futures), desc="fetching videos"):
            with lock:
                log.writerow(f.result())
                log_file.flush()
    log_file.close()


def video_fps(file_fps, c):
    """The frame rate the cut uses. Some videos (signers U5, U6) say 1000 fps in the file although they have as many
    frames per sentence as the 30 fps videos; a rate outside bansign9k.fps_range is replaced by bansign9k.fallback_fps."""
    lo, hi = c["fps_range"]
    return float(file_fps) if lo <= float(file_fps) <= hi else float(c["fallback_fps"])


def cut_all(chosen, cfg, raw_dir, cut_dir, sentences):
    c = cfg["bansign9k"]
    by_id = {s["sentence_id"]: s for s in sentences}
    partners = B.last_word_partners(sentences)
    have = {(v["sentence_id"], v["signer"]): B.video_stem(v) for v in chosen if is_done(raw_dir, B.video_stem(v))}
    cache = {}

    def raw_of(stem):
        if stem not in cache:
            cache[stem] = load_raw(raw_dir, stem)
        return cache[stem]

    rows = []
    for v in progress(chosen, total=len(chosen), desc="cutting"):
        stem = B.video_stem(v)
        if stem not in have.values():
            continue
        raw, meta = raw_of(stem)
        words = B.sign_words(by_id[v["sentence_id"]]["sign"])
        mine = [have[(p, v["signer"])] for p in partners[v["sentence_id"]] if (p, v["signer"]) in have]
        fps = video_fps(meta["fps"], c)
        cut = cut_last_sign(raw, fps, len(words), [raw_of(p)[0] for p in mine], c["cut"])
        row = dict(stem=stem, sentence_id=v["sentence_id"], signer=v["signer"], role=v["role"], word=words[-1],
                   sign=by_id[v["sentence_id"]]["sign"], n_signs=len(words), fps=fps, file_fps=meta["fps"],
                   width=meta["width"],
                   height=meta["height"], video_frames=len(raw), partners=" ".join(mine), **cut)
        if cut["ok"]:
            trial = raw[cut["start"]:cut["end"]]
            q = trial_quality(trial, cfg["quality"])
            row.update(hand=q["dominant_hand"], best_hand_frac=round(q["best_hand_frac"], 3), quality_flags=q["flags"],
                       good=q["ok"])
            if q["ok"]:
                atomic_save_npy(cut_dir / f"{WORD_IDS.get(words[-1], 'other')}__{stem}.npy", trial)
        else:
            row.update(hand="", best_hand_frac="", quality_flags="", good=False)
        rows.append(row)
    return rows


def word_report(rows):
    out = []
    for word in B.MARKERS:
        rs = [r for r in rows if r["word"] == word]
        good = [r for r in rs if r["good"]]
        out.append(dict(word=word, word_id=WORD_IDS[word], videos=len(rs), cut=sum(r["ok"] for r in rs),
                        good_clips=len(good), dtw=sum(r["method"] == "dtw" for r in good),
                        motion=sum(r["method"] == "motion" for r in good),
                        dropped_short=sum("shorter" in r["reason"] for r in rs),
                        dropped_long=sum("longer" in r["reason"] for r in rs),
                        signers=" ".join(f"{s}:{n}" for s, n in sorted(Counter(r["signer"] for r in good).items(),
                                                                       key=lambda x: int(x[0][1:]))),
                        right_hand=sum(r["hand"] == "R" for r in good), left_hand=sum(r["hand"] == "L" for r in good),
                        median_s=round(float(np.median([r["seconds"] for r in good])), 2) if good else ""))
    return out


def previews(rows, cfg, cut_dir, out_dir):
    s2s, c = cfg["speech2sign"], cfg["bansign9k"]
    opts = dict(render_options(s2s["render"]), style="skeleton")
    written = []
    for word in B.MARKERS:
        good = [r for r in rows if r["word"] == word and r["good"]]
        if not good:
            continue
        good.sort(key=lambda r: (r["method"] != "dtw", int(r["signer"][1:]), r["sentence_id"]))
        per_signer, picked = Counter(), []
        for r in good:  # spread the preview over the signers
            if per_signer[r["signer"]] < max(1, c["preview_clips"] // 5 + 1) and len(picked) < c["preview_clips"]:
                picked.append(r)
                per_signer[r["signer"]] += 1
        seqs, tiles = [], []
        for r in good[:COLUMNS * 4]:
            cand = candidate(np.load(cut_dir / f"{WORD_IDS[word]}__{r['stem']}.npy"), r["fps"], r["width"],
                             r["height"], cfg, r["stem"], trim=False)
            if cand is None:
                continue
            if r in picked:
                seqs.append(cand["seq"])
            seq = cand["seq"]
            img = draw_frame(seq[len(seq) // 2], "skeleton", TILE, opts["view"], supersample=2)
            bar = np.full((LABEL_H, TILE, 3), 255, np.uint8)
            cv2.putText(bar, f"{r['stem']} {r['method']}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1,
                        cv2.LINE_AA)
            cv2.putText(bar, f"{r['seconds']:.2f} s, hand {r['hand']}", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (90, 90, 90), 1, cv2.LINE_AA)
            tiles.append(np.vstack([img, bar]))
        if seqs:
            path = out_dir / f"{WORD_IDS[word]}.mp4"
            save_animation(np.concatenate(seqs), path, fps=c["preview_fps"], captions=word,
                           font_path=ROOT / s2s["font"], **opts)
            written.append(path)
        if tiles:
            blank = np.full_like(tiles[0], 255)
            tiles += [blank] * (-len(tiles) % COLUMNS)
            sheet = np.vstack([np.hstack(tiles[i:i + COLUMNS]) for i in range(0, len(tiles), COLUMNS)])
            path = out_dir / f"sheet_{WORD_IDS[word]}.png"
            cv2.imwrite(str(path), sheet)
            written.append(path)
    return written


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--time-test", type=int, metavar="N", help="extract only N videos and print a time estimate")
    ap.add_argument("--cut-only", action="store_true")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["bansign9k"]
    base, videos_dir = subdir(cfg, c["dir"]), subdir(cfg, c["videos_dir"])
    raw_dir, cut_dir = subdir(cfg, f"{c['keypoints_dir']}/raw"), subdir(cfg, f"{c['keypoints_dir']}/last_sign")
    sentences = B.read_sentences(ROOT / c["xlsx"])

    subset_path = base / "subset.csv"
    if subset_path.exists():
        with open(subset_path, encoding="utf-8-sig") as f:
            chosen = [dict(r, sentence_id=int(r["sentence_id"])) for r in csv.DictReader(f)]
        reader = members = None
    else:
        reader, members = zip_reader(cfg)
        videos = [dict(B.parse_video(m["name"]), name=m["name"]) for m in members if B.parse_video(m["name"])]
        chosen = B.choose_subset(videos, sentences, c["subset"]["per_marker"], c["subset"]["seed"],
                                 c["subset"]["max_partners"])
        write_csv(subset_path, chosen)
    print(f"Subset: {len(chosen)} videos, {dict(Counter((r['marker'], r['role']) for r in chosen))}", flush=True)

    if not args.cut_only:
        todo = chosen[:args.time_test] if args.time_test else chosen
        if any(not (videos_dir / v["name"]).exists() for v in todo):
            if reader is None:
                reader, members = zip_reader(cfg)
            fetch_videos(todo, members, reader, videos_dir, base / "fetch_log.csv")
        clips = {B.video_stem(v): (videos_dir / v["name"], dict(sentence_id=v["sentence_id"], signer=v["signer"],
                                                                  role=v["role"], marker=v["marker"]))
                 for v in todo if (videos_dir / v["name"]).exists()}
        stats = extract_clips(clips, cfg, raw_dir, base / "extract_log.csv", args.workers or c["extract_workers"])
        if args.time_test:
            if stats["done"]:
                per = stats["seconds"] / stats["done"]
                left = sum(not is_done(raw_dir, B.video_stem(v)) for v in chosen)
                print(f"Time test: {stats['done']} videos in {stats['seconds']:.0f} s ({per:.1f} s per video with "
                      f"{args.workers or c['extract_workers']} workers, {stats['frames'] / stats['done']:.0f} frames "
                      f"per video). The other {left} videos: about {left * per / 60:.0f} min.")
            return

    rows = cut_all(chosen, cfg, raw_dir, cut_dir, sentences)
    if not rows:
        print("No extracted videos yet.")
        return
    write_csv(base / "cuts.csv", rows)
    words = word_report(rows)
    write_csv(base / "word_quality.csv", words)
    out_dir = subdir(cfg, c["preview_dir"])
    written = previews(rows, cfg, cut_dir, out_dir)
    print(f"\n{len(rows)} videos cut: {dict(Counter(r['method'] for r in rows if r['ok']))} kept, "
          f"{sum(not r['ok'] for r in rows)} dropped ({dict(Counter(r['reason'] for r in rows if not r['ok']))})")
    print("| word | videos | good clips | dtw | motion | signers | R | L | median s |")
    for w in words:
        print(f"| {w['word']} | {w['videos']} | {w['good_clips']} | {w['dtw']} | {w['motion']} | {w['signers']} | "
              f"{w['right_hand']} | {w['left_hand']} | {w['median_s']} |")
    print(f"Previews: {', '.join(p.name for p in written)} in {out_dir.relative_to(ROOT)}")
    with_clips = {w["word"]: w["word_id"] for w in words if w["good_clips"]}
    added = add_words(ROOT / cfg["speech2sign"]["core_vocab"]["word_list"], B.SOURCE, "S",
                      B.word_list_rows(B.vocabulary(sentences), with_clips))
    print(f"Word list: {len(added)} rows added (source {B.SOURCE}, verified false)"
          + "".join(f"\n  {r['id']} {r['bangla']} ({r['category']}) folder={r['folder']} core_id={r['core_id']}"
                    for r in added))
    free = shutil.disk_usage(base).free / 1e9
    print(f"Free on the drive: {free:.0f} GB")


if __name__ == "__main__":
    main()
