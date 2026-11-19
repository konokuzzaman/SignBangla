"""The core sign dictionary: BdSLW60 + BdSLW401 + core vocabulary (BdSL40, BDSL49 digits, INCLUDE if allowed) + the
BdSL-300, Ban-Sign-Sent-9K (last signs) and DU dynamic words marked verified = true in data/core_vocab/word_list.csv
(never the unverified ones).

    work/sign_dictionary_core/   same layout as work/sign_dictionary (see dictionary.py), plus per word: source,
                                 verified and category; and digit_poses.npy (the handshape pose of each digit ০-৯), so
                                 any number can be signed digit by digit.
Order of preference for a word: BdSLW60 (copied from work/sign_dictionary), then the core sources, then BdSLW401.
The BdSLW60, BdSLW401 and core keypoint files are only read, never changed.

Each word uses one signing hand: right-hand trials when there are any, else left-hand ones, never a mix.
"""
import json
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

from ..bdslw401 import priority_first
from ..labels import LABELS
from ..labels401 import LABELS_401, parse_clip_name
from ..preprocess import PreprocessError
from ..quality import trial_quality
from ..utils import ROOT, atomic_save_npy, atomic_write_json, progress, subdir, work_dir
from . import bdsl300 as b3
from . import core_vocab as cv
from .core_signs import hold_sign, signing_span
from .dictionary import SignDictionary, is_upright, split_variants, trial_to_fps

BDSLW60_NUMBERS = {n for n, *_ in LABELS}


def choose(candidates, n_trials, hand_weight):
    """candidates: dicts with seq, hand ('R'/'L'), hand_frac, duration_s -> the best n_trials of one hand. Right hand
    first; inside a hand: duration near the median, plus a penalty for frames without the signing hand."""
    if not candidates:
        return []
    hand = "R" if any(c["hand"] == "R" for c in candidates) else "L"
    same = [c for c in candidates if c["hand"] == hand]
    median = np.median([c["duration_s"] for c in same])
    for c in same:
        c["score"] = abs(c["duration_s"] - median) / median + hand_weight * (1 - c["hand_frac"])
    return sorted(same, key=lambda c: (c["score"], c["ref"]))[:n_trials]


def candidate(raw, fps, width, height, cfg, ref, trim=True):
    """One raw recording (NaN = missing) -> a candidate dict, or None when it fails the checks of step 5."""
    c = cfg["speech2sign"]["core_vocab"]
    if trim:
        span = signing_span(raw, c["trim_raise_frac"], c["trim_margin_frames"])
        if span is None:
            return None
        raw = raw[span[0]:span[1]]
    q = trial_quality(raw, cfg["quality"])
    if not q["ok"] or q["dominant_hand"] not in ("R", "L"):
        return None
    try:
        seq = trial_to_fps(raw, fps, width, height, cfg["preprocess"], cfg["speech2sign"]["fps"])
    except PreprocessError:
        return None
    if not is_upright(seq):
        return None
    hand = q["dominant_hand"]
    return dict(seq=seq, hand=hand, hand_frac=q["rh_frac"] if hand == "R" else q["lh_frac"],
                duration_s=len(raw) / fps, ref=ref)


def bdslw401_words(cfg, skip):
    """Words of BdSLW401 that are not in `skip`: up to bdslw401_candidates clips each, spread over signers."""
    c, s2s = cfg["speech2sign"]["core_vocab"], cfg["speech2sign"]
    raw_dir = subdir(cfg, f"{cfg['bdslw401']['work_subdir']}/raw_keypoints")
    by_word = defaultdict(list)
    for p in raw_dir.glob("*.json"):
        info = parse_clip_name(p.stem)
        if info and info["view"] == "F":
            by_word[info["word_number"]].append(p.stem)
    out = {}
    for number, bangla, roman, english in progress(LABELS_401, desc="BdSLW401 words"):
        key = f"W{number}"
        if key in skip or number not in by_word:
            continue
        stems = priority_first(by_word[number], c["bdslw401_candidates"], [])[: c["bdslw401_candidates"]]
        cands = []
        for stem in stems:
            meta = json.loads((raw_dir / f"{stem}.json").read_text(encoding="utf-8"))
            cand = candidate(np.load(raw_dir / f"{stem}.npy"), meta["fps"], meta["width"], meta["height"], cfg, stem)
            if cand:
                cands.append(cand)
        chosen = choose(cands, s2s["n_trials"], s2s["hand_weight"])
        if chosen:
            out[key] = (dict(bangla=split_variants(bangla), romanized=split_variants(roman), english=english,
                             category="word", source="BdSLW401", verified=True), chosen)
    return out


def video_words(cfg, source, ids):
    """Core words from the BdSL40 or INCLUDE videos (scripts/41): their checked, trimmed trials."""
    c, s2s = cfg["speech2sign"]["core_vocab"], cfg["speech2sign"]
    base = work_dir(cfg) / c["work_subdir"] / source
    if not (base / "quality_report.csv").exists():
        print(f"  {source}: no quality_report.csv (run scripts/41_core_vocab_extract.py); skipped")
        return {}
    q = pd.read_csv(base / "quality_report.csv")
    out = {}
    for word_id, rows in q[q["ok"].astype(bool)].groupby("id"):
        if word_id not in ids:
            continue
        cands = []
        for r in rows.itertuples():
            cand = candidate(np.load(base / "trials" / f"{r.stem}.npy"), r.fps, r.width, r.height, cfg, r.stem, trim=False)
            if cand:
                cands.append(cand)
        chosen = choose(cands, s2s["n_trials"], s2s["hand_weight"])
        if chosen:
            out[word_id] = (core_entry(word_id, source), chosen)
    return out


def bdsl300_words(cfg, verified):
    """BdSL-300 words verified by hand -> {key: (entry, chosen trials)}. verified: bdsl300.verified_words rows.
    A word linked to a core word (core_id, e.g. Eat -> C009 খাওয়া) gets that key, so verb forms and question words
    work with it; others get their B### id. Folders with the same key (Write and lekha) share their clips."""
    if not verified:
        return {}
    s2s, b = cfg["speech2sign"], cfg["speech2sign"]["bdsl300"]
    report = work_dir(cfg) / b["dir"] / "quality_report.csv"
    if not report.exists():
        print("  BdSL300: no quality_report.csv (run scripts/44_bdsl300_extract.py); skipped")
        return {}
    q = pd.read_csv(report, keep_default_na=False)
    q = q[q["ok"].astype(str) == "True"]
    # SignBD-Word films each signing with a close (c) and a full-body (f) camera: use one copy, the close one if it passed
    close = set(zip(q.loc[q["view"] == "c", "folder"], q.loc[q["view"] == "c", "signer"]))
    has_close = np.array([k in close for k in zip(q["folder"], q["signer"])], bool)
    q = q[~((q["view"] == "f").to_numpy() & has_close)]
    trial_dir = work_dir(cfg) / b["keypoints_dir"] / "trials"
    groups = defaultdict(lambda: dict(folders=[], meta=None))
    for word_id, folder, core_id, bangla, english, category in verified:
        key = core_id or word_id
        g = groups[key]
        g["folders"].append(folder)
        if g["meta"] is None:
            g["meta"] = (dict(core_entry(core_id, b3.SOURCE), verified=True) if core_id else
                         dict(bangla=split_variants(bangla.replace(" / ", "/")), romanized=[], english=english,
                              category=category, source=b3.SOURCE, verified=True))
    out = {}
    for key, g in groups.items():
        cands = []
        for r in q[q["folder"].isin(g["folders"])].itertuples():
            path = trial_dir / f"{r.stem}.npy"
            if path.exists():
                cand = candidate(np.load(path), float(r.fps), int(r.width), int(r.height), cfg, r.stem, trim=False)
                if cand:
                    cands.append(cand)
        chosen = choose(cands, s2s["n_trials"], s2s["hand_weight"])
        if chosen:
            out[key] = (dict(g["meta"], bdsl300_folders=g["folders"]), chosen)
        else:
            print(f"  BdSL300 {key} ({', '.join(g['folders'])}): verified but no usable clip")
    return out


EXTRA_SOURCES = ("BanSign9K", "DynamicDU")  # word-list sources with their own clips (scripts/47 and 49)


def extra_source_clips(cfg, source):
    """{clip group: [(npy path, fps, width, height, stem)]} of the good clips of one extra source. Groups: the
    Ban-Sign-Sent-9K word id (na, nei, ...) or the DU class folder."""
    out = defaultdict(list)
    if source == "BanSign9K":
        c = cfg["bansign9k"]
        base, clips = work_dir(cfg) / c["dir"], work_dir(cfg) / c["keypoints_dir"] / "last_sign"
        if not (base / "cuts.csv").exists() or not (base / "word_quality.csv").exists():
            return out
        ids = {_nfc(r.word): r.word_id for r in pd.read_csv(base / "word_quality.csv", encoding="utf-8-sig",
                                                            keep_default_na=False).itertuples()}
        cuts = pd.read_csv(base / "cuts.csv", encoding="utf-8-sig", keep_default_na=False)
        for r in cuts[cuts["good"].astype(str) == "True"].itertuples():
            wid = ids.get(_nfc(r.word))
            if wid:
                out[wid].append((clips / f"{wid}__{r.stem}.npy", float(r.fps), int(r.width), int(r.height), r.stem))
    elif source == "DynamicDU":
        c = cfg["dynamic_du"]
        base, trials = work_dir(cfg) / c["dir"], work_dir(cfg) / c["keypoints_dir"] / "trials"
        if not (base / "quality_report.csv").exists():
            return out
        q = pd.read_csv(base / "quality_report.csv", encoding="utf-8-sig", keep_default_na=False)
        for r in q[q["ok"].astype(str) == "True"].itertuples():
            out[r.folder].append((trials / f"{r.stem}.npy", float(r.fps), int(r.width), int(r.height), r.stem))
    return out


def _nfc(text):
    import unicodedata
    return unicodedata.normalize("NFC", str(text)).strip()


def core_id_for(bangla):
    """The core word (C###) spelled like `bangla` (any of its variants), or ''. For rows without a core_id."""
    names = {_nfc(v) for v in split_variants(bangla.replace(" / ", "/"))}
    return next((key for key, b, _e, _c in cv.CORE_WORDS if names & {_nfc(v) for v in split_variants(b)}), "")


def extra_source_words(cfg, rows):
    """Verified Ban-Sign-Sent-9K and DU rows of the word list -> {key: (entry, chosen trials)}, chosen like the other
    sources (quality checks, one signing hand, duration near the word's median). A row linked to a core word (core_id,
    e.g. না -> C035) gets that key; others keep their S### / D### id. Phrases (DU) keep all their words in the entry,
    so the text matcher can find them as a whole."""
    s2s = cfg["speech2sign"]
    out, missing = {}, []
    clips = {source: extra_source_clips(cfg, source) for source in EXTRA_SOURCES}
    for r in rows:
        if r.get("source") not in EXTRA_SOURCES or not b3.is_true(r.get("verified")):
            continue
        source = r["source"]
        found = clips[source].get(r.get("folder", ""), [])
        if not found and source == "BanSign9K":  # rows without a folder: match the word itself
            ids = {_nfc(w): w for w in clips[source]}
            found = clips[source].get(ids.get(_nfc(r["bangla"]), ""), [])
        cands = []
        for path, fps, width, height, stem in found:
            if path.exists():
                cand = candidate(np.load(path), fps, width, height, cfg, f"{source}/{stem}", trim=False)
                if cand:
                    cands.append(cand)
        chosen = choose(cands, s2s["n_trials"], s2s["hand_weight"])
        if not chosen:
            missing.append((r["id"], r["bangla"], "no clip passed the checks" if found else "no clips found"))
            continue
        core_id = r.get("core_id", "") or core_id_for(r["bangla"])
        key = core_id or r["id"]
        meta = (dict(core_entry(core_id, source), verified=True) if core_id else
                dict(bangla=split_variants(r["bangla"].replace(" / ", "/")), romanized=[], english=r["english"],
                     category=r["category"], source=source, verified=True))
        meta["word_list_id"] = r["id"]
        out[key] = (meta, chosen)
    for word_id, bangla, why in missing:
        print(f"  {word_id} {bangla}: verified but {why}")
    return out


def core_entry(word_id, source):
    _, bangla, english, category = next(r for r in cv.CORE_WORDS if r[0] == word_id)
    return dict(bangla=split_variants(bangla), romanized=[], english=english, category=category, source=source,
                verified=source != "INCLUDE")


def digit_words(cfg):
    """The digit signs of scripts/42 and their handshape poses."""
    c = cfg["speech2sign"]["core_vocab"]
    base = work_dir(cfg) / c["work_subdir"] / "BDSL49"
    out, poses, timing = {}, np.zeros((10, 67, 3), np.float32), None
    for i, d in enumerate(cv.BANGLA_DIGITS):
        path = base / f"digit_{d}.json"
        if not path.exists():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        poses[i] = np.array(meta["pose"], np.float32)
        seq = np.load(base / f"digit_{d}.npy")
        out[cv.DIGIT_IDS[d]] = (core_entry(cv.DIGIT_IDS[d], "BDSL49"),
                                [dict(seq=seq, hand="R", hand_frac=1.0, duration_s=len(seq) / meta["fps"],
                                      ref=f"BDSL49/{d}/{meta['photo']}")])
        timing = dict(move_frames=c["digit_move_frames"], hold_frames=int(round(c["digit_hold_s"] * meta["fps"])),
                      change_frames=c["digit_change_frames"])
    if not out:
        print("  BDSL49: no digit signs (run scripts/42_digits_from_images.py); skipped")
    return out, poses, timing


def build_core_dictionary(cfg, use_unverified=None):
    """-> dictionary.json content. use_unverified: None = speech2sign.core_vocab.use_unverified_signs."""
    s2s = cfg["speech2sign"]
    c = s2s["core_vocab"]
    use_unverified = c["use_unverified_signs"] if use_unverified is None else use_unverified
    base_dir = work_dir(cfg) / s2s["dictionary_dir"]
    base = SignDictionary(base_dir)
    out = subdir(cfg, c["dictionary_dir"])
    for old in out.glob("*.npy"):  # this folder is rebuilt from the sources every time
        old.unlink()
    shutil.copyfile(base_dir / base.info["rest_pose"], out / "rest_pose.npy")

    words = {}
    for key, e in base.words.items():  # BdSLW60 as built by scripts/20, unchanged
        shutil.copyfile(base_dir / e["file"], out / e["file"])
        words[key] = dict(e, category="word", source="BdSLW60", verified=True)

    found = {}
    found.update(video_words(cfg, "BdSL40", set(cv.BDSL40_SOURCES)))
    digits, poses, timing = digit_words(cfg)
    found.update(digits)
    if use_unverified:
        for key, value in video_words(cfg, "INCLUDE", set(cv.INCLUDE_SOURCES)).items():
            found.setdefault(key, value)
    if timing:  # 10, 20, 100 as their own entries, signed digit by digit
        for word_id, number in cv.COMPOSED_NUMBERS.items():
            idx = ["০১২৩৪৫৬৭৮৯".index(ch) for ch in number]
            if all(np.abs(poses[i]).sum() > 0 for i in idx):
                seq = hold_sign([poses[i] for i in idx], base.rest_pose, timing["move_frames"], timing["hold_frames"],
                                timing["change_frames"])
                found[word_id] = (dict(core_entry(word_id, "BDSL49 digit by digit"), verified=False),
                                  [dict(seq=seq, hand="R", hand_frac=1.0, duration_s=len(seq) / s2s["fps"], ref=number)])
    for key, value in bdsl300_words(cfg, b3.verified_words(ROOT / c["word_list"])).items():
        if key not in found or not found[key][0]["verified"]:  # a verified BdSL-300 sign replaces an unverified one
            found[key] = value
    for key, value in extra_source_words(cfg, b3.read_word_list(ROOT / c["word_list"])).items():
        if key not in found or not found[key][0]["verified"]:  # the same rule for Ban-Sign-Sent-9K and DU signs
            found[key] = value
    found.update(bdslw401_words(cfg, skip=set(words)))

    order = ([r[0] for r in cv.CORE_WORDS] + sorted(k for k in found if k[0] in "BSD")
             + sorted((k for k in found if k.startswith("W")), key=lambda k: int(k[1:])))
    for key in order:
        if key not in found or key in words:
            continue
        entry, chosen = found[key]
        atomic_save_npy(out / f"{key}.npy", np.concatenate([ch["seq"] for ch in chosen]))
        words[key] = dict(entry, class_idx=-1, file=f"{key}.npy",
                          trials=[dict(trial_id=ch["ref"], hand=ch["hand"], duration_s=round(float(ch["duration_s"]), 3),
                                       frames=len(ch["seq"])) for ch in chosen])
    info = dict(fps=s2s["fps"], n_keypoints=67, rest_pose="rest_pose.npy", rest_info=base.info.get("rest_info"),
                use_unverified_signs=bool(use_unverified), words=words)
    if timing:
        atomic_save_npy(out / "digit_poses.npy", poses)
        info["digits"] = dict(poses="digit_poses.npy", **timing)
    atomic_write_json(out / "dictionary.json", info)
    return info


def coverage(info):
    """The core word list with what the dictionary has for each word: a DataFrame."""
    rows = []
    for word_id, bangla, english, category in cv.CORE_WORDS:
        e = info["words"].get(word_id)
        rows.append(dict(id=word_id, bangla=bangla, english=english, category=category, has_sign=e is not None,
                         source=e["source"] if e else "", verified=("yes" if e["verified"] else "no") if e else "",
                         trials=len(e["trials"]) if e else 0, original_list=int(word_id[1:]) <= 70))
    return pd.DataFrame(rows)
