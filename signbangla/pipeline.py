"""The data pipeline steps as plain functions (the scripts in scripts/ are thin wrappers around these).

work/ layout produced:
    manifest.csv              every video found (signer, word, fps, frames ...)
    raw_keypoints/            one (T,67,3) .npy + .json per VIDEO   (step 3)
    trials/                   one (T,67,3) .npy per TRIAL           (step 4)
    trials_metadata.csv
    quality_report.csv        per-trial quality stats + flags       (step 5)
    processed/                one (64,67,3) .npy per TRIAL          (step 6)
    processed_metadata.csv
    splits.csv                trial_id, signer, split               (step 7)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .annotations import frames_slice, load_annotation_file
from .labels import N_CLASSES, parse_video_stem
from .preprocess import PreprocessError, detect_dominant_hand, mirror_sequence, preprocess_trial
from .quality import trial_quality
from .utils import atomic_save_npy, dataset_root, progress, subdir, work_dir


# ----------------------------------------------------------------------------- step 1: manifest
def scan_videos(cfg):
    root = dataset_root(cfg)
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root}\nEdit paths.dataset_root in the config file.")
    exts = {e.lower() for e in cfg["video"]["extensions"]}
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            info = parse_video_stem(p.stem)
            rows.append(dict(path=str(p), rel_path=str(p.relative_to(root)), folder=p.parent.name, stem=p.stem,
                             parsed=info is not None, signer=info["signer"] if info else -1,
                             word=info["word"] if info else "", view=info["view"] if info else "",
                             class_idx=info["class_idx"] if info else -1))
    return pd.DataFrame(rows, columns=["path", "rel_path", "folder", "stem", "parsed", "signer", "word", "view", "class_idx"])


def probe_videos(df):
    """Add fps / n_frames / width / height by opening each video with OpenCV (fast, no decoding)."""
    import cv2
    fps, nf, w, h = [], [], [], []
    for p in progress(df["path"], desc="probing videos"):
        cap = cv2.VideoCapture(p)
        fps.append(cap.get(cv2.CAP_PROP_FPS))
        nf.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        w.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        h.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
    out = df.copy()
    out["fps"], out["n_frames"], out["width"], out["height"] = fps, nf, w, h
    return out


def rotated_stems(cfg):
    """Videos stored upside down: their frames are turned upright before MediaPipe sees them."""
    return set(cfg["video"].get("rotate_180") or [])


def needs_extraction(stem, out_dir, rotated):
    """True if a video has no finished keypoints, or has keypoints made before it was listed as upside down."""
    side = out_dir / f"{stem}.json"
    if not side.exists():
        return True
    return stem in rotated and not json.loads(side.read_text()).get("rotated_180", False)


def usable_videos(cfg, manifest):
    m = manifest[manifest["parsed"] & (manifest["class_idx"] >= 0)]
    view = cfg["video"].get("view")
    if view:
        m = m[m["view"] == view.upper()]
    return m.reset_index(drop=True)


# ----------------------------------------------------------------------------- step 4: trials
def collect_annotations(cfg):
    """All JSON files under dataset_root -> ({video stem lower-case: [trial dicts]}, report rows)."""
    root = dataset_root(cfg)
    by_video, report = defaultdict(list), []
    for p in sorted(root.rglob("*.json")):
        try:
            trials = load_annotation_file(p, cfg["annotation"])
            err = ""
        except Exception as e:  # not an annotation file, or unreadable
            trials, err = [], f"{type(e).__name__}: {e}"
        no_video = sum(t["video"] is None for t in trials)
        for t in trials:
            if t["video"]:
                by_video[t["video"].lower()].append(t)
        report.append(dict(file=str(p.relative_to(root)), n_trials=len(trials), n_without_video_name=no_video, error=err))
    for v in by_video.values():
        v.sort(key=lambda t: t["start"])
    return dict(by_video), pd.DataFrame(report, columns=["file", "n_trials", "n_without_video_name", "error"])


def build_trials(cfg, manifest, whole_video=False, limit=None):
    """Cut every extracted video into trials using the annotations. Returns (trials_metadata DataFrame, problems list)."""
    raw_dir, out_dir = subdir(cfg, "raw_keypoints"), subdir(cfg, "trials")
    ann, _ = collect_annotations(cfg)
    vids = usable_videos(cfg, manifest)
    if limit:
        vids = vids.head(limit)
    rows, problems = [], []
    for v in progress(list(vids.itertuples()), desc="building trials"):
        npy = raw_dir / f"{v.stem}.npy"
        if not npy.exists():
            problems.append(f"{v.stem}: no raw keypoints yet (run step 3)")
            continue
        raw = np.load(npy)
        side = json.loads((raw_dir / f"{v.stem}.json").read_text()) if (raw_dir / f"{v.stem}.json").exists() else {}
        T = raw.shape[0]
        trials = ann.get(v.stem.lower(), [])
        if not trials:
            if whole_video:
                trials = [dict(start=0, end=T - 1, hand=None, whole=True)]
            else:
                problems.append(f"{v.stem}: no annotation found (use --whole-video if each file is one sign)")
                continue
        k = 0
        for t in trials:
            sl = (0, T) if t.get("whole") else frames_slice(t["start"], t["end"], T, cfg["annotation"])
            if sl is None:
                problems.append(f"{v.stem}: trial {t['start']}-{t['end']} unusable (video has {T} frames)")
                continue
            lo, hi = sl
            tid = f"{v.stem}_t{k:03d}"
            atomic_save_npy(out_dir / f"{tid}.npy", raw[lo:hi])
            rows.append(dict(trial_id=tid, video_stem=v.stem, signer=v.signer, word=v.word, class_idx=v.class_idx,
                             view=v.view, trial_idx=k, start=t["start"], end=t["end"], lo=lo, hi=hi, n_frames=hi - lo,
                             hand_annot=t.get("hand"), fps=side.get("fps"), width=side.get("width"),
                             height=side.get("height")))
            k += 1
    df = pd.DataFrame(rows)
    df.to_csv(work_dir(cfg) / "trials_metadata.csv", index=False)
    return df, problems


# ----------------------------------------------------------------------------- step 5: quality
def run_quality(cfg):
    meta = pd.read_csv(work_dir(cfg) / "trials_metadata.csv")
    tdir = subdir(cfg, "trials")
    stats = [trial_quality(np.load(tdir / f"{tid}.npy"), cfg["quality"]) for tid in progress(meta["trial_id"], desc="quality")]
    stats = pd.DataFrame(stats)
    df = pd.concat([meta.drop(columns=[c for c in stats.columns if c in meta]), stats], axis=1)
    df.to_csv(work_dir(cfg) / "quality_report.csv", index=False)
    return df


# ----------------------------------------------------------------------------- step 6: preprocess
def run_preprocess(cfg, include_bad=False, mirror=None):
    wd = work_dir(cfg)
    qpath = wd / "quality_report.csv"
    meta = pd.read_csv(qpath if qpath.exists() else wd / "trials_metadata.csv")
    if "ok" in meta and not include_bad:
        meta = meta[meta["ok"].astype(bool)]
    if meta.empty:
        raise ValueError("No trials to preprocess: every trial was flagged in step 5. Look at work/quality_plots/ "
                         "and work/bad_trials.csv, or relax the thresholds in the config (quality:).")
    mirror = cfg["preprocess"]["mirror_left_dominant"] if mirror is None else mirror
    tdir, pdir = subdir(cfg, "trials"), subdir(cfg, "processed")
    rows = []
    for r in progress(list(meta.itertuples()), desc="preprocessing"):
        try:
            seq, info = preprocess_trial(np.load(tdir / f"{r.trial_id}.npy"), r.width, r.height, cfg["preprocess"])
            dom = detect_dominant_hand(seq)
            if mirror and dom == "L":
                seq = mirror_sequence(seq)
            atomic_save_npy(pdir / f"{r.trial_id}.npy", seq)
            rows.append(dict(trial_id=r.trial_id, status="ok", error="", dominant_hand=dom, mirrored=bool(mirror and dom == "L"), **info))
        except PreprocessError as e:
            rows.append(dict(trial_id=r.trial_id, status="failed", error=str(e)))
    meta = meta.drop(columns=[c for c in ("ok", "flags") if c in meta])
    meta = meta.rename(columns={c: "raw_" + c for c in ("pose_frac", "lh_frac", "rh_frac", "dominant_hand") if c in meta})
    out = meta.merge(pd.DataFrame(rows), on="trial_id")
    out.to_csv(wd / "processed_metadata.csv", index=False)
    return out


# ----------------------------------------------------------------------------- step 7: splits
def choose_signer_split(counts, scfg):
    """counts: Series signer -> n trials. Returns {signer: 'train'|'val'|'test'}."""
    signers = sorted(counts.index)
    test = [s for s in scfg["test_signers"] if s in signers]
    rest = [s for s in signers if s not in test]
    cand = [s for s in rest if counts[s] >= scfg["min_val_trials"]] or sorted(rest, key=lambda s: -counts[s])
    rng = np.random.default_rng(scfg["seed"])
    val = sorted(rng.choice(cand, size=min(scfg["n_val_signers"], len(cand)), replace=False).tolist())
    return {s: ("test" if s in test else "val" if s in val else "train") for s in signers}


def check_splits(df):
    problems = []
    for split in ("train", "val", "test"):
        missing = sorted(set(range(N_CLASSES)) - set(df.loc[df["split"] == split, "class_idx"]))
        if missing:
            problems.append(f"split '{split}' has no samples for classes {missing}")
    if (df.groupby("signer")["split"].nunique() > 1).any():
        problems.append("a signer appears in more than one split")
    return problems


def make_splits(cfg):
    wd = work_dir(cfg)
    meta = pd.read_csv(wd / "processed_metadata.csv")
    meta = meta[meta["status"] == "ok"]
    if meta.empty:
        raise ValueError("No successfully preprocessed trials. Check work/quality_report.csv and work/processed_metadata.csv.")
    mapping = choose_signer_split(meta.groupby("signer").size(), cfg["splits"])
    df = meta[["trial_id", "signer", "word", "class_idx"]].copy()
    df["split"] = df["signer"].map(mapping)
    df.to_csv(wd / "splits.csv", index=False)
    problems = check_splits(df)
    n_train = sum(v == "train" for v in mapping.values())
    if n_train != cfg["splits"]["expected_train_signers"]:
        problems.append(f"{n_train} train signers (plan expects {cfg['splits']['expected_train_signers']})")
    return df, mapping, problems
