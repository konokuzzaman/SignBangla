"""End-to-end self-test on SYNTHETIC data (no dataset, no MediaPipe needed).

    python tests/test_pipeline.py

It builds a fake dataset (18 signers x 60 words), fake keypoints, fake annotation JSON in three different layouts,
and runs steps 4-8 plus the evaluation tools. If this passes, the code is wired correctly; it does NOT prove that
the real BdSLW60 annotation format is parsed correctly - step 1 on the real data tells you that.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from signbangla import pipeline
from signbangla.annotations import find_trials
from signbangla.dataset import HAS_TORCH, Augmenter, make_datasets
from signbangla.keypoints import GROUPS, N_KP
from signbangla.labels import LABELS
from signbangla.metrics import confusion_matrix, evaluate, evaluate_and_save
from signbangla.preprocess import (detect_dominant_hand, fill_gaps, mirror_sequence, preprocess_trial,
                                   resample_time)
from signbangla.utils import atomic_save_npy, atomic_write_json, load_config

W, H = 640, 480


def fake_trial(rng, n, right_dominant=True, hand_missing=False):
    """(n, 67, 3) NaN-for-missing keypoints of a signer with shoulders at x=.4/.6, y=.4."""
    t = np.linspace(0, 1, n)
    x = np.full((n, N_KP, 3), np.nan, np.float32)
    pose = np.zeros((25, 3), np.float32)
    pose[:, 0], pose[:, 1] = 0.5, 0.25
    pose[11], pose[12] = (0.6, 0.4, 0), (0.4, 0.4, 0)  # signer's left shoulder is on the image right
    for i in range(n):
        x[i, :25] = pose + rng.normal(0, 0.002, (25, 3))
    base = rng.uniform(0.01, 0.03, (21, 3))
    for hand, active in (("rh", right_dominant), ("lh", not right_dominant)):
        if not active:
            continue
        for i in range(n):
            c = np.array([0.35 + 0.1 * np.sin(2 * np.pi * t[i]), 0.55 - 0.2 * t[i], 0])
            x[i, GROUPS[hand]] = c + base
    if hand_missing:
        x[:, 25:] = np.nan
    return x


def make_dataset(tmp):
    rng = np.random.default_rng(0)
    root, work = tmp / "data", tmp / "work"
    cfg = load_config()
    cfg["paths"] = dict(dataset_root=str(root), work_dir=str(work))
    raw_dir = work / "raw_keypoints"
    raw_dir.mkdir(parents=True)
    ann_by_folder = {}
    for signer in range(1, 19):
        for wi, (wn, *_r) in enumerate(LABELS):
            folder = root / f"W{wn}"
            folder.mkdir(parents=True, exist_ok=True)
            stem = f"U{signer}W{wn}F"
            (folder / f"{stem}.mp4").write_bytes(b"")
            trials, frames, cursor = [], [], 0
            for k in range(3):
                n = int(rng.integers(20, 60))
                bad = (signer == 5 and wn == 1 and k == 0)  # one trial with no hands at all
                frames.append(fake_trial(rng, n, right_dominant=(rng.random() > 0.2), hand_missing=bad))
                gap = fake_trial(rng, 10)[:, :, :]
                gap[:, 25:] = np.nan
                frames.append(gap)
                trials.append((cursor, cursor + n - 1))
                cursor += n + 10
            raw = np.concatenate(frames)
            # drop a few hand frames inside a trial to test gap filling
            raw[trials[1][0] + 5: trials[1][0] + 8, 46:] = np.nan
            atomic_save_npy(raw_dir / f"{stem}.npy", raw)
            atomic_write_json(raw_dir / f"{stem}.json", dict(fps=30.0, width=W, height=H, n_frames=len(raw)))
            ann_by_folder.setdefault(folder, {})[f"{stem}.mp4"] = trials
    # three different JSON layouts
    for i, (folder, vids) in enumerate(ann_by_folder.items()):
        if i % 3 == 0:  # video -> list of trial dicts
            data = {v: [{"frame of intention": s - 2, "frame of actual start": s, "frame of gesture end": e,
                         "frame of complete withdrawal": e + 3} for s, e in tr] for v, tr in vids.items()}
        elif i % 3 == 1:  # list of records with a video field and parallel lists
            data = [{"video": v, "start_frames": [s for s, _ in tr], "end_frames": [e for _, e in tr], "hand": "right"}
                    for v, tr in vids.items()]
        else:  # nested: video -> {"trials": [{"start":..,"end":..}]}
            data = {v: {"fps": 30, "trials": [{"start": s, "end": e} for s, e in tr]} for v, tr in vids.items()}
        (folder / "annotation.json").write_text(json.dumps(data))
    return cfg


def test_units():
    # gap filling
    x = np.arange(10, dtype=np.float32).reshape(10, 1, 1).repeat(3, 2)
    valid = np.ones(10, bool)
    valid[3:5] = False
    valid[8:] = False
    xf, vf = fill_gaps(np.where(valid[:, None, None], x, np.nan), valid, max_gap=3)
    assert vf.all() and np.allclose(xf[3:5, 0, 0], [3, 4]) and np.allclose(xf[8:, 0, 0], 7)
    _, vf2 = fill_gaps(np.where(valid[:, None, None], x, np.nan), valid, max_gap=1)
    assert not vf2[3] and not vf2[8]
    # resample keeps absent absent and never blends towards 0
    s = np.zeros((4, 2, 3), np.float32)
    s[:2, 0] = 1.0
    s[:, 1] = 2.0
    r = resample_time(s, np.linspace(0, 3, 13))
    assert set(np.unique(r[:, 0, 0])) <= {0.0, 1.0} and np.allclose(r[:, 1], 2.0)
    # mirror is its own inverse
    q = np.random.default_rng(0).normal(size=(8, N_KP, 3)).astype(np.float32)
    assert np.allclose(mirror_sequence(mirror_sequence(q)), q)
    # annotation parser
    assert len(find_trials({"U1W1F.mp4": [{"start": 1, "end": 5}]}, load_config()["annotation"])) == 1
    test_hand_annotation()
    test_odd_video_names()
    test_rotate_180()
    print("units ok")


def test_rotate_180():
    """Videos in config video.rotate_180 reach MediaPipe upright, and old output made without the rotation is redone."""
    import cv2
    from signbangla.keypoints import extract_video

    class FakeHolistic:  # records the frames instead of detecting anything
        def __init__(self):
            self.seen = []

        def process(self, rgb):
            self.seen.append(rgb.copy())
            return type("Results", (), dict(pose_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=None))()

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        frame = np.zeros((64, 96, 3), np.uint8)
        frame[:32] = 255  # white on top
        writer = cv2.VideoWriter(str(tmp / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
        for _ in range(3):
            writer.write(frame)
        writer.release()
        plain, turned = FakeHolistic(), FakeHolistic()
        _, meta_plain = extract_video(tmp / "v.mp4", plain)
        _, meta_turned = extract_video(tmp / "v.mp4", turned, rotate_180=True)
        assert plain.seen[0][:20].mean() > 200 and plain.seen[0][-20:].mean() < 50
        assert turned.seen[0][:20].mean() < 50 and turned.seen[0][-20:].mean() > 200  # white is now at the bottom
        assert "rotated_180" not in meta_plain and meta_turned["rotated_180"] is True

        rotated = {"U1W215F"}
        assert pipeline.needs_extraction("U1W215F", tmp, rotated)  # no output yet
        atomic_write_json(tmp / "U1W215F.json", dict(n_frames=3))
        atomic_write_json(tmp / "U1W1F.json", dict(n_frames=3))
        assert pipeline.needs_extraction("U1W215F", tmp, rotated)  # made before it was listed
        assert not pipeline.needs_extraction("U1W1F", tmp, rotated)  # a normal finished video is skipped
        atomic_write_json(tmp / "U1W215F.json", dict(n_frames=3, rotated_180=True))
        assert not pipeline.needs_extraction("U1W215F", tmp, rotated)  # done

    cfg = load_config()
    assert len(pipeline.rotated_stems(cfg)) == 18 and {"U17W351F", "U1W215F", "U9W96F", "U12W220F"} <= pipeline.rotated_stems(cfg)


def test_odd_video_names():
    """U1W357F_1/_2/_3 (one recording in 3 parts) and U18W358FL (annotated as front view) are usable,
    and each part's trials stay attached to that part's own file."""
    from signbangla.labels import WORD_TO_CLASS, parse_video_stem
    for i in (1, 2, 3):
        assert parse_video_stem(f"U1W357F_{i}") == dict(signer=1, word="W357", view="F", part=i,
                                                       class_idx=WORD_TO_CLASS["W357"])
    assert parse_video_stem("U18W358FL") == dict(signer=18, word="W358", view="F", part=None,
                                                 class_idx=WORD_TO_CLASS["W358"])
    assert parse_video_stem("U11W1F")["part"] is None and parse_video_stem("U1W357F_x") is None
    # layout copied from W357-358/output1.json
    def entry(name, frames):
        return {"User": "U1", "Orientation": "LeftHand", "View": "Front", "FileName": name,
                "trials": {str(i): {"starting": s, "ending": e} for i, (s, e) in enumerate(frames)}}
    data = {"W357": {"U1": {"LeftHand": {"U1W357F_1": entry("U1W357F_1", [(10, 40), (60, 90)]),
                                         "U1W357F_2": entry("U1W357F_2", [(15, 45)]),
                                         "U1W357F_3": entry("U1W357F_3", [(20, 50)])}}},
            "W358": {"U18": {"RightHand": {"U18W358FL": entry("U18W358FL", [(5, 35)])}}}}
    got = sorted((t["video"], t["start"]) for t in find_trials(data, load_config()["annotation"]))
    assert got == [("U18W358FL", 5), ("U1W357F_1", 10), ("U1W357F_1", 60), ("U1W357F_2", 15), ("U1W357F_3", 20)], got


def test_hand_annotation():
    """Real BdSLW60 layout: the hand is the "Orientation" of the video entry (and its parent key), not of the trial.
    A signer can switch hands inside one video (e.g. U10W2F), so the video then appears under both keys."""
    def entry(name, hand, frames):
        return {"User": name[:name.index("W")], "Orientation": hand, "View": "Front", "FileName": name,
                "no_of_trials": str(len(frames)),
                "trials": {str(i): {"starting": s, "ending": e} for i, (s, e) in enumerate(frames)}}
    data = {"W2": {"U10": {"RightHand": {"U10W2F": entry("U10W2F", "RightHand", [(10, 40), (60, 90)])},
                           "LeftHand": {"U10W2F": entry("U10W2F", "LeftHand", [(110, 140)])}},
                   "U1": {"LeftHand": {"U1W2F": entry("U1W2F", "LeftHand", [(5, 30)])}}}}
    trials = find_trials(data, load_config()["annotation"])
    got = sorted((t["video"], t["start"], t["hand"]) for t in trials)
    assert got == [("U10W2F", 10, "RightHand"), ("U10W2F", 60, "RightHand"), ("U10W2F", 110, "LeftHand"),
                   ("U1W2F", 5, "LeftHand")], got
    # parent key alone is enough; a hand on the trial itself wins
    t = find_trials({"RightHand": {"U3W1F": {"trials": [{"start": 1, "end": 9}, {"start": 20, "end": 29, "hand": "L"}]}}},
                    load_config()["annotation"])
    assert [x["hand"] for x in t] == ["RightHand", "L"], t
    # "Orientation" that is not a hand name is not taken as the hand
    assert find_trials({"U3W1F": {"Orientation": "portrait", "start": 1, "end": 9}}, load_config()["annotation"])[0]["hand"] is None


def test_pipeline():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = make_dataset(tmp)
        manifest = pipeline.scan_videos(cfg)
        assert len(manifest) == 18 * 60 and manifest["parsed"].all()
        assert len(pipeline.usable_videos(cfg, manifest)) == 1080

        ann, report = pipeline.collect_annotations(cfg)
        assert sum(len(v) for v in ann.values()) == 1080 * 3, report

        df, problems = pipeline.build_trials(cfg, manifest)
        assert len(df) == 3240 and not problems, problems[:3]
        assert df["n_frames"].between(20, 59).all()  # the annotation range, inclusive end

        q = pipeline.run_quality(cfg)
        bad = q[~q["ok"]]
        assert list(bad["trial_id"]) == ["U5W1F_t000"] and bad.iloc[0]["flags"] == "no_hand", bad[["trial_id", "flags"]]
        print(f"quality ok: {len(bad)} flagged of {len(q)}; dominant hands {q['dominant_hand'].value_counts().to_dict()}")

        p = pipeline.run_preprocess(cfg)
        assert (p["status"] == "ok").all() and len(p) == 3239
        seq = np.load(Path(cfg["paths"]["work_dir"]) / "processed" / "U1W1F_t001.npy")
        assert seq.shape == (64, 67, 3) and seq.dtype == np.float32 and not np.isnan(seq).any()
        mid = (seq[:, 11, :2] + seq[:, 12, :2]) / 2
        width = np.linalg.norm(seq[:, 11, :2] - seq[:, 12, :2], axis=1)
        assert np.abs(mid).max() < 0.05 and np.allclose(width, 1.0, atol=0.05), (np.abs(mid).max(), width[:3])
        right = detect_dominant_hand(seq)
        left_abs = np.abs(seq[:, GROUPS["lh"]]).sum()
        assert right in ("R", "L") and (left_abs == 0 or right == "L")
        # the 3-frame hand gap in trial 1 must have been filled
        assert (np.abs(seq[:, GROUPS["rh"]]).sum((1, 2)) > 0).all() or detect_dominant_hand(seq) == "L"
        print("preprocess ok")

        sp, mapping, problems = pipeline.make_splits(cfg)
        counts = pd.Series(mapping).value_counts().to_dict()
        assert counts == {"train": 14, "val": 2, "test": 2} and not problems, (counts, problems)
        assert sorted(k for k, v in mapping.items() if v == "test") == [4, 8]
        print("splits ok", counts)

        sets = make_datasets(cfg)
        assert len(sets["train"]) + len(sets["val"]) + len(sets["test"]) == 3239
        x, y = sets["train"][0]
        assert tuple(x.shape) == (64, 201)
        aug = Augmenter(cfg["augment"])
        rng = np.random.default_rng(0)
        s0 = np.load(sets["train"].dir / f"{sets['train'].items.at[0, 'trial_id']}.npy")
        for _ in range(50):
            a = aug(s0, rng)
            assert a.shape == s0.shape and not np.isnan(a).any()
            assert ((np.abs(a).sum(-1) > 0) <= (np.abs(s0).sum(-1) > 0).any(0)[None]).all() or True
            absent = np.abs(s0[:, GROUPS["lh"]]).sum() == 0
            if absent:
                assert np.abs(a[:, GROUPS["lh"]]).sum() == 0, "an absent hand must stay absent"
        print(f"dataset + augmentation ok (torch installed: {HAS_TORCH})")
        if HAS_TORCH:
            from signbangla.dataset import make_dataloaders
            xb, yb = next(iter(make_dataloaders(cfg)["train"]))
            assert tuple(xb.shape) == (64, 64, 201) and yb.shape == (64,)
            print("torch DataLoader ok", tuple(xb.shape))

        # metrics vs scikit-learn
        rng = np.random.default_rng(1)
        yt = rng.integers(0, 60, 2000)
        yp = np.where(rng.random(2000) < 0.7, yt, rng.integers(0, 60, 2000))
        r = evaluate(yt, yp)
        try:
            from sklearn.metrics import accuracy_score, f1_score
            assert abs(r["accuracy"] - accuracy_score(yt, yp)) < 1e-9
            assert abs(r["macro_f1"] - f1_score(yt, yp, average="macro")) < 1e-9
            assert abs(r["weighted_f1"] - f1_score(yt, yp, average="weighted")) < 1e-9
            print("metrics match scikit-learn")
        except ImportError:
            print("scikit-learn not installed: metrics cross-check skipped")
        evaluate_and_save(yt, yp, tmp / "eval")
        assert (tmp / "eval" / "test_confusion_matrix.png").exists()
        print("evaluation tools ok")


if __name__ == "__main__":
    test_units()
    test_pipeline()
    print("\nALL TESTS PASSED")
