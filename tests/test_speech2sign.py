"""Self-test of Module 2 (Speech -> Sign) on SYNTHETIC keypoints (no dataset, no MediaPipe needed).

    python tests/test_speech2sign.py

Covers the sign dictionary, stitching and rendering. It does NOT judge whether a chosen sign looks right to a signer -
watch work/m2_demo.mp4 for that.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from signbangla.keypoints import GROUPS, N_KP
from signbangla.speech2sign.dictionary import (SignDictionary, build_dictionary, group_present, is_upright, rank_trials,
                                               split_variants, trial_to_fps)
from signbangla.speech2sign.render import CaptionDrawer, draw_skeleton, render_video
from signbangla.speech2sign.stitch import (EASINGS, fill_absent_hands, frame_captions, get_easing, stitch, stitch_words,
                                           transition, trim_idle)
from signbangla.utils import ROOT, atomic_save_npy, atomic_write_json, load_config

CFG = load_config()
PCFG = CFG["preprocess"]
FONT = ROOT / CFG["speech2sign"]["font"]


def fake_raw(T, upside_down=False, right_hand=True):
    """Raw trial (T, 67, 3): upright body in image coordinates, right wrist rises and falls, NaN = missing."""
    x = np.full((T, N_KP, 3), np.nan, np.float32)
    x[:, GROUPS["pose"]] = [0.5, 0.3, 0.0]
    x[:, 11], x[:, 12] = [0.6, 0.4, 0.0], [0.4, 0.4, 0.0]  # shoulders: the signer's left is on the image right
    x[:, 23], x[:, 24] = [0.55, 0.7, 0.0], [0.45, 0.7, 0.0]  # hips
    x[:, 15] = [0.65, 0.7, 0.0]
    x[:, 16] = [0.35, 0.7, 0.0]
    x[:, 16, 1] = 0.7 - 0.3 * np.sin(np.linspace(0, np.pi, T))
    if right_hand:
        x[:, GROUPS["rh"]] = x[:, 16:17] + np.linspace(0, 0.05, 21)[:, None] * [1.0, 1.0, 0.0]
    if upside_down:
        x[..., :2] = 1 - x[..., :2]
    return x


def norm_sign(T, y0=0.0):
    """Normalized sign (T, 67, 3) with both a pose and a right hand; the right wrist moves all the time."""
    s = np.zeros((T, N_KP, 3), np.float32)
    s[:, GROUPS["pose"]] = [0.1, 0.1, 0.1]
    s[:, 11], s[:, 12] = [0.5, 0, 0], [-0.5, 0, 0]
    s[:, 23], s[:, 24] = [0.4, 1.3, 0], [-0.4, 1.3, 0]
    s[:, 16, 1] = y0 + np.linspace(0, 1, T)
    s[:, GROUPS["rh"]] = s[:, 16:17] + np.linspace(0, 0.2, 21)[:, None]
    return s


def rest_pose():
    r = norm_sign(2)[0]
    r[16, 1] = 1.2
    r[GROUPS["rh"]] = r[16] + np.linspace(0, 0.2, 21)[:, None]
    return r


# ---------------------------------------------------------------------------------- dictionary
def test_split_variants():
    assert split_variants("বাবা/আব্বা") == ["বাবা", "আব্বা"]
    assert split_variants("মা") == ["মা"]
    assert split_variants("অডিও ক্যাসেট") == ["অডিও ক্যাসেট"]  # a space is not a variant separator


def test_resample_uses_real_duration():
    # the same 2 second sign at 15 and at 30 fps -> the same number of 30 fps frames
    a = trial_to_fps(fake_raw(31), 15.0, 1280, 720, PCFG, 30)
    b = trial_to_fps(fake_raw(61), 30.0, 1280, 720, PCFG, 30)
    assert a.shape == b.shape == (61, N_KP, 3), (a.shape, b.shape)
    assert np.abs(a[:, 16] - b[:, 16]).max() < 0.02  # and the same movement
    assert not group_present(a, "lh").any() and group_present(a, "rh").all()  # an absent hand stays exactly 0
    shoulders = np.linalg.norm(a[:, 11, :2] - a[:, 12, :2], axis=1)
    assert np.allclose(shoulders, 1.0, atol=1e-3) and np.allclose((a[:, 11] + a[:, 12])[:, :2], 0, atol=1e-3)


def test_is_upright():
    assert is_upright(trial_to_fps(fake_raw(20), 30.0, 1280, 720, PCFG, 30))
    assert not is_upright(trial_to_fps(fake_raw(20, upside_down=True), 30.0, 1280, 720, PCFG, 30))


def test_rank_prefers_righthand_near_median_in_seconds():
    rows = pd.DataFrame([
        dict(trial_id="left_perfect", n_frames=40, fps=30.0, hand_annot="LeftHand", dominant_hand="L", rh_frac=0.0, lh_frac=1.0),
        dict(trial_id="right_15fps", n_frames=20, fps=15.0, hand_annot="RightHand", dominant_hand="R", rh_frac=1.0, lh_frac=0.0),
        dict(trial_id="right_median", n_frames=40, fps=30.0, hand_annot="RightHand", dominant_hand="R", rh_frac=1.0, lh_frac=0.0),
        dict(trial_id="right_short", n_frames=20, fps=30.0, hand_annot="RightHand", dominant_hand="R", rh_frac=1.0, lh_frac=0.0),
        dict(trial_id="right_blurry", n_frames=40, fps=30.0, hand_annot="RightHand", dominant_hand="R", rh_frac=0.5, lh_frac=0.0),
        dict(trial_id="right_says_left", n_frames=40, fps=30.0, hand_annot="RightHand", dominant_hand="L", rh_frac=1.0, lh_frac=1.0),
    ])
    order = rank_trials(rows)["trial_id"].tolist()
    # 20 frames at 15 fps is as long as 40 at 30 fps: only the 30 fps 20-frame trial is really short
    assert set(order[:2]) == {"right_15fps", "right_median"}, order
    assert order.index("right_blurry") < order.index("right_short"), order
    assert order[-2:] == ["right_says_left", "left_perfect"], order


def make_dictionary(tmp):
    """Fake work folder with 2 words and an upside-down trial that would otherwise rank first; builds the dictionary."""
    cfg = load_config()
    cfg["paths"]["work_dir"] = str(tmp)
    tdir, raw_dir = tmp / "trials", tmp / "raw_keypoints"
    tdir.mkdir()
    raw_dir.mkdir()
    rows = []
    for word, signer in (("W1", 1), ("W20", 2)):
        for k, (T, fps, flip) in enumerate([(40, 30.0, True), (40, 30.0, False), (20, 15.0, False), (44, 30.0, False), (90, 30.0, False)]):
            stem = f"U{signer + k}{word}F"
            np.save(raw_dir / f"{stem}.npy", np.concatenate([fake_raw(40, flip, right_hand=False), fake_raw(T, flip)]))
            np.save(tdir / f"{stem}_t000.npy", fake_raw(T, flip))
            rows.append(dict(trial_id=f"{stem}_t000", video_stem=stem, signer=signer + k, word=word, start=40, end=39 + T,
                             n_frames=T, fps=fps, width=1280, height=720, ok=True, hand_annot="RightHand", dominant_hand="R",
                             rh_frac=1.0, lh_frac=0.0))
    pd.DataFrame(rows).to_csv(tmp / "quality_report.csv", index=False)
    return cfg, build_dictionary(cfg)


def test_dictionary_build_and_lookup():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg, info = make_dictionary(tmp)
        assert set(info["words"]) == {"W1", "W20"}
        e = info["words"]["W1"]
        assert e["bangla"] == ["বাবা", "আব্বা"] and e["romanized"] == ["baba", "abba"] and e["english"] == "Father"
        ids = [t["trial_id"] for t in e["trials"]]
        assert len(ids) == 3 and "U1W1F_t000" not in ids, ids  # the upside-down trial is never chosen
        assert "U5W1F_t000" not in ids, ids  # 90 frames is far from the median
        on_disk = json.loads((tmp / "sign_dictionary" / "dictionary.json").read_text(encoding="utf-8"))
        assert on_disk["words"]["W1"]["bangla"] == ["বাবা", "আব্বা"]  # stored as real Bangla text

        sd = SignDictionary.from_config(cfg)
        arr = np.load(tmp / "sign_dictionary" / "W1.npy")
        assert arr.shape == (sum(t["frames"] for t in e["trials"]), N_KP, 3) and sd.fps == 30
        for name in ("বাবা", "আব্বা", "baba", "abba", "father", "W1"):
            assert sd.find(name) == "W1", name
        for v, t in enumerate(e["trials"]):
            assert len(sd.sequence("বাবা", v)) == t["frames"]
        assert np.array_equal(sd.sequence("বাবা", 1), arr[e["trials"][0]["frames"]:e["trials"][0]["frames"] + e["trials"][1]["frames"]])
        assert sd.tokenize("মা বাবা") == ["মা", "বাবা"]
        try:
            sd.find("কুকুর")
            raise AssertionError("unknown word must raise")
        except KeyError:
            pass
        try:
            sd.tokenize("মা কুকুর")
            raise AssertionError("unknown word must raise")
        except KeyError:
            pass
        assert np.abs(sd.rest_pose[GROUPS["pose"]]).sum() > 0 and sd.rest_pose.shape == (N_KP, 3)
        st = stitch_words(["মা", "বাবা"], sd, n_transition=5)
        assert [s["label"] for s in st.segments if s["kind"] == "sign"] == ["মা", "বাবা"]


def test_lookup_unicode_and_multiword():
    """Two Unicode spellings of য় (U+09DF, or U+09AF + U+09BC) and labels that contain a space."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        np.save(tmp / "rest_pose.npy", rest_pose())
        np.save(tmp / "W218.npy", norm_sign(10))
        word = dict(bangla=["আয়না"], romanized=["ayna"], english="Mirror", file="W218.npy", trials=[dict(frames=10)])
        multi = dict(bangla=["অডিও ক্যাসেট"], romanized=["audio cassette"], english="Audio Cassette", file="W218.npy", trials=[dict(frames=10)])
        atomic_write_json(tmp / "dictionary.json", dict(fps=30, rest_pose="rest_pose.npy", words={"W218": word, "W217": multi}))
        sd = SignDictionary(tmp)
        assert sd.find("আয়না") == sd.find("আয়না") == "W218"
        assert sd.tokenize("আয়না অডিও ক্যাসেট") == ["আয়না", "অডিও ক্যাসেট"]


# ---------------------------------------------------------------------------------- stitching
def test_easing_curves():
    assert set(EASINGS) >= {"linear", "cosine", "cubic", "minimum_jerk"}
    grid = np.linspace(0, 1, 101)
    for name in EASINGS:
        v = np.array([float(get_easing(name)(t)) for t in grid])
        assert abs(v[0]) < 1e-9 and abs(v[-1] - 1) < 1e-9 and (np.diff(v) >= -1e-9).all(), name
        assert abs(v[50] - 0.5) < 1e-9, name  # all are symmetric
    assert get_easing(lambda t: t ** 2)(0.5) == 0.25  # your own curve works too
    try:
        get_easing("nope")
        raise AssertionError("unknown easing must raise")
    except ValueError:
        pass
    # gentle at the ends, unlike linear: first step smaller, and easing changes the frames
    a, b = np.ones((N_KP, 3), np.float32), 3 * np.ones((N_KP, 3), np.float32)
    lin, cos = transition(a, b, 9, "linear"), transition(a, b, 9, "cosine")
    assert (cos[0] - a).max() < (lin[0] - a).max()
    assert transition(a, b, 0).shape == (0, N_KP, 3) and len(transition(a, b, 7)) == 7


def test_trim_idle():
    moving = norm_sign(40)
    padded = np.concatenate([np.repeat(moving[:1], 5, axis=0), moving, np.repeat(moving[-1:], 5, axis=0)])
    out = trim_idle(padded)
    assert 40 <= len(out) <= 46, len(out)  # idle ends cut; margin may keep a frame or two
    # never eats a held sign: a quick rise followed by a long hold loses at most max_frac from the end
    hold = np.concatenate([norm_sign(8), np.repeat(norm_sign(8)[-1:], 32, axis=0)])
    assert len(trim_idle(hold)) >= int(0.85 * len(hold)), len(trim_idle(hold))
    # a sign with no movement at all, and a very short one, are returned whole
    still = np.repeat(moving[:1], 20, axis=0)
    assert len(trim_idle(still)) == 20
    tiny = np.concatenate([np.repeat(moving[:1], 4, axis=0), norm_sign(3), np.repeat(moving[:1], 4, axis=0)])
    assert len(trim_idle(tiny, min_frames=30, max_frac=0.5)) == len(tiny)


def test_stitch_structure_and_parameters():
    rest, signs = rest_pose(), [norm_sign(20), norm_sign(30, 0.5)]
    for n in (0, 4, 12):
        st = stitch(signs, rest, labels=["a", "b"], n_transition=n, n_rest=6, trim=False)
        assert len(st.seq) == 6 + n + 20 + n + 30 + n + 6, (n, len(st.seq))
        kinds = [s["kind"] for s in st.segments]
        assert kinds == ["rest", "transition", "sign", "transition", "sign", "transition", "rest"]
        assert st.segments[0]["start"] == 0 and st.segments[-1]["end"] == len(st.seq)
        assert all(x["end"] == y["start"] for x, y in zip(st.segments, st.segments[1:]))
        assert np.array_equal(st.seq[0], rest) and np.array_equal(st.seq[-1], rest)  # begins and ends in the rest pose
        assert len(frame_captions(st, "x y")) == len(st.seq)
    # more transition frames -> smaller biggest jump between neighbouring frames
    jump = lambda s: np.abs(np.diff(s.seq, axis=0)).max()  # noqa: E731
    assert (jump(stitch(signs, rest, n_transition=20, trim=False)) < jump(stitch(signs, rest, n_transition=5, trim=False))
            < jump(stitch(signs, rest, n_transition=0, trim=False)))
    # easing really changes the transition frames; n_rest_transition is separate
    a, b = (stitch(signs, rest, n_transition=9, easing=e, trim=False).seq for e in ("linear", "minimum_jerk"))
    assert a.shape == b.shape and not np.allclose(a, b)
    assert len(stitch(signs, rest, n_transition=4, n_rest_transition=10, n_rest=3, trim=False).seq) == 3 + 10 + 20 + 4 + 30 + 10 + 3
    for bad in (dict(signs=[]), dict(labels=["only one"])):
        try:
            stitch(bad.get("signs", signs), rest, labels=bad.get("labels"))
            raise AssertionError("bad input must raise")
        except ValueError:
            pass


def test_stitch_trims_signs():
    rest, s = rest_pose(), norm_sign(40)
    padded = np.concatenate([np.repeat(s[:1], 5, axis=0), s, np.repeat(s[-1:], 5, axis=0)])
    st = stitch([padded], rest, labels=["a"])
    sign = next(x for x in st.segments if x["kind"] == "sign")
    assert sign["end"] - sign["start"] < len(padded)


def test_absent_hand_never_blends_to_zero():
    a, b = norm_sign(2)[0], norm_sign(2, 1.0)[1]
    b[GROUPS["rh"]] = 0.0  # right hand absent on one side
    t = transition(a, b, 8)
    assert group_present(t, "rh").all()
    span = t[:, GROUPS["rh"]][:, :, :2]
    assert np.allclose(span - span[:, :1], (a[GROUPS["rh"]] - a[GROUPS["rh"]][:1])[:, :2], atol=1e-4)  # same hand shape ...
    assert np.allclose(t[:, GROUPS["rh"].start, :2] - t[:, 16, :2], (a[GROUPS["rh"].start] - a[16])[:2], atol=1e-4)  # ... on the wrist
    # absent on both sides stays absent
    assert not group_present(transition(a, b, 8), "lh").any()


def test_fill_absent_hands():
    rest = rest_pose()
    s = norm_sign(10)
    s[3:6, GROUPS["rh"]] = 0.0
    out = fill_absent_hands(s, rest)
    assert group_present(out, "rh").all() and not group_present(out, "lh").any()  # the rest pose here has no left hand
    assert np.allclose(out[4, GROUPS["rh"].start, :2], s[4, 16, :2])  # attached to that frame's wrist
    assert np.array_equal(out[0], s[0])  # frames with a hand are untouched
    rest2 = rest.copy()
    rest2[GROUPS["lh"]] = rest2[15] + 0.1
    filled = fill_absent_hands(s, rest2)
    assert group_present(filled, "lh").all()  # hand missing in the whole sign -> the rest pose's hand
    assert np.allclose(filled[2, GROUPS["lh"].start, :2] - s[2, 15, :2], [0.1, 0.1])


# ---------------------------------------------------------------------------------- rendering
def test_caption_shaping():
    """Bangla must be shaped: without it ক্ল is three separate glyphs and ি stands after its consonant."""
    c = CaptionDrawer(FONT, 720)
    width = lambda t: c.text_mask(t)[0].shape[1]  # noqa: E731
    assert width("ক্ল") < width("ক") + width("্") + width("ল"), "conjunct not formed"
    vowel_first = c.text_mask("কি")[0]
    ink = np.flatnonzero(vowel_first[vowel_first.shape[0] // 2:].max(0) > 128)
    assert ink[0] < width("ক") * 0.25  # ি drawn before ক
    bar = c.bar("মা চা আপেল")
    assert bar.shape == (110, 720, 3) and (bar.min(-1) < 100).any()  # ink on the bar
    assert (c.bar("") == 245).all() and c.bar("মা") is c.bar("মা")  # empty is blank; repeated captions are cached


def test_draw_skeleton():
    s = norm_sign(2)[0]
    img = draw_skeleton(s, size=240)
    assert img.shape == (240, 240, 3) and (img < 200).any()
    assert (img[:, :, 2] > img[:, :, 0] + 50).any()  # the right hand is red (BGR)
    assert (draw_skeleton(np.zeros((N_KP, 3), np.float32), size=64) == 255).all()  # nothing present -> blank


def test_render_video():
    seq = np.stack([norm_sign(30)[i] for i in range(30)])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "out.mp4"
        n = render_video(seq, path, fps=30, captions=["মা"] * 15 + ["চা"] * 15, font_path=FONT, size=240, caption_height=60, font_size=30)
        assert n == 30 and path.exists() and not list(path.parent.glob("*.tmp.mp4"))
        cap = cv2.VideoCapture(str(path))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30 and abs(cap.get(cv2.CAP_PROP_FPS) - 30) < 0.01
        codec = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode().lower()
        assert codec in ("h264", "avc1"), codec
        assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (240, 300)
        ok, first = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
        ok2, later = cap.read()
        cap.release()
        assert ok and ok2 and np.abs(first[240:].astype(int) - later[240:]).max() > 50  # the caption changed
        odd = Path(d) / "odd.mp4"  # H.264 needs even sizes: an odd one is padded, not an error
        assert render_video(seq[:3], odd, size=241) == 3 and cv2.VideoCapture(str(odd)).get(cv2.CAP_PROP_FRAME_WIDTH) == 242
        for bad in (dict(captions=["x"], font_path=FONT), dict(captions="x")):
            try:
                render_video(seq, Path(d) / "bad.mp4", **bad)
                raise AssertionError("bad captions must raise")
            except ValueError:
                pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"{t.__name__} ok")
    print("\nALL TESTS PASSED")
