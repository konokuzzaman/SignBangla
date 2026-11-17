"""Tests for the core vocabulary: word list, verb forms, text matching, trimming, digit signs, numbers in the dictionary,
choosing trials and the non-manual markers. Synthetic data only, so they run without the datasets.

    python tests/test_core_vocab.py
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from signbangla.keypoints import GROUPS, N_KP
from signbangla.speech2sign import core_vocab as cv
from signbangla.speech2sign.core_dictionary import choose
from signbangla.speech2sign.core_signs import (canonical_shape, elbow_for, handshape_pose, hold_sign, hand_angle,
                                               medoid, place_hand, signing_span)
from signbangla.speech2sign.dictionary import SignDictionary
from signbangla.speech2sign.nonmanual import FACE, Markers, apply_markers, sentence_markers
from signbangla.speech2sign.render import caption_drawer, draw_frame
from signbangla.speech2sign.text_processing import UNKNOWN, WordMatcher, clean_text, full_synonyms, full_vocabulary
from signbangla.utils import ROOT, load_config

CFG = load_config()
FONT = ROOT / CFG["speech2sign"]["font"]
M = WordMatcher.full()


def matched(text):
    return [(m.word, m.how, m.key, m.category) for m in M.match(text)]


# ------------------------------------------------------------------ word list
def test_word_list():
    ids = [r[0] for r in cv.CORE_WORDS]
    assert len(ids) == len(set(ids)) and all(i.startswith("C") for i in ids)
    original = [r for r in cv.CORE_WORDS if int(r[0][1:]) <= 70]
    assert len(original) == 70
    assert {r[3] for r in cv.CORE_WORDS} <= set(cv.CATEGORIES)
    counts = {c: sum(r[3] == c for r in original) for c in cv.CATEGORIES}
    assert counts["pronoun"] == 8 and counts["verb"] == 19 and counts["question"] == 7 and counts["number"] == 13
    assert set(cv.BDSL40_SOURCES) <= set(ids) and set(cv.INCLUDE_SOURCES) <= set(ids)
    assert not set(cv.BDSL40_SOURCES) & set(cv.INCLUDE_SOURCES)
    assert sorted(cv.BDSL49_DIGIT_FOLDERS) == list(range(36, 46))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "word_list.csv"
        cv.write_word_list(path)
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0]) == ["id", "bangla", "english", "category", "source", "verified", "folder", "core_id"]
        assert len(rows) == len(cv.CORE_WORDS)
        with open(path, "a", encoding="utf-8", newline="") as f:  # a BdSL-300 row verified by hand ...
            csv.writer(f).writerow(["B999", "এক্স", "x", "noun", "BdSL300", "true", "x", ""])
        cv.write_word_list(path)  # ... survives a rewrite of the core rows
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    assert len(rows) == len(cv.CORE_WORDS) + 1 and rows[-1]["id"] == "B999" and rows[-1]["verified"] == "true"
    by_id = {r["id"]: r for r in rows}
    assert by_id["C001"]["source"] == "BdSL40" and by_id["C001"]["verified"] == "yes"
    assert by_id["C061"]["source"] == "INCLUDE" and by_id["C061"]["verified"] == "no"
    assert by_id["C042"]["source"] == "BDSL49" and by_id["C049"]["verified"] == "no"
    assert by_id["C003"]["source"] == "" and by_id["C003"]["verified"] == ""


def test_include_label_fixes_swap_i_and_you():
    assert cv.INCLUDE_LABEL_FIXES[("I", "MVI_0001")] == "you" and cv.INCLUDE_LABEL_FIXES[("you", "MVI_9996")] == "I"


# ------------------------------------------------------------------ verb forms and matching
def test_verb_forms():
    forms = cv.verb_forms()
    for form in ("খাব", "খাবে", "খাচ্ছি", "খেয়েছি"):
        assert forms[form] == "খাওয়া", form
    for form in ("যাব", "যাবে", "যাচ্ছি", "গিয়েছি"):
        assert forms[form] == "যাওয়া", form
    assert forms["সাহায্য করব"] == "সাহায্য করা" and forms["অপেক্ষা করছি"] == "অপেক্ষা করা"
    for verb in cv.VERBS:
        assert sum(v == verb for v in forms.values()) >= 15, verb


def test_verb_forms_never_hide_another_word():
    names = set()
    for key, bangla, *_ in full_vocabulary():
        if not (key.startswith("C") and next(r for r in cv.CORE_WORDS if r[0] == key)[3] == "verb"):
            names |= {clean_text(n) for n in bangla.split("/")}
    clash = {f for f in cv.verb_forms() if clean_text(f) in names}
    assert not clash, clash


def test_every_core_word_matches_itself():
    for key, bangla, _english, category in cv.CORE_WORDS:
        for name in bangla.split("/"):
            m = M.match(name)
            assert len(m) == 1 and m[0].key == key and m[0].category == category, (name, m)


def test_test_sentences():
    assert matched("আমি অসুস্থ") == [("আমি", "exact", "C001", "pronoun"), ("অসুস্থ", "exact", "W387", "word")]
    assert [x[0] for x in matched("তুমি কেমন আছ?")] == ["তুমি", "কেমন", ""]  # আছ is dropped
    assert matched("আমার মাথা ব্যথা নেই")[1][2] == "W376"  # মাথা ব্যথা -> headache
    m = matched("আমি ৩টা ট্যাবলেট খাব")
    assert m[1] == ("৩", "number", "C042", "number") and m[2][2] == "W395" and m[3][:3] == ("খাওয়া", "synonym", "C009")
    assert [x[2] for x in matched("গতকাল বাবার জ্বর")] == ["C053", "W1", "W371"]
    assert matched("ধন্যবাদ") == [("ধন্যবাদ", "exact", "C061", "greeting")]


def test_numbers():
    assert matched("25")[0][:3] == ("২৫", "number", "২৫")
    assert matched("১০টা")[0][2] == "C049"  # 10 has its own entry
    assert matched("তিনটা")[0][:3] == ("৩", "number", "C042")
    assert matched("দশ")[0][0] == "১০"
    assert matched("নয়")[0][3] == "negation"  # নয় is read as "not", not as nine


def test_default_matcher_unchanged():
    m = WordMatcher().match("আমি মা")
    assert m[0].word == UNKNOWN and m[1].word == "মা" and m[1].key == "W20"


def test_synonym_targets_exist():
    WordMatcher(labels=full_vocabulary(), synonyms=full_synonyms())  # raises if a target is missing


# ------------------------------------------------------------------ trimming and digits
def raw_video(T=60, up=(20, 40)):
    """Image coordinates: shoulders at y 0.4, hips at 0.8; the right wrist rises between frames up[0] and up[1]."""
    raw = np.full((T, N_KP, 3), np.nan, np.float32)
    raw[:, :25] = 0.5
    raw[:, 11, :2], raw[:, 12, :2] = [0.6, 0.4], [0.4, 0.4]
    raw[:, 23, :2], raw[:, 24, :2] = [0.58, 0.8], [0.42, 0.8]
    raw[:, 15, :2], raw[:, 16, :2] = [0.62, 0.85], [0.38, 0.85]
    raw[up[0]:up[1], 16, 1] = 0.45
    return raw


def test_signing_span():
    assert signing_span(raw_video(), raise_frac=0.7, margin=3) == (17, 43)
    assert signing_span(raw_video(up=(0, 0))) is None


def fake_hand(angle_deg=-90.0, scale=100.0):
    """21 points: wrist at (50, 300), fingers fanning out along `angle`."""
    a = np.radians(angle_deg)
    d, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    h = np.zeros((21, 3), np.float32)
    h[0, :2] = [50, 300]
    for f in range(5):
        for j in range(4):
            h[1 + 4 * f + j, :2] = h[0, :2] + scale * ((0.6 + 0.25 * j) * d + (f - 2) * 0.2 * n)
    return h


def test_handshape_geometry():
    h = fake_hand(-60)
    c = canonical_shape(h)
    assert np.allclose(c[0], 0) and abs(np.linalg.norm(c[9, :2]) - 1) < 1e-5 and c[9, 1] < 0 and abs(c[9, 0]) < 1e-5
    placed = place_hand(h, (-0.4, 0.15), 0.3, hand_angle(h))
    assert np.allclose(placed[0, :2], [-0.4, 0.15], atol=1e-5)
    assert abs(np.linalg.norm(placed[9, :2] - placed[0, :2]) - 0.3) < 1e-5
    assert abs(hand_angle(placed) - hand_angle(h)) < 1e-4
    assert medoid([canonical_shape(fake_hand(a)) for a in (-90, -60, -80)]) in (0, 1, 2)
    odd = fake_hand()
    odd[5:9, :2] += 80  # one wrong detection
    assert medoid([canonical_shape(x) for x in (fake_hand(), fake_hand(-80), odd)]) != 2


def test_elbow_keeps_bone_lengths():
    s, w = np.array([-0.5, 0.0]), np.array([-0.4, 0.15])
    e = elbow_for(s, w, 0.8, 0.7)
    assert abs(np.linalg.norm(e - s) - 0.8) < 1e-6 and abs(np.linalg.norm(w - e) - 0.7) < 1e-6
    assert e[0] < min(s[0], w[0])  # bends outward, to the signer's right


def rest_pose():
    f = np.zeros((N_KP, 3), np.float32)
    f[11], f[12], f[13], f[14], f[15], f[16] = [.5, 0, 0], [-.5, 0, 0], [.6, .8, 0], [-.6, .8, 0], [.6, 1.5, 0], [-.6, 1.5, 0]
    f[:11, 1] = -0.5
    f[GROUPS["rh"]] = f[16] + np.linspace(0, 0.3, 21)[:, None] * [0, 1, 0]
    return f


def test_digit_sign():
    rest = rest_pose()
    hand = place_hand(fake_hand(), (-0.4, 0.15), 0.3, -np.pi / 2)
    pose = handshape_pose(rest, hand, CFG)
    assert np.allclose(pose[16], hand[0]) and np.allclose(pose[GROUPS["rh"]], hand)
    seq = hold_sign([pose], rest, 10, 15, 8)
    assert len(seq) == 10 + 15 + 9 and np.allclose(seq[0], rest) and np.allclose(seq[-1], rest)
    assert np.allclose(seq[10:25], pose)
    two = hold_sign([pose, pose * 1.01], rest, 10, 15, 8)
    assert len(two) == 10 + 15 + 7 + 15 + 9


def test_dictionary_signs_numbers_digit_by_digit():
    rest = rest_pose()
    poses = np.zeros((10, N_KP, 3), np.float32)
    for i in range(10):
        poses[i] = handshape_pose(rest, place_hand(fake_hand(-90 + 3 * i), (-0.4, 0.15), 0.3, -np.pi / 2), CFG)
    poses[7] = 0  # no sign for ৭
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        np.save(d / "rest_pose.npy", rest)
        np.save(d / "digit_poses.npy", poses)
        np.save(d / "C040.npy", hold_sign([poses[1]], rest, 10, 15, 8))
        info = dict(fps=30, rest_pose="rest_pose.npy", digits=dict(poses="digit_poses.npy", move_frames=10,
                                                                     hold_frames=15, change_frames=8),
                    words={"C040": dict(bangla=["১"], romanized=[], english="1", file="C040.npy",
                                        trials=[dict(frames=34)])})
        (d / "dictionary.json").write_text(json.dumps(info), encoding="utf-8")
        sd = SignDictionary(d)
        assert sd.has("১") and sd.has("২৫") and sd.has("25") and not sd.has("৭০") and not sd.has("চা")
        assert len(sd.sequence("২৫")) == 10 + 15 + 7 + 15 + 9
        assert len(sd.sequence("১", variant=2)) == 34  # fewer trials than the variant: the last one


def test_choose_keeps_one_hand():
    c = [dict(hand="L", hand_frac=1.0, duration_s=2.0, ref="a"), dict(hand="R", hand_frac=0.9, duration_s=2.0, ref="b"),
         dict(hand="R", hand_frac=0.5, duration_s=2.1, ref="c"), dict(hand="R", hand_frac=1.0, duration_s=5.0, ref="d")]
    chosen = choose(c, 2, 0.5)
    assert [x["ref"] for x in chosen] == ["b", "c"]
    assert [x["ref"] for x in choose(c[:1], 3, 0.5)] == ["a"]


# ------------------------------------------------------------------ non-manual markers
def test_sentence_markers():
    assert sentence_markers("তুমি কেমন আছ?", M.match("তুমি কেমন আছ?")) == Markers(True, False)
    assert sentence_markers("তুমি যাবে?") == Markers(True, False)
    assert sentence_markers("আমার মাথা ব্যথা নেই", M.match("আমার মাথা ব্যথা নেই")) == Markers(False, True)
    assert sentence_markers("এটা ভালো নয়") == Markers(False, True)
    assert sentence_markers("আমি অসুস্থ") == Markers(False, False)


def test_apply_markers_changes_only_the_face_of_the_last_signs():
    T = 200
    seq = np.repeat(rest_pose()[None], T, 0)
    words = np.full(T, -1)
    words[20:90], words[90:160] = 0, 1
    ncfg = CFG["speech2sign"]["nonmanual"]
    out, brow = apply_markers(seq, words, 2, Markers(True, True), 60, ncfg)
    assert np.allclose(np.delete(out, FACE, axis=1), np.delete(seq, FACE, axis=1))  # body and hands untouched
    assert np.allclose(out[:5], seq[:5]) and brow[:5].max() == 0
    assert brow[120] > 0.9 and out[120, 0, 1] > seq[120, 0, 1]  # head tilted forward, eyebrows up
    assert np.abs(out[90:160, 0, 0] - seq[90:160, 0, 0]).max() > 0.03  # shaking during the last sign
    assert np.allclose(out[20:60, 0, 0], seq[20:60, 0, 0])  # not during the first one
    assert np.allclose(out[:, 7:9, 0], seq[:, 7:9, 0])  # the ears do not shake: the face turns inside the head
    assert np.array_equal(seq, np.repeat(rest_pose()[None], T, 0))  # the input is not changed


def test_brows_and_word_colours_are_drawn():
    f = rest_pose()
    f[23], f[24] = [.35, 1.3, 0], [-.35, 1.3, 0]
    for style in ("figure", "neon"):
        assert not np.array_equal(draw_frame(f, style, 256), draw_frame(f, style, 256, brow=1.0)), style
    assert np.array_equal(draw_frame(f, "skeleton", 256), draw_frame(f, "skeleton", 256, brow=1.0))
    d = caption_drawer(FONT, "figure", 400, 64, 30)
    plain = d.bar((("আমি", "খাব"), 0))
    coloured = d.bar((("আমি", "খাব"), 0, (None, cv.CATEGORY_COLORS["verb"])))
    assert not np.array_equal(plain, coloured)
    purple = np.array(cv.CATEGORY_COLORS["verb"])
    assert (np.abs(coloured.astype(int) - purple).sum(-1) < 40).sum() > 20


def test_remote_zip_reads_members_with_one_request_each():
    import io
    import zipfile

    from signbangla.speech2sign import remote_zip as rz
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Pronouns/40. I/MVI_0001.MOV", b"stored video bytes" * 50, compress_type=zipfile.ZIP_STORED)
        z.writestr("Pronouns/40. I/MVI_0002.MOV", b"deflated video bytes" * 500, compress_type=zipfile.ZIP_DEFLATED)
        z.mkdir("Pronouns/41. you")
    data = buf.getvalue()
    calls = []

    def fake_request(url, start, end):
        calls.append((start, end))
        return data[start:end + 1]

    old = rz._request, rz.remote_size, rz.MIN_INTERVAL
    rz._request, rz.remote_size, rz.MIN_INTERVAL = fake_request, lambda url: len(data), 0
    try:
        members = rz.list_members("zip")
        assert [m["name"] for m in members] == ["Pronouns/40. I/MVI_0001.MOV", "Pronouns/40. I/MVI_0002.MOV"]
        with tempfile.TemporaryDirectory() as tmp:
            for m, expected in zip(members, (b"stored video bytes" * 50, b"deflated video bytes" * 500)):
                calls.clear()
                dest = Path(tmp) / Path(m["name"]).name
                assert rz.fetch_member("zip", m, dest) == len(expected) and len(calls) == 1
                assert dest.read_bytes() == expected
                assert rz.fetch_member("zip", m, dest) == 0  # already there
            bad = dict(members[0], crc=members[0]["crc"] ^ 1)
            try:
                rz.fetch_member("zip", bad, Path(tmp) / "bad.MOV")
                raise AssertionError("a wrong CRC must raise")
            except IOError:
                assert not (Path(tmp) / "bad.MOV").exists()
    finally:
        rz._request, rz.remote_size, rz.MIN_INTERVAL = old


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"{t.__name__} ok")
    print("\nALL TESTS PASSED")
