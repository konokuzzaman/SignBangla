"""Module 3 screen: the sentence endpoint of the local server, sign segmentation on recorded keypoint sequences, and the
recogniser's input preparation. No camera is used."""
import copy
import json
import socket
import urllib.error
import urllib.request

import numpy as np
import pytest

from signbangla.sign2text import app_server as A
from signbangla.sign2text import lexicon as L
from signbangla.sign2text.builder import SentenceBuilder
from signbangla.sign2text.recognizer import read_word_list, velocity
from signbangla.sign2text.segmentation import READY, SIGNING, SignSegmenter, body_line
from signbangla.utils import ROOT, load_config

CFG = load_config()
SEG = CFG["module3"]["screen"]["segment"]


# ---------------------------------------------------------------- a test copy of the Module 3 data
@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("screen")
    c = copy.deepcopy(CFG)
    c["paths"]["work_dir"] = str(d / "work")
    for k in ("lexicon", "verbs", "predicates", "test_sentences"):
        c["module3"][k] = str(d / f"{k}.csv")
    L.write_missing({k: d / f"{k}.csv" for k in ("lexicon", "verbs", "predicates", "test_sentences")})
    c["module3"]["screen"]["checkpoint"] = str(d / "no_model.pt")  # camera mode must fail politely
    return c


@pytest.fixture(scope="module")
def screen(cfg):
    s = A.SentenceScreen(cfg, open_browser=False, port=_free_port())
    yield s
    s.close()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read(), r.headers.get("Content-Type")


# ---------------------------------------------------------------- the sentence endpoint
def test_build_endpoint(screen):
    r = _post(screen.url + "api/build", {"items": [{"word": "আমি"}, {"word": "পানি"}, {"word": "চাওয়া"}]})
    assert r["sentence"] == "আমি পানি চাই।" and r["known"] == [True, True, True] and not r["unknown"]
    assert any(t.startswith("d. verb চাওয়া") for t in r["trace"])
    r = _post(screen.url + "api/build", {"items": [{"word": "তুমি"}, {"word": "পানি"}, {"word": "চাওয়া"}],
                                         "is_question": True})
    assert r["sentence"] == "তুমি পানি চাও?"


def test_build_endpoint_unknown_and_empty(screen):
    r = _post(screen.url + "api/build", {"items": [{"word": "আমি"}, {"word": "ঝুমঝুমি"}, {"word": "চাওয়া"}]})
    assert r["known"] == [True, False, True] and r["unknown"] == ["ঝুমঝুমি"] and "ঝুমঝুমি" in r["sentence"]
    assert _post(screen.url + "api/build", {"items": []})["sentence"] == ""


def test_build_endpoint_with_camera_candidates(screen):
    cam = [["চা", 0.6], ["পানি", 0.3], ["ভাত", 0.1]]
    items = [{"word": "আমি"}, {"word": "চা", "candidates": cam}, {"word": "চাওয়া"}]
    assert _post(screen.url + "api/build", {"items": items})["sentence"] == "আমি চা চাই।"
    items[1]["word"] = "পানি"  # corrected by hand from the top-5
    assert _post(screen.url + "api/build", {"items": items})["sentence"] == "আমি পানি চাই।"


def test_bad_requests(screen):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(screen.url + "api/build", {"items": "আমি"})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(screen.url + "../configs/config.yaml")
    assert e.value.code == 404


def test_page_config_split_and_files(screen):
    c = json.loads(_get(screen.url + "api/config")[0])
    assert set(c["groups"]) == set(A.GROUPS) and len(c["examples"]) == 5 and "401 BdSLW401 words" in c["note"]
    assert any(w["word"] == "যাওয়া" for w in c["groups"]["verb"])
    assert any(w["has_sign"] for w in c["groups"]["noun"])
    assert _post(screen.url + "api/split", {"text": "আমার মাথা ব্যথা"})["words"] == ["আমার", "মাথা ব্যথা"]
    for path, kind in (("", "text/html"), ("app.js", "javascript"), ("style.css", "text/css"), ("font.ttf", "font/ttf")):
        body, ct = _get(screen.url + path)
        assert kind in ct and body
    assert c["camera"]["requested"] is False


def test_examples_build(screen):
    b = screen.builder
    expected = ["আমি পানি চাই।", "তুমি কোথায় যাও?", "গতকাল আমি ডাক্তারের কাছে গিয়েছিলাম।", "আমার মাথা ব্যথা করছে।",
                "আমি ভাত খাই না।"]
    assert [b.build(x)[0] for x in CFG["module3"]["screen"]["examples"]] == expected


def test_speak_without_speech_module(screen):
    r = _post(screen.url + "api/speak", {"text": "আমি পানি চাই।"})
    assert r["ok"] or "texttovoice" in r["message"] or "Speech failed" in r["message"]
    assert _post(screen.url + "api/speak", {"text": ""})["ok"] is False


def test_camera_without_model_does_not_crash(cfg):
    s = A.SentenceScreen(cfg, camera_index=0, open_browser=False, port=_free_port())
    try:
        cam = json.loads(_get(s.url + "api/camera?since=0")[0])
        assert cam["requested"] and not cam["enabled"] and "Model not found" in cam["message"]
        assert _post(s.url + "api/build", {"items": [{"word": "আমি"}, {"word": "ভালো"}]})["sentence"] == "আমি ভালো আছি।"
    finally:
        s.close()


def test_build_items_function(cfg):
    b = SentenceBuilder.from_config(cfg, write=False)
    r = A.build_items(b, [{"word": "সে"}, {"word": " "}, {"word": "আসা"}, {"word": "না"}])
    assert r["sentence"] == "সে আসে না।" and r["known"] == [True, True, True]


# ---------------------------------------------------------------- segmentation on recorded sequences
def _frame(wrist_y, wrist_x=0.35, hips=True):
    """A front-view frame in image coordinates: shoulders at y 0.4, hips 0.7, right wrist at (wrist_x, wrist_y)."""
    f = np.full((67, 3), np.nan, np.float32)
    f[:25] = [0.5, 0.3, 0.0]
    f[11], f[12] = [0.6, 0.4, 0], [0.4, 0.4, 0]
    if hips:
        f[23], f[24] = [0.55, 0.7, 0], [0.45, 0.7, 0]
    else:
        f[23:25] = np.nan
    f[15] = [0.65, 0.8, 0]
    f[16] = [wrist_x, wrist_y, 0]
    return f


def _sequence(parts, fps=30):
    """parts: (seconds, kind) with kind rest / sign (raised and moving) / hold (raised, still)."""
    frames = []
    for seconds, kind in parts:
        for i in range(int(round(seconds * fps))):
            if kind == "rest":
                frames.append(_frame(0.8))
            elif kind == "sign":
                frames.append(_frame(0.45, 0.35 + 0.05 * np.sin(i / 3)))
            else:
                frames.append(_frame(0.45, 0.35))
    return frames


def test_two_signs_and_a_short_movement():
    fps = 30
    frames = _sequence([(1, "rest"), (1, "sign"), (1, "rest"), (0.2, "sign"), (1, "rest"), (1.5, "sign"), (1, "rest")], fps)
    seg = SignSegmenter(**SEG)
    found = seg.segments(frames, fps)
    assert len(found) == 2
    assert abs(len(found[0]) / fps - 1.3) < 0.25 and abs(len(found[1]) / fps - 1.8) < 0.25  # + pre-roll and margin
    assert seg.state == READY


def test_short_movement_is_reported():
    seg = SignSegmenter(**SEG)
    assert seg.segments(_sequence([(1, "rest"), (0.2, "sign"), (1, "rest")]), 30) == []
    assert seg.last_event.startswith("ignored")


def test_still_hands_end_the_sign_and_need_movement_to_restart():
    fps = 30
    seg = SignSegmenter(**SEG)
    found = seg.segments(_sequence([(1, "rest"), (1, "sign"), (2, "hold"), (1, "rest")], fps), fps)
    assert len(found) == 1 and "hands still" in seg.last_event
    assert len(found[0]) / fps < 1.6  # ends about end_s after the movement stopped, not when the hands drop


def test_long_sign_is_cut():
    fps = 30
    found = SignSegmenter(**SEG).segments(_sequence([(0.5, "rest"), (6, "sign"), (1, "rest")], fps), fps)
    assert len(found) == 2 and len(found[0]) / fps <= SEG["max_s"] + SEG["preroll_s"] + 0.1


def test_state_and_missing_body():
    seg = SignSegmenter(**SEG)
    empty = np.full((67, 3), np.nan, np.float32)
    assert seg.update(empty, 0.0) is None and seg.state == READY
    for i, f in enumerate(_sequence([(0.3, "sign")])):
        seg.update(f, 1 + i / 30)
    assert seg.state == SIGNING
    assert body_line(empty, 0.5, 1.6) is None
    y, width = body_line(_frame(0.8, hips=False), 0.5, 1.6)  # no hips: assumed below the shoulders
    assert abs(width - 0.2) < 1e-6 and abs(y - (0.4 + 0.5 * 1.6 * 0.2)) < 1e-6


RECORDED = ROOT / "work" / "raw_keypoints" / "U6W91F.npy"


@pytest.mark.skipif(not RECORDED.exists(), reason="recorded BdSLW60 keypoints not on this computer")
def test_recorded_video_matches_its_annotation():
    """A whole recorded BdSLW60 video (one word signed 15 times, annotated): the segmenter finds every repetition."""
    raw = np.load(RECORDED)
    fps = json.loads(RECORDED.with_suffix(".json").read_text())["fps"]
    found = SignSegmenter(**SEG).segments(raw, fps)
    assert abs(len(found) - 15) <= 1
    assert all(SEG["min_s"] <= len(s) / fps <= SEG["max_s"] + 1 for s in found)


# ---------------------------------------------------------------- recogniser input
def test_velocity_features():
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    v = velocity(x)
    assert v.shape == (4, 6) and (v[0, 3:] == 0).all() and (v[1:, 3:] == 3).all() and (v[:, :3] == x).all()


@pytest.mark.skipif(not (ROOT / CFG["bdslw401"]["root"] / "word_list.csv").exists(), reason="no BdSLW401 word list")
def test_word_list_mapping():
    words = read_word_list(ROOT / CFG["bdslw401"]["root"] / "word_list.csv")
    assert len(words) == 401 and words[0] == ("বাবা", "Father") and words[2][0] == "ভাই"  # class_idx = W number - 1
