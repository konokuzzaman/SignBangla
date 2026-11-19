"""SignBangla app, Sign -> Text (signbangla/app/pipeline_sign2text.py): segmentation and recognition on recorded keypoints
(not a live camera), the sentence state with top-5 choices, the sentence-end pause, speaking, and the endpoints."""
import copy
import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from signbangla.app.pipeline_sign2text import SignToText
from signbangla.app.server import AppServer
from signbangla.sign2text.segmentation import SignSegmenter
from signbangla.utils import ROOT, load_config

CFG = load_config(None)
SCFG = CFG["app"]["sign"]
RECORDED = ROOT / "work" / "raw_keypoints" / "U6W91F.npy"
CHECKPOINT = ROOT / SCFG["checkpoint"]


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def s2t(clock):
    spoken = []

    def speak(text):
        spoken.append(text)
        return "/audio/" + "0" * 64 + ".mp3"

    s = SignToText(CFG, ROOT, camera_index=None, speak=speak, clock=clock, monitor=False, predict=False)
    s.spoken = spoken
    s.state.update_camera(enabled=True, state="ready")  # as if a camera were running
    yield s
    s.close()


def camera_sign(s, *top):
    """A recognised sign as the camera worker reports it: top-k words with scores."""
    s.state.add_event([dict(word=w, english="", score=p, class_idx=i) for i, (w, p) in enumerate(top)])


def wait_spoken(s):
    """Speaking runs in a background thread: wait until every finished sentence has its result."""
    import time
    for _ in range(100):
        if all(f["done"] for f in s.finished):
            return
        time.sleep(0.01)


# ------------------------------------------------------------------ recorded keypoints
@pytest.mark.skipif(not RECORDED.exists(), reason="recorded BdSLW60 keypoints not on this computer")
def test_segmentation_on_a_recorded_sequence():
    """The app's segment settings on a whole recorded video (one word signed 15 times)."""
    raw = np.load(RECORDED)
    fps = json.loads(RECORDED.with_suffix(".json").read_text())["fps"]
    seg = SignSegmenter.from_config(SCFG["segment"])
    found = seg.segments(raw, fps)
    assert abs(len(found) - 15) <= 1
    assert all(len(s) / fps >= SCFG["segment"]["min_s"] for s in found)


@pytest.mark.skipif(not (RECORDED.exists() and CHECKPOINT.exists()), reason="recorded keypoints or model missing")
def test_recorded_signs_give_top5(s2t):
    raw = np.load(RECORDED)
    meta = json.loads(RECORDED.with_suffix(".json").read_text())
    seg = SignSegmenter.from_config(SCFG["segment"]).segments(raw, meta["fps"])[0]
    top = s2t.recognizer.top_k(seg, meta["width"], meta["height"], SCFG["top_k"])
    assert len(top) == 5 and all(a["score"] >= b["score"] for a, b in zip(top, top[1:]))
    assert s2t.vocabulary == 401


# ------------------------------------------------------------------ the sentence
def test_camera_sign_adds_top1_and_top5_can_replace_it(s2t):
    camera_sign(s2t, ("পাখি", 0.30), ("ভাত", 0.28), ("পা", 0.15), ("মা", 0.1), ("বাবা", 0.05))
    s2t.tick()
    v = s2t.view()
    assert [i["word"] for i in v["items"]] == ["পাখি"] and v["items"][0]["own"] == "পাখি"
    assert [w for w, _ in v["items"][0]["candidates"]] == ["পাখি", "ভাত", "পা", "মা", "বাবা"]
    v = s2t.replace(0, "ভাত")
    assert v["items"][0]["word"] == "ভাত" and "ভাত" in v["sentence"]


def test_sentence_built_after_every_word(s2t):
    s2t.add("আমি")
    assert s2t.view()["sentence"]
    s2t.add("ভাত")
    s2t.add("খাওয়া")
    assert s2t.view()["sentence"] == "আমি ভাত খাই।"
    assert s2t.set_question(True)["sentence"].endswith("?")
    s2t.set_question(False)
    assert s2t.undo()["sentence"] != "আমি ভাত খাই।"
    assert s2t.clear()["items"] == []
    with pytest.raises(ValueError):
        s2t.replace(3, "x")


def test_pause_ends_the_sentence_after_a_camera_word(s2t, clock):
    finished = []
    s2t.on_finish = finished.append
    s2t.add("আমি")  # by hand: no automatic end
    clock.t += 5
    s2t.tick()
    assert finished == []
    camera_sign(s2t, ("ভাত", 0.9))
    s2t.tick()
    s2t.state.update_camera(state="signing")  # the next sign starts: no end
    clock.t += 2
    s2t.tick()
    assert finished == []
    s2t.state.update_camera(state="ready")
    clock.t += SCFG["sentence_end_s"] - 0.2
    s2t.tick()
    assert finished == []
    clock.t += 0.3
    s2t.tick()
    assert len(finished) == 1 and finished[0]["words"] == ["আমি", "ভাত"] and finished[0]["auto"]
    assert s2t.view()["items"] == []
    wait_spoken(s2t)
    assert s2t.spoken == [finished[0]["sentence"]]
    assert s2t.view()["finished"][0]["audio"].startswith("/audio/")


def test_done_button_and_failed_speech(clock):
    def broken(text):
        raise RuntimeError("no internet")

    s = SignToText(CFG, ROOT, camera_index=None, speak=broken, clock=clock, monitor=False, predict=False)
    try:
        assert not s.finish()["ok"]  # nothing to finish
        s.add("বাবা")
        out = s.finish()
        assert out["ok"] and out["finished"]["sentence"]
        wait_spoken(s)
        f = s.view()["finished"][0]
        assert f["done"] and not f["audio"] and "no internet" in f["message"]
    finally:
        s.close()


def test_missing_model_and_camera_are_reported(clock):
    cfg = copy.deepcopy(CFG)
    cfg["app"]["sign"]["checkpoint"] = "work/no_such_model.pt"
    s = SignToText(cfg, ROOT, camera_index=0, speak=lambda t: "", clock=clock, monitor=False)
    try:
        st = s.status()
        assert not st["ok"] and not st["model"]["ok"] and "not found" in st["model"]["message"]
        assert s.worker is None and s.vocabulary == 0
        assert s.add("আমি")["sentence"]  # the word buttons still work
    finally:
        s.close()


# ------------------------------------------------------------------ endpoints
def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, json.loads(r.read())


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_sign_endpoints(s2t):
    cfg = copy.deepcopy(CFG)
    cfg["app"]["port"] = 8880
    server = AppServer(cfg, ROOT, None, s2t)
    server.start()
    u = server.url
    try:
        _, st = get(u + "api/status")
        assert st["sign_to_text"]["model"]["vocabulary"] in (0, 401)
        _, words = get(u + "api/sign/words")
        assert sum(len(g) for g in words["groups"].values()) > 100
        for w in ("আমি", "ভাত", "খাওয়া"):
            status, v = post(u + "api/sign/add", {"word": w})
        assert status == 200 and v["sentence"] == "আমি ভাত খাই।"
        status, v = post(u + "api/sign/replace", {"index": 9, "word": "x"})
        assert status == 400 and not v["ok"]
        status, v = post(u + "api/sign/done", {})
        assert v["ok"] and v["finished"]["sentence"] == "আমি ভাত খাই।"
        wait_spoken(s2t)
        _, v = get(u + "api/sign/state?since=0")
        assert v["finished"][0]["done"]
        _, hist = get(u + "api/history")
        assert hist[-1]["direction"] == "sign" and hist[-1]["text"] == "আমি ভাত খাই।"
        with pytest.raises(urllib.error.HTTPError):  # no camera running: no picture
            urllib.request.urlopen(u + "video.mjpg", timeout=5)
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(u + "audio/" + "1" * 64 + ".mp3", timeout=5)
    finally:
        server.close()


def test_sign_endpoints_without_sign_part():
    cfg = copy.deepcopy(CFG)
    cfg["app"]["port"] = 8890
    server = AppServer(cfg, ROOT, None, None)
    server.start()
    try:
        _, v = get(server.url + "api/sign/state")
        assert not v["ok"]
        status, v = post(server.url + "api/sign/add", {"word": "আমি"})
        assert status == 400 and "not loaded" in v["message"]
        _, st = get(server.url + "api/status")
        assert not st["sign_to_text"]["ok"]
    finally:
        server.close()
