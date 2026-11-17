"""The studio3d style: 3D lifting, sentence payload, local server and viewer page."""
import json
import socket
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from signbangla.speech2sign.lift3d import LEFT_HAND, RIGHT_HAND, fill_missing, lift_to_3d
from signbangla.speech2sign.stitch import Sentence
from signbangla.speech2sign.studio3d import Studio3DServer, bgr_to_hex, page_settings, sequence_payload
from signbangla.utils import ROOT, load_config

HANDS = (LEFT_HAND, RIGHT_HAND)


def fake_rest():
    """A front-view rest pose in the normalized format (x < 0 = signer's right, y down)."""
    rest = np.zeros((67, 3), np.float32)
    face = {0: (0, -0.55), 2: (0.1, -0.6), 5: (-0.1, -0.6), 7: (0.18, -0.58), 8: (-0.18, -0.58), 9: (0.05, -0.45),
            10: (-0.05, -0.45)}
    for i in range(11):
        rest[i, :2] = face.get(i, (0.0, -0.55))
    body = {11: (0.5, 0), 12: (-0.5, 0), 13: (0.65, 0.7), 14: (-0.7, 0.7), 15: (0.6, 1.2), 16: (-0.65, 1.2),
            23: (0.3, 1.33), 24: (-0.3, 1.33)}
    for i, xy in body.items():
        rest[i, :2] = xy
    for i in range(17, 23):
        rest[i, :2] = rest[15 if i % 2 else 16, :2]
    rng = np.random.default_rng(0)
    for sl, wrist in ((LEFT_HAND, 15), (RIGHT_HAND, 16)):
        rest[sl, :2] = rest[wrist, :2] + rng.normal(0, 0.05, (21, 2))
        rest[sl, 2] = rng.normal(0, 0.03, 21)
        rest[sl.start, 2] = 0
    return rest


def fake_sequence(rest, n=40):
    """The right hand rises towards the face and back; the left hand is absent in some frames (zeros and NaN)."""
    seq = np.repeat(rest[None], n, 0).copy()
    lift = np.sin(np.linspace(0, np.pi, n))[:, None]
    for i in (14, 16):
        seq[:, i, 1] -= 0.4 * lift[:, 0]
    seq[:, RIGHT_HAND, 1] -= 0.9 * lift
    seq[:, RIGHT_HAND, 0] += 0.3 * lift
    seq[:, 16, 2] -= 0.5 * lift[:, 0]  # towards the camera
    seq[5:10, LEFT_HAND] = 0
    seq[12:14, LEFT_HAND] = np.nan
    return seq


def test_lift_shape_no_nan_and_input_unchanged():
    rest = fake_rest()
    seq = fake_sequence(rest)
    before = seq.copy()
    out = lift_to_3d(seq, rest)
    assert out.shape == seq.shape and out.dtype == np.float32
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(np.isnan(seq), np.isnan(before))
    np.testing.assert_array_equal(np.nan_to_num(seq), np.nan_to_num(before))


def test_lift_axes_and_body_plane():
    rest = fake_rest()
    out = lift_to_3d(fake_sequence(rest), rest)
    np.testing.assert_allclose(out[:, 11, :2], [[0.5, 0.0]] * len(out), atol=1e-6)  # x kept, y up
    assert (out[:, 0, 1] > out[:, 11, 1]).all()  # the head is above the shoulders
    assert (out[:, :13, 2] == 0).all() and (out[:, 23:25, 2] == 0).all()  # the body stays in the plane z = 0


def test_hands_stay_in_front():
    rest = fake_rest()
    out = lift_to_3d(fake_sequence(rest), rest, wrist_front_min=0.12, hand_front_min=0.05)
    for sl, wrist in zip(HANDS, (15, 16)):
        assert (out[:, sl, 2] >= 0.05 - 1e-6).all()
        assert (out[:, wrist, 2] >= 0.12 - 1e-6).all()


def test_wrist_meets_hand():
    rest = fake_rest()
    out = lift_to_3d(fake_sequence(rest), rest)
    for sl, wrist in zip(HANDS, (15, 16)):
        np.testing.assert_allclose(out[:, wrist, :2], out[:, sl.start, :2], atol=1e-6)


def test_raised_forearm_comes_forward():
    rest = fake_rest()
    out = lift_to_3d(fake_sequence(rest), rest)
    mid = len(out) // 2
    assert out[mid, 16, 2] > out[0, 16, 2]  # foreshortened, pointing at the camera


def test_missing_hand_becomes_rest_hand():
    rest = fake_rest()
    seq = fake_sequence(rest)
    filled = fill_missing(seq, rest)
    for t in (5, 9, 12, 13):
        np.testing.assert_allclose(filled[t, LEFT_HAND], rest[LEFT_HAND])
    out = lift_to_3d(seq, rest)
    np.testing.assert_allclose(out[7, LEFT_HAND, 0], rest[LEFT_HAND, 0], atol=1e-6)
    np.testing.assert_allclose(out[7, LEFT_HAND, 1], -rest[LEFT_HAND, 1], atol=1e-6)


def test_short_sequences_and_bad_options():
    rest = fake_rest()
    for n in (1, 2, 5):
        out = lift_to_3d(fake_sequence(rest, 40)[:n], rest)
        assert out.shape == (n, 67, 3) and np.isfinite(out).all()
    with pytest.raises(TypeError):
        lift_to_3d(fake_sequence(rest), rest, not_an_option=1)


def fake_sentence(rest, n=40):
    words = np.full(n, -1)
    words[10:30] = 0
    return Sentence(fake_sequence(rest, n), np.zeros(n, int), words, ["বাবা"], 60.0)


def test_payload():
    rest = fake_rest()
    layout = (("বাবা", "খাব"), (None, (170, 50, 140)), [0])
    p = sequence_payload(fake_sentence(rest), layout, rest)
    assert p["fps"] == 60.0 and len(p["frames"]) == 40 and len(p["word"]) == 40
    assert np.isfinite(np.array(p["frames"])).all()
    assert p["tokens"] == [{"text": "বাবা", "color": None}, {"text": "খাব", "color": "#8c32aa"}]
    assert p["word"][0] == -1 and p["word"][15] == 0
    json.dumps(p, ensure_ascii=False)
    assert bgr_to_hex((40, 40, 200)) == "#c82828"


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read(), r.headers.get("Content-Type")


@pytest.fixture
def server():
    cfg = load_config()
    s = Studio3DServer.from_config(cfg, ROOT)
    s.open_browser = False
    yield s
    s.close()


def test_server_routes_and_updates(server):
    assert server.url.startswith("http://127.0.0.1:")
    page, kind = _get(server.url)
    assert b'/vendor/three.module.js' in page and kind.startswith("text/html")
    js, kind = _get(server.url + "vendor/three.module.js")
    assert b"REVISION = '160'" in js and "javascript" in kind
    font, kind = _get(server.url + "font.ttf")
    assert len(font) > 10000 and kind == "font/ttf"
    assert json.loads(_get(server.url + "settings.json")[0])["camera"]["fov"] > 0
    assert json.loads(_get(server.url + "version")[0])["version"] == 0
    rest = fake_rest()
    v = server.publish(sequence_payload(fake_sentence(rest), (("বাবা",), (None,), [0]), rest))
    assert json.loads(_get(server.url + "version")[0])["version"] == v == 1
    data = json.loads(_get(server.url + "sequence.json")[0].decode("utf-8"))
    assert data["version"] == 1 and data["tokens"][0]["text"] == "বাবা"
    assert server.wait_until_loaded(1, 1)
    server.publish(sequence_payload(fake_sentence(rest), (("ভাই",), (None,), [0]), rest))
    assert json.loads(_get(server.url + "version")[0])["version"] == 2
    with pytest.raises(urllib.error.HTTPError):
        _get(server.url + "../configs/config.yaml")


def test_busy_port_moves_on():
    cfg = load_config()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    cfg["speech2sign"]["studio3d"] = dict(cfg["speech2sign"]["studio3d"], port=port, port_tries=5, open_browser=False)
    try:
        s = Studio3DServer.from_config(cfg, ROOT)
        try:
            used = int(s.url.rstrip("/").rsplit(":", 1)[1])
            assert port < used < port + 5
        finally:
            s.close()
    finally:
        blocker.close()


def test_page_is_local_and_unlabelled():
    cfg = load_config()
    scfg = cfg["speech2sign"]["studio3d"]
    web = ROOT / scfg["web_dir"]
    html = (web / "index.html").read_text(encoding="utf-8")
    js = (web / "studio3d.js").read_text(encoding="utf-8")
    for text in (html, js):
        assert "http://" not in text and "https://" not in text  # no CDN
        assert "holo" not in text.lower() and "bloom" not in text.lower() and "renderFrame" not in text
    assert 'id="tag"' not in html  # no style label on screen
    assert Path(ROOT / scfg["three_js"]).exists()
    assert set(page_settings(scfg)) == {"camera", "floor_y", "colors", "poll_ms", "caption_font_size", "face",
                                        "hand_clearance", "hand_clearance_frames"}
    assert "function curve(" in js and "function headGeom(" in js and "function pushHands(" in js
    assert scfg["face"]["show_nose_mouth"] is False  # only the eye dots by default
    assert "F.show_nose_mouth ? lineParts() : null" in js  # no nose / mouth meshes are made when it is false
