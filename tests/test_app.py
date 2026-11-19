"""SignBangla app (signbangla/app): text -> sign words, the shared animation sequence, the server endpoints, and a page
that still works without a microphone or speech model."""
import copy
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from signbangla.app import pipeline_speech2sign as P
from signbangla.app.gloss import Glosser
from signbangla.app.server import AppServer
from signbangla.speech2sign.dictionary import normalize_word
from signbangla.utils import ROOT, load_config, work_dir

CFG = load_config(None)
GCFG = CFG["app"]["gloss"]
CORE = work_dir(CFG) / CFG["speech2sign"]["core_vocab"]["dictionary_dir"]
needs_core = pytest.mark.skipif(not (CORE / "dictionary.json").exists(), reason="work/sign_dictionary_core missing")


class FakeDictionary:
    """Enough of SignDictionary for the glosser: words with names, sources and verified flags."""

    def __init__(self, words):
        self.words = words
        self.index = {}
        for key, e in words.items():
            self.index[key.lower()] = key
            for name in e["bangla"]:
                self.index.setdefault(normalize_word(name).lower(), key)

    def find(self, name):
        k = normalize_word(name).lower()
        if k not in self.index:
            raise KeyError(name)
        return self.index[k]

    def number_digits(self, name):
        return None


def word(bangla, source="BdSL40", verified=True, category="word"):
    return dict(bangla=[bangla], romanized=[], english="", category=category, source=source, verified=verified)


FAKE = FakeDictionary({
    "C001": word("আমি", category="pronoun"),
    "C002": word("তুমি", category="pronoun"),
    "C009": word("খাওয়া", category="verb"),
    "C052": word("আজ", category="time"),
    "W81": word("ভাত", "BdSLW60"),
    "W1": word("বাবা", "BdSLW60"),
    "W20": word("মা", "BdSLW60"),
    "W72": word("পান", "BdSLW401"),
    "C030": word("কোথায়", "BdSL300", verified=False, category="question"),
    "C028": word("কী", "BdSL40", category="question"),
})


@pytest.fixture(scope="module")
def glosser():
    return Glosser(FAKE, GCFG, CFG["speech2sign"]["matching"])


def signs(g, text):
    return [t.caption for t in g.gloss(text).tokens if t.signed]


def test_basic_sentence_and_verb_forms(glosser):
    out = glosser.gloss("আমি ভাত খাব")
    assert [t.caption for t in out.tokens] == ["আমি", "ভাত", "খাওয়া"]
    assert out.keys == ["C001", "W81", "C009"]
    assert out.unknown == []


def test_time_words_first_and_question_words_last(glosser):
    assert signs(glosser, "আমি আজ ভাত খাব") == ["আজ", "আমি", "ভাত", "খাওয়া"]
    assert [t.caption for t in glosser.gloss("কী তুমি খাবে").tokens][-1] == "কী"


def test_phrase_verb_synonym_beats_the_noun_sign(glosser):
    # পান alone is the betel-leaf sign; পান করা (drink) has no sign and becomes খাওয়া
    out = glosser.gloss("আমি পান করেছি")
    assert [t.caption for t in out.tokens] == ["আমি", "খাওয়া"]
    assert out.tokens[1].status == "synonym" and "পান করা" in out.tokens[1].note
    assert signs(glosser, "আমি পান খাই") == ["আমি", "পান", "খাওয়া"]


def test_filler_words_are_dropped_and_do_not_merge(glosser):
    out = glosser.gloss("মা ও বাবা")
    assert [t.caption for t in out.tokens] == ["মা", "বাবা"]
    assert [t.text for t in out.dropped] == ["ও"]


def test_copula_dropped_and_unknown_words_kept_red(glosser):
    out = glosser.gloss("তুমি কেমন আছ?")
    assert [t.text for t in out.dropped] == ["আছ"]
    assert out.question
    nosign = [t for t in out.tokens if not t.signed]
    assert [t.text for t in nosign] == ["কেমন"]
    captions, colors, positions = out.layout()
    assert colors[captions.index("কেমন")] is not None and positions == [captions.index("তুমি")]


def test_unverified_signs_are_not_used(glosser):
    out = glosser.gloss("বাবা কোথায়")
    assert out.keys == ["W1"]
    assert [t.status for t in out.tokens if t.caption == "কোথায়"] == ["no_sign"]


def test_negation_and_question_markers(glosser):
    assert glosser.gloss("আমি ভাত খাই না").negation
    assert not glosser.gloss("আমি ভাত খাই").negation
    assert glosser.gloss("তুমি কি ভাত খাবে").question
    assert "কি" in [t.text for t in glosser.gloss("তুমি কি ভাত খাবে").dropped]


# ------------------------------------------------------------------ the sequence both views play
@pytest.fixture(scope="module")
def speech():
    if not (CORE / "dictionary.json").exists():
        pytest.skip("work/sign_dictionary_core missing")
    s = P.SpeechToSign(CFG, load_asr=False)
    s.predictor.log = None  # tests do not write to the app's prediction log
    return s


@needs_core
def test_sequence_payload(speech):
    out = speech.from_text("আমি ভাত খাব")
    s = out["sequence"]
    frames = np.array(s["frames"])
    assert frames.ndim == 3 and frames.shape[1:] == (67, 3)
    assert len(s["word"]) == len(frames)
    assert [t["text"] for t in s["tokens"]] == out["gloss"]["signs"]
    assert set(s["word"]) == {-1, 0, 1, 2}
    assert s["fps"] == 60.0


@needs_core
def test_head_shake_is_in_the_frames(speech):
    """The head movement is in the keypoints themselves, so the 3D and the 2D view both show it."""
    gloss = speech.glosser.gloss("আমি ভাত খাই না")
    assert gloss.negation
    shaken = speech.sequence(gloss)
    gloss.negation = False
    plain = speech.sequence(gloss)
    a, b = np.array(plain["frames"]), np.array(shaken["frames"])
    assert a.shape == b.shape
    eyes = [2, 5]
    assert np.abs(b[:, eyes, 0] - a[:, eyes, 0]).max() > 0.02
    assert np.allclose(b[:, 11:, :], a[:, 11:, :])  # only the face moves
    assert shaken["markers"]["negation"] and not plain["markers"]["negation"]


@needs_core
def test_question_tilt_is_in_the_frames(speech):
    plain = speech.sequence(speech.glosser.gloss("বাবা আমি"))
    tilted = speech.sequence(speech.glosser.gloss("বাবা আমি?"))
    a, b = np.array(plain["frames"]), np.array(tilted["frames"])
    assert np.abs(b[:, 0, 1] - a[:, 0, 1]).max() > 0.02


def test_both_views_get_the_same_data():
    """The page hands the one sequence object to both views and drives both from one clock."""
    js = (ROOT / CFG["app"]["web_dir"] / "app.js").read_text(encoding="utf-8")
    load = js[js.index("function load(data)"):js.index("function replay()")]
    assert "skeleton.view.setData(data)" in load and "send3d({ type: 'load', data })" in load
    tick = js[js.index("function tick(now)"):]
    assert "skeleton.view.draw(i)" in tick and "send3d({ type: 'frame', i })" in tick


def test_no_words_with_a_sign_gives_no_sequence(glosser):
    assert glosser.gloss("xyz").keys == []


def test_phrases_are_matched_as_a_whole():
    words = dict(FAKE.words)
    words["D005"] = word("আপনি কেমন আছেন", "DynamicDU", category="phrase")
    words["C063"] = word("দয়া করে", "DynamicDU", category="greeting")
    g = Glosser(FakeDictionary(words), GCFG, CFG["speech2sign"]["matching"])
    assert g.gloss("আপনি কেমন আছেন?").keys == ["D005"]
    assert g.gloss("দয়া করে আপনি কেমন আছেন").keys == ["C063", "D005"]
    assert g.gloss("তুমি কেমন আছ").keys == ["C002"]  # not the phrase: only its words


# ------------------------------------------------------------------ Ban-Sign-Sent-9K and DU signs in the dictionary
def test_extra_sources_use_verified_rows_only():
    from signbangla.speech2sign.core_dictionary import extra_source_words
    rows = [dict(id="S001", bangla="না", english="not", category="negation", source="BanSign9K", verified="false",
                 folder="na", core_id="C035"),
            dict(id="D003", bangla="হ্যালো", english="hello", category="greeting", source="DynamicDU", verified="no",
                 folder="hello", core_id="")]
    assert extra_source_words(CFG, rows) == {}


def test_core_id_found_by_spelling():
    from signbangla.speech2sign.core_dictionary import core_id_for
    assert core_id_for("দয়া করে") == "C063"
    assert core_id_for("না") == "C035"
    assert core_id_for("আপনি কেমন আছেন") == ""


@needs_core
def test_new_source_signs_in_the_app(speech):
    """না (Ban-Sign-Sent-9K), দয়া করে and the phrase আপনি কেমন আছেন (DU) are signed once verified and rebuilt."""
    d = speech.dictionary.words
    if not any(e.get("source") == "DynamicDU" for e in d.values()):
        pytest.skip("dictionary built before the DU / Ban-Sign-Sent-9K signs were added")
    out = speech.from_text("আমি ভাত খাই না")["gloss"]
    assert out["signs"] == ["আমি", "ভাত", "খাওয়া", "না"] and out["unknown"] == []
    assert speech.from_text("দয়া করে")["gloss"]["signs"] == ["দয়া করে"]
    assert speech.from_text("আপনি কেমন আছেন")["gloss"]["signs"] == ["আপনি কেমন আছেন"]


# ------------------------------------------------------------------ server
def app_config(port):
    cfg = copy.deepcopy(CFG)
    cfg["app"]["port"] = port
    return cfg


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def post(url, body, raw=None):
    data = raw if raw is not None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(scope="module")
def server(speech):
    s = AppServer(app_config(8850), ROOT, speech)
    s.start()
    yield s
    s.close()


@needs_core
def test_pages_and_files(server):
    for path in ("/", "/app.js", "/style.css", "/skeleton.js", "/viewer3d.html", "/studio3d.js",
                 "/vendor/three.module.js", "/font.ttf"):
        status, kind, body = get(server.url + path.lstrip("/"))
        assert status == 200 and body, path
    with pytest.raises(urllib.error.HTTPError):
        get(server.url + "no-such-file")
    with pytest.raises(urllib.error.HTTPError):
        get(server.url + "../configs/config.yaml")


@needs_core
def test_settings_use_the_orthographic_camera(server):
    _, _, body = get(server.url + "settings.json")
    s = json.loads(body)
    assert s["camera"]["type"] == "orthographic"
    assert s["face"]["show_nose_mouth"] is False  # eyes-only face


@needs_core
def test_speech_text_endpoint_and_history(server):
    status, out = post(server.url + "api/speech/text", {"text": "আমি ভাত খাব"})
    assert status == 200 and out["ok"] and out["gloss"]["signs"] == ["আমি", "ভাত", "খাওয়া"]
    assert "server_s" in out["timings"] and out["sequence"]["frames"]
    _, _, body = get(server.url + "api/history")
    assert json.loads(body)[-1]["text"] == "আমি ভাত খাব"
    _, _, body = get(server.url + "api/speech/last")
    assert json.loads(body)["text"] == "আমি ভাত খাব"
    status, out = post(server.url + "api/speech/text", {"text": "  "})
    assert status == 200 and not out["ok"]
    status, out = post(server.url + "api/speech/text", None, raw=b"{not json")
    assert status == 400 and not out["ok"]


@needs_core
def test_page_works_without_microphone_and_speech_model(monkeypatch):
    monkeypatch.setattr(P, "check_microphone", lambda device=None: "no microphone found")
    speech = P.SpeechToSign(CFG, load_asr=False, predict=False)
    s = AppServer(app_config(8860), ROOT, speech)
    s.start()
    try:
        status, _, page = get(s.url)
        assert status == 200 and b"Speech" in page
        st = json.loads(get(s.url + "api/status")[2])
        sp = st["speech_to_sign"]
        assert not sp["microphone"]["ok"] and "no microphone" in sp["microphone"]["message"]
        assert not sp["speech"]["ok"] and sp["speech"]["message"]
        assert not st["sign_to_text"]["ok"] and st["sign_to_text"]["message"]
        status, out = post(s.url + "api/speech/start", {})
        assert status == 200 and not out["ok"] and "Microphone" in out["message"]
        status, out = post(s.url + "api/speech/text", {"text": "বাবা"})
        assert out["ok"] and out["sequence"]  # typed text still works
    finally:
        s.close()


def test_server_without_speech_part():
    s = AppServer(app_config(8870), ROOT, None)
    s.start()
    try:
        assert get(s.url)[0] == 200
        status, out = post(s.url + "api/speech/text", {"text": "বাবা"})
        assert not out["ok"] and "not loaded" in out["message"]
    finally:
        s.close()


def test_studio3d_default_is_unchanged():
    """The Module 2 viewer keeps its perspective camera; only the app asks for the orthographic one."""
    assert CFG["speech2sign"]["studio3d"]["camera"].get("type", "perspective") != "orthographic"
    js = (ROOT / CFG["speech2sign"]["studio3d"]["web_dir"] / "studio3d.js").read_text(encoding="utf-8")
    assert "if (EMBED)" in js and "poll();" in js and "new THREE.PerspectiveCamera(cam.fov" in js


def test_push_to_talk_reports_missing_device(monkeypatch):
    talk = P.PushToTalk(16000, device=10_000, max_s=1)  # no such device
    talk.start()
    audio, error = talk.stop()
    assert audio is None and error
