"""SignBangla app, phase 3: tense / negation markers in Speech -> Sign, context prediction (the chosen word follows the
context; না / নেই / question words are never inserted or removed), the prediction log, accept / undo / alternatives,
test-sentence coverage and the framing check."""
import copy
import csv
import json
import os
import urllib.error
import urllib.request

import pytest

from signbangla.app.gloss import Glosser, TenseMarkers
from signbangla.app.pipeline_sign2text import SignToText, framing_hint
from signbangla.app.predict import GAP, PROTECTED, ContextPredictor, NgramLM, PredictionLog
from signbangla.app.server import AppServer, read_sentences
from signbangla.utils import ROOT, load_config, work_dir
from tests.test_app import CORE, FAKE, FakeDictionary, word

CFG = load_config(None)
PCFG = CFG["app"]["predict"]
needs_core = pytest.mark.skipif(not (CORE / "dictionary.json").exists(), reason="work/sign_dictionary_core missing")

VERBS = [dict(verb="খাওয়া", person="3_informal", present_simple="খায়", present_continuous="খাচ্ছে",
              past_simple="খেল", past_perfect="খেয়েছিল", future="খাবে", present_perfect="খেয়েছে",
              past_habitual="খেত"),
         dict(verb="খাওয়া", person="1", present_simple="খাই", present_continuous="খাচ্ছি", past_simple="খেলাম",
              past_perfect="খেয়েছিলাম", future="খাব", present_perfect="খেয়েছি", past_habitual="খেতাম")]
RULES = {"markers": [
    dict(marker="ছিল", habitual=False, will=False, tense="past_perfect", neg="", support=170),
    dict(marker="ছিল", habitual=True, will=False, tense="past_habitual", neg="", support=157),
    dict(marker="না", habitual=False, will=False, tense="present_simple", neg="na", support=192),
    dict(marker="নেই", habitual=False, will=False, tense="present_simple", neg="ni", support=170),
    dict(marker="হবে", habitual=False, will=False, tense="future", neg="", support=168),
    dict(marker="হয়েছে", habitual=False, will=False, tense="present_perfect", neg="", support=164)]}


def with_markers(**extra):
    words = dict(FAKE.words, **extra)
    return Glosser(FakeDictionary(words), CFG["app"]["gloss"], CFG["speech2sign"]["matching"],
                   TenseMarkers(VERBS, RULES))


VERIFIED_MARKERS = dict(C035=word("না", "BanSign9K", category="negation"),
                        C036=word("নেই", "BanSign9K", category="negation"),
                        S002=word("ছিল", "BanSign9K", category="other"),
                        S009=word("হয়েছে", "BanSign9K", category="other"),
                        S005=word("হবে", "BanSign9K", verified=False, category="other"))


def captions(g, text):
    return [(t.caption, t.status) for t in g.gloss(text).tokens]


# ------------------------------------------------------------------ tense and negation markers
def test_markers_from_verb_endings():
    g = with_markers(**VERIFIED_MARKERS)
    assert captions(g, "আমি ভাত খাইনি")[-2:] == [("খাওয়া", "sign"), ("নেই", "marker")]
    assert captions(g, "আমি ভাত খেয়েছিলাম")[-2:] == [("খাওয়া", "sign"), ("ছিল", "marker")]
    assert captions(g, "মা ভাত খেয়েছে")[-2:] == [("খাওয়া", "sign"), ("হয়েছে", "marker")]
    assert g.gloss("আমি ভাত খাইনি").negation


def test_unverified_marker_keeps_the_plain_verb_and_the_time_word():
    g = with_markers(**VERIFIED_MARKERS)
    assert [c for c, _ in captions(g, "আমি আজ ভাত খাব")] == ["আজ", "আমি", "ভাত", "খাওয়া"]  # হবে unverified
    assert [c for c, _ in captions(g, "মা ভাত খেত")] == ["মা", "ভাত", "খাওয়া"]  # আগে has no sign


def test_separate_negation_word_is_kept():
    g = with_markers(**VERIFIED_MARKERS)
    assert [c for c, _ in captions(g, "আমি ভাত খাই না")] == ["আমি", "ভাত", "খাওয়া", "না"]


def test_negation_marker_without_a_sign_stays_red():
    g = with_markers()  # no না / নেই signs at all
    out = captions(g, "আমি ভাত খাইনি")
    assert out[-2][0] == "খাওয়া" and out[-1] == ("নেই", "no_sign")


def test_a_fuzzy_match_never_swallows_a_known_word():
    g = with_markers()
    assert [c for c, _ in captions(g, "দাদি ভাত")][-1] == "ভাত"


def test_fuzzy_match_with_another_first_letter_is_unknown():
    """রান্না (cooking) must not become কান্না (crying): a near spelling with another first letter is no match."""
    words = dict(FAKE.words, C090=word("কান্না", "BdSL300", category="verb"))
    g = Glosser(FakeDictionary(words), CFG["app"]["gloss"], CFG["speech2sign"]["matching"])
    out = g.gloss("মা রান্না")
    assert [(t.text, t.signed) for t in out.tokens] == [("মা", True), ("রান্না", False)]


def test_yes_no_particle_is_dropped_even_with_a_sign():
    g = with_markers()  # FAKE has কী (question) with a sign
    out = g.gloss("তুমি কি ভাত খাবে")
    assert "কি" in [t.text for t in out.dropped] and "কী" not in [t.caption for t in out.tokens]
    assert out.question


def test_negation_kept_when_the_verb_has_no_sign():
    verbs = VERBS + [dict(verb="পারা", person="1", present_simple="পারি", present_continuous="পারছি",
                          past_simple="পারলাম", past_perfect="পেরেছিলাম", future="পারব",
                          present_perfect="পেরেছি", past_habitual="পারতাম")]
    g = Glosser(FakeDictionary(dict(FAKE.words, **VERIFIED_MARKERS)), CFG["app"]["gloss"],
                CFG["speech2sign"]["matching"], TenseMarkers(verbs, RULES))
    out = g.gloss("আমি পারিনি")
    assert [(t.caption, t.signed) for t in out.tokens] == [("আমি", True), ("পারিনি", False), ("নেই", True)]
    assert out.negation


# ------------------------------------------------------------------ context prediction
TRAIN = [["আমি", "ভাত", "খাওয়া"]] * 6 + [["আমি", "মা", "দেখা"]] * 6 + [["বাবা", "চা", "খাওয়া"]] * 4 + \
        [["মা", "ভাত", "রান্না"]] * 3


@pytest.fixture
def predictor(tmp_path):
    p = ContextPredictor(NgramLM(TRAIN), PCFG, ["ভাত", "মা", "চা", "পাখি", "পা", "বাবা", "না", "কী"],
                         PredictionLog(tmp_path / "predictions.csv"))
    return p


TOP5 = [["পাখি", 0.30], ["ভাত", 0.28], ["পা", 0.15], ["মা", 0.10], ["বাবা", 0.05]]


def items(*words):
    out = []
    for w in words:
        out.append(dict(word=TOP5[0][0], candidates=TOP5, source="camera") if w == "?" else
                   dict(word=w, candidates=None, source="hand"))
    return out


def test_chosen_word_follows_the_context(predictor):
    a = items("আমি", "?", "খাওয়া")
    predictor.predict_items(a)
    assert a[1]["prediction"]["word"] == "ভাত" and a[1]["prediction"]["top1"] == "পাখি"
    b = items("আমি", "?", "দেখা")
    predictor.predict_items(b)
    assert b[1]["prediction"]["word"] == "মা"
    assert a[1]["prediction"]["alternatives"] and "ভাত" not in a[1]["prediction"]["alternatives"]


def test_later_words_change_the_prediction(predictor):
    a = items("আমি", "?")
    predictor.predict_items(a)
    first = a[1]["prediction"]["word"]
    a.append(dict(word="দেখা", candidates=None, source="hand"))
    predictor.predict_items(a)
    assert a[1]["prediction"]["word"] == "মা" and first in ("ভাত", "মা")


def test_two_unclear_neighbours_give_a_gap(predictor):
    a = items("আমি", "?", "?", "খাওয়া")
    predictor.predict_items(a)
    assert a[1]["prediction"]["word"] == GAP and a[2]["prediction"]["word"] == GAP


def test_clear_signs_and_user_choices_are_not_predicted(predictor):
    a = items("আমি", "?", "খাওয়া")
    a[1]["candidates"] = [["পাখি", 0.9], ["ভাত", 0.05]]
    predictor.predict_items(a)
    assert not a[1].get("prediction")
    b = items("আমি", "?", "খাওয়া")
    b[1]["user"] = True
    predictor.predict_items(b)
    assert not b[1].get("prediction")


def test_negation_and_question_words_are_never_inserted_or_removed(predictor):
    # a sign whose best word is না keeps it (never removed) ...
    a = [dict(word="আমি", candidates=None, source="hand"),
         dict(word="না", candidates=[["না", 0.2], ["ভাত", 0.19]], source="camera")]
    predictor.predict_items(a)
    assert not a[1].get("prediction")
    # ... and না / কী are never suggested, even when they are candidates or in the vocabulary
    c = items("আমি", "?", "খাওয়া")
    c[1]["candidates"] = [["কী", 0.3], ["ভাত", 0.29]]  # best word is a question word: kept, not predicted
    predictor.predict_items(c)
    assert not c[1].get("prediction")
    b = items("আমি", "?", "খাওয়া")
    b[1]["candidates"] = [["পাখি", 0.3], ["না", 0.29], ["কী", 0.2], ["ভাত", 0.1]]
    predictor.predict_items(b)
    assert b[1]["prediction"]["word"] not in PROTECTED
    assert all(w not in PROTECTED for w in b[1]["prediction"]["alternatives"])
    assert not set(predictor.vocabulary) & PROTECTED
    gap = predictor.fill_gap(["আমি", GAP, "খাওয়া"], 1, extra=["না", "কী"])
    assert gap["word"] not in PROTECTED and all(w not in PROTECTED for w, _ in gap["ranked"])


def test_speech_gaps_never_touch_protected_words(predictor):
    from signbangla.app.gloss import Token
    predictor.signs = {"ভাত": "W81", "মা": "W20"}
    toks = [Token("আমি", "আমি", "C001", "sign"), Token("না", "না", "", "no_sign", "negation"),
            Token("কী", "কী", "", "no_sign", "question")]
    predictor.fill_speech_gaps(toks)
    assert [t.status for t in toks] == ["sign", "no_sign", "no_sign"]


def test_predictions_are_logged(predictor, tmp_path):
    predictor.predict_items(items("আমি", "?", "খাওয়া"))
    with open(tmp_path / "predictions.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["time", "direction", "sentence", "position", "candidates", "chosen", "accepted"]
    r = rows[-1]
    assert r["direction"] == "sign" and r["position"] == "1" and r["chosen"] == "ভাত" and r["accepted"] == "pending"
    assert json.loads(r["candidates"])[0][0] == "ভাত"


def test_accept_undo_and_alternative_in_the_sentence(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["work_dir"] = str(tmp_path)
    s = SignToText(cfg, ROOT, camera_index=None, speak=lambda t: "", monitor=False, lm=NgramLM(TRAIN))
    try:
        s.add("আমি")
        s.state.add_event([dict(word=w, english="", score=p, class_idx=0) for w, p in TOP5])
        s.tick()
        s.add("খাওয়া")
        v = s.view()
        assert v["items"][1]["word"] == "ভাত" and v["items"][1]["prediction"]["top1"] == "পাখি"
        assert "ভাত" in v["sentence"]
        v = s.reject(1)
        assert v["items"][1]["word"] == "পাখি" and v["items"][1]["prediction"] is None
        s.clear()
        s.add("আমি")
        s.state.add_event([dict(word=w, english="", score=p, class_idx=0) for w, p in TOP5])
        s.tick()
        s.add("খাওয়া")
        v = s.accept(1)
        assert v["items"][1]["word"] == "ভাত" and v["items"][1]["user"]
        v = s.replace(1, "মা")
        assert v["items"][1]["word"] == "মা"
        with open(tmp_path / PCFG["log"], encoding="utf-8-sig", newline="") as f:
            accepted = [r["accepted"] for r in csv.DictReader(f)]
        assert "no" in accepted and "yes" in accepted
        with pytest.raises(ValueError):
            s.accept(0)  # not a prediction
    finally:
        s.close()


# ------------------------------------------------------------------ Speech -> Sign with the real dictionary
@pytest.fixture(scope="module")
def speech():
    if not (CORE / "dictionary.json").exists():
        pytest.skip("work/sign_dictionary_core missing")
    from signbangla.app.pipeline_speech2sign import SpeechToSign
    s = SpeechToSign(CFG, load_asr=False)
    s.predictor.log = None  # tests do not write to the app's prediction log
    return s


@needs_core
def test_real_markers(speech):
    d = speech.dictionary.words
    if "S002" not in d:
        pytest.skip("dictionary built before the Ban-Sign-Sent-9K signs were added")
    g = speech.glosser
    assert [t.caption for t in g.gloss("আমি ভাত খাইনি").tokens] == ["আমি", "ভাত", "খাওয়া", "নেই"]
    assert [t.caption for t in g.gloss("বাবা চা খেয়েছিল").tokens][-1] == "ছিল"
    assert [t.caption for t in g.gloss("আমি আগামীকাল ভাত খাব").tokens] == ["আগামীকাল", "আমি", "ভাত", "খাওয়া"]


@needs_core
def test_choose_endpoint_rules(speech):
    speech.from_text("আমি ভাত খাই")
    with pytest.raises(ValueError):
        speech.choose(0, "accept")  # not a prediction
    out = speech.choose(1, "alternative", "মা")
    assert [t["caption"] for t in out["gloss"]["tokens"]] == ["আমি", "মা", "খাওয়া"] and out["sequence"]
    with pytest.raises(ValueError):
        speech.choose(1, "alternative", "না")  # never inserted by a prediction choice


@needs_core
def test_coverage_endpoint(speech, tmp_path):
    tests = tmp_path / "test_sentences.csv"
    tests.write_text("sentence\nআমি ভাত খাই না\nআমি xyz খাই\n", encoding="utf-8-sig")
    cfg = copy.deepcopy(CFG)
    cfg["app"]["test_sentences"] = str(tests)
    cfg["app"]["port"] = 8895
    cfg["paths"]["work_dir"] = str(tmp_path / "work")
    server = AppServer(cfg, ROOT, speech, None)
    server.start()
    try:
        with urllib.request.urlopen(server.url + "api/tests", timeout=30) as r:
            out = json.loads(r.read())
        assert out["ok"] and out["summary"]["sentences"] == 2 and out["summary"]["fully_signed"] == 1
        assert out["rows"][0]["coverage_pct"] == 100.0 and "xyz" in out["rows"][1]["missing"]
        saved = tmp_path / "work" / CFG["app"]["coverage"]
        assert saved.exists() and len(read_sentences(saved)) == 2
    finally:
        server.close()
    cfg["app"]["test_sentences"] = str(tmp_path / "missing.csv")
    s = AppServer(dict(cfg, app=dict(cfg["app"], port=8896)), ROOT, speech, None)
    assert not s.test_sentences()["ok"]
    s.httpd.server_close()


def test_read_sentences_columns(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("id,sentence\n1,আমি ভাত খাই\n2,\n", encoding="utf-8-sig")
    b = tmp_path / "b.csv"
    b.write_text("বাক্য\nমা চা\n", encoding="utf-8")
    assert read_sentences(a) == ["আমি ভাত খাই"] and read_sentences(b) == ["মা চা"]


# ------------------------------------------------------------------ framing
def test_framing_hint():
    f = CFG["app"]["sign"]["framing"]
    good = [(0.27, 0.5, True)] * 30
    assert framing_hint(good, f)["hint"] == "framing like the training videos"
    assert "mirrored" in framing_hint([(0.27, 0.5, False)] * 30, f)["hint"]
    assert "move back" in framing_hint([(0.5, 0.5, True)] * 30, f)["hint"]
    assert "middle" in framing_hint([(0.27, 0.2, True)] * 30, f)["hint"]


# ------------------------------------------------------------------ the pretrained fill-mask model (when downloaded)
def test_fill_mask_model_offline(monkeypatch):
    from signbangla.app.predict import MaskedLM
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        lm = MaskedLM(PCFG["mlm_model"], "cpu")
    except Exception:  # noqa: BLE001
        pytest.skip("fill-mask model not downloaded")
    s = lm.scores(["আমি", GAP, "খাই"], 1, ["ভাত", "চা", "আকাশ"])
    assert set(s) == {"ভাত", "চা", "আকাশ"} and s["চা"] > s["আকাশ"]
