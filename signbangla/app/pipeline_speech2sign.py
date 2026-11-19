"""Speech -> Sign for the app: microphone (push to talk) or typed text -> sign words -> one animation sequence.

Everything is loaded once (SpeechToSign.load). A missing microphone or speech model only switches that part off; typed
text keeps working. The sequence is the stitched sentence (stitch.sentence_motion) with the head movement of
nonmanual.py for questions and negation already in the keypoints, lifted to 3D (lift3d.py). Both views of the page play
these same frames: the 3D mannequin uses x, y, z and the 2D skeleton x and y.
"""
import dataclasses
import threading
import time

import numpy as np

from ..speech2sign.dictionary import SignDictionary
from ..speech2sign.nonmanual import Markers, apply_markers
from ..speech2sign.stitch import sentence_motion
from ..speech2sign.studio3d import sequence_payload
from ..utils import ROOT, work_dir
from .gloss import Glosser, TenseMarkers
from .predict import PROTECTED, build_predictor


def check_microphone(device=None):
    """'' when an input device can be opened, else the reason."""
    try:
        import sounddevice as sd
    except Exception as e:  # noqa: BLE001 - missing package or audio library
        return f"no audio library ({type(e).__name__}: {e})"
    try:
        info = sd.query_devices(device, kind="input")
    except Exception as e:  # noqa: BLE001
        return f"no microphone found ({e})"
    return "" if info and info.get("max_input_channels", 0) > 0 else "no microphone found"


class PushToTalk:
    """Records with speech2sign/audio.record while the page's button is held: start() opens the microphone, stop()
    closes it and returns the audio. Recording stops by itself after max_s seconds."""

    def __init__(self, sample_rate=16000, device=None, max_s=20):
        self.sample_rate, self.device, self.max_s = sample_rate, device, max_s
        self.thread, self.audio, self.error = None, None, None
        self.stop_event = threading.Event()
        self.started = threading.Event()

    @property
    def recording(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        from ..speech2sign.audio import record
        if self.recording:
            return
        self.stop_event.clear()
        self.started.clear()
        self.audio, self.error = None, None
        calls = iter([self.started.set, lambda: self.stop_event.wait(self.max_s)])

        def wait(_message):
            next(calls)()
            return ""

        def run():
            try:
                self.audio = record(self.sample_rate, self.device, wait=wait)
            except Exception as e:  # noqa: BLE001 - shown on the page
                self.error = f"{type(e).__name__}: {e}"
                self.started.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self.started.wait(2)

    def stop(self):
        """-> (audio float32 or None, error text)."""
        if self.thread is None:
            return None, "not recording"
        self.stop_event.set()
        self.thread.join(5)
        self.thread = None
        return self.audio, self.error or ""


class SpeechToSign:
    def __init__(self, cfg, load_asr=None, predict=True):
        self.cfg = cfg
        self.s2s = cfg["speech2sign"]
        acfg = cfg["app"]
        self.load_times = {}
        started = time.perf_counter()
        folder = work_dir(cfg) / self.s2s["core_vocab"]["dictionary_dir"]
        if not (folder / "dictionary.json").exists():
            folder = work_dir(cfg) / self.s2s["dictionary_dir"]
        self.dictionary = SignDictionary(folder)
        self.glosser = Glosser(self.dictionary, acfg["gloss"], self.s2s["matching"],
                               TenseMarkers.from_config(cfg, ROOT))
        self.load_times["dictionary"] = round(time.perf_counter() - started, 2)
        self.last_gloss = None
        self.predictor, self.predict_message = None, ""
        if predict:
            signs, categories = self.signed_words()
            self.predictor, self.load_times["language_model"], self.predict_message = build_predictor(
                cfg, ROOT, self.glosser.matcher, signs, signs, categories)
            if acfg["predict"]["speech"]:
                self.glosser.predictor = self.predictor

        audio = self.s2s["audio"]
        self.mic_error = check_microphone(audio["device"])
        self.talk = PushToTalk(audio["sample_rate"], audio["device"], acfg["speech"]["max_record_s"])
        self.asr, self.asr_error = None, ""
        load = acfg["speech"]["load_asr"] if load_asr is None else load_asr
        if not load:
            self.asr_error = "speech recognition is switched off (--no-asr or app.speech.load_asr)"
        else:
            started = time.perf_counter()
            try:
                from ..speech2sign.asr import WhisperASR, vocabulary_words
                a = self.s2s["asr"]
                self.asr = WhisperASR(a["model"], a["language"], a["device"], a["max_new_tokens"], vocabulary_words(),
                                      a["generation_config_from"])
                self.load_times["speech_model"] = round(time.perf_counter() - started, 2)
            except Exception as e:  # noqa: BLE001 - no model files, no internet for the first download, no package
                self.asr_error = f"speech model could not be loaded ({type(e).__name__}: {str(e)[:200]})"

    def signed_words(self):
        """-> ({word: sign key} of every usable sign by its first Bangla name, {word: category}); না / নেই / question
        words left out, because prediction never inserts them."""
        signs, cats = {}, {}
        for key, e in self.dictionary.words.items():
            names = e.get("bangla") or []
            if not names or not self.glosser.usable(key):
                continue
            w = names[0]
            if w in PROTECTED or w in signs:
                continue
            signs[w], cats[w] = key, e.get("category", "")
        return signs, cats

    def status(self):
        return dict(
            dictionary=dict(folder=self.dictionary.folder.name, words=len(self.dictionary.words)),
            microphone=dict(ok=not self.mic_error, message=self.mic_error),
            speech=dict(ok=self.asr is not None, message=self.asr_error,
                        device=getattr(self.asr, "device", "") if self.asr else ""),
            prediction=dict(ok=self.predictor is not None, model=getattr(self.predictor.lm, "name", "")
                            if self.predictor else "", message=self.predict_message),
            load_times=self.load_times)

    # ---- text -> animation
    def sequence(self, gloss):
        """Gloss -> the payload both views play: fps, frames (T, 67, 3), word (caption index per frame), tokens."""
        keys = gloss.keys
        if not keys:
            return None
        sentence = sentence_motion(keys, self.dictionary, **self.s2s["sentence"])
        markers = Markers(gloss.question, gloss.negation)
        if markers.question or markers.negation:
            seq, _ = apply_markers(sentence.seq, sentence.words, len(keys), markers, sentence.fps,
                                   self.s2s["nonmanual"])
            sentence = dataclasses.replace(sentence, seq=seq)
        payload = sequence_payload(sentence, gloss.layout(), self.dictionary.rest_pose, self.s2s["studio3d"]["lift"])
        payload["markers"] = dict(question=markers.question, negation=markers.negation)
        payload["duration_s"] = round((len(sentence.seq) - 1) / sentence.fps, 2)
        return payload

    def from_text(self, text, heard=False, timings=None):
        timings = dict(timings or {})
        started = time.perf_counter()
        gloss = self.glosser.gloss(text)
        timings["text_s"] = round(time.perf_counter() - started, 3)
        return self._result(gloss, heard, timings)

    def _result(self, gloss, heard, timings):
        self.last_gloss, self.last_heard = gloss, heard
        started = time.perf_counter()
        payload = self.sequence(gloss)
        timings["animation_s"] = round(time.perf_counter() - started, 3)
        return dict(ok=True, text=gloss.text, heard=heard, gloss=gloss.as_dict(), sequence=payload, timings=timings,
                    message="" if payload else "None of these words has a sign yet.")

    def choose(self, index, action, word=""):
        """A predicted word (≈) in the last sentence: accept it, undo it (back to red) or take an alternative.
        -> the new result, with the animation made again."""
        from .gloss import Token
        g = self.last_gloss
        if g is None or not 0 <= int(index) < len(g.tokens):
            raise ValueError("no such word")
        t = g.tokens[int(index)]
        if t.status != "predicted" and action != "alternative":
            raise ValueError("this word is not a prediction")
        old = t.word
        if action == "accept":
            t.status, t.note = "sign", f"{t.text} -> {t.word} (prediction accepted)"
        elif action == "undo":
            g.tokens[int(index)] = Token(t.text, "", "", "unknown", t.category, "prediction undone")
        elif action == "alternative":
            m = self.glosser.matcher.match(word)
            key = self.glosser.sign_of(m[0]) if len(m) == 1 else ""
            if not key or m[0].word in PROTECTED:
                raise ValueError(f"{word} has no sign")
            g.tokens[int(index)] = Token(t.text, m[0].word, key, "sign", m[0].category,
                                         f"{t.text} -> {m[0].word} (chosen)")
        else:
            raise ValueError(f"unknown action {action}")
        if self.predictor and self.predictor.log:
            self.predictor.log.write("speech", g.text, int(index), [], word or old,
                                     {"accept": "yes", "undo": "no"}.get(action, f"alternative:{word}"))
        return self._result(g, self.last_heard, {})

    def coverage(self, sentences):
        """Test sentences -> one row each: how many of its words get a sign (predictions counted apart)."""
        rows = []
        for s in sentences:
            g = self.glosser.gloss(s)
            signed = [t for t in g.tokens if t.signed and t.status != "predicted"]
            predicted = [t for t in g.tokens if t.status == "predicted"]
            missing = [t.text for t in g.tokens if not t.signed]
            n = len(g.tokens)
            rows.append(dict(sentence=s, signs=" ".join(t.caption for t in g.tokens if t.signed),
                             words=n, signed=len(signed), predicted=len(predicted), missing=" ".join(missing),
                             coverage_pct=round(100 * len(signed) / n, 1) if n else 0.0,
                             all_signed=bool(n) and len(signed) == n))
        return rows

    # ---- microphone
    def listen_start(self):
        if self.mic_error:
            return dict(ok=False, message=f"Microphone: {self.mic_error}")
        if self.asr is None:
            return dict(ok=False, message=f"Speech recognition: {self.asr_error}")
        self.talk.start()
        if self.talk.error:
            return dict(ok=False, message=f"Microphone: {self.talk.error}")
        return dict(ok=True, recording=True)

    def listen_stop(self):
        from ..speech2sign.audio import has_speech, loudest_rms
        audio, error = self.talk.stop()
        if error:
            return dict(ok=False, message=f"Microphone: {error}")
        a = self.s2s["audio"]
        rate = a["sample_rate"]
        audio = np.zeros(0, np.float32) if audio is None else audio
        if not has_speech(audio, rate, a["min_seconds"], a["min_rms"]):
            return dict(ok=False, message=f"Heard nothing ({len(audio) / rate:.1f} s, volume "
                                          f"{loudest_rms(audio, rate):.3f}; needs {a['min_rms']}). Speak closer to the "
                                          "microphone and hold the button while speaking.")
        started = time.perf_counter()
        text = self.asr.transcribe(audio, self.glosser.matcher, rate)
        asr_s = round(time.perf_counter() - started, 2)
        if not text.strip():
            return dict(ok=False, message="Speech recognition returned no text.", timings=dict(speech_s=asr_s))
        return self.from_text(text, heard=True, timings=dict(speech_s=asr_s, audio_s=round(len(audio) / rate, 2)))
