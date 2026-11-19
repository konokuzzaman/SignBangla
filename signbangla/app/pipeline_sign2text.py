"""Sign -> Text for the app: camera -> MediaPipe -> single signs -> top-5 words -> Bangla sentence -> speech.

Reuses Module 3's camera worker (sign2text/app_server.CameraWorker: OpenCV + MediaPipe Holistic with the dataset
settings of signbangla/keypoints.py, segmentation.py, the MJPEG picture), Module 1's recogniser (sign2text/recognizer.py)
and the sentence builder (build_items -> builder.build_from_candidates, corpus rules on). Settings: app.sign.

Each recognised sign adds its top-1 word; the page can swap it for another of its top-5 words. The sentence is rebuilt
after every change. It ends when the hands stay down for app.sign.sentence_end_s after a camera word, or with the Done
button: then it is spoken (sign2speech/speech.py, female voice, cached in work/app/speech_cache) and added to the
history. A missing camera, model or speech package only switches that part off; the word buttons always work.
"""
import threading
import time
from pathlib import Path

from ..sign2text.app_server import CameraWorker, ScreenState, build_items, word_groups
from ..sign2text.builder import SentenceBuilder
from ..utils import subdir

CAMERA_START_TIMEOUT_S = 20


class AppCameraWorker(CameraWorker):
    """Module 3's camera worker with the app's settings, and the time each recognition takes."""

    def __init__(self, index, cfg, recognizer, state, scfg):
        super().__init__(index, cfg, recognizer, state)
        self.scfg = scfg
        self.framing = []  # recent (shoulder width / frame width, centre x, left shoulder on the image right?)

    def recognise(self, segment, w, h):
        started = time.perf_counter()
        super().recognise(segment, w, h)
        self.state.update_camera(recognise_s=round(time.perf_counter() - started, 3),
                                 sign_frames=int(len(segment)))

    def picture(self, bgr, frame, cv2):
        """Also checks the framing against the dataset videos over the last 30 frames (see framing_hint)."""
        import numpy as np
        ls, rs = frame[11, :2], frame[12, :2]
        if not (np.isnan(ls).any() or np.isnan(rs).any()):
            self.framing.append((abs(ls[0] - rs[0]), (ls[0] + rs[0]) / 2, ls[0] > rs[0]))
            self.framing = self.framing[-30:]
            if len(self.framing) == 30:
                self.state.update_camera(framing=framing_hint(self.framing, self.scfg["framing"]))
        return super().picture(bgr, frame, cv2)


def framing_hint(samples, f):
    """Recent shoulder measurements -> dict(hint, shoulder_width, mirrored). The dataset videos (BdSLW401, BdSLW60) show
    the signer unmirrored, shoulders about 0.26 of the frame width, in the middle."""
    import numpy as np
    s = np.array(samples, float)
    width, centre, unmirrored = float(np.median(s[:, 0])), float(np.median(s[:, 1])), float(s[:, 2].mean())
    hints = []
    if unmirrored < 0.5:
        hints.append("the camera picture is mirrored: set app.sign.flip_input to true")
    lo, hi = f["shoulder_width"]
    if width > hi:
        hints.append("move back a little (hands may leave the picture)")
    elif width < lo:
        hints.append("move closer")
    if not f["centre_x"][0] <= centre <= f["centre_x"][1]:
        hints.append("stand in the middle of the picture")
    return dict(hint="; ".join(hints) or "framing like the training videos", shoulder_width=round(width, 2),
                centre_x=round(centre, 2), mirrored=unmirrored < 0.5)


def speech_output_problem():
    """'' when the Bangla voice can be used, else what is missing."""
    try:
        from ..sign2speech.speech import edge_module
        edge_module()
    except Exception as e:  # noqa: BLE001 - not installed, or the branch is missing
        return f"speech output not available ({e})"
    return ""


class SignToText:
    def __init__(self, cfg, root, camera_index="config", speak=None, clock=time.monotonic, monitor=True, lm=None,
                 predict=True):
        self.cfg, self.root = cfg, root
        self.a = a = cfg["app"]["sign"]
        self.clock = clock
        self.load_times = {}
        started = time.perf_counter()
        self.builder = SentenceBuilder.from_config(cfg, root)
        self.groups = word_groups(self.builder.lx)
        self.load_times["sentence_builder"] = round(time.perf_counter() - started, 2)

        self.state = ScreenState()
        self.recognizer, self.model_message = self._load_recognizer()
        self.predictor = self._predictor(lm) if predict else None
        self.audio_dir = subdir(cfg, a["speech_cache"])
        self.speech_message = "" if speak else speech_output_problem()
        self.speak_fn = speak or self._synthesize
        self.on_finish = None  # set by the server: adds the sentence to the history

        self.lock = threading.RLock()
        self.items = []  # dict(word, candidates [[word, score], ...] or None, source camera | hand)
        self.question = False
        self.result = dict(sentence="", trace=[], unknown=[], known=[])
        self.build_s = self.predict_s = 0.0
        self.finished, self.next_finish = [], 1
        self.last_event_id = 0
        self.last_activity = clock()
        self.camera_word_pending = False  # a camera word was added since the last finished sentence

        self.worker = None
        index = a["camera_index"] if camera_index == "config" else camera_index
        if index is None:
            self.state.update_camera(message="Camera switched off (--no-camera). The word buttons still work.")
        elif self.recognizer is None:
            self.state.update_camera(message=f"{self.model_message} The word buttons still work.")
        else:
            self._start_camera(index)

        self.stop_event = threading.Event()
        self.monitor = threading.Thread(target=self._monitor, daemon=True) if monitor else None
        if self.monitor:
            self.monitor.start()

    # ---- loading
    def _load_recognizer(self):
        checkpoint, word_list = self.root / self.a["checkpoint"], self.root / self.a["word_list"]
        if not checkpoint.exists():
            return None, f"Sign model not found: {self.a['checkpoint']}."
        if not word_list.exists():
            return None, f"Word list not found: {self.a['word_list']}."
        started = time.perf_counter()
        try:
            from ..sign2text.recognizer import SignRecognizer
            rec = SignRecognizer(checkpoint, word_list, self.cfg["preprocess"], self.a["device"])
        except Exception as e:  # noqa: BLE001 - wrong checkpoint, no torch, ...
            return None, f"The sign model could not be loaded: {type(e).__name__}: {e}."
        self.load_times["recogniser"] = round(time.perf_counter() - started, 2)
        return rec, ""

    def _predictor(self, lm):
        """Context prediction for unclear signs; the language model is shared with Speech -> Sign when given."""
        from ..speech2sign.text_processing import WordMatcher
        from .predict import ContextPredictor, Lemmatiser, NgramLM, PredictionLog, training_sentences
        from ..utils import work_dir
        started = time.perf_counter()
        if lm is None:
            lm = NgramLM(training_sentences(self.cfg, self.root, Lemmatiser(WordMatcher.full())))
        vocab = [w for g in self.groups.values() for w in (x["word"] for x in g)]
        vocab += [w for w, _ in self.recognizer.words] if self.recognizer else []
        p = self.cfg["app"]["predict"]
        self.load_times["prediction"] = round(time.perf_counter() - started, 2)
        return ContextPredictor(lm, p, vocab, PredictionLog(work_dir(self.cfg) / p["log"]))

    def _start_camera(self, index):
        started = time.perf_counter()
        self.state.update_camera(requested=True, message=f"Starting camera {index} and MediaPipe ...")
        self.worker = AppCameraWorker(index, self.cfg, self.recognizer, self.state, self.a)
        self.worker.start()
        while time.perf_counter() - started < CAMERA_START_TIMEOUT_S:  # MediaPipe is loaded once, here
            cam = self.state.snapshot(10 ** 9)
            if cam["enabled"] or not self.worker.is_alive():
                break
            time.sleep(0.05)
        self.load_times["camera_and_mediapipe"] = round(time.perf_counter() - started, 2)

    @property
    def vocabulary(self):
        return int(self.recognizer.config["num_classes"]) if self.recognizer else 0

    def status(self):
        cam = self.state.snapshot(10 ** 9)
        return dict(ok=bool(cam["enabled"]), message=cam["message"],
                    model=dict(ok=self.recognizer is not None, name=getattr(self.recognizer, "name", ""),
                               vocabulary=self.vocabulary, message=self.model_message),
                    speech_output=dict(ok=not self.speech_message, message=self.speech_message),
                    load_times=self.load_times, poll_ms=self.a["poll_ms"])

    # ---- the sentence
    @staticmethod
    def shown_word(it):
        """The word used in the sentence: a pending prediction, else the sign's own word."""
        p = it.get("prediction")
        return p["word"] if p and not it.get("user") else it["word"]

    def _rebuild(self):
        """Re-scores the unclear signs with the context (later words add context), then builds the sentence."""
        started = time.perf_counter()
        if self.predictor is not None:
            self.predictor.predict_items(self.items)
        self.predict_s = round(time.perf_counter() - started, 4)
        started = time.perf_counter()
        shown = [dict(word=self.shown_word(i), candidates=i["candidates"]) for i in self.items]
        self.result = (build_items(self.builder, shown, self.question) if self.items
                       else dict(sentence="", trace=[], unknown=[], known=[]))
        self.build_s = round(time.perf_counter() - started, 4)

    def _log(self, index, accepted, chosen):
        if self.predictor is not None and self.predictor.log is not None:
            words = " ".join(self.shown_word(i) for i in self.items)
            self.predictor.log.write("sign", words, index, [], chosen, accepted)

    def accept(self, index):
        """Keep the predicted word of sign `index`."""
        with self.lock:
            it = self._item(index)
            p = it.get("prediction")
            if not p:
                raise ValueError("this word is not a prediction")
            if p["word"] != "…":
                it["word"] = p["word"]
            it["user"] = True
            self._log(index, "yes", p["word"])
            self._rebuild()
            return self.view()

    def reject(self, index):
        """Undo a prediction: back to the recogniser's own top-1 word."""
        with self.lock:
            it = self._item(index)
            p = it.get("prediction")
            if not p:
                raise ValueError("this word is not a prediction")
            it["word"], it["user"], it["prediction"] = p["top1"], True, None
            self._log(index, "no", p["top1"])
            self._rebuild()
            return self.view()

    def _item(self, index):
        if not 0 <= int(index) < len(self.items):
            raise ValueError("no such word")
        return self.items[int(index)]

    def add(self, word, candidates=None, source="hand"):
        word = str(word).strip()
        if not word:
            return self.view()
        with self.lock:
            self.items.append(dict(word=word, candidates=candidates, source=source))
            if source == "camera":
                self.camera_word_pending = True
            self.last_activity = self.clock()
            self._rebuild()
            return self.view()

    def replace(self, index, word):
        """A word picked by the user (another top-5 word or a prediction's alternative): never predicted again."""
        with self.lock:
            it = self._item(index)
            word = str(word).strip()
            if it.get("prediction") and not it.get("user"):
                self._log(index, f"alternative:{word}", word)
            it["word"], it["user"], it["prediction"] = word, True, None
            self.last_activity = self.clock()
            self._rebuild()
            return self.view()

    def undo(self):
        with self.lock:
            if self.items:
                self.items.pop()
            self._rebuild()
            return self.view()

    def clear(self):
        with self.lock:
            self.items = []
            self.camera_word_pending = False
            self._rebuild()
            return self.view()

    def set_question(self, on):
        with self.lock:
            self.question = bool(on)
            self._rebuild()
            return self.view()

    def finish(self, auto=False):
        """Ends the sentence: speaks it (in the background) and adds it to the history."""
        with self.lock:
            if not self.items:
                return dict(self.view(), ok=False, message="No words yet.")
            sentence = self.result["sentence"]
            for k, it in enumerate(self.items):  # a prediction still shown when the sentence ends counts as kept
                if it.get("prediction") and not it.get("user"):
                    self._log(k, "kept", it["prediction"]["word"])
            entry = dict(id=self.next_finish, sentence=sentence, words=[self.shown_word(i) for i in self.items],
                         auto=auto,
                         question=self.question, audio="", message="", speech_s=None, done=False)
            self.next_finish += 1
            self.finished.append(entry)
            self.finished = self.finished[-20:]
            self.items, self.camera_word_pending = [], False
            self._rebuild()
        if self.on_finish:
            self.on_finish(entry)
        threading.Thread(target=self._speak_entry, args=(entry,), daemon=True).start()
        return dict(self.view(), ok=True, finished=entry)

    def _speak_entry(self, entry):
        started = time.perf_counter()
        try:
            entry["audio"] = self.speak_fn(entry["sentence"])
        except Exception as e:  # noqa: BLE001 - no internet, no edge-tts, ...
            entry["message"] = f"Not spoken: {type(e).__name__}: {e}"
        entry["speech_s"] = round(time.perf_counter() - started, 2)
        entry["done"] = True

    def _synthesize(self, text):
        from ..sign2speech.speech import synthesize
        return f"/audio/{Path(synthesize(text, self.audio_dir, self.a['voice'])).name}"

    # ---- camera events and the sentence-end pause
    def tick(self, now=None):
        now = self.clock() if now is None else now
        snap = self.state.snapshot(self.last_event_id)
        auto_finish = False
        with self.lock:
            for e in snap["events"]:
                self.last_event_id = e["id"]
                top = [[t["word"], float(t["score"])] for t in e["top"]]
                if top:
                    self.items.append(dict(word=top[0][0], candidates=top, source="camera"))
                    self.camera_word_pending = True
                    self.last_activity = now
                    self._rebuild()
            if snap["state"] == "signing":
                self.last_activity = now
            elif (self.items and self.camera_word_pending
                  and now - self.last_activity >= self.a["sentence_end_s"]):
                auto_finish = True
        if auto_finish:
            self.finish(auto=True)

    def _monitor(self):
        while not self.stop_event.wait(0.1):
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - keep the camera loop alive; shown on the page
                self.state.update_camera(last=f"error: {type(e).__name__}: {e}")

    def view(self, since_finish=0):
        cam = self.state.snapshot(10 ** 9)
        cam.pop("events", None)
        with self.lock:
            return dict(ok=True, camera=cam, question=self.question, build_s=self.build_s, predict_s=self.predict_s,
                        items=[dict(word=self.shown_word(i), own=i["word"], source=i["source"],
                                    candidates=i["candidates"] or [], user=bool(i.get("user")),
                                    prediction=i.get("prediction") if not i.get("user") else None)
                               for i in self.items],
                        sentence=self.result["sentence"], unknown=self.result["unknown"],
                        known=self.result["known"], trace=self.result["trace"],
                        finished=[dict(f) for f in self.finished if f["id"] > since_finish])

    def close(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.stop_event.set()
            self.worker.join(timeout=5)
