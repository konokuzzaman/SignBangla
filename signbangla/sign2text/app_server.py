"""The Module 3 screen: a local web page that builds a Bangla sentence from sign words.

Words mode needs nothing but the lexicon. Camera mode (optional) reads the webcam with OpenCV, runs MediaPipe Holistic
with the dataset settings (signbangla.keypoints), cuts the stream into signs (segmentation.py), recognises each with
Module 1's 401-word model (recognizer.py) and sends the top-k words to the page; the page adds the top-1 word and lets
the user pick another one. The camera picture with the skeleton is streamed as MJPEG (/video.mjpg).

The server uses the Python standard library only and listens on 127.0.0.1 (if the port is busy, the next ones are
tried). Endpoints:
    GET  /  /app.js  /style.css  /font.ttf          the page
    GET  /api/config                                  word groups, examples, camera status
    POST /api/build   {items, is_question}            -> {sentence, trace, unknown, known}
    POST /api/split   {text}                          -> {words}: the lexicon words of a typed sequence
    GET  /api/camera?since=<id>                       -> camera status and new recognised signs
    GET  /video.mjpg                                  camera picture with the skeleton
    POST /api/speak   {text}                          -> {ok, url} (the project's Bangla speech output) or a message
    GET  /audio/<hash>.mp3                            the spoken sentence
"""
import json
import re
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from ..speech2sign.studio3d import _LocalServer
from ..utils import ROOT, subdir
from .builder import SentenceBuilder, build_from_candidates
from .lexicon import norm
from .tables import COPULA

GROUPS = ("pronoun", "verb", "question", "time", "negation", "noun", "other")
CAMERA_NOTE = "Camera vocabulary: 401 BdSLW401 words (nouns). Verbs and question words come from the word list for now."
AUDIO_RE = re.compile(r"^/audio/([0-9a-f]{64}\.mp3)$")
BOUNDARY = "frame"


# ---------------------------------------------------------------- sentences
def word_groups(lexicon):
    """{group: [{word, english, has_sign}]} for the word buttons, in lexicon order."""
    groups = {g: [] for g in GROUPS}
    for w, e in lexicon.entries.items():
        if w == COPULA:
            continue
        g = e["pos"] if e["pos"] in GROUPS else "other"
        groups[g].append(dict(word=w, english=e.get("english", ""), has_sign=e.get("has_sign") == "yes"))
    return groups


def build_items(builder, items, is_question=False):
    """items: [{word}] from the word buttons or [{word, candidates: [[word, score], ...]}] from the camera.
    With any camera item the words go through build_from_candidates; a word picked by hand counts as the best
    candidate of its sign. -> dict(sentence, trace, unknown, known)."""
    items = [i for i in items if str(i.get("word", "")).strip()]
    if any(i.get("candidates") for i in items):
        cands = []
        for i in items:
            pairs = [(str(w), float(s)) for w, s in i.get("candidates") or []]
            if not pairs or i["word"] != max(pairs, key=lambda p: p[1])[0]:
                top = max([s for _, s in pairs], default=0.0)
                pairs = [(i["word"], top + 1.0)] + [p for p in pairs if p[0] != i["word"]]
            cands.append(pairs)
        sentence, trace, unknown = build_from_candidates(cands, is_question, builder=builder)
    else:
        sentence, trace, unknown = builder.build([i["word"] for i in items], is_question)
    known = [norm(i["word"]) in builder.lx.surface for i in items]
    return dict(sentence=sentence, trace=trace, unknown=unknown, known=known)


def split_words(builder, text):
    """A typed sequence -> its lexicon words (multi-word entries kept together, unknown words as they are)."""
    return [t.word or t.text for t in builder.normalise(text, [], [])]


# ---------------------------------------------------------------- camera
class ScreenState:
    """What the camera thread shares with the page."""

    def __init__(self):
        self.lock = threading.Condition()
        self.events, self.next_id = [], 1
        self.camera = dict(requested=False, enabled=False, message="", state="ready", fps=0.0, last="")
        self.jpeg = None
        self.jpeg_id = 0

    def update_camera(self, **kw):
        with self.lock:
            self.camera.update(kw)

    def add_event(self, top):
        with self.lock:
            self.events.append(dict(id=self.next_id, top=top, time=time.time()))
            self.next_id += 1
            self.events = self.events[-50:]

    def set_jpeg(self, data):
        with self.lock:
            self.jpeg, self.jpeg_id = data, self.jpeg_id + 1
            self.lock.notify_all()

    def snapshot(self, since):
        with self.lock:
            return dict(self.camera, events=[e for e in self.events if e["id"] > since])


class CameraWorker(threading.Thread):
    def __init__(self, index, cfg, recognizer, state):
        super().__init__(daemon=True)
        self.index, self.cfg, self.rec, self.state = index, cfg, recognizer, state
        self.scfg = cfg["module3"]["screen"]
        self.stop_event = threading.Event()

    def run(self):
        import cv2

        from ..keypoints import make_holistic, results_to_frame
        from .segmentation import SIGNING, SignSegmenter
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            self.state.update_camera(enabled=False, message=f"No camera found at index {self.index}. Words mode works.")
            return
        try:
            holistic = make_holistic(self.cfg)
        except Exception as e:  # noqa: BLE001
            cap.release()
            self.state.update_camera(enabled=False, message=f"MediaPipe could not start: {e}")
            return
        seg = SignSegmenter.from_config(self.scfg["segment"])
        self.state.update_camera(enabled=True, message=f"Camera {self.index}, model {self.rec.name}")
        frames, t_fps = 0, time.perf_counter()
        try:
            while not self.stop_event.is_set():
                ok, bgr = cap.read()
                if not ok:
                    self.state.update_camera(enabled=False, message="The camera stopped sending pictures.")
                    break
                t = time.perf_counter()
                h, w = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                frame = results_to_frame(holistic.process(rgb))
                segment = seg.update(frame, t)
                if segment is not None:
                    self.recognise(segment, w, h)
                frames += 1
                if t - t_fps >= 1.0:
                    self.state.update_camera(fps=round(frames / (t - t_fps), 1))
                    frames, t_fps = 0, t
                self.state.update_camera(state="signing" if seg.state == SIGNING else "ready", last=seg.last_event)
                self.state.set_jpeg(self.picture(bgr, frame, cv2))
        finally:
            cap.release()
            holistic.close()

    def recognise(self, segment, w, h):
        from .recognizer import PreprocessError
        try:
            self.state.add_event(self.rec.top_k(segment, w, h, self.scfg["top_k"]))
        except PreprocessError as e:
            self.state.update_camera(last=f"sign not recognised: {e}")

    def picture(self, bgr, frame, cv2):
        """The camera picture with the skeleton, mirrored like a mirror if set, as JPEG bytes."""
        import numpy as np

        from ..viz import CONNECTIONS
        colors = {"black": (235, 235, 235), "tab:blue": (235, 130, 47), "tab:red": (40, 140, 245)}
        h, w = bgr.shape[:2]
        img = bgr.copy()
        pts = frame[:, :2] * [w, h]
        ok = ~np.isnan(pts).any(1)
        for a, b, c in CONNECTIONS:
            if ok[a] and ok[b]:
                cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), colors[c], 2, cv2.LINE_AA)
        if self.scfg["mirror_display"]:
            img = cv2.flip(img, 1)
        scale = self.scfg["video_width"] / w
        img = cv2.resize(img, (self.scfg["video_width"], int(round(h * scale))), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.scfg["jpeg_quality"]])[1].tobytes()


# ---------------------------------------------------------------- server
class SentenceScreen:
    def __init__(self, cfg, camera_index=None, root=ROOT, open_browser=None, port=None):
        m3 = cfg["module3"]
        self.scfg = s = m3["screen"]
        self.cfg, self.root = cfg, root
        self.builder = SentenceBuilder.from_config(cfg, root)
        self.groups = word_groups(self.builder.lx)
        self.state = ScreenState()
        self.audio_dir = subdir(cfg, m3["speech_cache"])
        web = root / s["web_dir"]
        self.files = {"/": (web / "index.html", "text/html; charset=utf-8"),
                      "/app.js": (web / "app.js", "text/javascript; charset=utf-8"),
                      "/style.css": (web / "style.css", "text/css; charset=utf-8"),
                      "/font.ttf": (root / cfg["speech2sign"]["font"], "font/ttf")}
        for path, _ in self.files.values():
            if not path.exists():
                raise FileNotFoundError(f"screen file missing: {path}")
        self.worker = None
        if camera_index is not None:
            self.start_camera(camera_index)
        self.httpd = self._bind(s["host"], port or s["port"], s["port_tries"])
        self.url = f"http://{s['host']}:{self.httpd.server_address[1]}/"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        if s["open_browser"] if open_browser is None else open_browser:
            webbrowser.open(self.url)

    def start_camera(self, index):
        self.state.update_camera(requested=True)
        checkpoint = self.root / self.scfg["checkpoint"]
        word_list = self.root / self.cfg["bdslw401"]["root"] / "word_list.csv"
        if not checkpoint.exists():
            self.state.update_camera(message=f"Model not found: {self.scfg['checkpoint']}. Words mode works.")
            return
        if not word_list.exists():
            self.state.update_camera(message=f"Word list not found: {word_list.relative_to(self.root)}.")
            return
        try:
            from .recognizer import SignRecognizer
            rec = SignRecognizer(checkpoint, word_list, self.cfg["preprocess"], self.scfg["device"])
        except Exception as e:  # noqa: BLE001
            self.state.update_camera(message=f"The model could not be loaded: {type(e).__name__}: {e}")
            return
        self.state.update_camera(message=f"Starting camera {index}...")
        self.worker = CameraWorker(index, self.cfg, rec, self.state)
        self.worker.start()

    def _bind(self, host, port, tries):
        handler = _make_handler(self)
        last = None
        for p in range(port, port + max(1, tries)):
            try:
                return _LocalServer((host, p), handler)
            except OSError as e:
                last = e
        raise OSError(f"no free port in {port}-{port + max(1, tries) - 1} on {host}: {last}")

    def page_config(self):
        return dict(groups=self.groups, examples=self.scfg["examples"], note=CAMERA_NOTE,
                    poll_ms=self.scfg["poll_ms"], camera=self.state.snapshot(10 ** 9))

    def speak(self, text):
        if not text.strip():
            return dict(ok=False, message="Nothing to say yet.")
        try:
            from ..sign2speech.speech import synthesize
        except ImportError:
            return dict(ok=False, message="Speech output (signbangla/sign2speech, edge-tts) is on the texttovoice "
                                          "branch and not installed here.")
        try:
            path = Path(synthesize(text, self.audio_dir))
        except Exception as e:  # noqa: BLE001 - no internet, edge-tts missing, ...
            return dict(ok=False, message=f"Speech failed: {type(e).__name__}: {e}")
        return dict(ok=True, url=f"/audio/{path.name}")

    def close(self):
        if self.worker is not None:
            self.worker.stop_event.set()
            self.worker.join(timeout=5)
        self.httpd.shutdown()
        self.httpd.server_close()


def _make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            url = urlsplit(self.path)
            if url.path in app.files:
                path, kind = app.files[url.path]
                return self._send(path.read_bytes(), kind)
            if url.path == "/api/config":
                return self._json(app.page_config())
            if url.path == "/api/camera":
                since = int((parse_qs(url.query).get("since") or ["0"])[0] or 0)
                return self._json(app.state.snapshot(since))
            if url.path == "/video.mjpg":
                return self._stream()
            m = AUDIO_RE.match(url.path)
            if m and (app.audio_dir / m.group(1)).exists():
                return self._send((app.audio_dir / m.group(1)).read_bytes(), "audio/mpeg")
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):  # noqa: N802
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except (ValueError, UnicodeDecodeError):
                return self.send_error(HTTPStatus.BAD_REQUEST, "body must be JSON")
            path = urlsplit(self.path).path
            if path == "/api/build":
                items = body.get("items")
                if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
                    return self.send_error(HTTPStatus.BAD_REQUEST, "items must be a list of objects")
                return self._json(build_items(app.builder, items, bool(body.get("is_question"))))
            if path == "/api/split":
                return self._json(dict(words=split_words(app.builder, str(body.get("text", "")))))
            if path == "/api/speak":
                return self._json(app.speak(str(body.get("text", ""))))
            self.send_error(HTTPStatus.NOT_FOUND)

        def _stream(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with app.state.lock:
                        app.state.lock.wait_for(lambda: app.state.jpeg_id != seen, timeout=1.0)
                        data, seen = app.state.jpeg, app.state.jpeg_id
                    if data is None:
                        if app.worker is None or not app.worker.is_alive():
                            return
                        continue
                    self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n"
                                     .encode() + data + b"\r\n")
                    if app.worker is None or not app.worker.is_alive():
                        return
            except (ConnectionError, OSError):
                return

        def _json(self, obj):
            self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _send(self, body, kind):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler
