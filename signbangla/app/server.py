"""The SignBangla app server: one local page with both directions (standard library only, bound to 127.0.0.1).

Pages and files:
    /                  the app page (web/index.html, app.js, style.css, skeleton.js)
    /viewer3d.html     the studio3d mannequin page (speech2sign/web3d), shown inside the app in embed mode
    /studio3d.js, /vendor/three.module.js, /settings.json (studio3d settings with the app's camera), /font.ttf
API (JSON):
    GET  /api/status          what is loaded and what is missing (microphone, speech model, ...), load times
    GET  /api/history         the conversation so far
    GET  /api/speech/last     the last Speech -> Sign result (a reloaded page shows it again)
    POST /api/speech/text     {"text": ...} -> sign words and the animation sequence
    POST /api/speech/start    start recording (push to talk)
    POST /api/speech/stop     stop recording -> recognised text, sign words, animation sequence
    GET  /api/sign/state?since=<id>   camera state, the words so far with their top-5, the sentence, finished sentences
    GET  /api/sign/words      word buttons (Module 3 lexicon) for adding words by hand
    POST /api/sign/add {word} | /replace {index, word} | /undo | /clear | /question {on} | /done
    POST /api/sign/accept {index} | /api/sign/reject {index}     keep or undo a predicted word (Sign -> Text)
    POST /api/speech/choose {index, action: accept | undo | alternative, word}   the same for Speech -> Sign
    GET  /api/tests           coverage of data/app/test_sentences.csv (saved to work/app/coverage.csv)
    GET  /video.mjpg          camera picture with the skeleton
    GET  /audio/<hash>.mp3    a spoken sentence
"""
import json
import re
import threading
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..speech2sign.studio3d import _LocalServer, page_settings

JSON = "application/json; charset=utf-8"
AUDIO_RE = re.compile(r"^/audio/([0-9a-f]{64}\.mp3)$")
BOUNDARY = "frame"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".ttf": "font/ttf"}
MAX_BODY = 1 << 20


SENTENCE_COLUMNS = ("sentence", "text", "bangla", "natural")


def read_sentences(path):
    """Sentences of a CSV: the 'sentence' (or text / bangla / natural) column, else the first column."""
    import csv
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    col = next((c for c in SENTENCE_COLUMNS if c in rows[0]), next(iter(rows[0])))
    return [r[col].strip() for r in rows if (r.get(col) or "").strip()]


def viewer_settings(cfg):
    """studio3d page settings with the app's straight-on camera."""
    settings = page_settings(cfg["speech2sign"]["studio3d"])
    settings["camera"] = dict(cfg["app"]["viewer3d"]["camera"])
    return settings


class History:
    def __init__(self, limit=200):
        self.items, self.limit, self.lock = [], limit, threading.Lock()

    def add(self, direction, text, detail=""):
        with self.lock:
            self.items.append(dict(time=datetime.now().strftime("%H:%M:%S"), direction=direction, text=text,
                                   detail=detail))
            del self.items[:-self.limit]

    def all(self):
        with self.lock:
            return list(self.items)


class AppServer:
    def __init__(self, cfg, root, speech=None, sign=None):
        a = cfg["app"]
        s2s = cfg["speech2sign"]
        web3d = root / s2s["studio3d"]["web_dir"]
        web = root / a["web_dir"]
        self.files = {
            "/": web / "index.html",
            "/index.html": web / "index.html",
            "/app.js": web / "app.js",
            "/style.css": web / "style.css",
            "/skeleton.js": web / "skeleton.js",
            "/viewer3d.html": web3d / "index.html",
            "/studio3d.js": web3d / "studio3d.js",
            "/vendor/three.module.js": root / s2s["studio3d"]["three_js"],
            "/font.ttf": root / s2s["font"],
        }
        missing = [str(p) for p in self.files.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"app files missing: {missing}")
        self.cfg, self.speech, self.sign, self.root = cfg, speech, sign, Path(root)
        self.settings = json.dumps(viewer_settings(cfg)).encode("utf-8")
        self.history = History(a["history_max"])
        if sign is not None:
            sign.on_finish = lambda e: self.history.add("sign", e["sentence"], "signs: " + " ".join(e["words"]))
        self.last_speech = None  # shown again when the page is reloaded
        self.load_times = {}
        self.lock = threading.Lock()  # one speech request at a time
        self.httpd = self._bind(a["host"], a["port"], a["port_tries"])
        self.url = f"http://{a['host']}:{self.httpd.server_address[1]}/"
        self.thread = None

    def _bind(self, host, port, tries):
        handler = _make_handler(self)
        last = None
        for p in range(port, port + max(1, tries)):
            try:
                return _LocalServer((host, p), handler)
            except OSError as e:  # port busy: try the next one
                last = e
        raise OSError(f"no free port in {port}-{port + max(1, tries) - 1} on {host}: {last}")

    def start(self, open_browser=False):
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        if open_browser:
            webbrowser.open(self.url)
        return self.url

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # ---- API
    def status(self):
        a = self.cfg["app"]
        speech = self.speech.status() if self.speech else dict(
            microphone=dict(ok=False, message="Speech -> Sign is not loaded"),
            speech=dict(ok=False, message="Speech -> Sign is not loaded"), dictionary=dict(words=0), load_times={})
        sign = self.sign.status() if self.sign else dict(
            ok=False, message="Sign -> Text is not loaded.", model=dict(ok=False, vocabulary=0, name="", message=""),
            speech_output=dict(ok=False, message=""), load_times={}, poll_ms=a["sign"]["poll_ms"])
        return dict(speech_to_sign=speech, sign_to_text=sign,
                    load_times=dict(self.load_times, **speech.get("load_times", {}), **sign.get("load_times", {})),
                    speeds=a["speeds"], skeleton=a["skeleton"])

    # ---- Sign -> Text
    def _sign(self):
        if self.sign is None:
            raise LookupError("Sign -> Text is not loaded.")
        return self.sign

    def sign_add(self, body):
        return self._sign().add(body.get("word", ""))

    def sign_replace(self, body):
        return self._sign().replace(int(body.get("index", -1)), body.get("word", ""))

    def sign_undo(self, _body):
        return self._sign().undo()

    def sign_clear(self, _body):
        return self._sign().clear()

    def sign_question(self, body):
        return self._sign().set_question(bool(body.get("on")))

    def sign_done(self, _body):
        return self._sign().finish()

    def sign_accept(self, body):
        return self._sign().accept(int(body.get("index", -1)))

    def sign_reject(self, body):
        return self._sign().reject(int(body.get("index", -1)))

    def speech_choose(self, body):
        if self.speech is None:
            raise LookupError("Speech -> Sign is not loaded.")
        with self.lock:
            return self.speech.choose(int(body.get("index", -1)), str(body.get("action", "")),
                                      str(body.get("word", "")))

    # ---- test sentences
    def test_sentences(self):
        """data/app/test_sentences.csv -> coverage per sentence, saved to work/app/coverage.csv."""
        import csv
        from ..utils import work_dir
        a = self.cfg["app"]
        path = self.root / a["test_sentences"]
        if not path.exists():
            return dict(ok=False, message=f"{a['test_sentences']} not found. Add a CSV with a 'sentence' column.",
                        rows=[])
        if self.speech is None:
            return dict(ok=False, message="Speech -> Sign is not loaded.", rows=[])
        sentences = read_sentences(path)
        with self.lock:
            rows = self.speech.coverage(sentences)
        out = work_dir(self.cfg) / a["coverage"]
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]) if rows else ["sentence"])
            w.writeheader()
            w.writerows(rows)
        full = sum(r["all_signed"] for r in rows)
        mean = round(sum(r["coverage_pct"] for r in rows) / len(rows), 1) if rows else 0.0
        try:
            saved = out.relative_to(self.root).as_posix()
        except ValueError:  # a work folder outside the project
            saved = str(out)
        return dict(ok=True, rows=rows, saved=saved,
                    summary=dict(sentences=len(rows), fully_signed=full, mean_coverage_pct=mean))

    def sign_words(self):
        return dict(groups=self.sign.groups if self.sign else {}, vocabulary=self.sign.vocabulary if self.sign else 0)

    def speech_text(self, body):
        text = str(body.get("text", "")).strip()
        if not text:
            return dict(ok=False, message="Type a sentence first.")
        if self.speech is None:
            return dict(ok=False, message="Speech -> Sign is not loaded.")
        with self.lock:
            out = self.speech.from_text(text)
        self._remember(out)
        return out

    def speech_start(self, _body):
        if self.speech is None:
            return dict(ok=False, message="Speech -> Sign is not loaded.")
        return self.speech.listen_start()

    def speech_stop(self, _body):
        if self.speech is None:
            return dict(ok=False, message="Speech -> Sign is not loaded.")
        with self.lock:
            out = self.speech.listen_stop()
        self._remember(out)
        return out

    def _remember(self, out):
        if out.get("ok"):
            self.last_speech = out
            g = out["gloss"]
            detail = "signs: " + (" ".join(g["signs"]) or "-")
            if g["unknown"]:
                detail += " | no sign: " + " ".join(g["unknown"])
            self.history.add("speech", out["text"], detail)

    def routes(self):
        return {"/api/speech/text": self.speech_text, "/api/speech/start": self.speech_start,
                "/api/speech/stop": self.speech_stop, "/api/sign/add": self.sign_add,
                "/api/sign/replace": self.sign_replace, "/api/sign/undo": self.sign_undo,
                "/api/sign/clear": self.sign_clear, "/api/sign/question": self.sign_question,
                "/api/sign/done": self.sign_done, "/api/sign/accept": self.sign_accept,
                "/api/sign/reject": self.sign_reject, "/api/speech/choose": self.speech_choose}

    def close_all(self):
        self.close()
        if self.sign is not None:
            self.sign.close()


def _make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - name required by http.server
            url = urlparse(self.path)
            path = url.path
            if path == "/api/status":
                return self._json(app.status())
            if path == "/api/sign/state":
                if app.sign is None:
                    return self._json(dict(ok=False, message="Sign -> Text is not loaded."))
                since = parse_qs(url.query).get("since", ["0"])[0]
                return self._json(app.sign.view(int(since) if since.isdigit() else 0))
            if path == "/api/sign/words":
                return self._json(app.sign_words())
            if path == "/api/tests":
                try:
                    return self._json(app.test_sentences())
                except Exception as e:  # noqa: BLE001 - a broken CSV is shown on the page
                    return self._json(dict(ok=False, message=f"{type(e).__name__}: {e}", rows=[]))
            if path == "/video.mjpg":
                return self._stream()
            m = AUDIO_RE.match(path)
            if m and app.sign is not None:
                file = app.sign.audio_dir / m.group(1)
                if file.exists():
                    return self._send(file.read_bytes(), "audio/mpeg")
                return self.send_error(HTTPStatus.NOT_FOUND)
            if path == "/api/history":
                return self._json(app.history.all())
            if path == "/api/speech/last":
                return self._json(app.last_speech or dict(ok=False, message=""))
            if path == "/settings.json":
                return self._send(app.settings, JSON)
            file = app.files.get(path)
            if file is None:
                return self.send_error(HTTPStatus.NOT_FOUND)
            return self._send(file.read_bytes(), TYPES.get(Path(file).suffix, "application/octet-stream"))

        def do_POST(self):  # noqa: N802
            route = app.routes().get(urlparse(self.path).path)
            if route is None:
                return self.send_error(HTTPStatus.NOT_FOUND)
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > MAX_BODY:
                    return self._json(dict(ok=False, message="request too large"), HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
                started = time.perf_counter()
                out = route(body)
                out.setdefault("timings", {})["server_s"] = round(time.perf_counter() - started, 3)
                return self._json(out)
            except json.JSONDecodeError:
                return self._json(dict(ok=False, message="bad JSON"), HTTPStatus.BAD_REQUEST)
            except (LookupError, ValueError) as e:
                return self._json(dict(ok=False, message=str(e)), HTTPStatus.BAD_REQUEST)
            except Exception as e:  # noqa: BLE001 - shown on the page instead of a dropped connection
                return self._json(dict(ok=False, message=f"{type(e).__name__}: {e}"), HTTPStatus.INTERNAL_SERVER_ERROR)

        def _json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._send(body, JSON, status)

        def _stream(self):
            """The camera picture as MJPEG, one part per new picture, until the page goes away."""
            if app.sign is None or app.sign.worker is None:
                return self.send_error(HTTPStatus.NOT_FOUND)
            state = app.sign.state
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while app.sign.worker.is_alive():
                    with state.lock:
                        state.lock.wait_for(lambda: state.jpeg_id != seen, timeout=1.0)
                        jpeg, seen = state.jpeg, state.jpeg_id
                    if jpeg is None:
                        continue
                    self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                                     .encode() + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _send(self, body, kind, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *args):  # keep the console for the app's own output
            pass

    return Handler
