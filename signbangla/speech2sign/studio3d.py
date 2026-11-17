"""The studio3d style: a 3D mannequin that plays in a browser window.

A small local web server (standard library only, bound to 127.0.0.1) serves the viewer page in web3d/, a local copy of
three.js, the project font and the current sentence as JSON. The page asks for the sentence's version every
`poll_ms` and loads a new sentence by itself, so nothing has to be reloaded by hand.

    server = Studio3DServer.from_config(cfg, ROOT)
    server.publish(sequence_payload(sentence, layout, rest, lift_options))   # opens the browser the first time
"""
import json
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .lift3d import lift_to_3d

STYLE = "studio3d"
DECIMALS = 3  # keypoint precision sent to the page (shoulder widths)


class _LocalServer(ThreadingHTTPServer):
    """Fails on a port that is already in use (the default reuse option lets Windows share it), so the next port is
    tried instead."""
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def bgr_to_hex(color):
    """(B, G, R) -> '#rrggbb'; None stays None (the page then uses its usual word colour)."""
    if color is None:
        return None
    b, g, r = (int(c) for c in color)
    return f"#{r:02x}{g:02x}{b:02x}"


def sequence_payload(sentence, layout, rest, lift_options=None):
    """A stitched Sentence -> the JSON the page plays.

    layout  (tokens, colours, positions) as for Sentence.captions: every word of the sentence, the BGR colour of words
            without a sign (None for signed words) and the token of each sign.
    The 3D lifting works on a copy; `sentence` is not changed."""
    frames = lift_to_3d(sentence.seq, rest, **(lift_options or {}))
    captions = sentence.captions(layout)
    tokens, _, colors = captions[0] if captions else (tuple(layout[0]), -1, tuple(layout[1]))
    return {
        "fps": float(sentence.fps),
        "frames": np.round(frames.astype(np.float64), DECIMALS).tolist(),  # float64: short numbers in the JSON
        "word": [int(c[1]) for c in captions],  # token highlighted in each frame, -1 = none
        "tokens": [{"text": t, "color": bgr_to_hex(c)} for t, c in zip(tokens, colors)],
    }


def page_settings(scfg):
    """The part of speech2sign.studio3d the page needs."""
    keys = ("camera", "floor_y", "colors", "poll_ms", "caption_font_size", "face", "hand_clearance",
            "hand_clearance_frames")
    return {k: scfg[k] for k in keys}


class Studio3DServer:
    """Serves the viewer and the current sentence on 127.0.0.1 from a background thread."""

    def __init__(self, web_dir, three_js, font, settings, host="127.0.0.1", port=8765, port_tries=10,
                 open_browser=True):
        web_dir = Path(web_dir)
        # a fixed list of files: nothing else on the disk can be reached through the server
        self.routes = {
            "/": (web_dir / "index.html", "text/html; charset=utf-8"),
            "/index.html": (web_dir / "index.html", "text/html; charset=utf-8"),
            "/studio3d.js": (web_dir / "studio3d.js", "text/javascript; charset=utf-8"),
            "/vendor/three.module.js": (Path(three_js), "text/javascript; charset=utf-8"),
            "/font.ttf": (Path(font), "font/ttf"),
        }
        for path, _ in self.routes.values():
            if not path.exists():
                raise FileNotFoundError(f"studio3d viewer file missing: {path}")
        self.settings = json.dumps(settings).encode("utf-8")
        self.open_browser = open_browser
        self.version = 0
        self.fetched = 0  # last version the page has loaded
        self.sequence = b"{}"
        self.lock = threading.Lock()
        self.fetched_event = threading.Condition(self.lock)
        self.httpd = self._bind(host, port, port_tries)
        self.url = f"http://{host}:{self.httpd.server_address[1]}/"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.browser_opened = False

    @classmethod
    def from_config(cls, cfg, root):
        """From the full config; paths are relative to the project root."""
        s2s = cfg["speech2sign"]
        scfg = s2s["studio3d"]
        return cls(root / scfg["web_dir"], root / scfg["three_js"], root / s2s["font"], page_settings(scfg),
                   scfg["host"], scfg["port"], scfg["port_tries"], scfg["open_browser"])

    def _bind(self, host, port, tries):
        handler = _make_handler(self)
        last = None
        for p in range(port, port + max(1, tries)):
            try:
                return _LocalServer((host, p), handler)
            except OSError as e:  # port busy: try the next one
                last = e
        raise OSError(f"no free port in {port}-{port + max(1, tries) - 1} on {host}: {last}")

    def publish(self, payload):
        """Make `payload` the current sentence. The page picks it up by itself; the browser opens the first time."""
        with self.lock:
            self.version += 1
            payload = dict(payload, version=self.version)
            self.sequence = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if self.open_browser and not self.browser_opened:
            self.browser_opened = True
            webbrowser.open(self.url)
        return self.version

    def wait_until_loaded(self, version, timeout):
        """Wait until the page has loaded `version`. False on timeout (for example, no browser is open)."""
        with self.fetched_event:
            return self.fetched_event.wait_for(lambda: self.fetched >= version, timeout)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _make_handler(server):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - name required by http.server
            path = self.path.split("?", 1)[0]
            if path == "/version":
                with server.lock:
                    self._send(json.dumps({"version": server.version}).encode(), "application/json")
            elif path == "/sequence.json":
                with server.lock:
                    body, version = server.sequence, server.version
                self._send(body, "application/json; charset=utf-8")
                with server.fetched_event:
                    server.fetched = max(server.fetched, version)
                    server.fetched_event.notify_all()
            elif path == "/settings.json":
                self._send(server.settings, "application/json")
            elif path in server.routes:
                file, kind = server.routes[path]
                self._send(file.read_bytes(), kind)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send(self, body, kind):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the console for the demo's own output
            pass

    return Handler
