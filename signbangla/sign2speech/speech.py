"""Cached edge-tts synthesis and nonblocking local audio playback."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import threading

VOICE = "bn-BD-NabanitaNeural"
ROOT = Path(__file__).resolve().parents[2]


def edge_module():
    # Optional project-local installation leaves the existing environment intact.
    local = ROOT / "work/speech_dependencies"
    if local.is_dir() and str(local) not in sys.path:
        sys.path.insert(0, str(local))
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Install speech dependencies: python -m pip install -r requirements-speech.txt") from exc
    return edge_tts


def synthesize(text, cache_dir, voice=VOICE):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{voice}\0{text}".encode()).hexdigest()
    target = cache_dir / f"{key}.mp3"
    if target.exists() and target.stat().st_size > 0:
        return target
    import tempfile
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".mp3", delete=False) as temp:
        partial = Path(temp.name)
    async def generate():
        edge = edge_module()
        for attempt in range(3):
            try:
                await edge.Communicate(text, voice).save(str(partial))
                return
            except edge.exceptions.NoAudioReceived:
                if attempt == 2:
                    raise
                await asyncio.sleep(attempt + 1)
    try:
        async def bounded_generate():
            await asyncio.wait_for(generate(), timeout=25)
        asyncio.run(bounded_generate())
        if not partial.stat().st_size:
            raise RuntimeError("Speech service returned empty audio")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


class SpeechWorker:
    def __init__(self, cache_dir, voice=VOICE):
        edge_module()
        player = shutil.which("afplay") if sys.platform == "darwin" else shutil.which("ffplay")
        if not player:
            raise RuntimeError("Audio playback needs afplay on macOS or ffplay on Windows/Linux")
        self.command = [player] if sys.platform == "darwin" else [player, "-nodisp", "-autoexit", "-loglevel", "error"]
        self.cache_dir, self.voice = cache_dir, voice
        self.status = "Ready"
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.process = None

    @property
    def busy(self):
        return self.future is not None and not self.future.done()

    def submit(self, text):
        if self.closed.is_set() or self.busy:
            return False
        self.status = "Preparing speech"
        self.future = self.pool.submit(self._speak, text)
        return True

    def _speak(self, text):
        try:
            path = synthesize(text, self.cache_dir, self.voice)
            with self.lock:
                if self.closed.is_set():
                    return
                self.status = "Speaking"
                self.process = subprocess.Popen(self.command + [str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                _, error = self.process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.communicate()
                raise RuntimeError("Audio playback timed out")
            if self.process.returncode and not self.closed.is_set():
                raise RuntimeError(error.decode(errors="replace").strip())
            self.status = "Ready"
        except Exception as exc:
            self.status = "Speech error - press c to retry"
            print(f"Speech error: {exc}", flush=True)

    def close(self):
        self.closed.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
        self.pool.shutdown(wait=True, cancel_futures=True)
