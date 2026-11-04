"""Draw a normalized sign sequence as a skeleton animation with a Bangla caption, and export it to H.264 mp4.

Bangla needs text shaping (vowel signs move, consonants join). OpenCV has none, and the Pillow wheels on Windows have
none either unless fribidi.dll is installed, so matplotlib's bundled shaper (FT2Font, matplotlib >= 3.11) turns the
text into a bitmap and Pillow composites that onto the caption bar.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib.ft2font as ft2font
import numpy as np
from PIL import Image

from ..viz import CONNECTIONS, _present

# viz.CONNECTIONS names matplotlib colors -> the same colors as BGR
_BGR = {"black": (60, 60, 60), "tab:blue": (180, 119, 31), "tab:red": (40, 39, 214)}
_SHIFT = 4  # sub-pixel bits for cv2 drawing, so slow movements do not jump between whole pixels


class CaptionDrawer:
    """Makes caption bars (BGR arrays), one per distinct text."""

    def __init__(self, font_path, width, height=110, font_size=56, color=(30, 30, 30), background=(245, 245, 245)):
        if not hasattr(ft2font, "__libraqm_version__"):
            raise RuntimeError("Bangla captions need matplotlib >= 3.11 (text shaping). Run: pip install -U matplotlib")
        if not Path(font_path).exists():
            raise FileNotFoundError(f"font not found: {font_path}")
        self.font = ft2font.FT2Font(str(font_path))
        self.font.set_size(font_size, 72)
        self.width, self.height, self.color, self.background = width, height, color, background
        self._cache = {}

    def text_mask(self, text):
        """Shaped text as a grayscale array (h, w), plus its descent in pixels."""
        self.font.set_text(text)
        self.font.draw_glyphs_to_bitmap(antialiased=True)
        return np.array(self.font.get_image(), dtype=np.uint8), self.font.get_descent() / 64

    def bar(self, text):
        if text not in self._cache:
            bar = Image.new("RGB", (self.width, self.height), self.background[::-1])
            if text:
                mask, descent = self.text_mask(text)
                x = (self.width - mask.shape[1]) // 2
                baseline = int(self.height * 0.68)  # same baseline for every caption, so the text does not hop
                bar.paste(self.color[::-1], (x, int(baseline + descent - mask.shape[0])), Image.fromarray(mask))
            self._cache[text] = np.asarray(bar)[:, :, ::-1].copy()
        return self._cache[text]


def draw_skeleton(frame, size=720, view=(-1.9, 1.9, -1.7, 2.1), background=(255, 255, 255)):
    """One frame (67, 3) -> BGR image (size, size, 3). `view` = (x min, x max, y min, y max) in shoulder widths; it stays
    fixed for the whole video, so the body does not jump. Absent parts are simply not drawn."""
    img = np.full((size, size, 3), background, np.uint8)
    x0, x1, y0, y1 = view
    unit = size / max(x1 - x0, y1 - y0)
    px = np.round((frame[:, :2] - [x0, y0]) * unit * (1 << _SHIFT)).astype(np.int32)
    ok = _present(frame)
    for a, b, color in CONNECTIONS:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(int(v) for v in px[a]), tuple(int(v) for v in px[b]), _BGR[color], 3, cv2.LINE_AA, _SHIFT)
    for p in px[ok]:
        cv2.circle(img, (int(p[0]), int(p[1])), 3 << _SHIFT, (30, 30, 30), -1, cv2.LINE_AA, _SHIFT)
    return img


def render_frames(seq, captions=None, font_path=None, size=720, view=(-1.9, 1.9, -1.7, 2.1), caption_height=110,
                  font_size=56):
    """Yield one BGR image per frame. `captions`: None, one text, or one text per frame."""
    if isinstance(captions, str):
        captions = [captions] * len(seq)
    if captions is not None and len(captions) != len(seq):
        raise ValueError(f"{len(captions)} captions for {len(seq)} frames")
    drawer = None
    if captions is not None:
        if font_path is None:
            raise ValueError("captions need font_path (speech2sign.font in configs/config.yaml)")
        drawer = CaptionDrawer(font_path, size, caption_height, font_size)
    for i, frame in enumerate(seq):
        img = draw_skeleton(frame, size, view)
        yield np.vstack([img, drawer.bar(captions[i])]) if drawer else img


def _h264_encoder(path, fps, width, height):
    """An ffmpeg process that reads raw BGR frames on stdin and writes H.264 (yuv420p, the format every player and
    browser supports; it needs an even width and height, and faststart lets browsers play while downloading)."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=tempfile.TemporaryFile())


def render_video(seq, path, fps=30, **kw):
    """Write the animation to an H.264 mp4 file. Returns the number of frames written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.mp4")  # finished files only appear under the real name
    encoder, n = None, 0
    try:
        for img in render_frames(seq, **kw):
            if encoder is None:
                encoder = _h264_encoder(tmp, fps, img.shape[1], img.shape[0])
            encoder.stdin.write(img.tobytes())
            n += 1
    finally:
        if encoder is not None:
            encoder.stdin.close()
            encoder.wait()
    if not n:
        raise ValueError("no frames to render")
    if encoder.returncode != 0:
        raise IOError(f"ffmpeg failed with exit code {encoder.returncode} while writing {tmp}")
    os.replace(tmp, path)
    return n


def render_gif(seq, path, fps=30, gif_fps=15, gif_width=480, **kw):
    """Write the animation to a GIF, at a lower frame rate and size because GIFs get large. One palette for the whole
    file stops colours flickering between frames. Returns the number of frames written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for i, img in enumerate(render_frames(seq, **kw))
              if int(i * gif_fps / fps) != int((i - 1) * gif_fps / fps) or i == 0]
    if not frames:
        raise ValueError("no frames to render")
    h, w = frames[0].shape[:2]
    size = (gif_width, int(round(h * gif_width / w)))
    images = [Image.fromarray(f).resize(size, Image.Resampling.LANCZOS) for f in frames]
    palette = Image.fromarray(np.hstack([np.asarray(im) for im in images[::max(1, len(images) // 16)]])).quantize(64)
    paletted = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    # GIF timing is in hundredths of a second. Rounding the running total keeps the length right for any gif_fps.
    ends = [int(round((i + 1) * 100 / gif_fps)) * 10 for i in range(len(paletted))]
    durations = [b - a for a, b in zip([0, *ends], ends)]
    tmp = path.with_name(path.stem + ".tmp.gif")
    paletted[0].save(tmp, save_all=True, append_images=paletted[1:], duration=durations, loop=0, optimize=False)
    os.replace(tmp, path)
    return len(paletted)


def save_animation(seq, path, fps=30, gif_fps=15, gif_width=480, **kw):
    """Write .gif or .mp4, chosen by the file extension. Returns the number of frames written."""
    suffix = Path(path).suffix.lower()
    if suffix == ".gif":
        return render_gif(seq, path, fps=fps, gif_fps=gif_fps, gif_width=gif_width, **kw)
    if suffix == ".mp4":
        return render_video(seq, path, fps=fps, **kw)
    raise ValueError(f"cannot save {path}: use .mp4 or .gif")
