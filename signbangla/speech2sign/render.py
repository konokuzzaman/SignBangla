"""Draw a normalized sign sequence as an animation with a Bangla caption, and export it to H.264 mp4 or GIF.

Three styles draw the same 67 keypoints: "figure" (default: a simple figure with natural hands), "neon" (glowing lines
on a dark background) and "skeleton" (plain lines and dots). Smooth 60 fps motion comes from stitch.sentence_motion;
this module only draws frames. Frames are 640 wide and 640 + caption tall (736 with the default caption): both
divisible by 16, which video encoders like.

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
from PIL import Image, ImageDraw, ImageFilter

from ..viz import CONNECTIONS, _present

# viz.CONNECTIONS names matplotlib colors -> the same colors as BGR
_BGR = {"black": (60, 60, 60), "tab:blue": (180, 119, 31), "tab:red": (40, 39, 214)}
_SHIFT = 4  # sub-pixel bits for cv2 drawing, so slow movements do not jump between whole pixels


def _lighten_bgr(c, f):
    return tuple(int(x + (255 - x) * f) for x in c)


class CaptionDrawer:
    """Makes caption bars (BGR arrays), one per distinct text."""

    def __init__(self, font_path, width, height=110, font_size=56, color=(30, 30, 30), background=(245, 245, 245),
                 highlight_text=None, highlight_back=None, other_text=None):
        if not hasattr(ft2font, "__libraqm_version__"):
            raise RuntimeError("Bangla captions need matplotlib >= 3.11 (text shaping). Run: pip install -U matplotlib")
        if not Path(font_path).exists():
            raise FileNotFoundError(f"font not found: {font_path}")
        self.font = ft2font.FT2Font(str(font_path))
        self.font.set_size(font_size, 72)
        self.width, self.height, self.color, self.background = width, height, color, background
        self.highlight_text = highlight_text or HIGHLIGHT_TEXT
        self.highlight_back = highlight_back or HIGHLIGHT_BACK
        self.other_text = other_text or OTHER_TEXT
        self.lighten = 0.0  # word colours are lightened this much on dark caption bars
        self._cache = {}

    def text_mask(self, text):
        """Shaped text as a grayscale array (h, w), plus its descent in pixels."""
        self.font.set_text(text)
        self.font.draw_glyphs_to_bitmap(antialiased=True)
        return np.array(self.font.get_image(), dtype=np.uint8), self.font.get_descent() / 64

    def bar(self, text):
        """A caption bar for a text, or for (words, current index) or (words, current index, colours): the sentence with
        the current word highlighted. A colour (BGR) per word, or None for the usual one, marks words without a sign."""
        if isinstance(text, tuple):
            return self._sentence_bar(*text)
        if text not in self._cache:
            bar = Image.new("RGB", (self.width, self.height), self.background[::-1])
            if text:
                mask, descent = self.text_mask(text)
                x = (self.width - mask.shape[1]) // 2
                baseline = int(self.height * 0.68)  # same baseline for every caption, so the text does not hop
                bar.paste(self.color[::-1], (x, int(baseline + descent - mask.shape[0])), Image.fromarray(mask))
            self._cache[text] = np.asarray(bar)[:, :, ::-1].copy()
        return self._cache[text]

    def _sentence_bar(self, words, current, colors=None):
        key = (tuple(words), int(current), tuple(colors) if colors else None)
        if key not in self._cache:
            bar = Image.new("RGB", (self.width, self.height), self.background[::-1])
            masks = [self.text_mask(w) for w in words]
            gap = int(self.height * 0.22)  # space between words
            total = sum(m.shape[1] for m, _ in masks) + gap * max(0, len(masks) - 1)
            scale = min(1.0, (self.width - 20) / total) if total else 1.0
            x = (self.width - total * scale) / 2
            baseline = int(self.height * 0.68)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(bar)
            for k, (mask, descent) in enumerate(masks):
                img = Image.fromarray(mask)
                if scale < 1:
                    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
                top = int(baseline + descent * scale - img.height)
                if k == current:
                    pad = int(self.height * 0.08)
                    draw.rounded_rectangle([x - pad, self.height * 0.1, x + img.width + pad, self.height * 0.9],
                                           radius=int(self.height * 0.15), fill=self.highlight_back[::-1])
                color = self.highlight_text if k == current else self.other_text
                if k != current and colors and colors[k] is not None:
                    color = _lighten_bgr(colors[k], self.lighten)
                bar.paste(color[::-1], (int(x), top), img)
                x += img.width + gap * scale
            self._cache[key] = np.asarray(bar)[:, :, ::-1].copy()
        return self._cache[key]


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


# ---------------------------------------------------------------- figure and neon styles
# Ported from the style experiments. Drawn with Pillow at SUPERSAMPLE x size, then reduced, so edges are smooth.
# Both place the body the same way: the shoulder midpoint at (size/2, 0.40 x size), 205 px per shoulder width at 640.
SUPERSAMPLE = 3
FIGURE = dict(bg=(240, 243, 246), bone=(55, 64, 78), joint=(255, 255, 255), torso=(226, 230, 235), torso_line=(160, 166, 175),
              head=(226, 230, 235), head_line=(150, 156, 166), face=(150, 156, 166), shadow=(214, 220, 228),
              right=((250, 140, 40), (196, 96, 20)), left=((40, 128, 240), (22, 84, 180)))  # RGB; right hand orange
NEON = dict(bg=(10, 14, 30), body=(70, 220, 255), spine=(40, 120, 170), joint=(255, 255, 255), grid=(18, 24, 48),
            right=(255, 170, 60), left=(255, 70, 190))  # RGB; right hand orange, left magenta
FINGER_CHAINS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]


class _Layout:
    def __init__(self, size, supersample=SUPERSAMPLE):
        self.size, self.s = size, supersample
        self.cx, self.cy, self.sc = size / 2, size * 0.40, 205 * size / 640

    def p(self, q):
        return np.array([(self.cx + q[0] * self.sc) * self.s, (self.cy + q[1] * self.sc) * self.s])

    def px(self, n):
        """n pixels at 640, in supersampled pixels at this size."""
        return n * self.s * self.size / 640


def _taper(draw, a, b, ra, rb, fill):
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = b - a
    length = np.hypot(*v)
    if length < 1e-6:
        draw.ellipse([a[0] - ra, a[1] - ra, a[0] + ra, a[1] + ra], fill=fill)
        return
    n = np.array([-v[1], v[0]]) / length
    draw.polygon([tuple(a + n * ra), tuple(b + n * rb), tuple(b - n * rb), tuple(a - n * ra)], fill=fill)
    for c, r in ((a, ra), (b, rb)):
        draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=fill)


def _cap(draw, a, b, r, fill):
    draw.line([tuple(a), tuple(b)], fill=fill, width=int(2 * r))
    for c in (a, b):
        draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=fill)


def _chaikin(pts, n=3):
    pts = np.asarray(pts, float)
    for _ in range(n):
        q = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            q += [a * 0.75 + b * 0.25, a * 0.25 + b * 0.75]
        pts = np.array(q)
    return [tuple(p) for p in pts]


def _lighten(c, f):
    return tuple(int(x + (255 - x) * f) for x in c)


def _natural_hand(draw, lay, hand, col, dark, highlight=True):
    """A hand from its 21 points: smoothed palm, tapered fingers with rounded tips drawn back to front (z), a soft
    highlight along each finger and knuckle creases."""
    from scipy.spatial import ConvexHull
    q = [lay.p(p) for p in hand]
    z = hand[:, 2]
    s = lay.px(1)
    pw = max(np.linalg.norm(q[5] - q[17]), np.linalg.norm(q[9] - q[0]) * 0.55, 14 * s)  # palm width
    ol = 1.3 * s  # outline
    hi, crease = _lighten(col, 0.35), tuple(int(x * 0.8) for x in col)
    hl = np.linalg.norm(q[9] - q[0])
    pr = max(pw * 0.5, hl * 0.26) * 0.92
    pts = np.array([q[0], q[1], q[2] * 0.35 + q[1] * 0.65, q[5], q[9], q[13], q[17]])
    try:
        palm = _chaikin(pts[ConvexHull(pts).vertices], 3)
    except Exception:  # points in a line (hand seen edge-on): the tapered wrist-to-knuckle shape is enough
        palm = None
    _taper(draw, q[0], q[9] * 0.8 + q[0] * 0.2, pr * 0.75 + ol, pr + ol, dark)
    if palm:
        pc = np.mean(np.array(palm), axis=0)
        draw.polygon([tuple(pc + (np.array(p) - pc) * (1 + ol / (np.linalg.norm(np.array(p) - pc) + 1e-6))) for p in palm],
                     fill=dark)
    _taper(draw, q[0], q[9] * 0.8 + q[0] * 0.2, pr * 0.75, pr, col)
    if palm:
        draw.polygon(palm, fill=col)
    for f in sorted(FINGER_CHAINS, key=lambda f: -np.mean(z[list(f)])):
        if f[0] == 1:
            r = [pw * 0.15, pw * 0.13, pw * 0.115, pw * 0.10]
        elif f[0] == 17:
            r = [pw * 0.095, pw * 0.085, pw * 0.075, pw * 0.068]
        else:
            r = [pw * 0.115, pw * 0.10, pw * 0.09, pw * 0.08]
        chain = [q[k] for k in f]
        v = chain[3] - chain[2]
        chain[3] = chain[3] - v / (np.linalg.norm(v) + 1e-6) * r[3] * 0.3  # the rounded end reaches the real fingertip
        for i in range(3):
            _taper(draw, chain[i], chain[i + 1], r[i] + ol, r[i + 1] + ol, dark)
        for i in range(3):
            _taper(draw, chain[i], chain[i + 1], r[i], r[i + 1], col)
        if highlight:
            for i in range(3):
                a, b = chain[i], chain[i + 1]
                v = b - a
                n = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-6)
                _taper(draw, a + n * r[i] * 0.35, b + n * r[i + 1] * 0.35, r[i] * 0.28, r[i + 1] * 0.28, hi)
            for i in (1, 2):
                p = chain[i]
                v = chain[i + 1] - chain[i - 1]
                n = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-6)
                draw.line([tuple(p + n * r[i] * 0.6), tuple(p - n * r[i] * 0.6)], fill=crease, width=max(1, int(1.1 * s)))


def _geometry(lay, p):
    """Positions shared by the figure and neon styles. The forearm always ends at the HAND's wrist (point 0 of the
    hand), not at the pose wrist, so arm and hand never come apart. The raised (active) arm comes last."""
    g = {k: lay.p(p[i]) for k, i in dict(ls=11, rs=12, le=13, re=14, nose=0, eL=2, eR=5, earL=7, earR=8, mL=9, mR=10).items()}
    g["lh"], g["rh"] = p[25:46], p[46:67]
    g["lw"], g["rw"] = lay.p(g["lh"][0]), lay.p(g["rh"][0])
    g["mid"] = (g["ls"] + g["rs"]) / 2
    g["sw"] = abs(g["ls"][0] - g["rs"][0])
    hw = max(abs(g["earL"][0] - g["earR"][0]) * 1.05, g["sw"] * 0.42)
    g["hw"], g["hh"] = hw, hw * 1.3
    g["eyey"] = (g["eL"][1] + g["eR"][1]) / 2
    g["mouthy"] = (g["mL"][1] + g["mR"][1]) / 2
    g["hcx"] = (g["earL"][0] + g["earR"][0]) / 2
    g["hcy"] = (g["eyey"] + g["mouthy"]) / 2 - g["hh"] * 0.06
    arms = [(g["ls"], g["le"], g["lw"], g["lh"], "L"), (g["rs"], g["re"], g["rw"], g["rh"], "R")]
    arms.sort(key=lambda a: -a[2][1])  # higher wrist (smaller y) drawn last, in front
    g["arms"] = arms
    return g


def _reduce(layer, size):
    """A supersampled RGBA layer -> (colour multiplied by coverage, coverage) at `size`, float32. A box filter is
    plenty for averaging supersampled pixels, and much faster than Pillow's high-quality filters."""
    arr = np.asarray(layer.resize((size, size), Image.BOX), np.float32)
    cover = arr[:, :, 3:] / 255.0
    return arr[:, :, :3] * cover, cover


def _soft_blur(img, sigma):
    """Gaussian blur done at a quarter of the size: the same soft result for shadows and glow, much faster."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 4), (w, h), interpolation=cv2.INTER_LINEAR)


def _to_bgr(rgb):
    return np.clip(rgb, 0, 255).astype(np.uint8)[:, :, ::-1].copy()


def _brows(d, g, s, raise_, color, width):
    """Eyebrow marks above both eyes, raised by `raise_` (0..1). Drawn only for non-manual markers (questions)."""
    for e in (g["eL"], g["eR"]):
        y = e[1] - (13 + 9 * raise_) * s
        d.arc([e[0] - 10 * s, y - 5 * s, e[0] + 10 * s, y + 7 * s], 200, 340, fill=color, width=int(width * s))


def draw_figure(frame, size=640, supersample=SUPERSAMPLE, brow=0.0):
    """The default style: grey oval head placed from the face keypoints with faint eye, nose and mouth marks, a
    trapezoid torso, dark arm bones with white joints, natural orange (right) and blue (left) hands, and a soft shadow.
    BGR (size, size, 3)."""
    lay, c = _Layout(size, supersample), FIGURE
    s = lay.px(1)
    w = size * supersample
    background = np.full((size, size, 3), c["bg"], np.float32)
    if np.abs(frame[[11, 12]]).sum() == 0:
        return _to_bgr(background)
    g = _geometry(lay, frame)
    ls, rs, mid, sw = g["ls"], g["rs"], g["mid"], g["sw"]
    hb = [g["hcx"] - g["hw"] / 2, g["hcy"] - g["hh"] / 2, g["hcx"] + g["hw"] / 2, g["hcy"] + g["hh"] / 2]
    torso = [(rs[0] + 3 * s, rs[1]), (ls[0] - 3 * s, ls[1]), (ls[0] - sw * 0.13, mid[1] + sw * 1.25),
             (rs[0] + sw * 0.13, mid[1] + sw * 1.25)]
    # soft shadow of the body, offset down-right, on the background
    shadow = Image.new("L", (w, w), 0)
    sd = ImageDraw.Draw(shadow)
    sd.polygon(torso, fill=90)
    sd.ellipse(hb, fill=90)
    for a, b in ((ls, g["le"]), (g["le"], g["lw"]), (rs, g["re"]), (g["re"], g["rw"])):
        sd.line([tuple(a), tuple(b)], fill=90, width=int(30 * s))
    unit = size / 640
    alpha = np.asarray(shadow.resize((size, size), Image.BOX), np.float32) / 255
    alpha = _soft_blur(alpha, 14 * unit)
    alpha = cv2.warpAffine(alpha, np.float32([[1, 0, 6 * unit], [0, 1, 8 * unit]]), (size, size))
    background += (np.array(c["shadow"], np.float32) - background) * alpha[:, :, None]
    # the figure itself, on a transparent layer
    layer = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.polygon(torso, fill=c["torso_line"])
    tc = np.mean(np.array(torso), axis=0)
    d.polygon([tuple(tc + (np.array(q) - tc) * 0.985) for q in torso], fill=c["torso"])
    _cap(d, mid, np.array([g["hcx"], hb[3] - 2 * s]), 7.5 * s, c["bone"])
    d.ellipse([hb[0] - 2.4 * s, hb[1] - 2.4 * s, hb[2] + 2.4 * s, hb[3] + 2.4 * s], fill=c["head_line"])
    d.ellipse(hb, fill=c["head"])
    for e in (g["eL"], g["eR"]):  # faint face marks, so contact with the face is readable
        d.ellipse([e[0] - 5 * s, e[1] - 5 * s, e[0] + 5 * s, e[1] + 5 * s], fill=c["face"])
    if brow > 0.02:  # fades in from the head colour
        _brows(d, g, s, brow, tuple(int(h + (f - h) * min(1.0, brow * 2)) for h, f in zip(c["head"], c["face"])), 3)
    nose = g["nose"]
    d.line([tuple(nose + np.array([0, -6 * s])), tuple(nose + np.array([0, 5 * s]))], fill=c["face"], width=int(3 * s))
    d.line([(g["mL"][0] - 2 * s, g["mouthy"]), (g["mR"][0] + 2 * s, g["mouthy"])], fill=c["face"], width=int(3 * s))
    _cap(d, ls, rs, 7.5 * s, c["bone"])
    for sh, el, wr, hand, side in g["arms"]:
        _cap(d, sh, el, 8.5 * s, c["bone"])
        _cap(d, el, wr, 8 * s, c["bone"])
        for j in (sh, el):
            d.ellipse([j[0] - 9.5 * s, j[1] - 9.5 * s, j[0] + 9.5 * s, j[1] + 9.5 * s], fill=c["bone"])
            d.ellipse([j[0] - 5.5 * s, j[1] - 5.5 * s, j[0] + 5.5 * s, j[1] + 5.5 * s], fill=c["joint"])
        if np.abs(hand).sum() > 0:
            col, dark = c["right"] if side == "R" else c["left"]
            _natural_hand(d, lay, hand, col, dark)
    colour, cover = _reduce(layer, size)
    return _to_bgr(colour + background * (1 - cover))


def draw_neon(frame, size=640, supersample=SUPERSAMPLE, brow=0.0):
    """Motion-capture look: glowing cyan body on a dark background with faint grid lines, orange (right) and magenta
    (left) hand bones with white joints, head placed from the face keypoints. BGR (size, size, 3)."""
    lay, c = _Layout(size, supersample), NEON
    s = lay.px(1)
    w = size * supersample
    unit = size / 640
    out = np.full((size, size, 3), c["bg"], np.float32)
    if np.abs(frame[[11, 12]]).sum() > 0:
        g = _geometry(lay, frame)
        layer = Image.new("RGBA", (w, w), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        hb = [g["hcx"] - g["hw"] / 2, g["hcy"] - g["hh"] / 2, g["hcx"] + g["hw"] / 2, g["hcy"] + g["hh"] / 2]
        d.ellipse(hb, outline=c["body"], width=int(4 * s))
        for e in (g["eL"], g["eR"]):
            d.ellipse([e[0] - 4 * s, e[1] - 4 * s, e[0] + 4 * s, e[1] + 4 * s], fill=c["body"])
        if brow > 0.02:
            _brows(d, g, s, brow, tuple(int(v * min(1.0, brow * 2)) for v in c["body"]), 3)
        mid = g["mid"]
        d.line([tuple(mid), (g["hcx"], hb[3])], fill=c["body"], width=int(5 * s))
        d.line([tuple(g["ls"]), tuple(g["rs"])], fill=c["body"], width=int(5 * s))
        d.line([tuple(mid), (mid[0], mid[1] + g["sw"] * 1.2)], fill=c["spine"], width=int(4 * s))
        for sh, el, wr, hand, side in g["arms"]:
            d.line([tuple(sh), tuple(el), tuple(wr)], fill=c["body"], width=int(6 * s), joint="curve")
            for j in (sh, el):
                d.ellipse([j[0] - 7 * s, j[1] - 7 * s, j[0] + 7 * s, j[1] + 7 * s], fill=c["joint"])
            if np.abs(hand).sum() > 0:
                col = c["right"] if side == "R" else c["left"]
                q = [lay.p(pt) for pt in hand]
                for f in FINGER_CHAINS:
                    d.line([tuple(q[0])] + [tuple(q[k]) for k in f], fill=col, width=int(4 * s), joint="curve")
                for k in range(21):
                    d.ellipse([q[k][0] - 3 * s, q[k][1] - 3 * s, q[k][0] + 3 * s, q[k][1] + 3 * s], fill=c["joint"])
        colour, cover = _reduce(layer, size)
        glow = _soft_blur(colour, 9 * unit)  # the lines on black, blurred: their glow
        out = colour + out * (1 - cover) + glow * 1.4
    out = _to_bgr(out)
    grid = c["grid"][::-1]
    for y in range(0, size, max(1, int(round(40 * unit)))):
        cv2.line(out, (0, y), (size, y), grid, 1)
    return out


STYLES = ("skeleton", "figure", "neon")
# caption bar colours per style (BGR): text, background, highlighted word text, highlight background, other words
CAPTION_COLORS = {
    "skeleton": ((30, 30, 30), (245, 245, 245), (40, 70, 160), (247, 232, 214), (90, 90, 90)),
    "figure": ((40, 32, 28), (251, 249, 248), (40, 70, 160), (247, 232, 214), (90, 90, 90)),
    "neon": ((255, 240, 230), (40, 24, 20), (60, 170, 255), (90, 60, 40), (200, 190, 180)),
}


def draw_frame(frame, style="figure", size=640, view=(-1.9, 1.9, -1.7, 2.1), supersample=SUPERSAMPLE, brow=0.0):
    """One frame (67, 3) in one style, without a caption: BGR (size, size, 3). A lower `supersample` draws faster with
    slightly rougher edges (the live window uses 2; saved videos use 3). `brow` (0..1) raises eyebrow marks, figure and
    neon only."""
    if style == "skeleton":
        return draw_skeleton(frame, size, view)
    if style == "figure":
        return draw_figure(frame, size, supersample, brow)
    if style == "neon":
        return draw_neon(frame, size, supersample, brow)
    raise ValueError(f"unknown style {style!r}; choose from {list(STYLES)}")


HIGHLIGHT_TEXT, HIGHLIGHT_BACK, OTHER_TEXT = (40, 70, 160), (247, 232, 214), (90, 90, 90)  # BGR


def render_options(render_cfg):
    """The drawing settings of speech2sign.render, as keyword arguments for render_frames (no fps or speed)."""
    keys = ("size", "view", "caption_height", "font_size", "style")
    return {k: render_cfg[k] for k in keys if k in render_cfg}


def caption_drawer(font_path, style="figure", size=640, caption_height=96, font_size=50):
    """A CaptionDrawer with the colours of a style."""
    text, back, hl_text, hl_back, other = CAPTION_COLORS[style]
    drawer = CaptionDrawer(font_path, size, caption_height, font_size, text, back, hl_text, hl_back, other)
    drawer.lighten = 0.45 if sum(back) < 300 else 0.0
    return drawer


def render_frames(seq, captions=None, font_path=None, size=640, view=(-1.9, 1.9, -1.7, 2.1), caption_height=96,
                  font_size=50, style="figure", brows=None):
    """Yield one BGR image per frame.

    captions  None, one text, one text per frame, or one (words, current word index) per frame: the whole sentence is
              shown and the current word highlighted (index -1 = none)
    style     "figure", "neon" or "skeleton"
    brows     None, or one eyebrow raise (0..1) per frame (nonmanual.py; figure and neon only)
    """
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {list(STYLES)}")
    if isinstance(captions, str):
        captions = [captions] * len(seq)
    if captions is not None and len(captions) != len(seq):
        raise ValueError(f"{len(captions)} captions for {len(seq)} frames")
    drawer = None
    if captions is not None:
        if font_path is None:
            raise ValueError("captions need font_path (speech2sign.font in configs/config.yaml)")
        drawer = caption_drawer(font_path, style, size, caption_height, font_size)
    for i, frame in enumerate(seq):
        img = draw_frame(frame, style, size, view, brow=float(brows[i]) if brows is not None else 0.0)
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
    return write_video(render_frames(seq, **kw), path, fps)


def write_video(images, path, fps=30):
    """Write BGR images (all the same size) to an H.264 mp4 file. Returns the number of frames written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.mp4")  # finished files only appear under the real name
    encoder, n = None, 0
    try:
        for img in images:
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


def render_gif(seq, path, fps=30, gif_fps=30, gif_width=640, **kw):
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


def save_animation(seq, path, fps=30, gif_fps=30, gif_width=640, **kw):
    """Write .gif or .mp4, chosen by the file extension. Returns the number of frames written."""
    suffix = Path(path).suffix.lower()
    if suffix == ".gif":
        return render_gif(seq, path, fps=fps, gif_fps=gif_fps, gif_width=gif_width, **kw)
    if suffix == ".mp4":
        return render_video(seq, path, fps=fps, **kw)
    raise ValueError(f"cannot save {path}: use .mp4 or .gif")
