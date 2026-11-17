"""Non-manual markers, added at render time: head and eyebrow movement that BdSL uses for questions and negation.

    question   the text ends with "?" or has a question word (কী কে কোথায় কখন কেন কেমন কত): the head tilts slightly
               forward and the eyebrows rise over the last signs.
    negation   the text has না, নেই or নয়: the head shakes side to side over the last sign.

This is an APPROXIMATION made without a BdSL signer: the movements, their size and timing are our own estimate, not taken
from BdSL recordings. It changes a copy of the face keypoints (0-10) only; the dictionary files are never changed, and
only the figure and neon styles use it (they draw a head and eyebrows). Settings: speech2sign.nonmanual.
"""
from dataclasses import dataclass

import numpy as np

from .core_vocab import NEGATION_WORDS, QUESTION_WORDS
from .text_processing import clean_text

FACE = list(range(11))  # nose, eyes, ears, mouth
NOSE, EARS = 0, (7, 8)
STYLES_WITH_MARKERS = ("figure", "neon")


@dataclass(frozen=True)
class Markers:
    question: bool = False
    negation: bool = False


def sentence_markers(text, matches=()):
    """Markers for a sentence, from the raw text (for "?") and its matched words."""
    words = set(clean_text(text).split()) | {m.word for m in matches}
    cats = {getattr(m, "category", "") for m in matches}
    question = text.strip().endswith(("?", "？")) or bool(words & QUESTION_WORDS) or "question" in cats
    negation = bool(words & NEGATION_WORDS) or "negation" in cats
    return Markers(question, negation)


def _envelope(mask, fps, ramp_s):
    """0/1 per frame -> smooth 0..1 that fades in and out over about `ramp_s` seconds."""
    n = max(1, int(round(ramp_s * fps)))
    if not mask.any():
        return np.zeros(len(mask), np.float32)
    win = np.hanning(2 * n + 1)
    win /= win.sum()
    env = np.convolve(mask.astype(np.float32), win, mode="same")
    return np.clip(env * 1.6, 0, 1).astype(np.float32)  # flat at the top, still smooth at the edges


def _last_signs(words, n_signs, k):
    """Frames that belong to the last k signs (moving to them, signing, holding)."""
    return np.isin(words, np.arange(max(0, n_signs - k), n_signs)) if n_signs else np.zeros(len(words), bool)


def apply_markers(seq, words, n_signs, markers, fps, ncfg):
    """seq (T, 67, 3), words (T,) the sign each frame belongs to (-1 none), n_signs -> (changed copy of seq, eyebrow
    raise per frame 0..1)."""
    out = np.array(seq, np.float32, copy=True)
    brow = np.zeros(len(out), np.float32)
    have = np.abs(out[:, FACE]).sum((1, 2)) > 0
    if markers.question:
        env = _envelope(_last_signs(words, n_signs, ncfg["question_signs"]), fps, ncfg["ramp_s"]) * have
        out[:, FACE, 1] += ncfg["tilt"] * env[:, None]
        out[:, NOSE, 1] += ncfg["nose_tilt"] * env
        brow = env
    if markers.negation:
        env = _envelope(_last_signs(words, n_signs, ncfg["negation_signs"]), fps, ncfg["ramp_s"]) * have
        t = np.arange(len(out)) / fps
        shift = ncfg["shake_amplitude"] * np.sin(2 * np.pi * ncfg["shake_hz"] * t) * env
        turn = [i for i in FACE if i not in EARS]  # the ears stay: the face turns inside the head outline
        out[:, turn, 0] += shift[:, None]
    return out, brow
