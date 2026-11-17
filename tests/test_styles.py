"""Tests for the three drawing styles (render.py) and the sentence motion (stitch.sentence_motion), on synthetic keypoints.

    python tests/test_styles.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from signbangla.keypoints import GROUPS, N_KP
from signbangla.speech2sign.render import (FIGURE, NEON, STYLES, _geometry, _Layout, caption_drawer, draw_frame,
                                           render_frames, save_animation, write_video)
from signbangla.speech2sign.stitch import SENTENCE_KINDS, minimum_jerk, sentence_motion
from signbangla.utils import ROOT, load_config

CFG = load_config()
S2S = CFG["speech2sign"]
FONT = ROOT / S2S["font"]


def body(right_wrist=(-0.6, 1.4), left_wrist=(0.6, 1.4), right_hand=True, left_hand=True):
    """One normalized frame: shoulders 1 apart at y=0, hips, a face, both arms and hands (hands point down)."""
    f = np.zeros((N_KP, 3), np.float32)
    f[11], f[12] = [0.5, 0, 0], [-0.5, 0, 0]
    f[23], f[24] = [0.35, 1.3, 0], [-0.35, 1.3, 0]
    for i, p in enumerate([[0, -0.5], [0.05, -0.6], [0.08, -0.6], [0.1, -0.6], [-0.05, -0.6], [-0.08, -0.6],
                           [-0.1, -0.6], [0.18, -0.55], [-0.18, -0.55], [0.05, -0.42], [-0.05, -0.42]]):
        f[i, :2] = p
    for group, (s, e, w), wrist, on in (("lh", (11, 13, 15), left_wrist, left_hand), ("rh", (12, 14, 16), right_wrist, right_hand)):
        f[e] = (f[s] + np.array([*wrist, 0], np.float32)) / 2 + [0, 0.2, 0]
        f[w] = [*wrist, 0]
        if on:
            hand = np.zeros((21, 3), np.float32)
            hand[0] = f[w]
            for k, chain in enumerate([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]):
                for j, idx in enumerate(chain):
                    hand[idx] = f[w] + [-0.12 + 0.06 * k, 0.05 + 0.05 * (j + 1), 0.01 * k]
            f[GROUPS[group]] = hand
    return f


class FakeDictionary:
    """What sentence_motion uses from SignDictionary."""

    def __init__(self):
        self.fps = 30
        self.rest_pose = body()
        t = np.linspace(0, 1, 20)[:, None]
        up = np.stack([body(right_wrist=(-0.6 + 0.3 * float(s), 1.4 - 1.2 * float(s))) for s in t[:, 0]])
        up[5:8, GROUPS["rh"]] = 0  # right hand missing for three frames
        self.words = {"W1": up, "W2": np.repeat(body(right_wrist=(-0.2, 0.0))[None], 12, axis=0)}

    def sequence(self, name, variant=0):
        return self.words[name].copy()


def test_three_styles_only():
    assert STYLES == ("skeleton", "figure", "neon")
    assert S2S["render"]["style"] == "figure"
    try:
        draw_frame(body(), "avatar")
        raise AssertionError("an unknown style must raise")
    except ValueError:
        pass


def test_every_style_draws_a_square_picture():
    for style in STYLES:
        img = draw_frame(body(), style, 640)
        assert img.shape == (640, 640, 3) and img.dtype == np.uint8 and img.std() > 5, style
        empty = draw_frame(np.zeros((N_KP, 3), np.float32), style, 160)
        assert empty.shape == (160, 160, 3)


def test_figure_and_neon_colours():
    fig = draw_frame(body(), "figure", 320)
    assert tuple(int(v) for v in fig[2, 2]) == FIGURE["bg"][::-1]  # light background (BGR)
    neon = draw_frame(body(), "neon", 320)
    assert neon.mean() < 60  # dark
    orange = np.array(FIGURE["right"][0][::-1])
    assert (np.abs(fig.astype(int) - orange).sum(-1) < 30).sum() > 50  # the right hand is orange
    blue = np.array(FIGURE["left"][0][::-1])
    assert (np.abs(fig.astype(int) - blue).sum(-1) < 30).sum() > 50  # the left hand is blue
    assert NEON["right"] != NEON["left"]


def test_forearm_ends_at_the_hand_wrist_and_active_arm_is_last():
    f = body(right_wrist=(-0.3, 0.2))  # right hand raised
    f[16] = [-0.35, 0.4, 0]  # pose wrist away from the hand's own wrist point
    lay = _Layout(640)
    g = _geometry(lay, f)
    assert np.allclose(g["rw"], lay.p(f[GROUPS["rh"]][0]))
    assert [a[4] for a in g["arms"]] == ["L", "R"]  # the raised right arm is drawn last, in front


def test_frame_size_is_divisible_by_16():
    r = S2S["render"]
    for style in STYLES:
        img = next(render_frames([body()], captions="মা", font_path=FONT, size=r["size"], caption_height=r["caption_height"],
                                 font_size=r["font_size"], style=style))
        assert img.shape[0] % 16 == 0 and img.shape[1] % 16 == 0, (style, img.shape)


def test_caption_colours_follow_the_style():
    light = caption_drawer(FONT, "figure", 320, 64, 30).bar((("মা", "চা"), 0))
    dark = caption_drawer(FONT, "neon", 320, 64, 30).bar((("মা", "চা"), 0))
    assert light.mean() > 180 and dark.mean() < 120


def test_minimum_jerk():
    a, b = np.zeros((67, 3)), np.ones((67, 3))
    m = minimum_jerk(a, b, 16)
    assert m.shape == (16, 67, 3) and np.allclose(m[0], a) and np.allclose(m[-1], b)
    steps = np.diff(m[:, 0, 0])
    assert steps[0] < steps[7] and steps[-1] < steps[7]  # starts and ends slowly


def test_sentence_motion_structure():
    d, sc = FakeDictionary(), S2S["sentence"]
    sen = sentence_motion(["W1", "W2"], d, **sc)
    assert sen.fps == 60 and len(sen.seq) == len(sen.kinds) == len(sen.words)
    kinds = [SENTENCE_KINDS[k] for k in sen.kinds]
    assert kinds[0] == kinds[-1] == "rest"
    up = sc["upsample"]
    source = (sc["rest_start_frames"] + 2 * (sc["transition_frames"] - 1) + 20 + 12 + 2 * sc["hold_frames"]
              + (sc["transition_frames"] - 1) + sc["rest_end_frames"])
    assert len(sen.seq) == source * up - (up - 1)
    assert set(sen.words[np.array(kinds) == "sign"]) == {0, 1}
    assert abs(kinds.count("hold") - 2 * sc["hold_frames"] * up) <= 4
    assert np.isfinite(sen.seq).all()


def test_missing_hands_are_filled_and_holds_stay_still():
    d = FakeDictionary()
    sen = sentence_motion(["W1"], d, **S2S["sentence"])
    sign = sen.seq[sen.kinds == SENTENCE_KINDS.index("sign")]
    assert (np.abs(sign[:, GROUPS["rh"]]).sum((1, 2)) > 0).all()  # no frame without a right hand
    hold = sen.seq[sen.kinds == SENTENCE_KINDS.index("hold")][2:-2]
    assert np.abs(np.diff(hold, axis=0)).max() < 0.02


def test_captions_and_speed():
    sen = sentence_motion(["W1", "W2"], FakeDictionary(), **S2S["sentence"])
    caps = sen.captions()
    assert all(c[0] == ("W1", "W2") for c in caps) and {c[1] for c in caps} == {-1, 0, 1}
    slow = sen.retimed(0.5)
    assert abs(len(slow.seq) - 2 * len(sen.seq)) <= 2 and len(slow.captions()) == len(slow.seq)
    assert sen.retimed(1.0) is sen


def test_export_in_every_style():
    seq = np.repeat(body()[None], 6, axis=0)
    with tempfile.TemporaryDirectory() as tmp:
        for style in STYLES:
            for name in ("a.mp4", "a.gif"):
                n = save_animation(seq, Path(tmp) / f"{style}_{name}", fps=60, gif_fps=30, gif_width=160, captions="মা",
                                   font_path=FONT, size=160, caption_height=48, font_size=24, style=style)
                assert n > 0
        assert write_video(iter([np.zeros((64, 64, 3), np.uint8)] * 3), Path(tmp) / "b.mp4", fps=30) == 3


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"{t.__name__} ok")
    print("\nALL TESTS PASSED")
