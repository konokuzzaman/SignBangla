"""Tests for Module 2 motion (signbangla/speech2sign/motion.py) and sentence captions, on synthetic keypoints.

    python tests/test_motion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from signbangla.keypoints import GROUPS, N_KP
from signbangla.speech2sign.motion import (ARM_BONES, HAND_BONES, KINDS, build_motion, fit_lengths, from_state,
                                           motion_metrics, remove_glitches, slerp, smooth, to_state, transition_length)
from signbangla.speech2sign.render import HIGHLIGHT_BACK, CaptionDrawer
from signbangla.utils import ROOT, load_config

CFG = load_config()
S2S = CFG["speech2sign"]
FONT = ROOT / S2S["font"]


def body(T=30, wave=0.0, seed=0):
    """Normalized frames (T, 67, 3): shoulders 1 apart, hips, face, arms, both hands. `wave` moves the right forearm."""
    rng = np.random.default_rng(seed)
    f = np.zeros((T, N_KP, 3), np.float32)
    t = np.linspace(0, 1, T)
    f[:, 11], f[:, 12] = [0.5, 0, 0], [-0.5, 0, 0]
    f[:, 23], f[:, 24] = [0.35, 1.3, 0.1], [-0.35, 1.3, 0.1]
    for i, p in enumerate([[0, -0.5], [0.05, -0.6], [0.08, -0.6], [0.1, -0.6], [-0.05, -0.6], [-0.08, -0.6],
                           [-0.1, -0.6], [0.18, -0.55], [-0.18, -0.55], [0.05, -0.42], [-0.05, -0.42]]):
        f[:, i, :2] = p
    f[:, 13], f[:, 15] = [0.7, 0.8, 0], [0.65, 1.5, 0]
    f[:, 14] = [-0.7, 0.8, 0]
    f[:, 16] = np.stack([-0.7 + 0.3 * np.sin(np.pi * t) * wave, 1.5 - 0.5 * np.sin(np.pi * t) * wave, 0 * t], 1)
    for k, (w, pinky, index, thumb) in enumerate(((15, 17, 19, 21), (16, 18, 20, 22))):
        f[:, pinky] = f[:, w] + [0.05, 0.08, 0]
        f[:, index] = f[:, w] + [-0.05, 0.08, 0]
        f[:, thumb] = f[:, w] + [-0.06, 0.03, 0]
        hand = np.zeros((T, 21, 3), np.float32)
        hand[:, 0] = f[:, w]
        for a, b in HAND_BONES:
            finger = (b - 1) // 4
            hand[:, b] = hand[:, a] + [-0.04 + 0.02 * finger, 0.04, 0.0]
        f[:, GROUPS[("lh", "rh")[k]]] = hand
    return f + rng.normal(0, 0.002, f.shape).astype(np.float32) * (np.abs(f) > 0)


class FakeDictionary:
    """The parts of SignDictionary that build_motion uses."""

    def __init__(self):
        self.fps = 30
        self.folder = Path("fake")
        self.words = {"W1": {"trials": [{}]}, "W2": {"trials": [{}]}}
        self.rest_pose = body(1)[0]
        self._seqs = {"W1": body(30, wave=1.0, seed=1), "W2": body(24, wave=0.6, seed=2)}

    def sequence(self, name, variant=0):
        return self._seqs[name].copy()


def test_spike_is_removed_and_real_change_is_kept():
    seq = body(30)
    spike = seq.copy()
    spike[12, 16] += [0.8, -0.6, 0]  # one-frame jump of the right wrist
    cleaned, fixed = remove_glitches(spike)
    assert fixed >= 1 and np.linalg.norm(cleaned[12, 16] - seq[12, 16]) < 0.05
    step = seq.copy()
    step[15:, GROUPS["rh"]] += [0, -0.3, 0]  # the hand moves and stays: a real change
    kept, _ = remove_glitches(step)
    assert np.allclose(kept[20, GROUPS["rh"]], step[20, GROUPS["rh"]], atol=1e-5)


def test_smoothing_reduces_jitter_but_keeps_a_handshape_change():
    rng = np.random.default_rng(3)
    seq = body(40)
    noisy = seq + rng.normal(0, 0.02, seq.shape).astype(np.float32) * (np.abs(seq) > 0)
    out = smooth(noisy, 9, 5, 3)
    err = lambda x: float(np.abs(x[:, 16] - seq[:, 16]).mean())
    assert err(out) < 0.6 * err(noisy)
    change = seq.copy()
    change[20:, GROUPS["rh"].start + 8] += [0, -0.15, 0]  # index fingertip bends at frame 20
    sm = smooth(change, 9, 5, 3)
    tip = GROUPS["rh"].start + 8
    assert abs(sm[24, tip, 1] - change[24, tip, 1]) < 0.03  # sharp again within a few frames


def test_retargeting_keeps_bone_lengths_and_directions():
    seq = body(30, wave=1.0)
    seq[10:15, 15] *= 1.3  # stretched keypoints in some frames
    lengths = fit_lengths([seq])
    out = from_state(to_state(seq), lengths)
    for (a, b), length in zip(ARM_BONES, lengths["arm"]):
        v = out[:, b] - out[:, a]
        v[:, 2] *= 0.3
        assert np.allclose(np.linalg.norm(v, axis=1), length, atol=1e-4)
    d_in = seq[:, 16, :2] - seq[:, 14, :2]
    d_out = out[:, 16, :2] - out[:, 14, :2]
    cos = (d_in * d_out).sum(1) / np.linalg.norm(d_in, axis=1) / np.linalg.norm(d_out, axis=1)
    assert cos.min() > 0.999  # direction on screen unchanged


def test_slerp():
    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert np.allclose(slerp(a, b, 0.0), a) and np.allclose(slerp(a, b, 1.0), b)
    mid = slerp(a, b, 0.5)
    assert np.isclose(np.linalg.norm(mid), 1) and np.allclose(mid, [2 ** -0.5, 2 ** -0.5, 0])


def test_transition_length():
    m = S2S["motion"]
    assert transition_length(0.0, m) == m["transition_min_frames"]
    assert transition_length(10.0, m) == m["transition_max_frames"]
    assert m["transition_min_frames"] <= transition_length(0.7, m) <= m["transition_max_frames"]


def test_build_motion_structure_holds_and_speed():
    d, m = FakeDictionary(), S2S["motion"]
    mo = build_motion(["W1", "W2"], d, S2S)
    kinds = [KINDS[k] for k in mo.kinds]
    assert kinds[0] == "rest" and kinds[-1] == "rest"
    assert kinds.count("hold") == 2 * int(round(m["hold_s"] * d.fps))
    end_rest = len(kinds) - max(i for i, k in enumerate(kinds) if k != "rest") - 1
    assert end_rest == int(round(m["sentence_pause_s"] * d.fps)) > int(round(m["rest_start_s"] * d.fps)) - 1
    assert set(mo.words[np.array(kinds) == "sign"]) == {0, 1}
    t60, t30 = mo.sample_times(60, 1.0), mo.sample_times(60, 0.5)
    assert abs(len(t30) - 2 * len(t60)) <= 2 and np.isclose(t60[1] - t60[0], 1 / 60)
    seq = mo.keypoints(t60)
    assert seq.shape == (len(t60), N_KP, 3) and np.isfinite(seq).all()


def test_held_pose_stays_still():
    mo = build_motion(["W1"], FakeDictionary(), S2S)
    ts = mo.sample_times(60)
    held = ts[mo.kind_at(ts) == "hold"][1:-1]
    seq = mo.keypoints(held)
    assert np.abs(np.diff(seq[:, [15, 16]], axis=0)).max() < 1e-4  # no overshoot from the curves


def test_bone_lengths_never_change_in_the_animation():
    mo = build_motion(["W1", "W2"], FakeDictionary(), S2S)
    seq = mo.keypoints(mo.sample_times(60))
    v = seq[:, 16] - seq[:, 14]
    v[:, 2] *= mo.pose_z
    lengths = np.linalg.norm(v, axis=1)
    assert lengths.max() - lengths.min() < 1e-3


def test_motion_metrics():
    seq = np.zeros((10, N_KP, 3), np.float32)
    seq[:, JOINT := 16, 0] = np.arange(10) * 0.1
    seq[:, :, 1] = 0.5  # every joint present
    r = motion_metrics(seq, 30)
    assert np.isclose(r["max_jump"], 0.1, atol=1e-6) and np.isclose(r["max_speed"], 3.0, atol=1e-4)
    assert r["mean_jerk"] < 1e-3  # constant speed has no jerk


def test_captions_follow_the_word_and_highlight_it():
    mo = build_motion(["W1", "W2"], FakeDictionary(), S2S)
    ts = mo.sample_times(60)
    caps = mo.captions(ts)
    assert all(c[0] == ("W1", "W2") for c in caps)
    assert {c[1] for c in caps} == {-1, 0, 1}
    drawer = CaptionDrawer(FONT, 400, 80, 36)
    plain, first, second = (drawer.bar((("মা", "চা"), k)) for k in (-1, 0, 1))
    assert plain.shape == first.shape == (80, 400, 3)
    assert not np.array_equal(plain, first) and not np.array_equal(first, second)
    highlight = np.array(HIGHLIGHT_BACK)  # BGR, like the bar
    assert (np.abs(first.astype(int) - highlight).sum(-1) == 0).any() and not (np.abs(plain.astype(int) - highlight).sum(-1) == 0).any()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"{t.__name__} ok")
    print("\nALL TESTS PASSED")
