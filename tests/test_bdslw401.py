"""Tests for the BdSLW401 helpers: extraction order, rotation list, orientation check. No dataset or MediaPipe needed.

    python tests/test_bdslw401.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from signbangla.bdslw401 import priority_first, rotated_clips, upright_vote
from signbangla.keypoints import N_KP
from signbangla.labels401 import parse_clip_name


def fake_clips(words=(1, 2, 3), signers=range(1, 19), per_pair=10):
    return [f"W{w:03d}S{s:02d}F_{c:02d}" for w in words for s in signers for c in range(1, per_pair + 1)]


def signer_of(stem):
    return parse_clip_name(stem)["signer"]


def test_every_clip_appears_once():
    stems = fake_clips()
    order = priority_first(stems, 12, [4, 8])
    assert sorted(order) == sorted(stems)


def test_first_group_size_and_excluded_signers():
    order = priority_first(fake_clips(), 12, [4, 8])
    first = order[:36]
    assert Counter(parse_clip_name(s)["word_number"] for s in first) == {1: 12, 2: 12, 3: 12}
    assert not {signer_of(s) for s in first} & {4, 8}
    assert {signer_of(s) for s in order[36:]} & {4, 8} == {4, 8}  # still extracted, only later


def test_first_group_is_spread_across_signers():
    first = priority_first(fake_clips(words=(1,)), 12, [4, 8])[:12]
    counts = Counter(signer_of(s) for s in first)
    assert max(counts.values()) == 1 and len(counts) == 12
    other = priority_first(fake_clips(words=(2,)), 12, [4, 8])[:12]
    assert {signer_of(s) for s in first} != {signer_of(s) for s in other}  # not the same signers for every word


def test_word_with_few_signers_takes_turns():
    stems = fake_clips(words=(7,), signers=(1, 2), per_pair=10)
    first = priority_first(stems, 12, [])[:12]
    assert Counter(signer_of(s) for s in first) == {1: 6, 2: 6}


def test_word_with_fewer_clips_than_requested():
    stems = fake_clips(words=(7,), signers=(1,), per_pair=3) + fake_clips(words=(7,), signers=(4,), per_pair=3)
    assert sorted(priority_first(stems, 12, [4])) == sorted(stems)
    assert {signer_of(s) for s in priority_first(stems, 12, [4])[:3]} == {1}


def test_word_with_only_excluded_signers():
    stems = fake_clips(words=(9,), signers=(4, 8), per_pair=2)
    assert sorted(priority_first(stems, 12, [4, 8])) == sorted(stems)


def test_order_never_changes():
    stems = fake_clips()
    assert priority_first(stems, 12, [4, 8]) == priority_first(list(reversed(stems)), 12, [4, 8])


def test_bad_name_raises():
    try:
        priority_first(["U4W1F"], 12, [])
        raise AssertionError("a BdSLW60 name must raise")
    except ValueError:
        pass


def test_rotated_clips():
    stems = ["W351S17F_01", "W351S17F_02", "W352S17F_01", "W351S01F_01"]
    assert rotated_clips(stems, ["W351S17"]) == {"W351S17F_01", "W351S17F_02"}
    assert rotated_clips(stems, ["W352S17F_01"]) == {"W352S17F_01"}
    assert rotated_clips(stems, []) == set() and rotated_clips(stems, None) == set()


def body(upside_down=False):
    f = np.full((N_KP, 3), np.nan, np.float32)
    f[:25] = [0.5, 0.5, 0.0]
    f[11], f[12] = [0.6, 0.4, 0.0], [0.4, 0.4, 0.0]  # shoulders: the signer's left is on the image's right
    f[23], f[24] = [0.55, 0.8, 0.0], [0.45, 0.8, 0.0]  # hips
    if upside_down:
        f[:25, :2] = 1 - f[:25, :2]
    return f


def test_upright_vote():
    assert upright_vote(body()) is True
    assert upright_vote(body(upside_down=True)) is False
    assert upright_vote(np.full((N_KP, 3), np.nan, np.float32)) is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"{t.__name__} ok")
    print("\nALL TESTS PASSED")
