"""Tests for the BdSLW401 word list and clip names. No dataset needed.

    python tests/test_labels401.py
"""
import csv
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signbangla.labels import LABELS, N_CLASSES, NAME_RE, parse_video_stem
from signbangla.labels401 import LABELS_401, N_WORDS_401, parse_clip_name, write_word_list


def test_word_list_is_complete():
    assert N_WORDS_401 == 401
    assert [row[0] for row in LABELS_401] == list(range(1, 402))
    for number, bangla, roman, english in LABELS_401:
        assert bangla and roman and english, number
        assert unicodedata.is_normalized("NFC", bangla), number
        assert all(ch.isspace() or ch == "/" or "ঀ" <= ch <= "৿" for ch in bangla), number


def test_same_numbers_as_the_60_words():
    """Each of the 60 words has the same number and Bangla word in both lists, apart from two known spelling differences."""
    by_number = {row[0]: row for row in LABELS_401}
    for number, bangla, _, _ in LABELS:
        first = bangla.split("/")[0].strip()
        assert number in by_number, number
        allowed = {5: "বৌ", 360: "দূর্বল"}.get(number, unicodedata.normalize("NFC", first))
        assert by_number[number][1].split("/")[0] == allowed, number
    assert by_number[1][3] == "Father" and by_number[351][3] == "AIDS" and by_number[358][3] == "Doctor"


def test_parse_clip_name():
    want = dict(word="W1", word_number=1, signer=4, view="F", clip=2)
    assert parse_clip_name("W001S04F_02") == want
    assert parse_clip_name("W001S04F_02.mp4") == want
    assert parse_clip_name("W401S18L_10.mp4") == dict(word="W401", word_number=401, signer=18, view="L", clip=10)


def test_parse_clip_name_rejects_other_names():
    for name in ("U4W1F", "U1W357F_2", "W1S4F_2", "W001S04_02", "W001S04X_02", "W001S04F", "x/W001S04F_02.mp4", ""):
        assert parse_clip_name(name) is None, name


def test_labels_py_is_unchanged():
    assert N_CLASSES == len(LABELS) == 60
    assert NAME_RE.pattern == r"^U(?P<user>\d+)W(?P<word>\d+)(?P<view>[A-Za-z]*)(?:_(?P<part>\d+))?$"
    assert parse_video_stem("U11W1F")["class_idx"] == 0
    assert parse_video_stem("W001S04F_02") is None


def test_write_word_list():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.csv"
        write_word_list(path)
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
    assert rows[0] == ["number", "bangla", "english"] and len(rows) == 402
    assert rows[1] == ["1", "বাবা/আব্বা", "Father"] and rows[401] == ["401", "এক্সরে", "X-Ray"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"{t.__name__} ok")
    print("\nALL TESTS PASSED")
