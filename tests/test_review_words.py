"""Saving of the word review page (scripts/tools/review_words.py): only the answered row changes, the encoding and line
endings stay, a backup is made before the first change, and every answer is logged."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("review_words", ROOT / "scripts" / "tools" / "review_words.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

HEADER = "id,bangla,english,category,source,verified,folder,core_id"
ROWS = [
    "C001,আমি,I,pronoun,BdSL40,yes,,",
    'B113,আপনাদের,"your (formal, plural)",pronoun,BdSL300,false,apnader,',
    "B225,নয়,nine,number,BdSL300,false,noy,",
    "S001,না,no / not,negation,BanSign9K,false,,C035",
    "D001,দয়া করে,please,other,DynamicDU,false,Please,",
]


def make_file(tmp_path, newline="\n", bom=True):
    path = tmp_path / "word_list.csv"
    text = newline.join([HEADER] + ROWS) + newline
    path.write_bytes((("﻿" if bom else "") + text).encode("utf-8"))
    return path


def store_for(tmp_path, path, ids=("B113", "B225", "S001", "D001")):
    return R.WordListFile(path, tmp_path / "backup", tmp_path / "review_log.csv", ids)


def lines_of(path):
    return path.read_bytes().decode("utf-8").splitlines(keepends=True)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_only_the_answered_row_changes(tmp_path, newline):
    path = make_file(tmp_path, newline)
    before = lines_of(path)
    old, new = R.answer(store_for(tmp_path, path), "B225", "correct")
    after = lines_of(path)
    assert old["verified"] == "false" and new["verified"] == "true"
    assert len(after) == len(before)
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert changed == [3]
    assert after[3] == f"B225,নয়,nine,number,BdSL300,true,noy,{newline}"


def test_encoding_kept(tmp_path):
    path = make_file(tmp_path)
    R.answer(store_for(tmp_path, path), "S001", "wrong", reason="looks like নেই")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.count(b"\xef\xbb\xbf") == 1
    assert b"\r\n" not in raw
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["id"] for r in rows] == ["C001", "B113", "B225", "S001", "D001"]
    assert rows[3]["verified"] == "no"
    assert rows[1]["english"] == "your (formal, plural)"


def test_file_without_bom_stays_without(tmp_path):
    path = make_file(tmp_path, bom=False)
    R.answer(store_for(tmp_path, path), "D001", "correct")
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_backup_made_once_before_the_first_change(tmp_path):
    path = make_file(tmp_path)
    original = path.read_bytes()
    store = store_for(tmp_path, path)
    assert not (tmp_path / "backup").exists()
    R.answer(store, "B225", "correct")
    backups = list((tmp_path / "backup").glob("word_list_*.csv"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    R.answer(store, "D001", "correct")
    assert len(list((tmp_path / "backup").glob("word_list_*.csv"))) == 1


def test_edits_and_quotes(tmp_path):
    path = make_file(tmp_path)
    before = lines_of(path)
    R.answer(store_for(tmp_path, path), "B113", "correct", bangla="আপনাদের", english='your, plural "formal"')
    after = lines_of(path)
    assert [i for i, (a, b) in enumerate(zip(before, after)) if a != b] == [2]
    with open(path, encoding="utf-8-sig", newline="") as f:
        row = [r for r in csv.DictReader(f) if r["id"] == "B113"][0]
    assert row["english"] == 'your, plural "formal"' and row["verified"] == "true" and row["folder"] == "apnader"


def test_log_has_every_answer(tmp_path):
    path = make_file(tmp_path)
    store = store_for(tmp_path, path)
    R.answer(store, "B225", "wrong", reason="this is নয় (nine)")
    R.answer(store, "D001", "unsure")
    R.answer(store, "D001", "correct", english="please (polite)")
    with open(tmp_path / "review_log.csv", encoding="utf-8-sig", newline="") as f:
        log = list(csv.DictReader(f))
    assert [c for c in log[0]] == R.LOG_COLUMNS
    assert [(r["id"], r["decision"]) for r in log] == [("B225", "wrong"), ("D001", "unsure"), ("D001", "correct")]
    assert log[0]["old"] == "verified=false" and log[0]["new"] == "verified=no" and log[0]["note"] == "this is নয় (nine)"
    assert log[1]["note"] == "unsure"
    assert "english=please (polite)" in log[2]["new"] and "verified=true" in log[2]["new"]
    assert R.read_log(tmp_path / "review_log.csv")["D001"]["decision"] == "correct"


def test_unsure_on_an_unverified_row_does_not_touch_the_file(tmp_path):
    path = make_file(tmp_path)
    original = path.read_bytes()
    R.answer(store_for(tmp_path, path), "S001", "unsure", reason="two signs?")
    assert path.read_bytes() == original
    assert not (tmp_path / "backup").exists()
    assert (tmp_path / "review_log.csv").exists()


def test_rows_not_under_review_and_other_columns_are_refused(tmp_path):
    path = make_file(tmp_path)
    original = path.read_bytes()
    store = store_for(tmp_path, path)
    with pytest.raises(ValueError):
        R.answer(store, "C001", "wrong")
    with pytest.raises(ValueError):
        store.update("B225", {"source": "BdSL40"}, "correct")
    with pytest.raises(ValueError):
        R.answer(store, "B225", "correct", bangla="")
    with pytest.raises(ValueError):
        R.answer(store, "B225", "correct", english="two\nlines")
    with pytest.raises(ValueError):
        R.answer(store, "B225", "maybe")
    assert path.read_bytes() == original


def test_review_rows_keep_answered_words():
    rows = [dict(id="B1", source="BdSL300", verified="true"), dict(id="B2", source="BdSL300", verified="false"),
            dict(id="C1", source="BdSL40", verified="false"), dict(id="B3", source="BdSL300", verified="true")]
    assert [r["id"] for r in R.review_rows(rows, {"B1": {}})] == ["B1", "B2"]


def test_priority_groups():
    def row(bangla, source, category, english="x"):
        return dict(bangla=bangla, source=source, category=category, english=english)
    assert R.group_of(row("না", "BanSign9K", "negation")) == 1
    assert R.group_of(row("হ্যালো", "DynamicDU", "greeting")) == 1
    assert R.group_of(row("খাওয়া", "BdSL300", "verb")) == 2
    assert R.group_of(row("হয়েছে", "BanSign9K", "other")) == 3
    assert R.group_of(row("আজ", "BdSL300", "time", "today")) == 3
    assert R.group_of(row("জুন", "BdSL300", "time", "June")) == 5
    assert R.group_of(row("শুভ সকাল", "DynamicDU", "greeting")) == 4
    assert R.group_of(row("নদী", "BdSL300", "noun")) == 5


def test_consistency_separates_clear_words():
    rng = np.random.default_rng(0)
    groups = {"A": [np.zeros(8) + rng.normal(0, .05, 8) for _ in range(4)],
              "B": [np.ones(8) + rng.normal(0, .05, 8) for _ in range(4)],
              "C": [rng.normal(0.5, 2, 8) for _ in range(4)]}
    s = R.consistency_scores(groups, warn_ratio=0.85, warn_nearest=0.0)
    assert s["A"]["ratio"] < 0.2 and s["A"]["nearest_same"] == 1.0 and not s["A"]["warn"]
    assert s["C"]["ratio"] > s["A"]["ratio"] and s["C"]["warn"]
