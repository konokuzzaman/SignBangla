"""BdSL-300: the word table, clip names, overlap rules, word list and the verified-only rule of the core dictionary."""
import copy

import numpy as np
import pandas as pd
import pytest
from test_speech2sign import fake_raw

from signbangla.labels401 import LABELS_401
from signbangla.speech2sign import bdsl300 as b3
from signbangla.speech2sign import core_vocab as cv
from signbangla.speech2sign.core_dictionary import bdsl300_words
from signbangla.utils import atomic_save_npy, load_config

CORE_IDS = {r[0] for r in cv.CORE_WORDS}
W_KEYS = {f"W{n}" for n, *_ in LABELS_401}


def test_table_is_consistent():
    assert len(b3.FOLDERS) == 300
    for folder, (english, bangla, category, status, matches, core_id, _note) in b3.FOLDERS.items():
        assert english and bangla, folder
        assert category in b3.CATEGORIES, folder
        assert status in b3.STATUSES, folder
        keys = [k for k in matches.split("|") if k]
        assert (status == b3.NEW) == (not keys), folder
        assert set(keys) <= W_KEYS | CORE_IDS, folder
        if core_id:
            assert core_id in CORE_IDS and status != b3.DUPLICATE, folder
    assert b3.UNSURE_ENGLISH <= set(b3.FOLDERS)


def test_overlap_rows_and_clashes():
    existing = b3.existing_words()  # BdSLW60 + BdSLW401 only
    existing.update({k: ("x", "x", "core") for k in CORE_IDS})
    rows = b3.overlap_rows(list(b3.FOLDERS), existing)
    assert len(rows) == 300 and [r["id"] for r in rows] == [f"B{i:03d}" for i in range(1, 301)]
    by = {r["folder"]: r for r in rows}
    assert by["kothay"]["status"] == "new" and by["kothay"]["needs_check"] is True
    assert by["Banana"]["status"] == "duplicate" and by["Banana"]["matches_existing_word"].startswith("W63 ")
    assert by["Eat"]["status"] == "ambiguous" and by["Eat"]["core_id"] == "C009"
    assert by["Go"]["needs_check"] is False
    assert b3.spelling_clashes(existing) == []
    with pytest.raises(KeyError):
        b3.overlap_rows(["Eat", "NotAFolder"], existing)


def test_parse_clip_and_stem():
    assert b3.parse_clip("Eat/Eat_A_3.mp4") == dict(folder="Eat", file="Eat_A_3.mp4", signer="A", view="", take=3)
    assert b3.parse_clip("Bad Smell/Bad Smell_D_5.MOV")["signer"] == "D"
    p = b3.parse_clip("chinta/p12_c_chintakora.mp4")
    assert (p["signer"], p["view"]) == ("p12", "c")
    assert b3.parse_clip("aat/p2_f_Aat.mp4")["view"] == "f"
    assert b3.parse_clip("Eat/readme.txt") is None and b3.parse_clip("Eat") is None
    assert b3.clip_stem("Bad Smell/Bad Smell_A_1.mp4") == "Bad-Smell__Bad-Smell_A_1"


def test_select_clips():
    names = ["aam/p1_c_aam.mp4", "aam/p1_f_aam.mp4", "aam/p2_f_aam.mp4", "Eat/Eat_A_1.mp4", "aam/notes.txt"]
    entries = [dict(name=n) for n in names]
    assert [e["name"] for e in b3.select_clips(entries, "both")] == names[:4]
    assert [e["name"] for e in b3.select_clips(entries, "close_first")] == ["aam/p1_c_aam.mp4", "aam/p2_f_aam.mp4",
                                                                            "Eat/Eat_A_1.mp4"]
    with pytest.raises(ValueError):
        b3.select_clips(entries, "front")


def test_word_list_keeps_hand_verified(tmp_path):
    path = tmp_path / "word_list.csv"
    cv.write_word_list(path)
    rows = b3.overlap_rows(list(b3.FOLDERS), {**b3.existing_words(), **{k: ("x", "x", "c") for k in CORE_IDS}})
    added = b3.add_to_word_list(path, rows)
    assert added == sum(r["status"] != "duplicate" for r in rows)
    assert b3.verified_words(path) == []  # nothing is switched on
    table = b3.read_word_list(path)
    assert all(r["verified"] == "false" for r in table if r["source"] == "BdSL300")
    for r in table:
        if r.get("folder") == "Go":
            r["verified"] = "TRUE"
    b3.write_rows(path, table)
    assert b3.add_to_word_list(path, rows) == 0  # running step 2 again adds nothing ...
    cv.write_word_list(path)  # ... and neither it nor scripts/40 undoes the hand-set flag
    assert [(v[1], v[2]) for v in b3.verified_words(path)] == [("Go", "C011")]


def test_only_verified_words_are_used(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["work_dir"] = str(tmp_path)
    b = cfg["speech2sign"]["bdsl300"]
    trials = tmp_path / b["keypoints_dir"] / "trials"
    trials.mkdir(parents=True)
    rows = []
    for folder, signer, view, raw in [("Go", "A", "", fake_raw(40)), ("Go", "B", "", fake_raw(44)),
                                      ("kukur", "p4", "c", fake_raw(40)), ("kukur", "p4", "f", fake_raw(40)),
                                      ("kukur", "p5", "c", fake_raw(42)), ("bah", "p1", "c", fake_raw(40))]:
        stem = f"{folder}__{signer}{view}"
        atomic_save_npy(trials / f"{stem}.npy", raw)
        rows.append(dict(stem=stem, folder=folder, signer=signer, view=view, fps=30.0, width=640, height=480, ok=True))
    (tmp_path / b["dir"]).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tmp_path / b["dir"] / "quality_report.csv", index=False)

    assert bdsl300_words(cfg, []) == {}
    verified = [("B034", "Go", "C011", "যাওয়া", "go", "verb"), ("B207", "kukur", "", "কুকুর", "dog", "noun")]
    out = bdsl300_words(cfg, verified)
    assert set(out) == {"C011", "B207"}  # bah is not verified, so it is not there
    entry, chosen = out["C011"]
    assert entry["source"] == "BdSL300" and entry["verified"] and entry["bangla"][0] == "যাওয়া"
    assert len(chosen) == 2 and all(np.isfinite(c["seq"]).all() for c in chosen)
    entry, chosen = out["B207"]
    assert entry["bangla"] == ["কুকুর"] and entry["category"] == "noun"
    assert sorted(c["ref"] for c in chosen) == ["kukur__p4c", "kukur__p5c"]  # one copy of each signing, close view
