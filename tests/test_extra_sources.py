"""Extra data sources: the standard-library xlsx reader, the Ban-Sign-Sent-9K sentence table and the last-sign cutting
(on synthetic keypoint sequences)."""
import zipfile

import numpy as np
import pytest

from signbangla.extra_sources import bansign9k as B
from signbangla.extra_sources import xlsx as X

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def _cell(ref, v):
    if v is None:
        return ""
    if isinstance(v, tuple):  # ("s", index) shared string, ("inline", text), ("b", 0/1), ("str", text)
        kind, x = v
        if kind == "inline":
            return f'<c r="{ref}" t="inlineStr"><is><t>{x}</t></is></c>'
        return f'<c r="{ref}" t="{kind}"><v>{x}</v></c>'
    return f'<c r="{ref}"><v>{v}</v></c>'


def make_xlsx(path, sheets, shared):
    """sheets: {name: {row number: {column letters: value}}}."""
    with zipfile.ZipFile(path, "w") as z:
        names = list(sheets)
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{MAIN}" xmlns:r="{RELS}"><sheets>' + "".join(
            f'<sheet name="{n}" sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i, n in enumerate(names)) + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{PKG}">' + "".join(
            f'<Relationship Id="rId{i + 1}" Type="x" Target="worksheets/sheet{i + 1}.xml"/>' for i in range(len(names)))
                   + "</Relationships>")
        si = "".join(f"<si><t>{s}</t></si>" if isinstance(s, str) else
                     "<si>" + "".join(f"<r><t>{p}</t></r>" for p in s) + "</si>" for s in shared)
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{MAIN}">{si}</sst>')
        for i, n in enumerate(names):
            rows = "".join(f'<row r="{r}">' + "".join(_cell(f"{col}{r}", v) for col, v in cells.items()) + "</row>"
                           for r, cells in sheets[n].items())
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", f'<worksheet xmlns="{MAIN}"><sheetData>{rows}</sheetData></worksheet>')


def test_column_index():
    assert [X.column_index(c) for c in ("A", "C", "Z", "AA", "AB")] == [0, 2, 25, 26, 27]


def test_xlsx_reader(tmp_path):
    p = tmp_path / "t.xlsx"
    make_xlsx(p, {"First": {1: {"A": ("s", 0), "B": ("s", 1)}, 2: {"A": 12, "C": 2.5}, 4: {"B": ("inline", "নিজে")}},
                  "Second": {1: {"A": ("b", 1), "B": ("str", "=x")}}},
              ["নাম", ["বা", "বা"]])
    assert X.sheet_names(p) == ["First", "Second"]
    rows = X.read_sheet(p)
    assert rows == [["নাম", "বাবা", None], [12, None, 2.5], [None, None, None], [None, "নিজে", None]]
    assert X.read_sheet(p, "Second") == [[True, "=x"]]
    with pytest.raises(KeyError):
        X.read_sheet(p, "Third")


def test_bansign9k_tables(tmp_path):
    p = tmp_path / "b.xlsx"
    shared = ["Sentence ID", "Natural Sentence", "Sign Sentence", "বাবা বিকালে চা খায়নি।", "বাবা বিকাল চা নেই।",
              "মা রাতে শাক-সব্জী খাবে।", "মা রাত শাক-সব্জী খাওয়ায়া হবে।"]
    timing_head = ["annotatedGlossId", "attachmentId", "sentenceId", "userId", "word", "start", "end", "contentUrl",
                   "metas"]
    shared += timing_head + ["বাবা", "https://x/y/REC_CS_S1_U2_20231017_1.mp4", "private"]
    t0 = len(shared) - len(timing_head) - 3
    timing = {1: {c: ("s", t0 + i) for c, i in zip("ABCDEFGHI", range(9))},
              2: {"A": 5, "B": 6, "C": 1, "D": 9, "E": ("s", t0 + 9), "F": 0.5, "G": 1.25, "H": ("s", t0 + 10),
                  "I": ("s", t0 + 11)}}
    make_xlsx(p, {B.SENTENCE_SHEET: {1: {"A": ("s", 0), "B": ("s", 1), "C": ("s", 2)},
                                     2: {"A": 27941, "B": ("s", 3), "C": ("s", 4)},
                                     3: {"A": 3, "B": ("s", 5), "C": ("s", 6)}},
                  B.TIMING_SHEET: timing}, shared)
    s = B.read_sentences(p)
    assert s[0] == dict(sentence_id=27941, natural="বাবা বিকালে চা খায়নি।", sign="বাবা বিকাল চা নেই।")
    assert B.sign_words(s[1]["sign"]) == ["মা", "রাত", "শাক-সব্জী", "খাওয়ায়া", "হবে"]
    assert [B.marker(x["sign"]) for x in s] == ["নেই", "হবে"]
    t = B.read_word_timings(p)
    assert t == [dict(gloss_id=5, sentence_id=1, user_id=9, word="বাবা", start=0.5, end=1.25,
                      video="REC_CS_S1_U2_20231017_1.mp4")]
    assert "private" not in str(t)
    v = B.vocabulary(s)
    assert {x["word"]: x["count"] for x in v}["বাবা"] == 1 and len(v) == 9
    typos = B.possible_typos(v, known=["বাবা", "চা"])
    assert [x["word"] for x in typos] == ["খাওয়ায়া"]


# ---------------------------------------------------------------- split zip reader
def _split_zip(tmp_path, files, part_size):
    """Write a normal zip, then cut it into parts the way a split zip is stored: disk numbers and offsets per part in
    the central directory and the end record; the index must fall in the last part."""
    import struct
    from signbangla.extra_sources import split_zip as S
    whole = tmp_path / "whole.zip"
    with zipfile.ZipFile(whole, "w") as z:
        for name, (data, method) in files.items():
            z.writestr(name, data, compress_type=method)
    b = bytearray(b"PK\x07\x08" + whole.read_bytes())  # split zips start with this marker; offsets include it
    i = b.rfind(S.EOCD_SIG)
    cd_size, cd_offset = struct.unpack("<II", b[i + 12:i + 20])
    cd_offset += 4
    j = cd_offset
    while b[j:j + 4] == S.CD_SIG:
        n_len, e_len, c_len = struct.unpack("<HHH", b[j + 28:j + 34])
        g = struct.unpack("<I", b[j + 42:j + 46])[0] + 4
        b[j + 34:j + 36] = struct.pack("<H", g // part_size)
        b[j + 42:j + 46] = struct.pack("<I", g % part_size)
        j += 46 + n_len + e_len + c_len
    n = len(b) // part_size + 1
    assert cd_offset // part_size == n - 1, "test zip: make part_size smaller than the data"
    b[i + 4:i + 8] = struct.pack("<HH", n - 1, n - 1)
    b[i + 16:i + 20] = struct.pack("<I", cd_offset % part_size)
    paths = S.part_paths(tmp_path / "v.zip", n)
    for k, p in enumerate(paths):
        p.write_bytes(bytes(b[k * part_size:(k + 1) * part_size]))
    return paths


def test_split_zip(tmp_path):
    from signbangla.extra_sources import split_zip as S
    rng = np.random.default_rng(0)
    files = {"Sign_Videos/a.mp4": (rng.integers(0, 256, 5000, dtype=np.uint8).tobytes(), zipfile.ZIP_STORED),
             "Sign_Videos/b.mp4": (b"bangla " * 3000, zipfile.ZIP_DEFLATED),
             "Sign_Videos/c.mp4": (b"x", zipfile.ZIP_STORED)}
    paths = _split_zip(tmp_path, files, part_size=1500)
    members, info = S.read_index(paths[-1])
    assert info["disk"] == len(paths) - 1 and [m["name"] for m in members] == list(files)
    assert members[0]["disk"] == 0 and members[1]["disk"] > 0  # the first file spans several parts
    reader = S.SplitReader(paths)
    try:
        for m in members:
            dest = tmp_path / "out" / m["name"]
            assert S.extract_member(reader, m, dest) == len(files[m["name"]][0])
            assert dest.read_bytes() == files[m["name"]][0]
            assert S.extract_member(reader, m, dest) == 0  # already there
        bad = dict(members[1], crc=members[1]["crc"] ^ 1)
        with pytest.raises(ValueError):
            S.extract_member(reader, bad, tmp_path / "bad.mp4")
        assert not (tmp_path / "bad.mp4").exists() and not (tmp_path / "bad.mp4.part").exists()
    finally:
        reader.close()


# ---------------------------------------------------------------- last-sign cutting on synthetic sequences
FPS = 30
SIGNS = {  # sign name -> (centre x, centre y, radius, turns): each sign is a circle of the right hand
    "A": (0.40, 0.45, 0.04, 1.0), "B": (0.30, 0.35, 0.05, 1.5), "C": (0.45, 0.30, 0.03, 2.0),
    "D": (0.25, 0.50, 0.06, 1.0), "E": (0.50, 0.40, 0.05, 2.5)}


def _body(wrist):
    f = np.full((67, 3), np.nan, np.float32)
    f[:25] = [0.5, 0.3, 0.0]
    f[11], f[12] = [0.6, 0.4, 0], [0.4, 0.4, 0]
    f[23], f[24] = [0.55, 0.8, 0], [0.45, 0.8, 0]
    f[15] = [0.62, 0.9, 0]
    f[16] = [wrist[0], wrist[1], 0]
    if wrist[1] < 0.8:  # the hand model sees a raised hand
        f[46:67] = [wrist[0], wrist[1], 0]
        f[46:67, 0] += np.linspace(0, 0.03, 21)
    return f


def _sentence(signs, sign_s=0.8, pause_s=0.15, rest_s=0.7):
    """rest, then each sign (a circle) with a short still pause between signs, then rest. -> (frames, sign starts)."""
    frames, starts = [], []
    rest = (0.42, 0.95)
    frames += [_body(rest)] * int(rest_s * FPS)
    for k, name in enumerate(signs):
        cx, cy, r, turns = SIGNS[name]
        if k:
            frames += [frames[-1]] * int(pause_s * FPS)
        starts.append(len(frames))
        n = int(sign_s * FPS)
        for i in range(n):
            a = 2 * np.pi * turns * i / n
            frames.append(_body((cx + r * np.cos(a), cy + r * np.sin(a))))
    frames += [_body(rest)] * int(rest_s * FPS)
    return np.stack(frames), starts


def test_hand_features_fill_missing_hand():
    from signbangla.extra_sources.last_sign import hand_features
    raw, _ = _sentence(["A"])
    f = hand_features(raw)
    assert f.shape == (len(raw), 84) and not np.isnan(f).any()
    # the left hand is never seen by the hand model: it sits at its pose wrist, (0.62 - 0.5) / 0.2 shoulder widths
    assert np.allclose(f[:, 0], 0.6, atol=1e-5)


def test_dtw_finds_where_two_sentences_part():
    from signbangla.extra_sources.last_sign import cut_last_sign
    raw, starts = _sentence(["A", "B", "C"])
    partner, _ = _sentence(["A", "B", "D"])
    cut = cut_last_sign(raw, FPS, 3, [partner])
    assert cut["method"] == "dtw" and cut["ok"] and cut["partners_used"] == 1
    assert abs(cut["start"] - starts[2]) <= 6  # within 0.2 s of the true start of C
    assert cut["end"] >= starts[2] + int(0.8 * FPS)


def test_motion_fallback_without_partner():
    from signbangla.extra_sources.last_sign import cut_last_sign
    raw, starts = _sentence(["A", "B", "C"])
    cut = cut_last_sign(raw, FPS, 3)
    assert cut["method"] == "motion" and cut["ok"]
    assert abs(cut["start"] - starts[2]) <= 6


def test_same_last_sign_gives_no_dtw_cut():
    from signbangla.extra_sources.last_sign import cut_last_sign
    raw, _ = _sentence(["A", "B", "C"])
    same, _ = _sentence(["E", "B", "C"])  # differs at the start, matches at the end
    assert cut_last_sign(raw, FPS, 3, [same])["method"] == "motion"


def test_cut_length_limits():
    from signbangla.extra_sources.last_sign import cut_last_sign
    raw, _ = _sentence(["A", "B", "C"], sign_s=3.5)
    partner, _ = _sentence(["A", "B", "D"], sign_s=3.5)
    cut = cut_last_sign(raw, FPS, 3, [partner])
    assert not cut["ok"] and "longer" in cut["reason"]
    still = np.stack([_body((0.42, 0.95))] * 60)
    assert cut_last_sign(still, FPS, 3)["reason"] == "no signing found"


def test_word_list_rows(tmp_path):
    import csv
    from signbangla.extra_sources.word_list import add_words
    path = tmp_path / "word_list.csv"
    cols = ["id", "bangla", "english", "category", "source", "verified", "folder", "core_id"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow(["C035", "না/নয়", "no / not", "negation", "", "", "", ""])
        w.writerow(["C009", "খাওয়া", "eat", "verb", "BdSL40", "yes", "", ""])
    vocab = [dict(word=x) for x in ("না", "খাওয়ায়া", "বাবা", "ঝুমঝুমি", "হবে")]
    rows = B.word_list_rows(vocab, {"না": "na", "হবে": "hobe"})
    added = add_words(path, "BanSign9K", "S", rows)
    got = {r["bangla"]: r for r in added}
    # না has clips: own row linked to C035; খাওয়া is known and has no clips; বাবা is a BdSLW60 word; ঝুমঝুমি is new
    assert set(got) == {"না", "ঝুমঝুমি", "হবে"}
    assert got["না"]["core_id"] == "C035" and got["না"]["category"] == "negation" and got["না"]["folder"] == "na"
    assert all(r["verified"] == "false" and r["source"] == "BanSign9K" for r in added)
    assert [r["id"] for r in added] == ["S001", "S002", "S003"]
    assert add_words(path, "BanSign9K", "S", rows) == []  # nothing added twice
    with open(path, encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == 5
