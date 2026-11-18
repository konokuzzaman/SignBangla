"""Adding the words of the extra sources to data/core_vocab/word_list.csv (always verified = false).

A word with clips from a source (a Ban-Sign-Sent-9K last sign, a DU class) always gets its own row, linked through
core_id to the core word with the same meaning when there is one (না -> C035). A word without clips is added only when
the project does not know it yet (not in the word list, BdSLW60 or BdSLW401). Rows already in the file are kept as
they are, so a 'true' set by hand is never undone; running again adds nothing twice.
"""
from ..labels import LABELS
from ..labels401 import LABELS_401
from ..speech2sign.bdsl300 import read_word_list, write_rows
from ..speech2sign.dictionary import split_variants
from ..speech2sign.text_processing import clean_text


def _forms(bangla):
    return {clean_text(v) for part in bangla.split("|") for v in split_variants(part)}


def known_words(rows):
    """{normalised word: id} from the word list, plus BdSLW60 / BdSLW401 words as 'W<n>'."""
    out = {}
    for n, bangla, *_ in list(LABELS_401) + list(LABELS):
        for f in _forms(bangla):
            out.setdefault(f, f"W{n}")
    for r in rows:
        for f in _forms(r["bangla"]):
            out[f] = r["id"]  # word-list ids win over W numbers
    return out


def add_words(path, source, prefix, words):
    """words: dicts with bangla, english, category, folder, has_clips. -> list of rows added."""
    rows = read_word_list(path)
    if not rows:
        raise FileNotFoundError(f"{path} is missing: run scripts/40_core_vocab_download.py first")
    known = known_words(rows)
    mine = {(r["source"], clean_text(r["bangla"])) for r in rows}
    used = {r["id"] for r in rows}
    number = 1 + max([int(i[len(prefix):]) for i in used if i.startswith(prefix) and i[len(prefix):].isdigit()],
                     default=0)
    added = []
    for w in words:
        key = clean_text(w["bangla"])
        if (source, key) in mine:
            continue
        match = next((known[f] for f in _forms(w["bangla"]) if f in known), "")
        if match and not w.get("has_clips"):
            continue  # already known and nothing new from this source
        while f"{prefix}{number:03d}" in used:
            number += 1
        row = dict(id=f"{prefix}{number:03d}", bangla=w["bangla"], english=w["english"], category=w["category"],
                   source=source, verified="false", folder=w.get("folder", ""),
                   core_id=match if match.startswith("C") else "")
        used.add(row["id"])
        mine.add((source, key))
        added.append(row)
    if added:
        write_rows(path, rows + added)
    return added
