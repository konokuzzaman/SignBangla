"""The Module 3 lexicon, verb table and predicate table: building the CSV files and reading them.

lexicon.csv  word (dictionary form), pos, person, possessive, locative, has_sign, verified, semantic, english,
             variants, source. Built from the words the project already knows:
               - the core vocabulary (data/core_vocab/word_list.csv, else core_vocab.CORE_WORDS)
               - BdSLW60 (labels.LABELS) and BdSLW401 (data/BdSLW401/word_list.csv, else labels401.LABELS_401)
               - BdSL-300 folders with status new or ambiguous (work/core_vocab/BdSL300/overlap.csv, else
                 bdsl300.FOLDERS), using their bangla_guess
               - the verbs and predicate words of tables.py
             Every row starts with verified = false.
verbs.csv    verb, person, the five tense forms, verified
predicates.csv   word, kind (fixed / copula), present, past, future, verified
"""
import csv
import re
from pathlib import Path

from ..labels import LABELS
from ..labels401 import LABELS_401
from ..speech2sign import bdsl300 as b3
from ..speech2sign import core_vocab as cv
from ..speech2sign.text_processing import clean_text
from . import tables as T

LEXICON_COLUMNS = ["word", "pos", "person", "possessive", "locative", "has_sign", "verified", "semantic", "english",
                   "variants", "source"]
VERB_COLUMNS = ["verb", "person", *T.TENSES, "verified"]
PREDICATE_COLUMNS = ["word", "kind", "present", "past", "future", "verified"]
TEST_COLUMNS = ["id", "sign_words", "expected_bangla", "category", "is_question", "verified"]
POS = ("pronoun", "verb", "noun", "question", "negation", "time", "adjective", "number", "greeting", "other")
CORE_POS = {"answer": "other", "extra": "noun", "word": "noun"}
NUKTA, HASANT = "়", "্"


def norm(text):
    """The project's Unicode normalisation (text_processing.clean_text): NFC, no zero-width characters, single spaces."""
    return clean_text(text)


# ---------------------------------------------------------------- case forms
def _is_consonant(ch):
    return "ক" <= ch <= "হ" or ch in ("ৎ", "ং", NUKTA, "ড়", "ঢ়", "য়")


def _is_vowel_sign(ch):
    return "া" <= ch <= "ৌ" or ch == "ৗ"


def possessive_of(word):
    """বাবা -> বাবার, ডাক্তার -> ডাক্তারের, ভাই -> ভাইয়ের (irregular forms in tables.POSSESSIVE)."""
    if word in T.POSSESSIVE:
        return T.POSSESSIVE[word]
    last = word[-1]
    if _is_vowel_sign(last):
        return word + "র"
    if _is_consonant(last) or last == HASANT:
        return word + "ের"
    return word + "য়ের"  # ends in a full vowel letter (ভাই)


def locative_of(word):
    """স্কুল -> স্কুলে, বারান্দা -> বারান্দায়, নদী -> নদীতে (irregular forms in tables.LOCATIVE)."""
    if word in T.LOCATIVE:
        return T.LOCATIVE[word]
    last = word[-1]
    if _is_consonant(last) or last == HASANT:
        return word + "ে"
    if last == "া":
        return word + "য়"
    return word + "তে"


# ---------------------------------------------------------------- building lexicon.csv
def _spellings(text):
    """'পানি/জল' -> ['পানি', 'জল']; 'ঠান্ডা / সর্দি' and 'চিন্তা (করা)' are cleaned the same way."""
    text = re.sub(r"\(.*?\)", " ", text)
    out = []
    for part in text.split("/"):
        w = norm(part)
        if w and w not in out:
            out.append(w)
    return out


def _read_csv(path):
    if path and Path(path).exists():
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    return None


def _sources(core_word_list=None, core_coverage=None, word_list_401=None, overlap=None):
    """(spellings, pos, english, has_sign, source) for every known word, in order of preference."""
    rows = []
    coverage = {r["id"]: r["has_sign"] == "True" for r in (_read_csv(core_coverage) or [])}
    core = _read_csv(core_word_list)
    core = ([r for r in core if r["id"].startswith("C")] if core else
            [dict(id=w, bangla=b, english=e, category=c) for w, b, e, c in cv.CORE_WORDS])
    for r in core:
        has = coverage.get(r["id"], bool(cv.source_of(r["id"])[0]) and cv.source_of(r["id"])[1] == "yes")
        rows.append((_spellings(r["bangla"]), CORE_POS.get(r["category"], r["category"]), r["english"], has,
                     "core_vocab"))
    for n, bangla, _roman, english in LABELS:
        rows.append((_spellings(bangla), "noun", english, True, "BdSLW60"))
    w401 = _read_csv(word_list_401)
    w401 = [(int(r["number"]), r["bangla"], r["english"]) for r in w401] if w401 else \
        [(n, b, e) for n, b, _r, e in LABELS_401]
    for n, bangla, english in w401:
        rows.append((_spellings(bangla), "noun", english, True, "BdSLW401"))
    ov = _read_csv(overlap)
    ov = ([(r["bangla_guess"], r["category"], r["english"], r["status"]) for r in ov] if ov else
          [(bg, cat, en, st) for en, bg, cat, st, *_ in b3.FOLDERS.values()])
    for bangla, category, english, status in ov:
        if status in (b3.NEW, b3.AMBIGUOUS):
            rows.append((_spellings(bangla), category, english, False, "BdSL300"))
    for verb in T.VERBS:
        rows.append(([verb], "verb", "", False, "verbs table"))
    for word, kind, *_ in T.PREDICATES:
        rows.append(([word], "noun" if kind == "fixed" else "adjective", "", False, "predicates table"))
    for word, pos, english in T.EXTRA_WORDS:
        rows.append(([word], pos, english, False, "module3"))
    for word in T.POSSESSIVE_PRONOUNS:
        rows.append(([word], "pronoun", "", False, "module3"))
    return rows


def semantic_of(word, pos):
    if word in T.POSSESSIVE_PRONOUNS:
        return "possessive"
    for name, group in (("kin", T.KIN), ("person", T.PERSON_ROLES), ("body", T.BODY), ("illness", T.ILLNESS),
                        ("place", T.PLACES)):
        if word in group:
            return name
    return ""


def build_lexicon(core_word_list=None, core_coverage=None, word_list_401=None, overlap=None, existing=None):
    """-> list of lexicon rows (dicts). `existing`: rows of a lexicon.csv already on disk; their verified values are
    kept, so a rebuild never undoes a check made by hand."""
    kept = {r["word"]: r.get("verified", "false") for r in (existing or [])}
    entries = {}
    for spellings, pos, english, has_sign, source in _sources(core_word_list, core_coverage, word_list_401, overlap):
        spellings = [T.SPELLING.get(s, s) for s in spellings] + [s for s in spellings if s in T.SPELLING]
        head, variants = spellings[0], [s for s in spellings[1:] if s != spellings[0]]
        pos = T.POS_OVERRIDE.get(head, "adjective" if head in T.ADJECTIVES else pos)
        pos = pos if pos in POS else "other"
        e = entries.get(head)
        if e is None:
            entries[head] = dict(word=head, pos=pos, english=english, has_sign=has_sign, variants=variants,
                                 source=[source])
            continue
        e["has_sign"] |= has_sign
        e["variants"] += [v for v in variants if v not in e["variants"]]
        if source not in e["source"]:
            e["source"].append(source)
        e["english"] = e["english"] or english
    rows = []
    for head, e in entries.items():
        pos, sem = e["pos"], semantic_of(head, e["pos"])
        person = possessive = locative = ""
        if head in T.PRONOUNS:
            person, possessive = T.PRONOUNS[head]
        elif head in T.POSSESSIVE_PRONOUNS:
            person = T.POSSESSIVE_PRONOUNS[head]
        elif pos == "noun":
            possessive = possessive_of(head)
            if sem in ("kin", "person"):
                person = "3"
                locative = f"{possessive} কাছে"
            elif sem == "place":
                locative = locative_of(head)
        variants = [v for v in e["variants"] if v not in entries]  # a variant must not hide another headword
        rows.append(dict(word=head, pos=pos, person=person, possessive=possessive, locative=locative,
                         has_sign="yes" if e["has_sign"] else "no", verified=kept.get(head, "false"), semantic=sem,
                         english=e["english"], variants="/".join(variants), source="+".join(e["source"])))
    return rows


# ---------------------------------------------------------------- writing and reading the files
def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def verb_rows():
    return [dict(verb=v, person=p, verified="false", **{t: forms[t].split("|")[i] for t in T.TENSES})
            for v, forms in T.VERBS.items() for i, p in enumerate(T.PERSONS)]


def predicate_rows():
    return [dict(word=w, kind=k, present=a, past=b, future=c, verified="false") for w, k, a, b, c in T.PREDICATES]


def test_rows():
    return [dict(id=i, sign_words=s, expected_bangla=e, category=c, is_question=str(q).lower(), verified="false")
            for i, s, e, c, q in T.TEST_SENTENCES]


def write_missing(paths, sources=None, rebuild_lexicon=False):
    """Write the Module 3 data files that do not exist yet (never overwrites a hand-edited file). With
    rebuild_lexicon, lexicon.csv is built again from the sources, keeping its verified values. -> files written."""
    written = []
    for key, rows, columns in (("verbs", verb_rows, VERB_COLUMNS), ("predicates", predicate_rows, PREDICATE_COLUMNS),
                               ("test_sentences", test_rows, TEST_COLUMNS)):
        if not Path(paths[key]).exists():
            write_csv(paths[key], rows(), columns)
            written.append(paths[key])
    if rebuild_lexicon or not Path(paths["lexicon"]).exists():
        existing = _read_csv(paths["lexicon"]) if Path(paths["lexicon"]).exists() else None
        write_csv(paths["lexicon"], build_lexicon(existing=existing, **(sources or {})), LEXICON_COLUMNS)
        written.append(paths["lexicon"])
    return written


def read_csv(path):
    rows = _read_csv(path)
    if rows is None:
        raise FileNotFoundError(f"{path} not found; run scripts/50_sentence_builder_eval.py once to write it")
    return [{k: norm(v) if k in ("word", "verb", "variants", "possessive", "locative", "present", "past", "future",
                                 *T.TENSES) and v else (v or "") for k, v in r.items()} for r in rows]


class Lexicon:
    """The three tables, ready for the rules."""

    def __init__(self, lexicon_rows, verb_rows_, predicate_rows_):
        self.entries = {r["word"]: r for r in lexicon_rows}
        self.verbs = {}
        for r in verb_rows_:
            self.verbs.setdefault(r["verb"], {})[r["person"]] = {t: r[t] for t in T.TENSES}
        self.predicates = {r["word"]: r for r in predicate_rows_}
        self.surface = {}  # any accepted spelling -> (headword, how)
        for w in self.entries:
            self.surface[w] = (w, "exact")
        for w, r in self.entries.items():
            for v in filter(None, r.get("variants", "").split("/")):
                self.surface.setdefault(norm(v), (w, "variant"))
        for form, base in cv.CORE_SYNONYMS.items():
            if norm(base) in self.entries:
                self.surface.setdefault(norm(form), (norm(base), "synonym"))
        for form, base in cv.verb_forms().items():
            if norm(base) in self.verbs:
                self.surface.setdefault(norm(form), (norm(base), "verb form"))
        self.max_words = max(len(s.split()) for s in self.surface)

    @classmethod
    def from_files(cls, paths):
        return cls(read_csv(paths["lexicon"]), read_csv(paths["verbs"]), read_csv(paths["predicates"]))

    def get(self, word):
        return self.entries.get(word)
