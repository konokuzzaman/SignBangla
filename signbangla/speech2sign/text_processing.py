"""Recognized or typed text -> dictionary words: Unicode cleanup, punctuation removal, synonyms, fuzzy matching.

Words the sign dictionary cannot show come back with word == UNKNOWN, so the caller can tell the user about them.

WordMatcher() knows the 60 BdSLW60 words. WordMatcher.full() also knows the BdSLW401 words and the core vocabulary
(core_vocab.py), with verb forms (খাব, খাচ্ছি, খেয়েছি -> খাওয়া), numbers (৩টা, 25, তিন), case endings (বাবার -> বাবা),
and drops the copula (আছ, আছে), which BdSL leaves out. Every match carries the word's key in the sign dictionary (W20,
C001, or a number like ১০) and its category, so words without a sign can still be shown as text, coloured by category.
"""
import difflib
import re
import unicodedata
from dataclasses import dataclass

from ..labels import LABELS
from ..labels401 import LABELS_401
from . import core_vocab as cv
from .dictionary import normalize_word, split_variants

UNKNOWN = "unknown"

# Everyday word -> the word in the sign dictionary. Hyphens and dots become spaces in clean_text, so "টি-শার্ট" and
# "টি.ভি." arrive here as two words.
SYNONYMS = {
    "আব্বু": "বাবা", "আব্বা": "বাবা", "আম্মা": "মা", "আম্মু": "মা", "ডাক্তার": "ডক্টর",
    "বৌ": "বউ", "আঙুর": "আঙ্গুর", "টি শার্ট": "টিশার্ট", "টি ভি": "টিভি", "এ সি": "এসি",
}
KHANDA_TA_OLD = "ত্‍"  # old spelling of ৎ: ta + hasant + zero-width joiner
_STRENGTH = {"exact": 3, "synonym": 3, "number": 3, "inflected": 3, "dropped": 1, "fuzzy": 2, "unknown": 0}
# Case and plural endings tried when a word is not found as it is (longest first): বাবার -> বাবা, বাড়িতে -> বাড়ি.
NOUN_ENDINGS = ("গুলোকে", "গুলো", "গুলি", "দেরকে", "দের", "টাকে", "টা", "টি", "কে", "তে", "রা", "ের", "য়ে", "য়", "র", "ে")
NUMBER_RE = re.compile(r"^([০-৯]+)(" + "|".join(cv.NUMBER_SUFFIXES) + r")?$")
EXTRA_SYNONYMS = {"মাথা ব্যথা": "মাথা ব্যাথা", "মাথাব্যথা": "মাথা ব্যাথা"}  # W376 is printed with ব্যাথা


def clean_text(text):
    """NFC form, no zero-width characters, punctuation and symbols turned into spaces (Bangla danda too), lower case."""
    text = normalize_word(text).replace(KHANDA_TA_OLD, "ৎ")
    kept = []
    for ch in text:
        kind = unicodedata.category(ch)
        if kind == "Cf":  # zero-width joiner and non-joiner: they change how a word looks, not which word it is
            continue
        kept.append(" " if kind[0] in "PS" else ch)
    return " ".join("".join(kept).lower().split())


def split_words(text):
    return clean_text(text).split()


@dataclass(frozen=True)
class Match:
    source: str  # the word or words as heard, after clean_text
    word: str  # the dictionary word (its first Bangla spelling), or UNKNOWN
    how: str  # exact | synonym | fuzzy | unknown
    score: float = 1.0  # similarity to the dictionary word, 0 to 1
    key: str = ""  # the word in the sign dictionary: W20, C001, or a number like ১০ (empty when unknown or dropped)
    category: str = ""  # core_vocab.CATEGORIES, "word" for the BdSLW words, "" when unknown


class WordMatcher:
    """Finds dictionary words in text.

    cutoff           fuzzy matches need at least this similarity
    min_fuzzy_chars  shorter words must match exactly: মা, চা and আম are one letter apart from other dictionary words
    margin           two different words this close to the best score make the match ambiguous, so it counts as unknown
    """

    def __init__(self, labels=LABELS, synonyms=None, cutoff=0.8, min_fuzzy_chars=4, margin=0.02, extended=False):
        """labels: (number, Bangla, romanized, English) like labels.py, or (key, Bangla, romanized, English, category).
        A name already taken keeps its first word. extended: also numbers, case endings and dropped words."""
        self.cutoff, self.min_fuzzy_chars, self.margin, self.extended = cutoff, min_fuzzy_chars, margin, extended
        self.exact, self.bangla, self.info = {}, {}, {}
        for row in labels:
            key, bangla, roman, english, category = (row if len(row) == 5 else (f"W{row[0]}", *row[1:], "word"))
            names = [clean_text(n) for n in split_variants(bangla)]
            display = self.info[key][0] if key in self.info else names[0]
            self.info.setdefault(key, (display, category))
            for name in names:
                self.bangla.setdefault(name, display)
            for name in (*names, *(clean_text(n) for n in split_variants(roman)), clean_text(english)):
                if name:
                    self.exact.setdefault(name, display)
        self.key_of = {}
        for key, (display, _) in self.info.items():
            self.key_of.setdefault(display, key)
        self.synonyms = {}
        for name, word in (SYNONYMS if synonyms is None else synonyms).items():
            word = clean_text(word)
            if word not in self.bangla:
                raise ValueError(f"synonym target {word!r} is not in the dictionary")
            self.synonyms.setdefault(clean_text(name), self.bangla[word])
        self.max_parts = max(len(n.split()) for n in (*self.exact, *self.synonyms))

    @classmethod
    def full(cls, extra=(), **kw):
        """The matcher for the core dictionary: BdSLW60, core vocabulary, BdSLW401, all synonyms and verb forms.
        extra: more (key, Bangla, romanized, English, category) rows, e.g. dictionary_words(dictionary, "B")."""
        return cls(labels=full_vocabulary(extra), synonyms=full_synonyms(), extended=True, **kw)

    def _make(self, phrase, display, how, score=1.0):
        key = self.key_of.get(display, "")
        return Match(phrase, display, how, score, key, self.info[key][1] if key else "")

    def _number(self, word):
        digits = cv.to_bangla_digits(word)
        for ending in sorted(cv.NUMBER_SUFFIXES, key=len, reverse=True) + [""]:
            stem = digits[: len(digits) - len(ending)] if ending else digits
            if (not ending or digits.endswith(ending)) and stem in cv.NUMBER_WORDS:
                num = cv.NUMBER_WORDS[stem]
                return Match(word, num, "number", 1.0, self.key_of.get(num, num), "number")
        m = NUMBER_RE.match(digits)
        if m:
            num = m.group(1)
            return Match(word, num, "number", 1.0, self.key_of.get(num, num), "number")
        return None

    def _special(self, word):
        """One word that is a number, a dropped word or a known word with a case ending, else None."""
        found = self._number(word)
        if found:
            return found
        if word in cv.DROPPED_WORDS:
            return Match(word, "", "dropped", 1.0, "", "")
        for ending in NOUN_ENDINGS:
            stem = word[: -len(ending)]
            if word.endswith(ending) and len(stem) >= 2:
                if stem in self.exact:
                    return self._make(word, self.exact[stem], "inflected")
                if stem in self.synonyms:
                    return self._make(word, self.synonyms[stem], "inflected")
        return None

    def _exact(self, phrase):
        # Speech recognition often splits one word in two (দুলা ভাই, টুথ পেস্ট), so the parts are also tried joined.
        for key in (phrase, phrase.replace(" ", "")):
            if key in self.exact:
                return self._make(phrase, self.exact[key], "exact")
            if key in self.synonyms:
                return self._make(phrase, self.synonyms[key], "synonym")
        return None

    def _fuzzy(self, phrase):
        if len(phrase) < self.min_fuzzy_chars:
            return None
        best = {}
        for key in {phrase, phrase.replace(" ", "")}:
            for name, word in self.bangla.items():
                if name.count(" ") == key.count(" "):
                    score = difflib.SequenceMatcher(None, key, name, autojunk=False).ratio()
                    best[word] = max(best.get(word, 0.0), score)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        if not ranked or ranked[0][1] < self.cutoff:
            return None
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < self.margin:
            return None
        return self._make(phrase, ranked[0][0], "fuzzy", round(ranked[0][1], 3))

    def _match_at(self, parts, i):
        """The longest match starting at parts[i], as (match, parts used). Exact matches always beat fuzzy ones."""
        lengths = range(min(self.max_parts, len(parts) - i), 0, -1)
        for n in lengths:
            found = self._exact(" ".join(parts[i:i + n]))
            if found:
                return found, n
        if self.extended:
            found = self._special(parts[i])
            if found:
                return found, 1
        for n in lengths:
            found = self._fuzzy(" ".join(parts[i:i + n]))
            if found:
                return found, n
        return Match(parts[i], UNKNOWN, "unknown", 0.0), 1

    def match(self, text):
        """Text -> one Match per word or multi-word dictionary entry, in spoken order."""
        parts, out, i = split_words(text), [], 0
        while i < len(parts):
            found, used = self._match_at(parts, i)
            out.append(found)
            i += used
        return out


def known_words(matches):
    return [m.word for m in matches if m.word != UNKNOWN and m.how != "dropped"]


def unknown_words(matches):
    return [m.source for m in matches if m.word == UNKNOWN]


def dictionary_words(dictionary, prefix):
    """(key, Bangla, '', '', category) rows for the dictionary words whose key starts with `prefix` (the verified
    BdSL-300 words have keys B###), for WordMatcher.full(extra=...)."""
    return [(k, "/".join(e.get("bangla", [])), "", "", e.get("category", "word"))
            for k, e in dictionary.words.items() if k.startswith(prefix) and e.get("bangla")]


def full_vocabulary(extra=()):
    """(key, Bangla, romanized, English, category) for BdSLW60, the core vocabulary and BdSLW401 (+ `extra` rows).
    English and romanized names are kept only for BdSLW60 (as before); for the others short English words would match
    too easily."""
    rows = [(f"W{n}", b, r, e, "word") for n, b, r, e in LABELS]
    rows += [(key, b, "", "", cat) for key, b, _e, cat in cv.CORE_WORDS]
    rows += [(f"W{n}", b, "", "", "word") for n, b, _r, _e in LABELS_401]  # BdSLW60 numbers: extra spellings only
    rows += list(extra)
    return rows


def full_synonyms():
    table = dict(SYNONYMS)
    table.update(EXTRA_SYNONYMS)
    table.update(cv.CORE_SYNONYMS)
    for form, base in cv.verb_forms().items():
        table.setdefault(form, base)
    return table


def best_candidate(candidates, matcher):
    """Of several transcriptions of one recording, the one that fits the dictionary best: its best match counts most
    (exact or synonym over fuzzy), and each unknown word counts against it. The first one wins a tie."""
    def fit(text):
        matches = matcher.match(text)
        strength = max((_STRENGTH[m.how] for m in matches), default=-1)
        return strength - 0.5 * len(unknown_words(matches))
    return max(candidates, key=fit)
