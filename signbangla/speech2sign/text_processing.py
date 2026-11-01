"""Recognized or typed text -> dictionary words: Unicode cleanup, punctuation removal, synonyms, fuzzy matching.

Words the sign dictionary cannot show come back with word == UNKNOWN, so the caller can tell the user about them.
"""
import difflib
import unicodedata
from dataclasses import dataclass

from ..labels import LABELS
from .dictionary import normalize_word, split_variants

UNKNOWN = "unknown"

# Everyday word -> the word in the sign dictionary. Hyphens and dots become spaces in clean_text, so "টি-শার্ট" and
# "টি.ভি." arrive here as two words.
SYNONYMS = {
    "আব্বু": "বাবা", "আব্বা": "বাবা", "আম্মা": "মা", "আম্মু": "মা", "ডাক্তার": "ডক্টর",
    "বৌ": "বউ", "আঙুর": "আঙ্গুর", "টি শার্ট": "টিশার্ট", "টি ভি": "টিভি", "এ সি": "এসি",
}
KHANDA_TA_OLD = "ত্‍"  # old spelling of ৎ: ta + hasant + zero-width joiner
_STRENGTH = {"exact": 3, "synonym": 3, "fuzzy": 2, "unknown": 0}


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


class WordMatcher:
    """Finds dictionary words in text.

    cutoff           fuzzy matches need at least this similarity
    min_fuzzy_chars  shorter words must match exactly: মা, চা and আম are one letter apart from other dictionary words
    margin           two different words this close to the best score make the match ambiguous, so it counts as unknown
    """

    def __init__(self, labels=LABELS, synonyms=None, cutoff=0.8, min_fuzzy_chars=4, margin=0.02):
        self.cutoff, self.min_fuzzy_chars, self.margin = cutoff, min_fuzzy_chars, margin
        self.exact, self.bangla = {}, {}
        for _, bangla, roman, english in labels:
            names = [clean_text(n) for n in split_variants(bangla)]
            for name in names:
                self.bangla[name] = names[0]
            for name in (*names, *(clean_text(n) for n in split_variants(roman)), clean_text(english)):
                self.exact[name] = names[0]
        self.synonyms = {}
        for name, word in (SYNONYMS if synonyms is None else synonyms).items():
            word = clean_text(word)
            if word not in self.bangla:
                raise ValueError(f"synonym target {word!r} is not in the dictionary")
            self.synonyms[clean_text(name)] = self.bangla[word]
        self.max_parts = max(len(n.split()) for n in (*self.exact, *self.synonyms))

    def _exact(self, phrase):
        # Speech recognition often splits one word in two (দুলা ভাই, টুথ পেস্ট), so the parts are also tried joined.
        for key in (phrase, phrase.replace(" ", "")):
            if key in self.exact:
                return Match(phrase, self.exact[key], "exact")
            if key in self.synonyms:
                return Match(phrase, self.synonyms[key], "synonym")
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
        return Match(phrase, ranked[0][0], "fuzzy", round(ranked[0][1], 3))

    def _match_at(self, parts, i):
        """The longest match starting at parts[i], as (match, parts used). Exact matches always beat fuzzy ones."""
        lengths = range(min(self.max_parts, len(parts) - i), 0, -1)
        for find in (self._exact, self._fuzzy):
            for n in lengths:
                found = find(" ".join(parts[i:i + n]))
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
    return [m.word for m in matches if m.word != UNKNOWN]


def unknown_words(matches):
    return [m.source for m in matches if m.word == UNKNOWN]


def best_candidate(candidates, matcher):
    """Of several transcriptions of one recording, the one that fits the dictionary best: its best match counts most
    (exact or synonym over fuzzy), and each unknown word counts against it. The first one wins a tie."""
    def fit(text):
        matches = matcher.match(text)
        strength = max((_STRENGTH[m.how] for m in matches), default=-1)
        return strength - 0.5 * len(unknown_words(matches))
    return max(candidates, key=fit)
