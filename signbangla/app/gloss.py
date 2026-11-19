"""Text -> sign words (gloss) for Speech -> Sign.

Steps:
  1. phrase verbs   "<noun> করা" counts as one verb in any form of করা (পান করেছি -> পান করা).
  2. synonyms       app.gloss.synonyms, used only when the spoken word has no sign and the other word has one
                    (পান করা -> খাওয়া).
  3. matching       speech2sign/text_processing.py (cleanup, synonyms, verb forms, case endings, numbers, fuzzy).
  4. dropping       words BdSL does not sign: the copula (done by the matcher) and app.gloss.drop_words without a sign.
  5. order          spoken Bangla order (subject object verb, negation after the verb) with time words moved to the
                    front and question words to the end (app.gloss.time_first / question_last).
  6. tense markers  a verb form that BdSL shows with a marker sign gets that sign after the verb (the reverse of
                    Module 3's corpus rules): খাইনি -> খাওয়া নেই, খেয়েছিল -> খাওয়া ছিল, খেয়েছে -> খাওয়া হয়েছে.
                    A marker is used only when all its signs are verified; otherwise the plain verb is signed (খাবে ->
                    খাওয়া while হবে is unverified; a time word in the sentence still carries the time). A negation
                    marker without a sign stays in the caption in red, so the meaning is never lost silently.
Only signs that are verified in the dictionary (or come from BdSLW60 / BdSLW401) are used; any other word stays in the
caption as text without a sign.
"""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..speech2sign import core_vocab as cv
from ..speech2sign.nonmanual import sentence_markers
from ..speech2sign.text_processing import UNKNOWN, Match, WordMatcher, clean_text, dictionary_words, split_words

TRUSTED_SOURCES = ("BdSLW60", "BdSLW401")  # dataset words: always usable
EXTRA_KEY_PREFIXES = ("B", "S", "D")  # dictionary keys of words that are not in the built-in word lists
NO_SIGN_COLOR = (40, 40, 200)  # BGR red, the caption colour of a word without a sign
PREDICTED_COLOR = (0, 130, 235)  # BGR orange: a word predicted from the context (≈)


TENSE_COLUMNS = ("present_simple", "present_continuous", "past_simple", "past_perfect", "future", "present_perfect",
                 "past_habitual")
NI = "নি"  # খাইনি = খাই + নি


class TenseMarkers:
    """Verb form -> (verb, tense, negation) from Module 3's verb tables, and (tense, negation) -> marker signs from the
    corpus rules (the most frequent marker of each outcome, with আগে for habitual and হবে for 'will')."""

    def __init__(self, verb_rows, rules):
        self.forms = {}
        for r in verb_rows:
            for col in TENSE_COLUMNS:
                form = clean_text(r.get(col, "") or "")
                if form and " " not in form:
                    self.forms.setdefault(form, (r["verb"], col, ""))
                    if col == "present_simple":
                        self.forms.setdefault(form + NI, (r["verb"], col, "ni"))
        best = {}
        for m in rules.get("markers", []):
            k = (m["tense"], m["neg"])
            if k not in best or m["support"] > best[k]["support"]:
                best[k] = m
        self.rules = {k: (["আগে"] if m["habitual"] else []) + (["হবে"] if m["will"] else []) + [m["marker"]]
                      for k, m in best.items()}
        # a separate না after the verb is its own word already: only fused forms need a marker here
        self.rules = {k: v for k, v in self.rules.items() if k[1] != "na"}

    @classmethod
    def from_config(cls, cfg, root):
        m3 = cfg["module3"]
        rows = []
        for key in ("verbs", "verb_tenses"):
            path = Path(root) / m3[key]
            if path.exists():
                with open(path, encoding="utf-8-sig", newline="") as f:
                    rows += list(csv.DictReader(f))
        path = Path(root) / m3["corpus"]["rules"]
        rules = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return cls(rows, rules)

    def lookup(self, word):
        """-> (verb, tense, neg, marker words) or None when the form needs no marker."""
        found = self.forms.get(clean_text(word))
        if not found:
            return None
        verb, tense, neg = found
        words = self.rules.get((tense, neg))
        return (verb, tense, neg, words) if words else None


@dataclass
class Token:
    text: str  # as heard or typed (after cleanup)
    word: str = ""  # the dictionary word, "" when unknown
    key: str = ""  # the sign to play, "" when the word has no usable sign
    status: str = "unknown"  # sign | synonym | marker | predicted | no_sign | unknown | dropped
    category: str = ""
    note: str = ""  # how a synonym, marker or prediction was chosen
    alternatives: list = field(default_factory=list)  # a prediction's next best words

    @property
    def signed(self):
        return bool(self.key)

    @property
    def caption(self):
        if self.status == "predicted":
            return "≈" + self.word
        return self.word if self.signed and self.word else self.text

    def as_dict(self):
        return dict(text=self.text, word=self.word, key=self.key, status=self.status, category=self.category,
                    note=self.note, caption=self.caption, alternatives=self.alternatives)


@dataclass
class Gloss:
    text: str
    tokens: list  # in signing order, without dropped words
    dropped: list = field(default_factory=list)
    question: bool = False
    negation: bool = False

    @property
    def keys(self):
        return [t.key for t in self.tokens if t.signed]

    @property
    def unknown(self):
        return [t.text for t in self.tokens if not t.signed]

    def layout(self):
        """(captions, BGR colours (None = signed), caption index of each sign) as stitch.Sentence.captions expects."""
        captions = [t.caption for t in self.tokens]
        colors = [PREDICTED_COLOR if t.status == "predicted" else None if t.signed else NO_SIGN_COLOR
                  for t in self.tokens]
        positions = [i for i, t in enumerate(self.tokens) if t.signed]
        return captions, colors, positions

    def as_dict(self):
        return dict(text=self.text, tokens=[t.as_dict() for t in self.tokens], dropped=[t.as_dict() for t in self.dropped],
                    signs=[t.caption for t in self.tokens if t.signed], unknown=self.unknown, question=self.question,
                    negation=self.negation)


class Glosser:
    def __init__(self, dictionary, gcfg, matching=None, markers=None):
        m = matching or {}
        fuzzy = dict(cutoff=m.get("fuzzy_cutoff", 0.8), min_fuzzy_chars=m.get("min_fuzzy_chars", 4),
                     margin=m.get("fuzzy_margin", 0.02))
        self.dictionary = dictionary
        # + the verified words with their own keys: BdSL-300 (B###), Ban-Sign-Sent-9K (S###) and DU signs and phrases
        # (D###); the matcher tries the longest entry first, so a phrase is found as a whole
        self.matcher = WordMatcher.full(extra=dictionary_words(dictionary, EXTRA_KEY_PREFIXES), **fuzzy)
        self.drop = {clean_text(w) for w in gcfg.get("drop_words", [])}
        self.always_drop = {clean_text(w) for w in gcfg.get("particles", [])}  # dropped even when a sign exists
        self.time_first = gcfg.get("time_first", True)
        self.question_last = gcfg.get("question_last", True)
        self.phrase_verbs = [clean_text(p).split() for p in gcfg.get("phrase_verbs", [])]
        self.synonyms = {clean_text(a): clean_text(b) for a, b in gcfg.get("synonyms", {}).items()}
        self.markers = markers if gcfg.get("tense_markers", True) else None
        self.predictor = None  # set by the pipeline: context prediction for words without a sign

    # ---- which signs may be used
    def usable(self, key):
        """A dictionary key or number -> the key when its sign may be used, else ""."""
        if not key:
            return ""
        try:
            found = self.dictionary.find(key)
        except KeyError:
            return key if self.dictionary.number_digits(key) else ""
        e = self.dictionary.words[found]
        return found if e.get("verified", False) or e.get("source") in TRUSTED_SOURCES else ""

    def sign_of(self, match):
        if match.word == UNKNOWN or match.how == "dropped":
            return ""
        return self.usable(match.key) or self.usable(match.word)

    def phrase_sign(self, phrase):
        """The sign of a whole phrase looked up as it is (no splitting into words)."""
        try:
            return self.usable(self.dictionary.find(phrase))
        except KeyError:
            return ""

    # ---- steps
    def _phrase_at(self, parts, i):
        """(canonical phrase, parts used) when a phrase verb or a synonym source starts at parts[i], else None."""
        for p in self.phrase_verbs:
            n = len(p)
            if parts[i:i + n - 1] == p[:-1] and i + n - 1 < len(parts) and parts[i + n - 1].startswith("কর"):
                return " ".join(p), n
        for source in sorted(self.synonyms, key=lambda s: -len(s.split())):
            n = len(source.split())
            if " ".join(parts[i:i + n]) == source:
                return source, n
        return None

    def _token(self, match):
        key = self.sign_of(match)
        if match.how == "dropped":
            return Token(match.source, "", "", "dropped")
        if match.word == UNKNOWN:
            return Token(match.source, "", "", "unknown")
        return Token(match.source, match.word, key, "sign" if key else "no_sign", match.category)

    def _unswallow(self, match):
        """Two checks on fuzzy matches, so a near-spelling never shows a wrong sign or loses a word:
        - a fuzzy match over several words (দাদি চা -> দাদা) is undone when one of its words is a known word on its
          own; the words are then matched one by one;
        - a fuzzy match whose first letter differs (রান্না -> কান্না, পারা -> আপনারা) counts as unknown: Bangla words
          that differ in their first letter are almost always different words."""
        if match.how != "fuzzy":
            return [match]
        if " " in match.source:
            singles = [self.matcher.match(w) for w in match.source.split()]
            if any(len(s) == 1 and s[0].how in ("exact", "synonym", "inflected", "number") for s in singles):
                out = []
                for s in singles:
                    for m in s:
                        out.extend(self._unswallow(m))
                return out
        if clean_text(match.source)[:1] != clean_text(match.word)[:1]:
            return [Match(match.source, UNKNOWN, "unknown", 0.0)]
        return [match]

    def _marker_tokens(self, word):
        """A verb form with a tense / negation marker -> [verb token, marker tokens], or None."""
        if self.markers is None or self.phrase_sign(word):
            return None  # a word with its own sign (ছিল, হয়েছে, ...) is that sign
        found = self.markers.lookup(word)
        if not found:
            return None
        verb, tense, neg, words = found
        m = self.matcher.match(verb)
        if len(m) != 1 or m[0].how not in ("exact", "synonym") or not self.sign_of(m[0]):
            # the verb has no sign (পারিনি -> পারা): shown in red, but a negation marker is still kept
            if neg != "ni":
                return None
            marks = [Token(w, w, self.phrase_sign(w), "marker" if self.phrase_sign(w) else "no_sign", "negation")
                     for w in words if w in cv.NEGATION_WORDS]
            return [Token(word, verb, "", "no_sign", "verb", f"{word} = {verb} + নেই; {verb} has no sign")] + marks
        marks = []
        for w in words:
            mm = self.matcher.match(w)
            key = self.sign_of(mm[0]) if len(mm) == 1 else ""
            marks.append(Token(w, mm[0].word if mm and mm[0].word != UNKNOWN else w, key,
                               "marker" if key else "no_sign", "negation" if w in cv.NEGATION_WORDS else "tense"))
        usable = all(t.signed for t in marks)
        note = f"{word} = {verb} + {' + '.join(words)} ({tense}{', negative' if neg else ''})"
        verb_token = Token(word, m[0].word, self.sign_of(m[0]), "sign", m[0].category,
                           note if usable else note + "; a marker has no verified sign, so the plain verb")
        if usable:
            return [verb_token] + marks
        # never lose a negation: an unsigned না / নেই stays in the caption (red)
        return [verb_token] + [t for t in marks if t.category == "negation"]

    def _phrase_token(self, phrase, text):
        own = self.phrase_sign(phrase)
        if own:
            return Token(text, phrase, own, "sign")
        target = self.synonyms.get(phrase)
        if target:
            m = self.matcher.match(target)
            if len(m) == 1 and self.sign_of(m[0]):
                return Token(text, m[0].word, self.sign_of(m[0]), "synonym", m[0].category, f"{phrase} -> {m[0].word}")
        if " " in phrase:  # a phrase verb without a sign: shown as one word
            return Token(text, phrase, "", "no_sign", "verb")
        return None

    def gloss(self, text):
        parts = split_words(text)
        tokens, chunk, pre_dropped, i = [], [], [], 0

        def flush():
            if chunk:
                for m in self.matcher.match(" ".join(chunk)):
                    tokens.extend(self._token(x) for x in self._unswallow(m))
                chunk.clear()

        while i < len(parts):
            if parts[i] in self.always_drop or (parts[i] in self.drop and not self.phrase_sign(parts[i])):
                # dropped before matching, so a short filler word cannot join its neighbour in a fuzzy match
                flush()
                pre_dropped.append(Token(parts[i], "", "", "dropped"))
                i += 1
                continue
            marked = self._marker_tokens(parts[i])
            if marked:
                flush()
                tokens.extend(marked)
                i += 1
                continue
            found = self._phrase_at(parts, i)
            if found:
                phrase, n = found
                tok = self._phrase_token(phrase, " ".join(parts[i:i + n]))
                if tok is not None:
                    flush()
                    tokens.append(tok)
                    i += n
                    continue
            chunk.append(parts[i])
            i += 1
        flush()

        kept, dropped = [], list(pre_dropped)
        for t in tokens:
            if t.status == "dropped" or (not t.signed and clean_text(t.text) in self.drop):
                dropped.append(Token(t.text, t.word, "", "dropped", t.category))
            else:
                kept.append(t)
        kept = self.reorder(kept)
        if self.predictor is not None:
            self.predictor.fill_speech_gaps(kept)
        markers = sentence_markers(text, self.matcher.match(text))
        negation = markers.negation or any(t.word in cv.NEGATION_WORDS for t in kept)
        return Gloss(text, kept, dropped, markers.question, negation)

    def reorder(self, tokens):
        def place(t):
            if self.time_first and t.category == "time":
                return 0
            if self.question_last and t.category == "question":
                return 2
            return 1
        return sorted(tokens, key=place)  # stable: the spoken order stays inside each part
