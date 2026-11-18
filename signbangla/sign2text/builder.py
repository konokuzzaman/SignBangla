"""Rule-based sentence builder: recognised BdSL sign words (dictionary forms, BdSL order) -> a Bangla sentence.

    sentence, trace, unknown = build_sentence(["আমি", "পানি", "চাওয়া"])     # "আমি পানি চাই।"

The rules run in this order; each one only changes what it is about and records it in the trace:
    a. normalise words and map spellings / synonyms / verb forms to the lexicon word (unknown words pass through)
    b. subject: the first pronoun or person noun (a pronoun right before a kin noun is its owner); default 1st person
    c. tense from time words (গতকাল past, আগামীকাল / পরে future, এখন present continuous, else simple present)
    d. verb agreement: the verb's form for the subject's person and the tense (verbs.csv)
    e. a place before যাওয়া / আসা / থাকা takes its locative form (স্কুল -> স্কুলে, ডাক্তার -> ডাক্তারের কাছে)
    f. possession: আমি মা -> আমার মা; with no verb আমি মাথা ব্যথা -> আমার মাথা ব্যথা; আমি টাকা নেই -> আমার টাকা নেই
    g. sentences without a verb: predicate words (জ্বর -> জ্বর হয়েছে) and state words with the copula (ভালো আছি)
    h. negation after the verb (চাই না, past: যাইনি); নেই for "there is not" and for the present copula (ভালো নেই)
    i. questions: question words before the verb, "?" at the end; is_question=True for yes/no questions
    j. greetings and fixed phrases stay as they are
    k. statements end with "।"
Words are never invented: the only words added are verb endings, copula forms and predicate phrases from the tables.
"""
from dataclasses import dataclass, field
from pathlib import Path

from ..utils import ROOT, load_config
from .lexicon import Lexicon, norm, write_missing
from .tables import COPULA

QUESTION_MARK, FULL_STOP = "?", "।"


@dataclass(eq=False)  # tokens are compared by identity: two equal words stay two tokens
class Token:
    text: str  # what is written in the sentence
    word: str = ""  # the lexicon word ("" for an unknown word)
    pos: str = ""
    semantic: str = ""
    person: str = ""
    possessive: str = ""
    locative: str = ""
    inserted: bool = False  # added by a rule (copula, না)
    tags: set = field(default_factory=set)


def module3_paths(cfg, root=ROOT):
    m = cfg["module3"]
    return {k: root / m[k] for k in ("lexicon", "verbs", "predicates", "test_sentences")}


def lexicon_sources(cfg, root=ROOT):
    """Where the lexicon's words come from (paths already in the config)."""
    from ..utils import work_dir
    s2s = cfg["speech2sign"]
    return dict(core_word_list=root / s2s["core_vocab"]["word_list"],
                core_coverage=work_dir(cfg) / s2s["core_vocab"]["dictionary_dir"] / "core_coverage.csv",
                word_list_401=root / cfg["bdslw401"]["root"] / "word_list.csv",
                overlap=work_dir(cfg) / s2s["bdsl300"]["dir"] / "overlap.csv")


class SentenceBuilder:
    def __init__(self, lexicon, settings):
        self.lx = lexicon
        self.s = settings
        self.past_words = {norm(w) for w in settings["past_words"]}
        self.future_words = {norm(w) for w in settings["future_words"]}
        self.continuous_words = {norm(w) for w in settings["continuous_words"]}
        self.motion_verbs = {norm(w) for w in settings["motion_verbs"]}
        self.possessive_subject_words = {norm(w) for w in settings["possessive_subject_words"]}

    @classmethod
    def from_config(cls, cfg=None, root=ROOT, write=True):
        """From configs/config.yaml -> module3. write: create missing data files from tables.py first."""
        cfg = cfg or load_config()
        paths = module3_paths(cfg, root)
        if write:
            write_missing(paths, lexicon_sources(cfg, root))
        return cls(Lexicon.from_files(paths), cfg["module3"])

    # ------------------------------------------------------------ a. normalise
    def normalise(self, words, trace, unknown):
        raw = norm(" ".join(words) if isinstance(words, (list, tuple)) else str(words)).split()
        tokens, i = [], 0
        while i < len(raw):
            for n in range(min(self.lx.max_words, len(raw) - i), 0, -1):
                phrase = " ".join(raw[i:i + n])
                hit = self.lx.surface.get(phrase)
                if hit:
                    word, how = hit
                    e = self.lx.get(word)
                    tokens.append(Token(word, word, e["pos"], e["semantic"], e["person"], e["possessive"],
                                        e["locative"]))
                    if phrase != word:
                        trace.append(f"a. {phrase} -> {word} ({how})")
                    i += n
                    break
            else:
                tokens.append(Token(raw[i]))
                unknown.append(raw[i])
                trace.append(f"a. unknown word {raw[i]}: kept as it is")
                i += 1
        return tokens

    # ------------------------------------------------------------ b. subject
    def subject(self, tokens, trace):
        """-> (subject token or None, owner token or None, person)."""
        for i, t in enumerate(tokens):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if t.pos == "pronoun":
                if nxt is not None and nxt.semantic == "kin":
                    trace.append(f"b. subject: {nxt.word} (person 3); {t.word} is its owner")
                    return nxt, t, "3"
                if t.semantic != "possessive":
                    trace.append(f"b. subject: {t.word} (person {t.person})")
                    return t, None, t.person
            elif t.pos == "noun" and t.semantic in ("kin", "person"):
                trace.append(f"b. subject: {t.word} (person 3)")
                return t, None, "3"
        for t in tokens:
            if t.word == norm("কে"):
                trace.append("b. subject: কে (person 3)")
                return t, None, "3"
        trace.append("b. no subject: 1st person assumed")
        return None, None, "1"

    # ------------------------------------------------------------ c. tense
    def tense(self, tokens, trace):
        for t in tokens:
            for words, tense in ((self.past_words, "past"), (self.future_words, "future"),
                                 (self.continuous_words, "present_continuous")):
                if t.word in words:
                    trace.append(f"c. tense: {tense} (from {t.word})")
                    return tense
        trace.append("c. tense: present (no time word)")
        return "present_simple"

    def _tense_column(self, tense):
        return self.s["past_tense"] if tense == "past" else tense

    # ------------------------------------------------------------ d. verb agreement
    def main_verb(self, tokens, trace):
        verbs = [t for t in tokens if t.pos == "verb"]
        for t in verbs[:-1]:
            trace.append(f"d. second verb {t.word}: left in its dictionary form")
        if verbs and verbs[-1].word not in self.lx.verbs:
            trace.append(f"d. no forms for {verbs[-1].word} in verbs.csv: left as it is")
            return None
        return verbs[-1] if verbs else None

    def agree(self, verb, person, tense, trace):
        form = self.lx.verbs[verb.word][person][self._tense_column(tense)]
        trace.append(f"d. verb {verb.word} -> {form} (person {person}, {self._tense_column(tense)})")
        verb.text = form

    # ------------------------------------------------------------ e. places
    def places(self, tokens, verb, skip, trace):
        if verb is None or verb.word not in self.motion_verbs:
            return
        for t in reversed(tokens[:tokens.index(verb)]):
            if t.pos == "noun" and t.locative and t not in skip:
                trace.append(f"e. place before {verb.word}: {t.word} -> {t.locative}")
                t.text = t.locative
                return

    # ------------------------------------------------------------ f. possession
    def possession(self, tokens, verb, subject, owner, trace):
        """-> the subject token after possession (a pronoun that became an owner is no longer the subject)."""
        def own(t, why):
            if t.pos == "pronoun" and t.possessive and "owner" not in t.tags:
                trace.append(f"f. possession ({why}): {t.word} -> {t.possessive}")
                t.text, t.tags = t.possessive, t.tags | {"owner"}
        if owner is not None:
            own(owner, f"before {subject.word}")
        if verb is None:
            for i, t in enumerate(tokens[:-1]):
                nxt = tokens[i + 1]
                if t.pos == "pronoun" and nxt.pos == "noun" and nxt.semantic != "person":
                    own(t, f"before {nxt.word}, no verb")
        if subject is not None and any(t.word in self.possessive_subject_words for t in tokens):
            own(subject, "with " + "/".join(sorted(t.word for t in tokens if t.word in self.possessive_subject_words)))
        if subject is not None and "owner" in subject.tags:
            return None
        return subject

    # ------------------------------------------------------------ g. sentences without a verb
    def predicates(self, tokens, person, tense, negations, trace):
        """-> the inserted copula token, if any."""
        for i, t in enumerate(tokens):
            p = self.lx.predicates.get(t.word)
            if p is None:
                continue
            column = {"past": "past", "future": "future"}.get(tense, "present")
            if p["kind"] == "fixed":
                if any(n.word == norm("নেই") for n in negations):
                    trace.append(f"g. {t.word} with নেই: no predicate phrase")
                    return None
                t.text = p[column]
                trace.append(f"g. predicate {t.word} -> {t.text}")
                return None
            form = self.lx.verbs[COPULA][person][self._tense_column(tense)]
            cop = Token(form, COPULA, "verb", inserted=True)
            tokens.insert(i + 1, cop)
            trace.append(f"g. state word {t.word}: copula {form} added (person {person})")
            return cop
        return None

    # ------------------------------------------------------------ h. negation
    def negation(self, tokens, negations, verb, copula, person, tense, trace):
        if not negations or len(tokens) == len(negations):
            return
        for n in negations:
            tokens.remove(n)
        n = negations[0]
        if verb is not None:
            if tense == "past":
                verb.text = self.lx.verbs[verb.word][person]["present_simple"] + "নি"
                trace.append(f"h. negation in the past: {verb.word} -> {verb.text}")
            else:
                tokens.insert(tokens.index(verb) + 1, Token("না", "না", "negation", inserted=True))
                trace.append(f"h. negation: না after {verb.text}")
        elif copula is not None:
            if tense in ("present_simple", "present_continuous"):
                trace.append(f"h. negation: copula {copula.text} -> নেই")
                copula.text = "নেই"
            else:
                tokens.insert(tokens.index(copula) + 1, Token("না", "না", "negation", inserted=True))
                trace.append(f"h. negation: না after {copula.text}")
        else:
            tokens.append(n)
            trace.append(f"h. negation: {n.word} at the end")

    # ------------------------------------------------------------ i. questions
    def question(self, tokens, verb, subject, is_question, trace):
        wh = [t for t in tokens if t.pos == "question"]
        for t in wh:
            if verb is not None and tokens.index(t) > tokens.index(verb):
                tokens.remove(t)
                tokens.insert(tokens.index(verb), t)
                trace.append(f"i. question word {t.word} moved before the verb")
            elif verb is None and t.word == norm("কত"):
                i = tokens.index(t)
                if i > 0 and tokens[i - 1].pos == "noun":
                    tokens.insert(i - 1, tokens.pop(i))
                    trace.append(f"i. কত moved before {tokens[i].word}")
        if wh or is_question:
            trace.append("i. question: ends with ?" + ("" if wh else " (yes/no, from is_question)"))
            if not wh and self.s.get("yes_no_particle") and subject is not None:
                tokens.insert(tokens.index(subject) + 1, Token("কি", "কি", "question", inserted=True))
            return True
        return False

    # ------------------------------------------------------------ the whole sentence
    def build(self, words, is_question=False):
        trace, unknown = [], []
        tokens = self.normalise(words, trace, unknown)
        if not tokens:
            return "", ["empty input"], []
        for t in tokens:
            if t.pos == "greeting":
                trace.append(f"j. fixed phrase {t.word}: kept as it is")
        if all(t.pos in ("greeting", "other") or not t.word for t in tokens) and not is_question:
            text = " ".join(t.text for t in tokens) + FULL_STOP
            trace.append("k. ends with ।")
            return text, trace, unknown
        subject, owner, person = self.subject(tokens, trace)
        tense = self.tense(tokens, trace)
        negations = [t for t in tokens if t.pos == "negation"]
        verb = self.main_verb(tokens, trace)
        if verb is not None:
            self.agree(verb, person, tense, trace)
        self.places(tokens, verb, {subject, owner}, trace)
        new_subject = self.possession(tokens, verb, subject, owner, trace)
        if subject is not None and new_subject is None:
            person = "3"
        copula = self.predicates(tokens, person, tense, negations, trace) if verb is None else None
        self.negation(tokens, negations, verb, copula, person, tense, trace)
        q = self.question(tokens, verb, new_subject, is_question, trace)
        if not q:
            trace.append("k. ends with ।")
        text = " ".join(t.text for t in tokens) + (QUESTION_MARK if q else FULL_STOP)
        return text, trace, unknown


_DEFAULT = None


def default_builder():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SentenceBuilder.from_config()
    return _DEFAULT


def build_sentence(words, is_question=False, builder=None):
    """words: sign words in dictionary form and BdSL order (a list, or one space-separated string).
    -> (sentence, trace, unknown_words)."""
    return (builder or default_builder()).build(words, is_question)


def build_from_candidates(candidates, is_question=False, scorer=None, builder=None):
    """candidates: one list per sign of (word, score) pairs, e.g. Module 1's top-5. For now the best-scoring word of each
    sign is used. scorer is the place for a language-model scorer later: it will get the candidate lists and pick the
    best word sequence; it is not used yet."""
    if scorer is not None:
        raise NotImplementedError("a language-model scorer is not available yet; call without scorer")
    words = [max(c, key=lambda ws: ws[1])[0] for c in candidates if c]
    return build_sentence(words, is_question, builder)
