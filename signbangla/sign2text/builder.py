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

With rules counted from sentence pairs (corpus_rules.py, module3.corpus in the config; used when the file exists):
    a. two or more kin signs said as one word (চাচা বউ -> চাচী); time-of-day words before a verb take the locative
       (সকাল -> সকালে; this one needs no corpus rules)
    c. a final marker sets tense and negation: হবে future, ছিল past perfect, হয়েছে present perfect, নেই past
       negative (খায়নি), না present negative, আগে ... ছিল / না habitual past (খেতেন / খেতেন না); a plain না
       with a time word (গতকাল) keeps rule h
    d. no verb sign but a marker: the verb is taken from the object (চা -> খাওয়া); a kin subject takes the person
       form most common for it with that tense (বাবা ... খান না, বাবা ... খায়নি)
    h. the marker signs themselves (ছিল, হবে, আগে, না, নেই) are not written; না / নি come back as the verb ending
"""
from dataclasses import dataclass, field
from pathlib import Path

from ..utils import ROOT, load_config
from . import corpus_rules as CR
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
    return {k: root / m[k] for k in ("lexicon", "verbs", "predicates", "test_sentences", "lexicon_extra", "verb_tenses")
            if k in m}


def corpus_rules_path(cfg, root=ROOT):
    c = cfg["module3"].get("corpus") or {}
    return root / c["rules"] if c.get("rules") else None


def lexicon_sources(cfg, root=ROOT):
    """Where the lexicon's words come from (paths already in the config)."""
    from ..utils import work_dir
    s2s = cfg["speech2sign"]
    return dict(core_word_list=root / s2s["core_vocab"]["word_list"],
                core_coverage=work_dir(cfg) / s2s["core_vocab"]["dictionary_dir"] / "core_coverage.csv",
                word_list_401=root / cfg["bdslw401"]["root"] / "word_list.csv",
                overlap=work_dir(cfg) / s2s["bdsl300"]["dir"] / "overlap.csv")


class SentenceBuilder:
    def __init__(self, lexicon, settings, corpus=None):
        """corpus: a corpus_rules.CorpusRules, or None for the hand-written rules only."""
        self.lx = lexicon
        self.s = settings
        self.corpus = corpus
        self.past_words = {norm(w) for w in settings["past_words"]}
        self.future_words = {norm(w) for w in settings["future_words"]}
        self.continuous_words = {norm(w) for w in settings["continuous_words"]}
        self.motion_verbs = {norm(w) for w in settings["motion_verbs"]}
        self.possessive_subject_words = {norm(w) for w in settings["possessive_subject_words"]}

    @classmethod
    def from_config(cls, cfg=None, root=ROOT, write=True, corpus=True):
        """From configs/config.yaml -> module3. write: create missing data files from tables.py first. corpus: use the
        rules in module3.corpus.rules when module3.corpus.use is true and the file exists."""
        cfg = cfg or load_config()
        paths = module3_paths(cfg, root)
        if write:
            write_missing(paths, lexicon_sources(cfg, root))
        rules = None
        path = corpus_rules_path(cfg, root)
        if corpus and (cfg["module3"].get("corpus") or {}).get("use") and path is not None and path.exists():
            rules = CR.CorpusRules(CR.load(path))
        return cls(Lexicon.from_files(paths), cfg["module3"], rules)

    def _token(self, word):
        e = self.lx.get(word)
        return Token(self.lx.written.get(word, word), word, e["pos"], e["semantic"], e["person"], e["possessive"],
                     e["locative"])

    # ------------------------------------------------------------ a. normalise
    def apply_compounds(self, tokens, trace, compounds=None):
        """Two or more signs that the corpus rules say are one word (চাচা বউ -> চাচী). compounds: {(words): rule};
        default the builder's corpus rules."""
        comp = compounds if compounds is not None else (self.corpus.compounds if self.corpus else {})
        if not comp:
            return tokens
        longest = max(len(k) for k in comp)
        out, i = [], 0
        while i < len(tokens):
            for n in range(min(longest, len(tokens) - i), 1, -1):
                key = tuple(t.word for t in tokens[i:i + n])
                rule = comp.get(key)
                if rule and self.lx.get(rule["word"]):
                    out.append(self._token(rule["word"]))
                    trace.append(f"a. compound {' '.join(key)} -> {rule['word']} "
                                 f"(train {rule['support']}/{rule['of']})")
                    i += n
                    break
            else:
                out.append(tokens[i])
                i += 1
        return out

    def time_locatives(self, tokens, verb, trace):
        """Time-of-day words in a sentence with a verb take their locative form (সকাল -> সকালে)."""
        if verb is None:
            return
        for t in tokens:
            if t.semantic == "time_of_day" and t.locative and t.text == t.word:
                rule = self.corpus.time_locative.get(t.word) if self.corpus else None
                t.text = t.locative
                trace.append(f"e. time of day: {t.word} -> {t.locative}"
                             + (f" (train {rule['support']}/{rule['of']})" if rule else ""))

    def normalise(self, words, trace, unknown):
        raw = norm(" ".join(words) if isinstance(words, (list, tuple)) else str(words)).split()
        tokens, i = [], 0
        while i < len(raw):
            for n in range(min(self.lx.max_words, len(raw) - i), 0, -1):
                phrase = " ".join(raw[i:i + n])
                hit = self.lx.surface.get(phrase)
                if hit:
                    word, how = hit
                    tokens.append(self._token(word))
                    if phrase != word:
                        trace.append(f"a. {phrase} -> {word} ({how})")
                    i += n
                    break
            else:
                tokens.append(Token(raw[i]))
                unknown.append(raw[i])
                trace.append(f"a. unknown word {raw[i]}: kept as it is")
                i += 1
        return self.apply_compounds(tokens, trace) if self.corpus else tokens

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

    # ------------------------------------------------------------ corpus rules: markers, dropped verb, person
    def corpus_marker(self, tokens, tense):
        """-> (key, rule) for a sentence ending in a marker the corpus rules know, else None. A plain final না with a
        time word that already sets the tense is left to rule h."""
        if not self.corpus:
            return None
        key = CR.marker_key(tokens)
        rule = self.corpus.markers.get(key) if key else None
        if rule is None:
            return None
        if key == (norm("না"), False, False) and tense != "present_simple":
            return None
        return key, rule

    def infer_verb(self, tokens, trace):
        """No verb sign: the verb that goes with the object (by word, else by its semantic), placed after it."""
        if any(self.lx.predicates.get(t.word) for t in tokens):
            return None
        for obj in reversed([t for t in tokens if t.pos == "noun" and t.semantic not in ("kin", "person")]):
            rule, how = self.corpus.verb_by_object.get(obj.word), obj.word
            if rule is None and obj.semantic:
                rule, how = self.corpus.verb_by_semantic.get(obj.semantic), f"{obj.semantic} words"
            if rule and rule["verb"] in self.lx.verbs and self.lx.get(rule["verb"]):
                verb = self._token(rule["verb"])
                verb.inserted = True
                tokens.insert(tokens.index(obj) + 1, verb)
                trace.append(f"d. no verb sign: {rule['verb']} added after {obj.word} (verb for {how}; train "
                             f"{rule['support']}/{rule['of']})")
                return verb
        return None

    def corpus_person(self, subject, person, column, neg, trace):
        """The person column for a kin subject: the most common one in train for (subject, tense, negation), else for
        (tense, negation) over all kin subjects; otherwise unchanged."""
        if not self.corpus or subject is None or subject.semantic != "kin":
            return person
        rule = self.corpus.person_by_subject.get((subject.word, column, neg))
        how = subject.word
        if rule is None:
            rule, how = self.corpus.person_by_tense.get((column, neg)), "kin subjects"
        if rule is None or rule["person"] == person:
            return person
        trace.append(f"d. person for {subject.word} with {column}{'+' + neg if neg else ''}: {rule['person']} "
                     f"(most common for {how}; train {rule['support']}/{rule['of']})")
        return rule["person"]

    def apply_marker(self, tokens, verb, subject, person, key, rule, trace, at=None):
        """Tense and negation from the marker; the marker signs are taken out. -> False if the verb has no such form.
        at: where in the trace the marker line goes (before the line of an added verb)."""
        column, neg = rule["tense"], rule["neg"]
        forms = self.lx.verbs.get(verb.word, {})
        person_trace = []
        person = self.corpus_person(subject, person, column, neg, person_trace)
        base = forms.get(person, {}).get("present_simple" if neg == CR.NEG_NI else column)
        if not base:
            return False
        trace.insert(len(trace) if at is None else at, f"c. marker {CR.describe_marker(key)}: {column}"
                     + {CR.NEG_NA: " + না", CR.NEG_NI: " negative (+নি)"}.get(neg, "")
                     + f" (train {rule['support']}/{rule['of']})")
        trace.extend(person_trace)
        verb.text = base + "নি" if neg == CR.NEG_NI else base
        trace.append(f"d. verb {verb.word} -> {verb.text} (person {person}, {column})")
        removed = [tokens[-1]]
        if key[2]:
            removed.append(tokens[-2])
        removed += [t for t in tokens[:-1] if t.word == norm("আগে")]
        for t in removed:
            tokens.remove(t)
        trace.append("h. marker signs not written: " + ", ".join(t.word for t in removed))
        if neg == CR.NEG_NA:
            tokens.insert(tokens.index(verb) + 1, Token("না", "না", "negation", inserted=True))
            trace.append(f"h. negation: না after {verb.text}")
        return True

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
        marker = self.corpus_marker(tokens, tense)
        at = len(trace)
        if verb is None and marker is not None:
            verb = self.infer_verb(tokens, trace)
        if verb is not None and marker is not None and self.apply_marker(tokens, verb, subject, person, *marker,
                                                                         trace, at):
            negations = []
        elif verb is not None:
            column = self._tense_column(tense)
            neg = (CR.NEG_NI if tense == "past" else CR.NEG_NA) if negations else CR.NEG_NONE
            person = self.corpus_person(subject, person, column, neg, trace)
            self.agree(verb, person, tense, trace)
        self.time_locatives(tokens, verb, trace)
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
