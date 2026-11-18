"""Rules counted from sentence pairs (sign words -> natural Bangla sentence), such as Ban-Sign-Sent-9K.

    rules = derive(train_rows, builder)        # builder: a SentenceBuilder without corpus rules
    save(rules, path); CorpusRules(load(path))  # what SentenceBuilder(corpus=...) uses

derive() counts, in the train rows only:
    compounds         two or more kin signs said as one word (চাচা বউ -> চাচী)
    verb_by_object    the verb of sentences that have no verb sign, per object word (চা -> খাওয়া), and
    verb_by_semantic  per object semantic (food -> খাওয়া) for objects not seen in train
    markers           tense column and negation for each final-marker pattern: (last sign, আগে before it, হবে
                      before না), e.g. (ছিল, no, no) -> past_perfect, (নেই, no, no) -> present_simple + নি
    person_by_subject the person column of the verb for a kin subject, per (subject, tense column, negation);
    person_by_tense   the same per (tense column, negation), for kin subjects not seen with that tense
    time_locative     how often a time-of-day sign is written in its locative form (the forms are in the lexicon)
Every rule keeps its support (train rows that agree) and 'of' (train rows with that key); a rule needs at least
min_support rows. Only rows whose split is 'train' are accepted: anything else raises ValueError.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .lexicon import norm
from .tables import HABITUAL_MARKER, MARKERS

NEG_NONE, NEG_NA, NEG_NI = "", "na", "ni"
PERSON_PREFERENCE = ("3", "2_formal", "1", "2_informal")
TENSE_PREFERENCE = ("present_simple", "future", "past_perfect", "present_perfect", "past_habitual", "past_simple",
                    "present_continuous")
_MARKERS = tuple(norm(m) for m in MARKERS)
_HABITUAL, _NA, _WILL = norm(HABITUAL_MARKER), norm("না"), norm("হবে")


def words_of(text):
    """Normalised words of a sentence, without । and ?."""
    return norm(str(text).replace("।", " ").replace("?", " ")).split()


# ---------------------------------------------------------------- splits
def group_key(sign, time_words):
    """'<subject signs> + <object sign>' of a sign sentence: the signs before the first time word (without আগে) and
    the sign after it. Sentences without a time word use their first two signs."""
    w = str(sign).replace("।", " ").split()
    t = next((i for i, x in enumerate(w) if norm(x) in time_words), None)
    if t is None:
        return " ".join(w[:2])
    subject = " ".join(x for x in w[:t] if norm(x) != _HABITUAL)
    return f"{subject} + {w[t + 1] if t + 1 < len(w) else ''}"


def make_splits(rows, time_words, fractions=(0.7, 0.15, 0.15), seed=0):
    """Whole (subject + object) groups go to one split, so dev and test have combinations train has not seen.
    Groups are shuffled with the seed and filled into train, then dev, until each has its share of the rows.
    -> [{sentence_id, split, group}] in the order of rows."""
    time_words = {norm(w) for w in time_words}
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r["sign"], time_words)].append(r["sentence_id"])
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    n, split_of = len(rows), {}
    limits = (("train", fractions[0] * n), ("dev", (fractions[0] + fractions[1]) * n))
    used = 0
    for g in order:
        name = next((s for s, limit in limits if used < limit), "test")
        for sid in groups[g]:
            split_of[sid] = (name, g)
        used += len(groups[g])
    return [dict(sentence_id=r["sentence_id"], split=split_of[r["sentence_id"]][0],
                 group=split_of[r["sentence_id"]][1]) for r in rows]


def check_train_only(rows):
    bad = [r.get("sentence_id", "?") for r in rows if r.get("split") != "train"]
    if bad:
        raise ValueError(f"rules may only be built from train rows; got {len(bad)} other rows (e.g. {bad[:3]})")


# ---------------------------------------------------------------- reading the natural sentences
def verb_forms(lexicon):
    """{one-word verb form: [(verb, tense column, person)]} from the verb tables."""
    out = defaultdict(list)
    for verb, persons in lexicon.verbs.items():
        for person, forms in persons.items():
            for column, form in forms.items():
                if form and " " not in form and (verb, column, person) not in out[form]:
                    out[form].append((verb, column, person))
    return out


def natural_outcome(words, forms):
    """(verb, tense column, person, negation) of the last verb form in a natural sentence, or None.
    negation: 'na' (the verb is followed by না), 'ni' (present form + নি: খায়নি) or ''."""
    for i in range(len(words) - 1, -1, -1):
        w, neg = words[i], NEG_NONE
        cands = forms.get(w)
        if not cands and w.endswith("নি"):
            cands, neg = [c for c in forms.get(w[:-2], []) if c[1] == "present_simple"], NEG_NI
        if not cands:
            continue
        if neg == NEG_NONE and i + 1 < len(words) and words[i + 1] == _NA:
            neg = NEG_NA

        def rank(c):
            return (TENSE_PREFERENCE.index(c[1]) if c[1] in TENSE_PREFERENCE else 99,
                    PERSON_PREFERENCE.index(c[2]) if c[2] in PERSON_PREFERENCE else 99)
        verb, column, person = min(cands, key=rank)
        return verb, column, person, neg
    return None


def marker_key(tokens):
    """(last marker sign, আগে before it, হবে right before না) for a sentence ending in a marker, else None."""
    words = [t.word or norm(t.text) for t in tokens]
    if not words or words[-1] not in _MARKERS:
        return None
    return words[-1], _HABITUAL in words[:-1], words[-1] == _NA and len(words) > 1 and words[-2] == _WILL


def describe_marker(key):
    last, habitual, will = key
    return " + ".join(filter(None, ("আগে ..." if habitual else "", "হবে" if will else "", last)))


# ---------------------------------------------------------------- deriving
def _best(counter, min_support):
    if not counter:
        return None
    value, support = counter.most_common(1)[0]
    total = sum(counter.values())
    if support < min_support:
        return None
    return dict(value=value, support=support, of=total,
                others={("|".join(map(str, v)) if isinstance(v, tuple) else v): n
                        for v, n in counter.most_common() if v != value})


def derive(train_rows, builder, min_support=3, min_share=0.6):
    """train_rows: [{sentence_id, sign, natural, split='train'}]. builder: SentenceBuilder without corpus rules (its
    lexicon and normalise / subject rules are used to read the sign words). -> rules dict (JSON-ready)."""
    check_train_only(train_rows)
    forms = verb_forms(builder.lx)
    parsed = []
    for r in train_rows:
        tokens = builder.normalise(r["sign"], [], [])
        parsed.append((r, tokens, words_of(r["natural"])))

    # compounds: a run of two or more kin signs at the start, said as one other kin word
    spans, said = Counter(), defaultdict(Counter)
    for r, tokens, nat in parsed:
        run = []
        for t in tokens:
            if t.pos == "noun" and t.semantic == "kin":
                run.append(t.word)
            else:
                break
        if len(run) >= 2 and nat:
            spans[tuple(run)] += 1
            e = builder.lx.get(nat[0])
            if nat[0] not in run and e and e["semantic"] == "kin":
                said[tuple(run)][nat[0]] += 1
    compounds = []
    for span, total in spans.items():
        b = _best(said[span], min_support)
        if b and b["support"] / total >= min_share:
            compounds.append(dict(signs=" ".join(span), word=b["value"], support=b["support"], of=total))
    comp = {tuple(c["signs"].split()): c for c in compounds}

    by_object, by_semantic, markers = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    person_subject, person_tense, time_of, time_ok = defaultdict(Counter), defaultdict(Counter), Counter(), Counter()
    locative_of = {}
    for r, tokens, nat in parsed:
        tokens = builder.apply_compounds(tokens, [], comp)
        out = natural_outcome(nat, forms)
        for t in tokens:
            if t.semantic == "time_of_day" and t.locative:
                time_of[t.word] += 1
                locative_of[t.word] = t.locative
                time_ok[t.word] += t.locative in nat
        if out is None:
            continue
        verb, column, person, neg = out
        if not any(t.pos == "verb" for t in tokens):
            objs = [t for t in tokens if t.pos == "noun" and t.semantic not in ("kin", "person")]
            if objs:
                by_object[objs[-1].word][verb] += 1
                if objs[-1].semantic:
                    by_semantic[objs[-1].semantic][verb] += 1
        key = marker_key(tokens)
        if key is not None:
            markers[key][(column, neg)] += 1
        subject = builder.subject(tokens, [])[0]
        if subject is not None and subject.semantic == "kin":
            person_subject[(subject.word, column, neg)][person] += 1
            person_tense[(column, neg)][person] += 1

    def table(counts, name):
        out = {}
        for k, c in sorted(counts.items()):
            b = _best(c, min_support)
            if b:
                out[k] = dict({name: b["value"]}, support=b["support"], of=b["of"], others=b["others"])
        return out

    marker_rules = []
    for (last, habitual, will), c in sorted(markers.items()):
        b = _best(c, min_support)
        if b:
            marker_rules.append(dict(marker=last, habitual=habitual, will=will, tense=b["value"][0], neg=b["value"][1],
                                     support=b["support"], of=b["of"], others=b["others"]))
    ids = ",".join(sorted(str(r["sentence_id"]) for r in train_rows))
    return dict(
        train_rows=len(train_rows), train_ids_sha1=hashlib.sha1(ids.encode()).hexdigest(),
        settings=dict(min_support=min_support, min_share=min_share),
        compounds=sorted(compounds, key=lambda c: -c["support"]),
        verb_by_object=table(by_object, "verb"), verb_by_semantic=table(by_semantic, "verb"),
        markers=marker_rules,
        person_by_subject=[dict(subject=s, tense=col, neg=neg, **v) for (s, col, neg), v in
                           table(person_subject, "person").items()],
        person_by_tense=[dict(tense=col, neg=neg, **v) for (col, neg), v in table(person_tense, "person").items()],
        time_locative=[dict(word=w, locative=locative_of[w], support=time_ok[w], of=n) for w, n in time_of.items()],
    )


def derive_from_split_rows(rows, builder, **settings):
    """rows: every pair with its 'split'. Only the train rows are passed on."""
    return derive([r for r in rows if r.get("split") == "train"], builder, **settings)


# ---------------------------------------------------------------- files and lookup
def save(rules, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class CorpusRules:
    """Lookup tables for the builder, from a rules dict."""

    def __init__(self, rules):
        self.rules = rules
        self.compounds = {tuple(norm(c["signs"]).split()): dict(c, word=norm(c["word"])) for c in rules["compounds"]}
        self.verb_by_object = {norm(k): dict(v, verb=norm(v["verb"])) for k, v in rules["verb_by_object"].items()}
        self.verb_by_semantic = {k: dict(v, verb=norm(v["verb"])) for k, v in rules["verb_by_semantic"].items()}
        self.markers = {(norm(m["marker"]), m["habitual"], m["will"]): m for m in rules["markers"]}
        self.person_by_subject = {(norm(p["subject"]), p["tense"], p["neg"]): p for p in rules["person_by_subject"]}
        self.person_by_tense = {(p["tense"], p["neg"]): p for p in rules["person_by_tense"]}
        self.time_locative = {norm(t["word"]): t for t in rules.get("time_locative", [])}
