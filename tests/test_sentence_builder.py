"""Module 3: the rule-based sentence builder (signbangla/sign2text). Uses its own copies of the tables in a temporary
folder, built from the code, so data/module3/ is neither needed nor changed."""
import copy
import csv
from collections import Counter

import pytest

from signbangla.sign2text import builder as B
from signbangla.sign2text import lexicon as L
from signbangla.sign2text import tables as T
from signbangla.utils import load_config

CFG = load_config()


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("module3")
    p = {k: d / f"{k}.csv" for k in ("lexicon", "verbs", "predicates", "test_sentences")}
    L.write_missing(p)  # no sources given: built from the tables in code
    return p


@pytest.fixture(scope="module")
def sb(paths):
    return B.SentenceBuilder(L.Lexicon.from_files(paths), CFG["module3"])


def build(sb, words, q=False):
    return sb.build(words, is_question=q)[0]


def with_settings(sb, **changes):
    return B.SentenceBuilder(sb.lx, dict(copy.deepcopy(CFG["module3"]), **changes))


# ---------------------------------------------------------------- the examples
def test_examples(sb):
    assert build(sb, ["আমি", "পানি", "চাওয়া"]) == "আমি পানি চাই।"
    assert build(sb, "তুমি কোথায় যাওয়া") == "তুমি কোথায় যাও?"


# ---------------------------------------------------------------- a. normalise
def test_spelling_variants_and_synonyms(sb):
    sentence, trace, unknown = sb.build("আমি বাসা যাওয়া")
    assert sentence == "আমি বাড়ি যাই।" and not unknown
    assert any(t.startswith("a. বাসা -> বাড়ি") for t in trace)
    assert build(sb, "আমি মাথা ব্যাথা") == "আমার মাথা ব্যথা করছে।"  # source spelling -> preferred spelling


def test_multiword_verbs(sb):
    assert build(sb, "আপনি চা পান করা") == "আপনি চা পান করেন।"
    assert build(sb, "আমি সাহায্য করা") == "আমি সাহায্য করি।"


def test_inflected_verb_is_mapped_to_its_dictionary_form(sb):
    sentence, trace, _ = sb.build("সে ভাত খাব")
    assert sentence == "সে ভাত খায়।" and any("verb form" in t for t in trace)


def test_unknown_words_pass_through(sb):
    sentence, trace, unknown = sb.build("আমি ঝুমঝুমি চাওয়া")
    assert sentence == "আমি ঝুমঝুমি চাই।" and unknown == ["ঝুমঝুমি"]
    assert any("unknown word ঝুমঝুমি" in t for t in trace)


def test_empty_input(sb):
    assert sb.build([]) == ("", ["empty input"], [])
    assert sb.build("   ") == ("", ["empty input"], [])


def test_never_invents_words(sb):
    """Every word of the output is an input word, a form from the tables, or না / নেই / নি."""
    for words in ("আমি পানি চাওয়া না", "গতকাল আমি জ্বর", "আপনি কেমন", "আমি গতকাল স্কুল যাওয়া না"):
        sentence, _, _ = sb.build(words)
        allowed = set(words.split()) | {"না", "নেই"}
        for forms in sb.lx.verbs.values():
            for person in forms.values():
                for f in person.values():
                    allowed |= set(f.split()) | {f + "নি"}
        for p in sb.lx.predicates.values():
            allowed |= set(" ".join((p["present"], p["past"], p["future"])).split())
        allowed |= {e["possessive"] for e in sb.lx.entries.values()} | {e["locative"] for e in sb.lx.entries.values()}
        allowed |= {w for e in sb.lx.entries.values() for w in e["locative"].split()}
        for w in sentence.rstrip("।?").split():
            assert w in allowed or w + "নি" in allowed, (words, w)


# ---------------------------------------------------------------- b. subject
def test_default_subject_is_first_person(sb):
    sentence, trace, _ = sb.build("পানি চাওয়া")
    assert sentence == "পানি চাই।" and "b. no subject: 1st person assumed" in trace


@pytest.mark.parametrize("words,expected", [
    ("আপনি ভাত খাওয়া", "আপনি ভাত খান।"),
    ("সে পানি চাওয়া", "সে পানি চায়।"),
    ("তারা টিভি দেখা", "তারা টিভি দেখে।"),
    ("আমরা স্কুল যাওয়া", "আমরা স্কুলে যাই।"),
    ("মা রুটি কেনা", "মা রুটি কেনে।"),
])
def test_subject_person(sb, words, expected):
    assert build(sb, words) == expected


def test_pronoun_before_kin_noun_is_its_owner(sb):
    sentence, trace, _ = sb.build("আমি মা আসা")
    assert sentence == "আমার মা আসে।"
    assert "b. subject: মা (person 3); আমি is its owner" in trace


# ---------------------------------------------------------------- c. tense and d. verb agreement
def test_tenses(sb):
    assert build(sb, "আমি গতকাল ভাত খাওয়া") == "আমি গতকাল ভাত খেয়েছিলাম।"
    assert build(sb, "আমি আগামীকাল ভাত খাওয়া") == "আমি আগামীকাল ভাত খাব।"
    assert build(sb, "আমরা পরে খাওয়া") == "আমরা পরে খাব।"
    assert build(sb, "আমি এখন ভাত খাওয়া") == "আমি এখন ভাত খাচ্ছি।"


def test_past_simple_setting(sb):
    assert build(with_settings(sb, past_tense="past_simple"), "সে গতকাল আসা") == "সে গতকাল এল।"


def test_irregular_forms_from_table(sb):
    assert build(sb, "তুমি গতকাল যাওয়া") == "তুমি গতকাল গিয়েছিলে।"
    assert build(sb, "আমি টাকা দেওয়া") == "আমি টাকা দিই।"
    assert build(sb, "আপনি বই নেওয়া") == "আপনি বই নেন।"


# ---------------------------------------------------------------- e. places
def test_places_before_motion_verbs(sb):
    assert build(sb, "সে হাসপাতাল যাওয়া") == "সে হাসপাতালে যায়।"
    assert build(sb, "আমি গতকাল ডাক্তার যাওয়া") == "আমি গতকাল ডাক্তারের কাছে গিয়েছিলাম।"
    assert build(sb, "আমরা বাংলাদেশ থাকা") == "আমরা বাংলাদেশে থাকি।"
    assert build(sb, "বাবা বাড়ি আসা") == "বাবা বাড়ি আসে।"  # বাড়ি has no locative ending


def test_no_locative_with_other_verbs(sb):
    assert build(sb, "আমি স্কুল দেখা") == "আমি স্কুল দেখি।"


# ---------------------------------------------------------------- f. possession
def test_possession_without_verb(sb):
    assert build(sb, "আমি মাথা ব্যথা") == "আমার মাথা ব্যথা করছে।"
    assert build(sb, "তুমি বাড়ি কোথায়") == "তোমার বাড়ি কোথায়?"
    assert build(sb, "আমি টাকা নেই") == "আমার টাকা নেই।"
    assert build(sb, "আমি ডাক্তার দরকার") == "আমার ডাক্তার দরকার।"


def test_no_possession_for_object_or_role(sb):
    assert build(sb, "আমি বই পড়া") == "আমি বই পড়ি।"
    assert build(sb, "আমি ছাত্র") == "আমি ছাত্র।"


# ---------------------------------------------------------------- g. sentences without a verb
def test_predicate_nouns(sb):
    assert build(sb, "আমি জ্বর") == "আমার জ্বর হয়েছে।"
    assert build(sb, "গতকাল আমি জ্বর") == "গতকাল আমার জ্বর ছিল।"
    assert build(sb, "আমি ক্ষুধা") == "আমার ক্ষুধা লেগেছে।"


def test_state_words_take_the_copula(sb):
    assert build(sb, "আমি ভালো") == "আমি ভালো আছি।"
    assert build(sb, "আপনি কেমন") == "আপনি কেমন আছেন?"
    assert build(sb, "গতকাল আমি ভালো") == "গতকাল আমি ভালো ছিলাম।"
    assert build(sb, "আমি অসুস্থ") == "আমি অসুস্থ।"  # অসুস্থ needs no copula


# ---------------------------------------------------------------- h. negation
def test_negation_after_verb(sb):
    assert build(sb, "আমি পানি চাওয়া না") == "আমি পানি চাই না।"
    assert build(sb, "আমি না পানি চাওয়া") == "আমি পানি চাই না।"  # moved after the verb
    assert build(sb, "আমি আগামীকাল যাওয়া না") == "আমি আগামীকাল যাব না।"


def test_negation_in_the_past(sb):
    assert build(sb, "আমি গতকাল স্কুল যাওয়া না") == "আমি গতকাল স্কুলে যাইনি।"
    assert build(sb, "সে গতকাল খাওয়া না") == "সে গতকাল খায়নি।"


def test_negation_without_verb(sb):
    assert build(sb, "আমি ভালো না") == "আমি ভালো নেই।"
    assert build(sb, "গতকাল আমি ভালো না") == "গতকাল আমি ভালো ছিলাম না।"
    assert build(sb, "আমি জ্বর নেই") == "আমার জ্বর নেই।"
    assert build(sb, "না") == "না।"


# ---------------------------------------------------------------- i. questions
def test_wh_questions(sb):
    assert build(sb, "তুমি যাওয়া কোথায়") == "তুমি কোথায় যাও?"  # BdSL puts the question word last
    assert build(sb, "আপনি কী চাওয়া") == "আপনি কী চান?"
    assert build(sb, "টাকা কত") == "কত টাকা?"
    assert build(sb, "সে কে") == "সে কে?"


def test_yes_no_questions(sb):
    assert build(sb, "তুমি পানি চাওয়া", q=True) == "তুমি পানি চাও?"
    assert build(sb, "তুমি পানি চাওয়া") == "তুমি পানি চাও।"
    assert build(with_settings(sb, yes_no_particle=True), "তুমি পানি চাওয়া", q=True) == "তুমি কি পানি চাও?"


# ---------------------------------------------------------------- j. fixed phrases, k. full stop
def test_fixed_phrases(sb):
    assert build(sb, "ধন্যবাদ") == "ধন্যবাদ।"
    sentence, trace, _ = sb.build("আসসালামু আলাইকুম")
    assert sentence == "আসসালামু আলাইকুম।" and any(t.startswith("j.") for t in trace)


def test_trace_lists_the_rules_in_order(sb):
    _, trace, _ = sb.build("আমি গতকাল স্কুল যাওয়া না")
    letters = [t[0] for t in trace]
    assert letters == sorted(letters) and {"b", "c", "d", "e", "h", "k"} <= set(letters)


# ---------------------------------------------------------------- the API
def test_build_from_candidates(sb):
    cands = [[("আমি", 0.9), ("তুমি", 0.1)], [("পানি", 0.4), ("চা", 0.6)], [("চাওয়া", 0.7), ("যাওয়া", 0.2)]]
    assert B.build_from_candidates(cands, builder=sb)[0] == "আমি চা চাই।"
    with pytest.raises(NotImplementedError):
        B.build_from_candidates(cands, scorer=lambda words: 0.0, builder=sb)


def test_build_sentence_function(sb):
    assert B.build_sentence(["সে", "আসা", "না"], builder=sb) == ("সে আসে না।", sb.build("সে আসা না")[1], [])


# ---------------------------------------------------------------- the tables
REQUIRED_VERBS = [v.replace("_", " ") for v in (
    "খাওয়া পান_করা যাওয়া আসা চাওয়া দেওয়া নেওয়া ঘুমানো বসা দেখা শোনা বলা জানা করা সাহায্য_করা "
    "অপেক্ষা_করা কেনা পড়া লেখা থাকা পারা বোঝা").split()]


def test_verb_table_is_complete(paths):
    with open(paths["verbs"], encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    have = {(L.norm(r["verb"]), r["person"]) for r in rows}
    for v in REQUIRED_VERBS:
        for p in T.PERSONS:
            assert (L.norm(v), p) in have, (v, p)
    assert all(r[t] for r in rows for t in T.TENSES) and all(r["verified"] == "false" for r in rows)
    yaoa = {r["person"]: r for r in rows if L.norm(r["verb"]) == L.norm("যাওয়া")}
    assert yaoa["1"]["past_simple"] == "গেলাম" and yaoa["1"]["past_perfect"] == "গিয়েছিলাম"


def test_case_forms():
    assert L.possessive_of("বাবা") == "বাবার" and L.possessive_of("ডাক্তার") == "ডাক্তারের"
    assert L.possessive_of("ভাই") == "ভাইয়ের" and L.possessive_of("মা") == "মায়ের"
    assert L.locative_of("স্কুল") == "স্কুলে" and L.locative_of("নদী") == "নদীতে"
    assert L.locative_of("বারান্দা") == "বারান্দায়" and L.locative_of("বাড়ি") == "বাড়ি"


def test_test_set(paths, sb):
    with open(paths["test_sentences"], encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 50 and len({r["id"] for r in rows}) == 50
    assert Counter(r["category"] for r in rows) == dict(simple=10, wh_question=8, yes_no_question=4, negation=6,
                                                        past=6, future=4, health=6, possession_place=6)
    assert all(r["verified"] == "false" for r in rows)
    for r in rows:  # only lexicon words
        assert not sb.build(r["sign_words"])[2], r["id"]


def test_files_are_never_overwritten(paths):
    before = paths["verbs"].read_text(encoding="utf-8-sig")
    paths["verbs"].write_text(before.replace("false", "true", 1), encoding="utf-8-sig")
    assert L.write_missing(paths) == []
    assert paths["verbs"].read_text(encoding="utf-8-sig").count("true") == 1
    paths["verbs"].write_text(before, encoding="utf-8-sig")


def test_lexicon_rebuild_keeps_verified(tmp_path):
    rows = L.build_lexicon()
    assert {r["word"] for r in rows} >= {"আমি", "পানি", "যাওয়া", "ডাক্তার", "স্কুল", "কোথায়"}
    assert all(r["verified"] == "false" for r in rows)
    rows[0]["verified"] = "true"
    again = L.build_lexicon(existing=rows)
    assert next(r for r in again if r["word"] == rows[0]["word"])["verified"] == "true"
