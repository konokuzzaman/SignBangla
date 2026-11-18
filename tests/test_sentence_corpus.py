"""Module 3: rules counted from sentence pairs (signbangla/sign2text/corpus_rules.py) and the builder rules that use
them. Synthetic pairs in the style of Ban-Sign-Sent-9K; the tables are written to a temporary folder, so data/module3/
is neither needed nor changed."""
import copy
import itertools

import pytest

from signbangla.sign2text import builder as B
from signbangla.sign2text import corpus_rules as CR
from signbangla.sign2text import lexicon as L
from signbangla.utils import load_config

CFG = load_config()
SUBJECTS = {"বাবা": "বাবা", "মা": "মা", "চাচা বউ": "চাচী"}
TIMES = {"সকাল": "সকালে", "রাত": "রাতে"}
FOODS = ("চা", "কফি")
# (signs before / after the object, natural verb)
PATTERNS = [("", "খাওয়ায়া হবে", "খাবে"), ("", "হবে", "খাবে"), ("", "খাওয়ায়া ছিল", "খেয়েছিল"),
            ("", "হয়েছে", "খেয়েছে"), ("", "নেই", "খায়নি"), ("", "না", "খান না"), ("আগে", "ছিল", "খেতেন"),
            ("আগে", "খাওয়ায়া না", "খেতেন না"), ("", "খাওয়ায়া হবে না", "খাবে না")]


def pairs(split="train"):
    rows = []
    for (subj, nat_subj), (time, loc), food, (before, after, verb) in itertools.product(
            SUBJECTS.items(), TIMES.items(), FOODS, PATTERNS):
        sign = " ".join(filter(None, (subj, before, time, food, after))) + "।"
        rows.append(dict(sentence_id=str(len(rows)), sign=sign, natural=f"{nat_subj} {loc} {food} {verb}।",
                         split=split))
    return rows


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("module3_corpus")
    p = {k: d / f"{k}.csv" for k in ("lexicon", "verbs", "predicates", "test_sentences", "lexicon_extra",
                                     "verb_tenses")}
    L.write_missing(p)
    return p


@pytest.fixture(scope="module")
def plain(paths):
    return B.SentenceBuilder(L.Lexicon.from_files(paths), CFG["module3"])


@pytest.fixture(scope="module")
def rules(plain):
    return CR.derive(pairs(), plain)


@pytest.fixture(scope="module")
def sb(plain, rules):
    return B.SentenceBuilder(plain.lx, CFG["module3"], CR.CorpusRules(rules))


def build(sb, words):
    return sb.build(words)[0]


# ---------------------------------------------------------------- lexicon additions
def test_extra_tables_are_written_unverified(paths):
    for key in ("lexicon_extra", "verb_tenses"):
        rows = L.read_csv(paths[key])
        assert rows and all(r["verified"] == "false" for r in rows)
    assert L.write_missing(paths) == []  # never written again


def test_typos_and_spellings_are_normalised(plain):
    for typo in ("খাওয়ায়া", "খাওয়ায়া"):  # য় as one character and as য + nukta
        tokens = plain.normalise(typo, [], [])
        assert [t.word for t in tokens] == [L.norm("খাওয়া")]
    assert [t.word for t in plain.normalise("দাদি", [], [])] == [L.norm("দাদী")]
    veg = plain.normalise("শাক-সব্জী", [], [])
    assert len(veg) == 1 and veg[0].text == "শাক-সব্জী" and veg[0].semantic == "food"
    for w, pos in (("ছিল", "marker"), ("হবে", "marker"), ("হয়েছে", "marker"), ("আগে", "marker"), ("বিকাল", "time"),
                   ("সন্ধ্যা", "time"), ("বউ", "noun"), ("জামাই", "noun")):
        e = plain.lx.get(L.norm(w))
        assert e and e["pos"] == pos, w
    assert plain.lx.get(L.norm("বউ"))["semantic"] == "kin"


def test_new_tense_columns(plain):
    eat = plain.lx.verbs[L.norm("খাওয়া")]
    assert eat["3"]["present_perfect"] == L.norm("খেয়েছে") and eat["2_formal"]["past_habitual"] == "খেতেন"


# ---------------------------------------------------------------- a. time of day and compounds
def test_time_of_day_takes_the_locative_with_a_verb(plain):
    sentence, trace, _ = plain.build("আমি সকাল ভাত খাওয়া")
    assert sentence == "আমি সকালে ভাত খাই।" and any("time of day: সকাল -> সকালে" in t for t in trace)
    assert build(plain, "আমি সন্ধ্যা খাওয়া") == "আমি সন্ধ্যায় খাই।"
    assert "সকালে" not in build(plain, "আমি সকাল")  # no verb: unchanged


def test_compound_is_derived_from_train(rules, sb):
    assert {(c["signs"], c["word"]) for c in rules["compounds"]} == {("চাচা বউ", "চাচী")}
    sentence, trace, _ = sb.build("চাচা বউ রাত কফি হবে")
    assert sentence == "চাচী রাতে কফি খাবে।" and any(t.startswith("a. compound চাচা বউ -> চাচী") for t in trace)


# ---------------------------------------------------------------- b. dropped verb
def test_dropped_verb_comes_from_the_object(rules, sb):
    assert rules["verb_by_object"][L.norm("চা")]["verb"] == L.norm("খাওয়া")
    assert rules["verb_by_semantic"]["food"]["verb"] == L.norm("খাওয়া")
    sentence, trace, _ = sb.build("বাবা সকাল চা হবে")
    assert sentence == "বাবা সকালে চা খাবে।" and any(t.startswith("d. no verb sign") for t in trace)
    assert build(sb, "মা সকাল মাংস হবে") == "মা সকালে মাংস খাবে।"  # not in train: by its semantic (food)


def test_no_verb_is_added_for_other_objects(sb):
    assert build(sb, "আমি টাকা নেই") == "আমার টাকা নেই।"
    assert build(sb, "আমি ভালো না") == "আমি ভালো নেই।"


# ---------------------------------------------------------------- c. markers
@pytest.mark.parametrize("signs,expected", [
    ("বাবা সকাল চা খাওয়ায়া হবে", "বাবা সকালে চা খাবে।"),
    ("বাবা সকাল চা খাওয়ায়া ছিল", "বাবা সকালে চা খেয়েছিল।"),
    ("বাবা সকাল চা হয়েছে", "বাবা সকালে চা খেয়েছে।"),
    ("বাবা সকাল চা নেই", "বাবা সকালে চা খায়নি।"),
    ("বাবা সকাল চা না", "বাবা সকালে চা খান না।"),
    ("বাবা আগে সকাল চা ছিল", "বাবা সকালে চা খেতেন।"),
    ("বাবা আগে সকাল চা খাওয়ায়া না", "বাবা সকালে চা খেতেন না।"),
    ("বাবা সকাল চা খাওয়ায়া হবে না", "বাবা সকালে চা খাবে না।"),
])
def test_final_marker_sets_tense_and_negation(sb, signs, expected):
    assert build(sb, signs) == expected


def test_marker_rules_are_derived_with_support(rules):
    got = {(m["marker"], m["habitual"], m["will"]): (m["tense"], m["neg"]) for m in rules["markers"]}
    n = L.norm
    assert got[(n("ছিল"), False, False)] == ("past_perfect", "")
    assert got[(n("ছিল"), True, False)] == ("past_habitual", "")
    assert got[(n("নেই"), False, False)] == ("present_simple", "ni")
    assert got[(n("না"), False, False)] == ("present_simple", "na")
    assert got[(n("না"), False, True)] == ("future", "na")
    assert got[(n("হয়েছে"), False, False)] == ("present_perfect", "")
    for m in rules["markers"]:
        assert rules["settings"]["min_support"] <= m["support"] <= m["of"]


def test_plain_na_with_a_time_word_keeps_the_old_rule(sb):
    assert build(sb, "সে গতকাল খাওয়া না") == "সে গতকাল খায়নি।"
    assert build(sb, "আমি আগামীকাল যাওয়া না") == "আমি আগামীকাল যাব না।"


# ---------------------------------------------------------------- d. person, e. markers not written
def test_kin_subject_takes_the_person_form_of_train(rules, sb):
    by = {(p["subject"], p["tense"], p["neg"]): p["person"] for p in rules["person_by_subject"]}
    assert by[("বাবা", "present_simple", "na")] == "2_formal" and by[("বাবা", "present_simple", "ni")] == "3"
    sentence, trace, _ = sb.build("দাদা সকাল চা না")  # দাদা is not in train: the rule for all kin subjects
    assert sentence == "দাদা সকালে চা খান না।" and any("most common for kin subjects" in t for t in trace)
    assert build(sb, "সে চা খাওয়া না") == "সে চা খায় না।"  # pronouns keep their person


def test_marker_signs_are_not_written(sb):
    for signs in ("মা আগে রাত কফি ছিল", "মা রাত কফি হয়েছে", "মা রাত কফি খাওয়ায়া হবে না", "মা রাত কফি নেই"):
        sentence, trace, _ = sb.build(signs)
        words = sentence.rstrip("।").split()
        assert not {"ছিল", "হবে", "আগে", "নেই", L.norm("হয়েছে")} & set(words), signs
        assert any(t.startswith("h. marker signs not written") for t in trace)


def test_old_behaviour_without_corpus_rules(plain):
    assert build(plain, "বাবা সকাল চা হবে") == "বাবা সকাল চা হবে।"  # nothing is guessed without the rules


# ---------------------------------------------------------------- splits and the test set
def test_splits_keep_groups_together():
    rows = pairs()
    a = CR.make_splits(rows, TIMES, (0.5, 0.25, 0.25), seed=1)
    assert a == CR.make_splits(rows, TIMES, (0.5, 0.25, 0.25), seed=1)  # fixed seed: same split
    split_of_group = {}
    for r in a:
        assert split_of_group.setdefault(r["group"], r["split"]) == r["split"]
    assert {r["split"] for r in a} == {"train", "dev", "test"} and len(split_of_group) == 6
    assert CR.group_key("চাচা বউ আগে সকাল চা না।", {"সকাল"}) == "চাচা বউ + চা"


def test_test_split_is_not_used_to_build_rules(plain):
    train = pairs("train")
    held = [dict(r, sentence_id=f"t{r['sentence_id']}", split=split) for split in ("dev", "test") for r in pairs()]
    changed = [dict(r, natural="বাবা কিছু বলে না।") for r in held]  # wrong sentences in dev / test
    a = CR.derive_from_split_rows(train + held, plain)
    b = CR.derive_from_split_rows(train + changed, plain)
    assert a == b == CR.derive(train, plain) and a["train_rows"] == len(train)
    with pytest.raises(ValueError):
        CR.derive(train + held[:1], plain)


def test_rules_file_round_trip(rules, tmp_path, plain):
    path = tmp_path / "rules.json"
    CR.save(rules, path)
    again = B.SentenceBuilder(plain.lx, copy.deepcopy(CFG["module3"]), CR.CorpusRules(CR.load(path)))
    assert build(again, "চাচা বউ আগে রাত চা ছিল") == "চাচী রাতে চা খেতেন।"
