"""Tests for Module 2 text processing: recognized or typed text -> dictionary words. No model or microphone needed.

    python tests/test_text_processing.py
"""
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signbangla.labels import LABELS
from signbangla.speech2sign.dictionary import split_variants
from signbangla.speech2sign.text_processing import (SYNONYMS, UNKNOWN, WordMatcher, best_candidate, clean_text, known_words,
                                                    split_words, unknown_words)

M = WordMatcher()


def test_clean_text_punctuation():
    assert clean_text("বাবা, মা।") == "বাবা মা"
    assert clean_text("  বাবা॥ (মা)  ?! ") == "বাবা মা"
    assert split_words("বাবা,মা") == ["বাবা", "মা"]  # no space after the comma
    assert clean_text("...") == "" and split_words("") == []


def test_clean_text_unicode():
    assert clean_text("আয়না") == clean_text("আয়না")  # the two Unicode spellings of য়
    assert [m.how for m in M.match("আয়না")] == [m.how for m in M.match("আয়না")] == ["exact"]
    assert clean_text("বাবা‌") == "বাবা" and clean_text("বা‍বা") == "বাবা"  # zero-width characters
    assert clean_text("চিকিত্‍সা") == clean_text("চিকিৎসা")  # the old spelling of ৎ


def test_clean_text_keeps_vowel_signs():
    for name in ("চুড়ি", "দুর্বল", "আত্মীয়", "অ্যাপার্টমেন্ট"):
        assert clean_text(name) == unicodedata.normalize("NFC", name)


def test_every_dictionary_word_matches_itself():
    for _, bangla, roman, english in LABELS:
        first = clean_text(split_variants(bangla)[0])
        for name in (*split_variants(bangla), *split_variants(roman), english):
            found = M.match(name)
            assert len(found) == 1 and found[0].word == first and found[0].how == "exact", name


def test_synonyms():
    for name, word in (("আব্বু", "বাবা"), ("আম্মা", "মা"), ("আম্মু", "মা"), ("ডাক্তার", "ডক্টর")):
        found = M.match(name)
        assert [(m.word, m.how) for m in found] == [(word, "synonym")], name
    assert [m.word for m in M.match("আব্বা")] == ["বাবা"]  # also a listed spelling of the word


def test_synonyms_with_punctuation():
    assert [m.word for m in M.match("টি-শার্ট")] == ["টিশার্ট"]
    assert [m.word for m in M.match("টি.ভি.")] == ["টিভি"]


def test_synonym_targets_are_in_the_dictionary():
    names = {clean_text(n) for _, bangla, *_ in LABELS for n in split_variants(bangla)}
    for word in SYNONYMS.values():
        assert clean_text(word) in names, word
    try:
        WordMatcher(synonyms={"কিছু": "নেই"})
        raise AssertionError("a synonym for a word outside the dictionary must raise")
    except ValueError:
        pass


def test_multi_word_entries():
    assert [m.word for m in M.match("অডিও ক্যাসেট")] == ["অডিও ক্যাসেট"]
    assert [m.word for m in M.match("বাবা চোখ ওঠা মা")] == ["বাবা", "চোখ ওঠা", "মা"]


def test_word_split_in_two_is_joined():
    for heard, word, how in (("দুলা ভাই", "দুলাভাই", "exact"), ("টুথ পেস্ট", "টুথপেস্ট", "exact"), ("চা চা", "চাচা", "exact"),
                             ("চক লেগট", "চকলেট", "fuzzy")):
        assert [(m.word, m.how) for m in M.match(heard)] == [(word, how)], heard
    assert [m.word for m in M.match("বাবা মা")] == ["বাবা", "মা"]  # two real words stay two words
    assert [m.word for m in M.match("মা চা আপেল")] == ["মা", "চা", "আপেল"]


def test_fuzzy_matches_small_spelling_errors():
    for heard, word in (("বালটি", "বালতি"), ("অডিও কাসেট", "অডিও ক্যাসেট"), ("চিরনি", "চিরুনি"), ("বাবার", "বাবা")):
        found = M.match(heard)
        assert [(m.word, m.how) for m in found] == [(word, "fuzzy")], heard
        assert 0.8 <= found[0].score < 1


def test_short_words_never_match_fuzzily():
    for heard in ("মে", "আ", "চি", "মাম"):  # one letter away from মা, আম, চা
        assert [m.word for m in M.match(heard)] == [UNKNOWN], heard


def test_close_words_stay_apart():
    """These dictionary words differ by one letter. Each must find itself, never its neighbour."""
    for a, b in (("চাচা", "চাচী"), ("দাদা", "দাদী"), ("চা", "চাল"), ("আম", "মা"), ("বালু", "বালতি"), ("বাত", "বাবা")):
        assert M.match(a)[0].word == a and M.match(b)[0].word == b


def test_ambiguous_fuzzy_match_is_unknown():
    labels = [(1, "কলম", "a", "A"), (2, "কলস", "b", "B")]
    m = WordMatcher(labels=labels, synonyms={}, cutoff=0.5)
    assert [x.word for x in m.match("কলক")] == [UNKNOWN]


def test_unknown_words_are_returned_as_unknown():
    found = M.match("বাবা কম্পিউটার মা")
    assert [m.word for m in found] == ["বাবা", UNKNOWN, "মা"]
    assert found[1].how == "unknown" and found[1].source == "কম্পিউটার"
    assert known_words(found) == ["বাবা", "মা"] and unknown_words(found) == ["কম্পিউটার"]


def test_empty_text():
    assert M.match("") == [] and M.match(" । ") == []


def test_best_candidate():
    assert best_candidate(["কম্পিউটার", "বাবা"], M) == "বাবা"  # known beats unknown
    assert best_candidate(["বালটি", "বালতি"], M) == "বালতি"  # exact beats fuzzy
    assert best_candidate(["বাবা কম্পিউটার", "বাবা"], M) == "বাবা"  # fewer unknown words
    assert best_candidate(["মা", "বাবা"], M) == "মা"  # a tie keeps the first
    assert best_candidate(["", "কম্পিউটার"], M) == "কম্পিউটার"  # empty text is the worst


def test_romanized_and_english_input():
    assert [m.word for m in M.match("baba Mother")] == ["বাবা", "মা"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"{t.__name__} ok")
    print("\nALL TESTS PASSED")
