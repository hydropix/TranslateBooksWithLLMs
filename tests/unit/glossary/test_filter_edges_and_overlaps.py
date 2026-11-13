"""Glossary matching: per-edge boundaries and longest-match suppression.

_has_word_char_at_edge treated both edges together, so a term with a word
character on one edge and punctuation on the other ("C++", "Ph.D.") was
wrapped in \\b...\\b and never matched. CJK terms were bare substring counts,
so a one-character name fired inside a longer glossary term containing it.
"""

from src.core.glossary.filter import filter_glossary


def test_terms_ending_in_punctuation_match():
    filtered, _ = filter_glossary(
        "She learned C++ before her Ph.D. in Paris.",
        {"C++": "C++", "Ph.D.": "doctorat", "Paris": "Paris"},
    )
    assert set(filtered) == {"C++", "Ph.D.", "Paris"}


def test_left_boundary_still_applies():
    filtered, _ = filter_glossary("ABC++ is not it.", {"C++": "C++"})
    assert filtered == {}


def test_one_character_name_does_not_fire_inside_a_longer_term():
    terms = {"李": "Li", "李明": "Li Ming"}
    filtered, _ = filter_glossary("李明走进了房间。", terms)
    assert filtered == {"李明": "Li Ming"}


def test_one_character_name_alone_still_matches():
    terms = {"李": "Li", "李明": "Li Ming"}
    filtered, _ = filter_glossary("李明和李说话。", terms)
    assert set(filtered) == {"李", "李明"}


def test_latin_word_boundaries_unchanged():
    filtered, _ = filter_glossary("Fantasy fans", {"Fan": "Fan"})
    assert filtered == {}
