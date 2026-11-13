"""The whitespace clean-up must not destroy typography.

It removed the space before ! ? ; : (mandatory in French), could eat
non-breaking spaces, and collapsed leading indentation (code, verse).
"""

import pytest

from src.core.post_processor import clean_translated_text, is_french_typography

NBSP = "\u00a0"
NNBSP = "\u202f"


@pytest.mark.parametrize("lang", ["French", "fr", "fr-CA", "Français"])
def test_french_keeps_the_space_before_high_punctuation(lang):
    text = "Quoi ? Vraiment ! Oui ; enfin : non."
    assert clean_translated_text(text, target_language=lang) == text


def test_other_targets_still_lose_the_space_before_punctuation():
    assert clean_translated_text("Really ? Yes !", target_language="English") == "Really? Yes!"


@pytest.mark.parametrize("lang", ["French", "English", ""])
def test_non_breaking_spaces_are_never_touched(lang):
    text = f"Quoi{NNBSP}? 10{NBSP}000{NBSP}{NBSP}euros{NBSP}!"
    assert clean_translated_text(text, target_language=lang) == text


def test_leading_indentation_is_kept():
    code = "def f():\n    if x:\n        return 1"
    assert clean_translated_text(code, target_language="English") == code


def test_runs_of_spaces_inside_a_line_still_collapse():
    assert clean_translated_text("a    b", target_language="French") == "a b"


def test_french_detection():
    assert is_french_typography("French")
    assert not is_french_typography("Frisian")
    assert not is_french_typography("English")
