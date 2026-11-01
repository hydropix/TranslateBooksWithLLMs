"""
Unit tests for src.core.glossary.name_scan.

Covers the recurring-name detection rules (mid-sentence capitals, sentence
starts, title abbreviations, common words in titles, uncased and German-like
texts) and the targeted sampler that builds the NER excerpts around them.
"""
import sys
from pathlib import Path

# Make the project importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.glossary.name_scan import (
    find_recurring_names,
    take_glossary_samples,
)
from src.utils.document_sampler import EXCERPT_SEPARATOR, take_distributed_samples

FILLER = "the road went on and on through the quiet hills under a grey sky. "


def _names(text):
    return [c.name for c in find_recurring_names(text)]


class TestFindRecurringNames:
    def test_mid_sentence_names_are_found_most_frequent_first(self):
        text = (
            "Yesterday the old man met Geoff near the river. " * 5
            + "Later she spoke with Amara at the market. " * 3
        )
        assert _names(text) == ["Geoff", "Amara"]

    def test_sentence_start_words_are_ignored(self):
        text = "But the door was locked. Then the wind rose. " * 10
        assert _names(text) == []

    def test_name_at_sentence_start_still_counts_when_mostly_mid_sentence(self):
        text = "Mira opened the door. " * 2 + "The guard saw Mira again. " * 4 + FILLER * 5
        assert _names(text) == ["Mira"]

    def test_too_few_occurrences_are_ignored(self):
        text = "They called Oskar twice. They called Oskar again. " + FILLER * 5
        assert _names(text) == []

    def test_possessive_is_folded_into_the_name(self):
        text = "We took Geoff's bike, then Geoff's tent, and asked Geoff. " * 2 + FILLER * 5
        assert _names(text) == ["Geoff"]

    def test_title_abbreviation_does_not_start_a_sentence(self):
        # "Smith" follows "Mr." every time: it must count as mid-sentence.
        text = "They visited Mr. Smith at noon. " * 4 + FILLER * 5
        assert _names(text) == ["Smith"]

    def test_pronoun_i_and_acronyms_are_ignored(self):
        text = "Then I said that I'm sure the NASA probe left. " * 5
        assert _names(text) == []

    def test_common_word_capitalized_in_titles_is_ignored(self):
        text = (
            "We played The Witcher all night and the next day. " * 4
            + "the cat and the dog and the bird sat on the wall. " * 3
        )
        assert _names(text) == ["Witcher"]

    def test_line_break_starts_a_new_sentence(self):
        text = "Chapter one\nHowever the night was long and dark\n" * 5
        assert _names(text) == []

    def test_uncased_script_yields_nothing(self):
        assert find_recurring_names("李凡走进了青玄宗大殿。" * 50) == []

    def test_capitals_everywhere_disables_the_scan(self):
        # German capitalizes every noun: capitals carry no name signal.
        text = "Der Mann sah den Hund und die Katze im Garten am Abend. " * 20
        assert find_recurring_names(text) == []

    def test_empty_text(self):
        assert find_recurring_names("") == []


class TestTakeGlossarySamples:
    def test_short_text_is_returned_untouched(self):
        assert take_glossary_samples("short text", 1000, 10) == ("short text", 1, 0)

    def test_no_names_falls_back_to_distributed_samples(self):
        text = FILLER * 2000
        joined, count, targeted = take_glossary_samples(text, 6000, 10, 500)
        assert (joined, count) == take_distributed_samples(text, 6000, 10, 500)
        assert targeted == 0

    def test_secondary_character_missed_by_even_spacing_is_reached(self):
        # Ignatius only lives in a short stretch of a long book.
        middle = "Then the guide spoke with Ignatius by the fire. " * 4
        text = FILLER * 1200 + middle + FILLER * 3000
        old, _ = take_distributed_samples(text, 6000, 10, 500)
        assert "Ignatius" not in old

        joined, count, targeted = take_glossary_samples(text, 6000, 10, 500)
        assert "Ignatius" in joined
        assert targeted == 1
        assert count >= 2

    def test_budget_is_respected(self):
        cast = [f"Name{chr(65 + i)}{chr(97 + i)}" for i in range(26)]
        text = "".join(
            FILLER * 200 + f"Then they met {name} at dawn. " * 3 for name in cast
        )
        joined, count, targeted = take_glossary_samples(text, 6000, 10, 500)
        assert targeted > 0
        # Separators included: suggest_terms truncates anything longer.
        assert len(joined) <= 6000
        excerpts = joined.split(EXCERPT_SEPARATOR)
        assert len(excerpts) == count
        assert all(excerpts)
