"""
Focused tests for the prompt builders in src/prompts/prompts.py:

- the refine prompt only carries placeholder instructions when the draft has placeholders;
- the text-cleanup section never asks to merge paragraphs when placeholders or the
  Plain Text Mode paragraph contract pin the layout;
- the subtitle prompts show their format example in the actual target language,
  with a neutral English fallback;
- the translation/refine output example lines come from the examples module.
"""
import pytest

from src.prompts.examples import (
    get_output_format_example,
    get_refinement_output_example,
    get_subtitle_example,
    get_subtitle_refinement_example,
)
from src.prompts.prompts import (
    generate_post_processing_prompt,
    generate_refinement_prompt,
    generate_subtitle_block_prompt,
    generate_subtitle_refinement_block_prompt,
    generate_translation_prompt,
)

MERGE_BULLET = "Merge incorrectly split paragraphs"
KEEP_BULLET = "never merge or split paragraphs"
PLACEHOLDER_SECTION = "# PLACEHOLDER PRESERVATION (CRITICAL)"
PLACEHOLDER_REMINDER = "PLACEHOLDER PRESERVATION IS ABSOLUTELY CRITICAL"


def _refine(has_placeholders, prompt_options=None, target_language="French"):
    return generate_refinement_prompt(
        draft_translation="Brouillon.",
        target_language=target_language,
        has_placeholders=has_placeholders,
        prompt_options=prompt_options,
    )


# ---------------------------------------------------------------------------
# Refine prompt: placeholder block gated on has_placeholders
# ---------------------------------------------------------------------------

def test_refine_without_placeholders_has_no_placeholder_block():
    system = _refine(has_placeholders=False).system
    assert PLACEHOLDER_SECTION not in system
    assert PLACEHOLDER_REMINDER not in system
    assert "[id0]" not in system


def test_refine_with_placeholders_keeps_placeholder_block():
    system = _refine(has_placeholders=True).system
    assert PLACEHOLDER_SECTION in system
    assert PLACEHOLDER_REMINDER in system


def test_refine_without_placeholders_keeps_reminder_layout():
    """Dropping the reminder leaves exactly one blank line before the output format."""
    system = _refine(has_placeholders=False).system
    assert "literary-quality French.\n\n# OUTPUT FORMAT" in system


@pytest.mark.parametrize("has_placeholders", [True, False])
def test_post_processing_follows_has_placeholders(has_placeholders):
    system = generate_post_processing_prompt(
        "Texte.", target_language="French", has_placeholders=has_placeholders
    ).system
    assert (PLACEHOLDER_REMINDER in system) is has_placeholders
    assert (PLACEHOLDER_SECTION in system) is has_placeholders


# ---------------------------------------------------------------------------
# Text cleanup: paragraph merging only when nothing pins the layout
# ---------------------------------------------------------------------------

def test_translation_cleanup_never_merges_with_placeholders():
    system = generate_translation_prompt(
        "Text.", "", "English", "French",
        has_placeholders=True, prompt_options={"text_cleanup": True},
    ).system
    assert MERGE_BULLET not in system
    assert KEEP_BULLET in system


def test_refine_cleanup_never_merges_with_placeholders():
    system = _refine(has_placeholders=True, prompt_options={"text_cleanup": True}).system
    assert MERGE_BULLET not in system
    assert KEEP_BULLET in system


def test_refine_cleanup_never_merges_in_plain_text_mode():
    system = _refine(
        has_placeholders=False,
        prompt_options={"text_cleanup": True, "plain_text_mode": True},
    ).system
    assert MERGE_BULLET not in system
    assert KEEP_BULLET in system


def test_refine_cleanup_merges_when_layout_is_free():
    """No placeholders and no plain-text contract: the merging variant stays."""
    system = _refine(has_placeholders=False, prompt_options={"text_cleanup": True}).system
    assert MERGE_BULLET in system
    assert KEEP_BULLET not in system


# ---------------------------------------------------------------------------
# Subtitle prompts: example in the target language
# ---------------------------------------------------------------------------

def _subtitle_translate(target_language):
    return generate_subtitle_block_prompt(
        [(1, "Hello"), (2, "World")], "", "English", target_language
    ).system


def _subtitle_refine(target_language):
    return generate_subtitle_refinement_block_prompt(
        [(0, "Hello"), (1, "World")], "", target_language
    ).system


def test_subtitle_translation_example_is_in_target_language():
    system = _subtitle_translate("Japanese")
    assert "[1]最初の翻訳行\n[2]2番目の翻訳行" in system
    assert "第一行翻译文本" not in system


def test_subtitle_translation_example_falls_back_to_english():
    system = _subtitle_translate("Klingon")
    assert "[1]First translated line\n[2]Second translated line" in system
    assert "第一行翻译文本" not in system


def test_subtitle_refinement_example_is_in_target_language():
    system = _subtitle_refine("Japanese")
    assert get_subtitle_refinement_example("Japanese") in system
    assert "[0]洗練された最初の行" in system
    assert "affinée" not in system


def test_subtitle_refinement_example_falls_back_to_english():
    system = _subtitle_refine("Klingon")
    assert "[0]First refined line\n[1]Second refined line" in system
    assert "affinée" not in system


def test_subtitle_refinement_french_example_unchanged():
    assert "[0]Première ligne affinée\n[1]Deuxième ligne affinée" in _subtitle_refine("French")


def test_subtitle_translation_chinese_example_unchanged():
    assert get_subtitle_example("Chinese") == "[1]第一行翻译文本\n[2]第二行翻译文本"
    assert "[1]第一行翻译文本\n[2]第二行翻译文本" in _subtitle_translate("Chinese")


# ---------------------------------------------------------------------------
# Output example lines (moved from inline dicts into the examples module)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("has_placeholders", [True, False])
def test_translation_output_example_comes_from_helper(has_placeholders):
    system = generate_translation_prompt(
        "Text.", "", "English", "German", has_placeholders=has_placeholders
    ).system
    assert get_output_format_example("German", has_placeholders) in system


def test_translation_output_example_fallback_is_mode_independent():
    assert get_output_format_example("Klingon", True) == "Your translated text here"
    assert get_output_format_example("Klingon", False) == "Your translated text here"


def test_refinement_output_example_comes_from_helper():
    assert get_refinement_output_example("French") == "Votre texte affiné ici"
    assert "Votre texte affiné ici" in _refine(has_placeholders=False).system
    assert get_refinement_output_example("Klingon") == "Your refined text here"
