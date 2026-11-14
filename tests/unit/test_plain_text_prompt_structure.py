"""
Tests for Phase 2 of issue #253: the plain-text prompt states an explicit
paragraph contract and stops instructing the model to merge paragraphs.

Plain Text Mode is identified by has_placeholders=False in
generate_translation_prompt. These tests guard the has_placeholders gate so a
future change cannot leak the paragraph-structure rules onto the placeholder
(EPUB HTML) path, or vice versa.
"""
import pytest

from src.prompts.prompts import (
    PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION,
    generate_translation_prompt,
)


def _prompt(has_placeholders, prompt_options=None):
    return generate_translation_prompt(
        main_content="Paragraph one.\n\nParagraph two.",
        previous_translation_context="",
        source_language="English",
        target_language="French",
        has_placeholders=has_placeholders,
        prompt_options=prompt_options,
    )


def test_plain_text_mode_states_paragraph_contract():
    """has_placeholders=False emits the explicit paragraph-count contract."""
    result = _prompt(has_placeholders=False)
    assert "Output EXACTLY the same number of paragraphs" in result.system


def test_placeholder_mode_does_not_state_paragraph_contract():
    """has_placeholders=True (EPUB/placeholder path) must not be perturbed."""
    result = _prompt(has_placeholders=True)
    assert "Output EXACTLY the same number of paragraphs" not in result.system


def test_plain_text_cleanup_forbids_merging():
    """With text_cleanup on, Plain Text Mode forbids merging/splitting paragraphs."""
    result = _prompt(has_placeholders=False, prompt_options={"text_cleanup": True})
    assert "never merge or split paragraphs" in result.system
    assert "Merge incorrectly split paragraphs" not in result.system


def test_placeholder_cleanup_does_not_merge():
    """The placeholder path demands the exact layout, so cleanup must not ask to merge."""
    result = _prompt(has_placeholders=True, prompt_options={"text_cleanup": True})
    assert "never merge or split paragraphs" in result.system
    assert "Merge incorrectly split paragraphs" not in result.system


def test_first_attempt_never_carries_the_retry_rules():
    """Without the count option the prompt is exactly the contract it always was."""
    for options in (None, {}, {"text_cleanup": True}, {"plain_text_mode": True}):
        system = _prompt(has_placeholders=False, prompt_options=options).system
        assert "RETRY:" not in system
        assert "Count the paragraphs before answering" not in system


def test_retry_states_the_expected_count():
    """The retry names the number and defends the blocks models drop (#253)."""
    system = _prompt(
        has_placeholders=False,
        prompt_options={PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION: 7},
    ).system
    # The base contract is still there; the retry rules are added to it.
    assert "Output EXACTLY the same number of paragraphs" in system
    assert "RETRY:" in system
    assert "EXACTLY 7 paragraph(s)" in system
    assert "author's note" in system


def test_retry_rules_never_reach_the_placeholder_path():
    """The placeholder pipeline has its own structural contract; leave it alone."""
    system = _prompt(
        has_placeholders=True,
        prompt_options={PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION: 7},
    ).system
    assert "RETRY:" not in system
    assert "EXACTLY 7 paragraph(s)" not in system


@pytest.mark.parametrize("value", [0, -1, None, "3", 3.0, True, False])
def test_a_non_positive_int_count_is_ignored(value):
    """A malformed hint degrades to the base contract instead of a broken rule.

    `True` is in the list on purpose: bool is an int subclass, and 'EXACTLY True
    paragraph(s)' is not an instruction.
    """
    system = _prompt(
        has_placeholders=False,
        prompt_options={PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION: value},
    ).system
    assert "Output EXACTLY the same number of paragraphs" in system
    assert "RETRY:" not in system


def test_other_sections_and_their_order_are_unchanged():
    """Placeholder, glossary and final-reminder sections and their relative
    order are unchanged in both Plain Text Mode and the placeholder path."""
    for has_placeholders in (False, True):
        result = generate_translation_prompt(
            main_content="Paragraph one.\n\nParagraph two.",
            previous_translation_context="",
            source_language="English",
            target_language="French",
            has_placeholders=has_placeholders,
            prompt_options={},
            glossary_block="GLOSSARY: foo -> bar",
        )
        system = result.system

        final_reminder_idx = system.index("# FINAL REMINDER: YOUR OUTPUT LANGUAGE")

        if has_placeholders:
            placeholder_idx = system.index("placeholder")
            assert placeholder_idx < final_reminder_idx

        # Glossary lives in the user prompt, not the system prompt, and must
        # stay there regardless of has_placeholders.
        assert "GLOSSARY: foo -> bar" in result.user
        assert "GLOSSARY: foo -> bar" not in system

        # The final reminder section must remain present and be the last
        # major section before the output format block in both modes.
        output_format_idx = system.index("# OUTPUT FORMAT")
        assert final_reminder_idx < output_format_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
