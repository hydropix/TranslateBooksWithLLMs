"""
Tests for Phase 2 of issue #253: the plain-text prompt states an explicit
paragraph contract and stops instructing the model to merge paragraphs.

Plain Text Mode is identified by has_placeholders=False in
generate_translation_prompt. These tests guard the has_placeholders gate so a
future change cannot leak the paragraph-structure rules onto the placeholder
(EPUB HTML) path, or vice versa.
"""
import re

import pytest

from src.prompts.prompts import (
    PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION,
    PLAIN_TEXT_KEEP_EMPHASIS_OPTION,
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


_EMPHASIS_RULE_START = "11. Keep each <i>...</i> or <em>...</em> pair"
_PLAIN_RULE_11 = "output plain translated text only"


@pytest.mark.parametrize("has_placeholders", [False, True])
@pytest.mark.parametrize("extra", [{}, {"text_cleanup": True}, {PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION: 4}])
@pytest.mark.parametrize("off_value", [False, None, 1, "true"])
def test_keep_emphasis_off_is_byte_identical(has_placeholders, extra, off_value):
    """Anything but True leaves both prompts exactly as without the key (#352)."""
    without_key = _prompt(has_placeholders, dict(extra))
    with_off = _prompt(has_placeholders, {**extra, PLAIN_TEXT_KEEP_EMPHASIS_OPTION: off_value})
    assert with_off.system == without_key.system
    assert with_off.user == without_key.user


def test_keep_emphasis_rewords_rule_11_only():
    """The option swaps rule 11 in place: same numbering, nothing else changes."""
    off = _prompt(False, {}).system
    on = _prompt(False, {PLAIN_TEXT_KEEP_EMPHASIS_OPTION: True}).system
    assert _PLAIN_RULE_11 in off and _PLAIN_RULE_11 not in on
    assert _EMPHASIS_RULE_START in on
    assert "add no other tag" in on
    assert "\n12. " not in on
    # Only the rule-11 line differs.
    diff = [(a, b) for a, b in zip(off.splitlines(), on.splitlines()) if a != b]
    assert len(off.splitlines()) == len(on.splitlines())
    assert len(diff) == 1 and diff[0][0].startswith("11. ") and diff[0][1].startswith("11. ")


def test_keep_emphasis_keeps_retry_rules_numbered_12_to_14():
    system = _prompt(False, {
        PLAIN_TEXT_KEEP_EMPHASIS_OPTION: True,
        PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION: 3,
    }).system
    assert _EMPHASIS_RULE_START in system
    assert _PLAIN_RULE_11 not in system
    numbers = [m.group(1) for m in re.finditer(r"^(\d+)\. ", system, re.MULTILINE)
               if int(m.group(1)) >= 6]
    assert numbers == ["6", "7", "8", "9", "10", "11", "12", "13", "14"]
    assert "12. RETRY:" in system
    assert "EXACTLY 3 paragraph(s)" in system


def test_keep_emphasis_never_reaches_the_placeholder_path():
    on = _prompt(True, {PLAIN_TEXT_KEEP_EMPHASIS_OPTION: True})
    off = _prompt(True, {})
    assert on == off
    assert "<em>...</em>" not in on.system


def test_keep_emphasis_leaves_the_user_prompt_alone():
    on = _prompt(False, {PLAIN_TEXT_KEEP_EMPHASIS_OPTION: True})
    off = _prompt(False, {})
    assert on.user == off.user


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
