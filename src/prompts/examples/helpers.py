"""
Helper functions for translation examples.

This module provides unified access to technical examples:
- Placeholder preservation (generated dynamically)
"""

from typing import Any, Dict, Optional, Tuple

from .constants import TAG0, TAG1, TAG2
from .placeholder_examples import get_example_for_pair
from .subtitle_examples import (
    SUBTITLE_EXAMPLES,
    SUBTITLE_EXAMPLE_FALLBACK,
    SUBTITLE_REFINEMENT_EXAMPLES,
    SUBTITLE_REFINEMENT_EXAMPLE_FALLBACK,
)
from .output_examples import (
    TRANSLATION_OUTPUT_EXAMPLES,
    TRANSLATION_OUTPUT_FALLBACK,
    REFINEMENT_OUTPUT_EXAMPLES,
    REFINEMENT_OUTPUT_FALLBACK,
)
from src.utils.lang_normalize import normalize_lang_key


def get_placeholder_example(
    source_lang: str,
    target_lang: str
) -> Tuple[Dict[str, str], str, str]:
    """
    Get placeholder preservation example for a language pair.

    Generates examples dynamically for any language pair using
    pre-translated sentences in each supported language.

    Returns:
        Tuple of (example_dict, actual_source_lang, actual_target_lang).
    """
    example = get_example_for_pair(source_lang, target_lang)
    return example, source_lang, target_lang


def get_subtitle_example(target_lang: str) -> str:
    """Get the subtitle translation example ([1]/[2] markers) in the target language.

    Falls back to a neutral English example when the language has no entry.
    """
    return SUBTITLE_EXAMPLES.get(normalize_lang_key(target_lang), SUBTITLE_EXAMPLE_FALLBACK)


def get_subtitle_refinement_example(target_lang: str) -> str:
    """Get the subtitle refinement example ([0]/[1] markers) in the target language.

    Falls back to a neutral English example when the language has no entry.
    """
    return SUBTITLE_REFINEMENT_EXAMPLES.get(
        normalize_lang_key(target_lang), SUBTITLE_REFINEMENT_EXAMPLE_FALLBACK
    )


def get_output_format_example(target_lang: str, has_placeholders: bool = True) -> str:
    """Get the example line the translation prompt shows between its output tags.

    Uses the "standard" variant (mentions the placeholder marker) when the input
    carries placeholders and the "plain" variant otherwise. An unknown target
    language gets the same English fallback in both modes.
    """
    examples = TRANSLATION_OUTPUT_EXAMPLES.get(normalize_lang_key(target_lang))
    if examples is None:
        return TRANSLATION_OUTPUT_FALLBACK
    return examples["standard" if has_placeholders else "plain"]


def get_refinement_output_example(target_lang: str) -> str:
    """Get the example line the refinement prompt shows between its output tags."""
    return REFINEMENT_OUTPUT_EXAMPLES.get(
        normalize_lang_key(target_lang), REFINEMENT_OUTPUT_FALLBACK
    )


def build_placeholder_section(
    source_lang: str,
    target_lang: str,
    placeholder_format: Optional[Tuple[str, str]] = None
) -> str:
    """
    Build the placeholder preservation section with language-specific examples.

    Args:
        source_lang: Source language name
        target_lang: Target language name
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders
                          e.g., ('[', ']') for [0] or ('[[', ']]') for [[0]]
                          If None, uses default [[0]] format

    Returns formatted instructions for preserving placeholders.
    """
    # Use TAG0, TAG1, TAG2 constants (always [idN] format)
    tag0, tag1, tag2 = TAG0, TAG1, TAG2

    example, actual_source, actual_target = get_placeholder_example(source_lang, target_lang)

    # Use examples as-is (already using [idN] format)
    example_source = example['source']
    example_correct = example['correct']

    return f"""# PLACEHOLDER PRESERVATION (CRITICAL)

You will encounter placeholders like: {tag0}, {tag1}, {tag2}
These represent HTML/XML tags that have been temporarily replaced.

**UNIFIED FORMAT:**
All placeholders use the [idN] format: [id0], [id1], [id2]...
This semantic format provides the highest accuracy for preservation.

**MANDATORY RULES:**
1. Keep ALL placeholders EXACTLY as they appear
2. Do NOT translate, modify, remove, or explain them
3. Maintain their EXACT position in the sentence structure
4. Do NOT add spaces around them unless present in the source

**Example ({actual_source.title()} → {actual_target.title()}):**

Source: "{example_source}"
✅ Correct: "{example_correct}"
❌ WRONG: "{example['wrong']}" (placeholders removed)
"""


# Removed _extract_format_from_tag and _get_format_description
# These functions are no longer needed with unified [idN] format


def has_example_for_pair(source_lang: str, target_lang: str) -> bool:
    """Check if a placeholder example exists for the given language pair.

    Always returns True since examples are generated dynamically
    with fallback to English for unsupported languages.
    """
    return True


async def ensure_example_ready(
    source_lang: str,
    target_lang: str,
    provider: Optional[Any] = None
) -> bool:
    """
    Ensure a placeholder example exists for the language pair.

    Always returns True since examples are generated dynamically
    with fallback to English for unsupported languages.
    """
    return True
