"""Utility functions for HTML processing with placeholders.

This module provides helper functions for working with HTML text that has
been processed with placeholder substitution. These utilities are used primarily
in EPUB translation workflows.
"""

import re
from typing import Dict, Optional, Tuple, Union
from src.common.cjk_punctuation import skip_closing_punctuation
from src.common.placeholder_format import PlaceholderFormat

# A Unicode letter: any word character that is neither a digit nor an
# underscore. Everything else (digits, Latin and CJK punctuation, symbols,
# whitespace) carries nothing an LLM could translate.
_UNICODE_LETTER_RE = re.compile(r'[^\W\d_]', re.UNICODE)


def _resolve_placeholder_format(
    placeholder_format: Optional[Union[Tuple[str, str], PlaceholderFormat]]
) -> PlaceholderFormat:
    """Resolve the several shapes callers pass as a placeholder format.

    The chunk loop passes a legacy (prefix, suffix) tuple; other callers pass
    a PlaceholderFormat or nothing at all. Same convention as
    reinsert_placeholders below.
    """
    if placeholder_format is None:
        return PlaceholderFormat.from_config()
    if hasattr(placeholder_format, 'remove_all'):
        return placeholder_format
    prefix, suffix = placeholder_format
    # Legacy tuple format: the pattern is always the unified [idN] one
    return PlaceholderFormat(prefix, suffix, r'\[id(\d+)\]')


def is_text_free_chunk(
    chunk_text: str,
    placeholder_format: Optional[Union[Tuple[str, str], PlaceholderFormat]] = None
) -> bool:
    """True if the chunk carries no translatable characters.

    Procedure:
      1. Remove every placeholder occurrence.
      2. Return True iff the remainder contains no Unicode letter.

    Consequences that are INTENDED, not accidental:
      - '[id0]'          -> True   (cover page: pure markup, an <svg> image)
      - '[id0]==[id1]'   -> True   (separator paragraph, nothing to translate)
      - '[id0]……[id1]'   -> True   (CJK ellipsis only)
      - '[id0]第1章[id1]' -> False  (has letters)
      - '[id0]2024[id1]' -> True   (digits are not letters; a bare number is
                                    correct as-is in every target language)

    Args:
        chunk_text: Chunk text with placeholders
        placeholder_format: Optional (prefix, suffix) tuple or PlaceholderFormat.
                          If None, uses the unified format [idN]

    Returns:
        True when the chunk must pass through verbatim instead of being sent
        to the LLM
    """
    if not chunk_text:
        return True

    fmt = _resolve_placeholder_format(placeholder_format)
    return _UNICODE_LETTER_RE.search(fmt.remove_all(chunk_text)) is None


def extract_text_and_positions(text_with_placeholders: str) -> Tuple[str, Dict[int, float]]:
    """
    Extract pure text and calculate relative positions of placeholders.

    Uses unified placeholder format: [idN]

    Args:
        text_with_placeholders: "[id0]Hello [id1]world[id2]"

    Returns:
        ("Hello world", {0: 0.0, 1: 0.46, 2: 1.0})
    """
    # Use centralized placeholder format
    fmt = PlaceholderFormat.from_config()

    # Text without placeholders
    pure_text = fmt.remove_all(text_with_placeholders)
    pure_length = len(pure_text)

    if pure_length == 0:
        # Edge case: only placeholders, no text
        placeholders = fmt.find_all(text_with_placeholders)
        return "", {idx: i / max(1, len(placeholders))
                    for i, (_, _, _, idx) in enumerate(placeholders)}

    # Calculate relative position of each placeholder
    positions = {}

    for start, end, placeholder, idx in fmt.find_all(text_with_placeholders):
        # Text before this placeholder (without previous placeholders)
        text_before = fmt.remove_all(text_with_placeholders[:start])
        relative_pos = len(text_before) / pure_length
        positions[idx] = relative_pos

    return pure_text, positions


def reinsert_placeholders(
    translated_text: str,
    positions: Dict[int, float],
    placeholder_format: Optional[Tuple[str, str]] = None
) -> str:
    """
    Reinsert placeholders at proportional positions.

    Args:
        translated_text: "Bonjour monde"
        positions: {0: 0.0, 1: 0.5, 2: 1.0}
        placeholder_format: Optional (prefix, suffix) tuple.
                          If None, uses unified format [idN]
                          Examples: ("[id", "]")

    Returns:
        "[id0]Bonjour [id1]monde[id2]"
    """
    if not positions:
        return translated_text

    # Use centralized PlaceholderFormat (supports the legacy tuple format)
    fmt = _resolve_placeholder_format(placeholder_format)

    text_length = len(translated_text)

    # Convert relative positions to absolute positions
    insertions = []
    for idx, rel_pos in positions.items():
        abs_pos = int(rel_pos * text_length)
        # Adjust to not cut a word (find nearest word boundary)
        abs_pos = find_nearest_word_boundary(translated_text, abs_pos)
        insertions.append((abs_pos, idx))

    # CRITICAL: Sort by position (descending) then by index (ASCENDING to preserve order)
    # When multiple placeholders have the same position, we must preserve their
    # sequential order (0, 1, 2...) to avoid tag mismatches.
    # We insert from end to start (reverse position) to avoid position shifting,
    # but within the same position, we insert in sequential order.
    insertions.sort(key=lambda x: (-x[0], x[1]))

    result = translated_text
    for abs_pos, idx in insertions:
        placeholder = fmt.create(idx)
        result = result[:abs_pos] + placeholder + result[abs_pos:]

    return result


def find_nearest_word_boundary(text: str, pos: int) -> int:
    """
    Find the nearest word boundary to the given position.
    Avoids cutting in the middle of a word.

    Handles multi-byte Unicode characters and various whitespace types.

    Args:
        text: The text to search within
        pos: The position to find a boundary near

    Returns:
        The position of the nearest word boundary
    """
    if pos <= 0:
        return 0
    if pos >= len(text):
        return len(text)

    # Word boundary characters (spaces, punctuation, etc.)
    # Includes various Unicode whitespace and CJK punctuation
    def is_boundary(char: str) -> bool:
        return char in ' \t\n\r\u00A0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u202F\u205F\u3000.,;:!?\u3001\u3002\uff0c\uff1a\uff1b\uff1f\uff01'

    # If we're already on a boundary, perfect
    if is_boundary(text[pos]):
        return skip_closing_punctuation(text, pos)

    # Find the nearest boundary (before or after)
    left = pos
    right = pos

    while left > 0 and not is_boundary(text[left]):
        left -= 1
    while right < len(text) and not is_boundary(text[right]):
        right += 1

    # Return the closest one, after any full-width closing mark ("。", "」"):
    # such a mark ends the text before it and must not open the next block.
    nearest = left if pos - left <= right - pos else right
    if nearest > 0:
        nearest = skip_closing_punctuation(text, nearest)
    return nearest
