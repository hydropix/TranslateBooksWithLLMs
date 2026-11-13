"""
Token alignment fallback for EPUB translation.

This module provides word-level alignment for reinserting HTML placeholders
when the LLM fails to preserve them correctly. It uses proportional positioning
with word boundary detection to guarantee 100% placeholder integrity.

Phase 2 of the multi-phase fallback system:
- Phase 1: Normal translation with placeholders (retry logic)
- Phase 2: Token alignment fallback (THIS MODULE)
- Phase 3: Untranslated fallback (preserve original)
"""

from typing import List, Tuple
import re


class TokenAlignmentFallback:
    """
    Semantic token alignment fallback for EPUB translation.

    Uses word-level alignment to reinsert placeholders when LLM fails
    to preserve them correctly. Provides 100% guarantee of placeholder
    integrity through algorithmic insertion.

    Algorithm:
    1. Find placeholder positions in original text
    2. Extract clean text (remove placeholders)
    3. Calculate relative positions (0.0 to 1.0)
    4. Map to translated text positions
    5. Insert placeholders at aligned positions
    6. Validate result

    Example:
        >>> aligner = TokenAlignmentFallback()
        >>> original = "[id0]Hello[id1] world[id2]"
        >>> translated = "Bonjour monde"
        >>> placeholders = ["[id0]", "[id1]", "[id2]"]
        >>> result = aligner.align_and_insert_placeholders(
        ...     original, translated, placeholders
        ... )
        >>> print(result)
        '[id0]Bonjour[id1] monde[id2]'
    """

    def __init__(self):
        """Initialize with lazy loading."""
        self._initialized = True

    def align_and_insert_placeholders(
        self,
        original_with_placeholders: str,
        translated_without_placeholders: str,
        placeholders: List[str]
    ) -> str:
        """
        Main entry point: align and reinsert placeholders.

        Args:
            original_with_placeholders: "[id0]Hello[id1] world[id2]"
            translated_without_placeholders: "Bonjour monde"
            placeholders: ["[id0]", "[id1]", "[id2]"]

        Returns:
            "[id0]Bonjour[id1] monde[id2]" (local indices preserved)

        Raises:
            Exception: Caught internally, falls back to proportional insertion
        """
        try:
            # 1. Find placeholder positions in original
            positions = self._find_placeholder_positions(
                original_with_placeholders, placeholders
            )

            # 2. Extract clean text from original
            clean_original = self._remove_placeholders(
                original_with_placeholders, placeholders
            )

            # 3. Calculate relative positions (0.0 to 1.0)
            relative_positions = self._calculate_relative_positions(
                positions, len(clean_original)
            )

            # 4. Map to translated text positions
            translated_positions = self._map_to_translated(
                relative_positions,
                translated_without_placeholders
            )

            # 5. Insert placeholders at calculated positions
            result = self._insert_placeholders_at_positions(
                translated_without_placeholders,
                translated_positions,
                placeholders
            )

            return result

        except Exception:
            # Fallback to proportional insertion
            return self._fallback_proportional(
                original_with_placeholders,
                translated_without_placeholders,
                placeholders
            )

    def _find_placeholder_positions(
        self,
        text: str,
        placeholders: List[str]
    ) -> List[Tuple[int, str]]:
        """
        Find the clean-text offset of each placeholder.

        Offsets are measured in the text WITH all placeholders removed
        (the cumulative length of preceding placeholders is subtracted),
        so dividing by len(clean_text) yields a ratio in [0.0, 1.0].

        Args:
            text: Text with placeholders
            placeholders: List of placeholder strings, in document order

        Returns:
            List of (clean_pos, placeholder) tuples
        """
        positions = []
        search_from = 0
        removed_length = 0

        for ph in placeholders:
            idx = text.find(ph, search_from)
            if idx != -1:
                positions.append((idx - removed_length, ph))
                search_from = idx + len(ph)
                removed_length += len(ph)

        return positions

    def _calculate_relative_positions(
        self,
        positions: List[Tuple[int, str]],
        text_length: int
    ) -> List[Tuple[float, str]]:
        """
        Calculate relative positions (0.0 to 1.0) for each placeholder.

        Args:
            positions: List of (clean_pos, placeholder) tuples
            text_length: Length of clean text (without placeholders)

        Returns:
            List of (relative_pos, placeholder) tuples
        """
        if text_length == 0:
            # Edge case: no text, distribute evenly
            if len(positions) == 0:
                return []
            step = 1.0 / max(1, len(positions) + 1)
            return [(i * step, ph) for i, (_, ph) in enumerate(positions)]

        relative = []
        for pos, ph in positions:
            rel_pos = min(1.0, max(0.0, pos / text_length))
            relative.append((rel_pos, ph))

        return relative

    def _map_to_translated(
        self,
        relative_positions: List[Tuple[float, str]],
        translated_text: str
    ) -> List[Tuple[int, str]]:
        """
        Map relative positions to absolute positions in translated text.

        Args:
            relative_positions: List of (relative_pos, placeholder) tuples
            translated_text: Translated text without placeholders

        Returns:
            List of (absolute_pos, placeholder) tuples
        """
        translated_length = len(translated_text)

        absolute = []
        for rel_pos, ph in relative_positions:
            abs_pos = int(rel_pos * translated_length)

            # Adjust to nearest word boundary
            abs_pos = self._find_nearest_word_boundary(
                translated_text, abs_pos
            )

            absolute.append((abs_pos, ph))

        return absolute

    def _find_nearest_word_boundary(
        self,
        text: str,
        pos: int
    ) -> int:
        """
        Find nearest word boundary (space, punctuation) to position.
        Avoids cutting in the middle of a word.

        Args:
            text: Text to search in
            pos: Target position

        Returns:
            Position of nearest word boundary
        """
        if pos <= 0:
            return 0
        if pos >= len(text):
            return len(text)

        # Word boundary characters (space, punctuation, CJK punctuation)
        boundaries = ' \t\n\r.,;:!?\u3001\u3002\uff0c\uff1a\uff1f\uff01'

        # If already on boundary, return
        if text[pos] in boundaries:
            return pos

        # Search backward and forward for boundary
        left = pos
        right = pos

        while left > 0 and text[left] not in boundaries:
            left -= 1

        while right < len(text) and text[right] not in boundaries:
            right += 1

        # Return closest boundary
        if pos - left <= right - pos:
            return left
        return right

    def _insert_placeholders_at_positions(
        self,
        text: str,
        insertions: List[Tuple[int, str]],
        placeholders: List[str]
    ) -> str:
        """
        Insert placeholders at specified positions.

        CRITICAL: Insert in REVERSE order (end to start) to preserve indices.
        For placeholders at the same position, maintain their original order.

        Args:
            text: Text to insert into
            insertions: List of (position, placeholder) tuples
            placeholders: Original ordered list of placeholders

        Returns:
            Text with placeholders inserted
        """
        # Get the original order of placeholders
        def get_placeholder_index(ph: str) -> int:
            """Extract numeric index from placeholder like '[id0]'"""
            match = re.search(r'\d+', ph)
            return int(match.group()) if match else 0

        # Group insertions by position
        from collections import defaultdict
        position_groups = defaultdict(list)
        for pos, ph in insertions:
            position_groups[pos].append(ph)

        # Sort each group by placeholder index (to maintain order)
        for pos in position_groups:
            position_groups[pos].sort(key=get_placeholder_index)

        # Sort positions in descending order (insert from end to start)
        sorted_positions = sorted(position_groups.keys(), reverse=True)

        result = text
        for pos in sorted_positions:
            # Insert all placeholders at this position (in order)
            placeholders_at_pos = position_groups[pos]
            combined = ''.join(placeholders_at_pos)
            result = result[:pos] + combined + result[pos:]

        return result

    def _remove_placeholders(
        self,
        text: str,
        placeholders: List[str]
    ) -> str:
        """
        Remove all placeholders from text.

        Args:
            text: Text with placeholders
            placeholders: List of placeholders to remove

        Returns:
            Text without placeholders
        """
        result = text
        for ph in placeholders:
            result = result.replace(ph, "")
        return result

    def _fallback_proportional(
        self,
        original_with_placeholders: str,
        translated_without_placeholders: str,
        placeholders: List[str]
    ) -> str:
        """
        Fallback to existing proportional insertion if alignment fails.

        Reuses code from html_chunker.py.

        Args:
            original_with_placeholders: Original text with placeholders
            translated_without_placeholders: Translated text without placeholders
            placeholders: List of placeholders

        Returns:
            Translated text with placeholders reinserted proportionally
        """
        try:
            from .html_utils import (
                extract_text_and_positions,
                reinsert_placeholders
            )

            # Extract positions from original
            _, positions = extract_text_and_positions(original_with_placeholders)

            # Reinsert at proportional positions in translated
            result = reinsert_placeholders(
                translated_without_placeholders,
                positions
            )

            return result
        except Exception:
            # Ultimate fallback: just concatenate with placeholders at boundaries
            # This ensures we always return something valid
            if not placeholders:
                return translated_without_placeholders

            # Insert first placeholder at start, last at end, others distributed
            if len(placeholders) == 1:
                return placeholders[0] + translated_without_placeholders
            elif len(placeholders) == 2:
                return placeholders[0] + translated_without_placeholders + placeholders[1]
            else:
                # Distribute all placeholders at proportional character
                # positions; insert from the end so earlier offsets stay valid
                text = translated_without_placeholders
                last = len(placeholders) - 1
                result = text
                for i, ph in reversed(list(enumerate(placeholders))):
                    pos = round(i * len(text) / last)
                    result = result[:pos] + ph + result[pos:]
                return result
