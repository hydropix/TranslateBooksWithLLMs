"""
Hierarchical text splitting strategies for HTML content.

This module provides text splitting functionality that tries different strategies
in order of preference: sentences -> punctuation -> newlines -> force split.
"""
import re
from typing import List, Callable

from src.core.chunking.token_chunker import TokenChunker


def _split_after(text: str, boundary_chars: str) -> List[str]:
    """Split ``text`` after each run of whitespace that follows a boundary char.

    The whitespace stays attached to the preceding part, so joining the parts
    gives back ``text`` exactly. A split without whitespace is also made right
    before a legacy ``[[`` placeholder that follows a boundary char.
    """
    pieces = re.split(rf'(?<=[{boundary_chars}])(\s+)(?=\S)', text)
    parts = []
    for index in range(0, len(pieces), 2):
        separator = pieces[index + 1] if index + 1 < len(pieces) else ''
        parts.extend(re.split(rf'(?<=[{boundary_chars}])(?=\[\[)', pieces[index] + separator))
    return parts


class TextSplitter:
    """Splits text using hierarchical strategies (sentences -> punctuation -> newlines -> force)."""

    def __init__(self, max_tokens: int, token_chunker: TokenChunker):
        """
        Initialize the TextSplitter.

        Args:
            max_tokens: Maximum tokens per segment
            token_chunker: TokenChunker instance for counting tokens
        """
        self.max_tokens = max_tokens
        self.token_chunker = token_chunker

    def split_oversized_segment(self, segment: str) -> List[str]:
        """
        Split oversized segment hierarchically.

        Tries strategies in order:
        1. Sentence boundaries
        2. Punctuation (; : ,)
        3. Newlines
        4. Force split at token boundaries

        Args:
            segment: Text segment to split

        Returns:
            List of smaller segments that fit within max_tokens
        """
        # Try sentence splitting first
        sentence_segments = self._split_on_sentences(segment)

        result_segments = []
        for sent_seg in sentence_segments:
            sent_tokens = self.token_chunker.count_tokens(sent_seg)

            if sent_tokens <= self.max_tokens:
                result_segments.append(sent_seg)
            else:
                # Sentence is still too large, try punctuation split
                punct_segments = self._split_on_punctuation(sent_seg)

                for punct_seg in punct_segments:
                    punct_tokens = self.token_chunker.count_tokens(punct_seg)

                    if punct_tokens <= self.max_tokens:
                        result_segments.append(punct_seg)
                    else:
                        # Still too large, try newline split
                        newline_segments = self._split_on_newlines(punct_seg)

                        for nl_seg in newline_segments:
                            nl_tokens = self.token_chunker.count_tokens(nl_seg)

                            if nl_tokens <= self.max_tokens:
                                result_segments.append(nl_seg)
                            else:
                                # Last resort: force split at max_tokens
                                force_segments = self._force_split_at_tokens(nl_seg)
                                result_segments.extend(force_segments)

        return result_segments

    def _split_with_strategy(
        self,
        text: str,
        split_func: Callable[[str], List[str]],
        joiner: str = ""
    ) -> List[str]:
        """
        Generic split method that eliminates code duplication.

        Splits text using the provided function and merges parts
        that fit within token limits.

        Args:
            text: Text to split
            split_func: Function that splits text into parts
            joiner: String to join parts with (e.g., '\n' for newlines)

        Returns:
            List of segments that fit within max_tokens
        """
        parts = split_func(text)
        segments = []
        current = ""

        for part in parts:
            if not part.strip():
                continue

            test_text = current + joiner + part if current else part
            if current and self.token_chunker.count_tokens(test_text) > self.max_tokens:
                if current.strip():
                    segments.append(current)
                current = part
            else:
                current = test_text

        if current.strip():
            segments.append(current)

        return segments if segments else [text]

    def _split_on_sentences(self, text: str) -> List[str]:
        """
        Split on sentence boundaries (. ! ?) while preserving placeholders.

        Returns:
            List of sentence segments
        """
        # Sentence boundaries (. ! ? followed by space or placeholder). The
        # whitespace stays with the preceding sentence: dropping it glued the
        # sentences back together as "end.Next".
        parts = _split_after(text, '.!?')
        return self._split_with_strategy(text, lambda t: parts)

    def _split_on_punctuation(self, text: str) -> List[str]:
        """
        Split on strong punctuation (; : ,) while preserving placeholders.

        Returns:
            List of punctuation-split segments
        """
        # Split on ; : , keeping the punctuation and the following whitespace
        # with the preceding text
        parts = _split_after(text, ';:,')
        return self._split_with_strategy(text, lambda t: parts)

    def _split_on_newlines(self, text: str) -> List[str]:
        """
        Split on newlines while preserving placeholders.

        Returns:
            List of newline-split segments
        """
        parts = text.splitlines(keepends=True)
        return self._split_with_strategy(text, lambda t: parts)

    def _force_split_at_tokens(self, text: str) -> List[str]:
        """
        Force split text at max_tokens boundary as last resort.
        Tries to split at word boundaries when possible.

        Returns:
            List of force-split segments
        """
        segments = []
        remaining = text

        while remaining:
            # If remaining text fits, we're done
            if self.token_chunker.count_tokens(remaining) <= self.max_tokens:
                segments.append(remaining)
                break

            # Binary search for the largest prefix that fits
            left, right = 0, len(remaining)
            best_pos = left

            while left <= right:
                mid = (left + right) // 2
                prefix = remaining[:mid]
                token_count = self.token_chunker.count_tokens(prefix)

                if token_count <= self.max_tokens:
                    best_pos = mid
                    left = mid + 1
                else:
                    right = mid - 1

            # Find word boundary near best_pos
            split_pos = self._find_word_boundary_near(remaining, best_pos)

            # Avoid infinite loop - ensure we make progress
            if split_pos == 0:
                split_pos = max(1, best_pos)

            segments.append(remaining[:split_pos])
            remaining = remaining[split_pos:].lstrip()

        return segments

    def _find_word_boundary_near(self, text: str, pos: int) -> int:
        """
        Find a word boundary (space) near the given position.
        Looks backward first to avoid cutting words.

        Args:
            text: Text to search in
            pos: Position to search near

        Returns:
            Position of word boundary
        """
        if pos >= len(text):
            return len(text)

        # Look backward for a space
        for i in range(pos, max(0, pos - 50), -1):
            if text[i] in ' \n\t':
                return i + 1

        # If no space found backward, look forward
        for i in range(pos, min(len(text), pos + 50)):
            if text[i] in ' \n\t':
                return i + 1

        # No word boundary found, split at pos
        return pos
