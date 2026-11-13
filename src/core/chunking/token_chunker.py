"""
Token-based text chunking with natural boundary preservation.

This module provides intelligent text chunking based on token counts
using tiktoken, while respecting natural text boundaries (paragraphs and sentences).
"""
import logging
import math
import re
import threading
from typing import List, Dict, Optional

from src.config import SENTENCE_TERMINATORS

logger = logging.getLogger(__name__)

_ENCODING_NAME = "cl100k_base"
_encoder = None
_encoder_failed = False
_encoder_lock = threading.Lock()

# Scripts where one character is roughly one token (or more) for cl100k_base:
# CJK ideographs, kana, Hangul, and the fullwidth/CJK punctuation blocks.
_DENSE_SCRIPT_RE = re.compile(
    r'[　-ヿ㐀-䶿一-鿿가-힣'
    r'ᄀ-ᇿ豈-﫿＀-￯]'
)


def get_encoder():
    """Return the shared tiktoken encoder, or None if it cannot be loaded.

    tiktoken downloads the encoding file on first use. Offline, behind a
    firewall, or in a build that did not bundle the file, that download fails
    (#339). The failure is logged once and cached, and callers fall back to
    an approximate count instead of aborting the translation.
    """
    global _encoder, _encoder_failed
    if _encoder is not None or _encoder_failed:
        return _encoder
    with _encoder_lock:
        if _encoder is not None or _encoder_failed:
            return _encoder
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding(_ENCODING_NAME)
        except Exception as e:
            _encoder_failed = True
            logger.warning(
                "Could not load the tiktoken '%s' encoding (%s). Token counts "
                "will be approximated from character counts. To get exact "
                "counts offline, set TIKTOKEN_CACHE_DIR to a folder holding "
                "the encoding file.", _ENCODING_NAME, e
            )
    return _encoder


def approximate_token_count(text: str) -> int:
    """Estimate cl100k_base tokens without the encoder.

    Dense scripts (CJK, kana, Hangul) count one token per character; other
    text counts one token per ~3.5 characters, slightly pessimistic for
    Latin text so chunks stay under the configured limit.
    """
    if not text:
        return 0
    dense = len(_DENSE_SCRIPT_RE.findall(text))
    other = len(text) - dense
    return dense + math.ceil(other / 3.5)


def count_tokens(text: str) -> int:
    """Count tokens with the same encoding the chunker sizes chunks with."""
    if not text:
        return 0
    encoder = get_encoder()
    if encoder is None:
        return approximate_token_count(text)
    return len(encoder.encode(text))


def _chunk(text: str, join_before: str) -> Dict[str, str]:
    """Build a raw chunk record.

    `join_before` is the separator that re-attaches this chunk to the previous
    one, so reassembly can tell a paragraph boundary from a sentence-level split
    inside a single source paragraph.
    """
    return {"text": text, "join_before": join_before}


class TokenChunker:
    """
    Token-based text chunker that respects natural boundaries.

    Uses a soft limit approach: accumulates content until reaching ~80% of max tokens,
    then completes at the next natural boundary (paragraph or sentence).
    """

    def __init__(self, max_tokens: int = 800, soft_limit_ratio: float = 0.8):
        """
        Initialize the TokenChunker.

        Args:
            max_tokens: Maximum tokens per chunk (hard limit)
            soft_limit_ratio: Ratio at which to start looking for boundaries (default 0.8 = 80%)
        """
        self.max_tokens = max_tokens
        self.soft_limit = int(max_tokens * soft_limit_ratio)
        self.encoder: Optional[object] = get_encoder()

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        if self.encoder is None:
            return approximate_token_count(text)
        return len(self.encoder.encode(text))

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs using double newlines.

        Args:
            text: Input text

        Returns:
            List of paragraphs (preserving single newlines within)
        """
        # Split on double newlines (or more)
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter out empty paragraphs but preserve whitespace-only ones as empty markers
        return [p for p in paragraphs if p.strip()]

    def split_paragraph_into_sentences(self, paragraph: str) -> List[str]:
        """
        Split a paragraph into sentences for finer-grained chunking.

        Used when a single paragraph exceeds max_tokens.

        Args:
            paragraph: Input paragraph text

        Returns:
            List of sentences
        """
        # Create regex pattern from sentence terminators
        sorted_terminators = sorted(list(SENTENCE_TERMINATORS), key=len, reverse=True)
        escaped_terminators = [re.escape(t) for t in sorted_terminators]
        pattern = '|'.join(escaped_terminators)

        sentences = []
        last_end = 0

        for match in re.finditer(pattern, paragraph):
            end = match.end()
            sentence = paragraph[last_end:end].strip()
            if sentence:
                sentences.append(sentence)
            last_end = end

        # Add remaining text if any
        remaining = paragraph[last_end:].strip()
        if remaining:
            sentences.append(remaining)

        # If no sentences found (no terminators), return the whole paragraph
        if not sentences and paragraph.strip():
            sentences = [paragraph.strip()]

        return sentences

    @staticmethod
    def split_paragraph_into_lines(paragraph: str) -> List[str]:
        """
        Split a block on single newlines, dropping blank lines.

        Many plain-text books (most Chinese web novels among them) put one
        paragraph per line with no blank line in between, so the blank-line
        split sees the whole file, or whole chapters, as a single paragraph.

        Args:
            paragraph: Input text block

        Returns:
            List of non-empty lines, trailing whitespace removed
        """
        return [line.rstrip() for line in paragraph.split('\n') if line.strip()]

    def _edge_context(self, chunk_text: str, last: bool) -> str:
        """
        Return the last (or first) paragraph of a chunk, for use as context.

        Falls back to the last (or first) line when the chunk has no blank
        lines, so a line-split chunk does not hand its whole body over as
        context to its neighbour.
        """
        paragraphs = self.split_into_paragraphs(chunk_text)
        if not paragraphs:
            return ""
        edge = paragraphs[-1] if last else paragraphs[0]
        lines = self.split_paragraph_into_lines(edge)
        if len(lines) > 1:
            return lines[-1] if last else lines[0]
        return edge

    def _chunk_units(self, units: List[str], separator: str = "\n\n") -> List[Dict[str, str]]:
        """
        Chunk a list of text units (paragraphs or sentences) into appropriately sized chunks.

        Each item is {"text": <chunk text>, "join_before": <separator that
        re-attaches this chunk to the previous one>}. `join_before` on the first
        item of the returned list is the caller's `separator` and is ignored by
        the top-level caller.

        Args:
            units: List of text units to chunk
            separator: Separator to use when joining units

        Returns:
            List of {"text", "join_before"} dictionaries
        """
        # Minimum chunk size threshold - chunks smaller than this will be merged
        # with adjacent content rather than saved separately
        min_chunk_tokens = int(self.max_tokens * 0.25)  # 25% of max_tokens

        chunks = []
        current_units = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)

            # If single unit exceeds max, we need to handle it specially
            if unit_tokens > self.max_tokens:
                # If current chunk is too small, don't save it separately
                # Instead, prepend it to the first sentence chunk
                prefix_units = []
                if current_units and current_tokens < min_chunk_tokens:
                    prefix_units = current_units
                    current_units = []
                    current_tokens = 0
                elif current_units:
                    # Current chunk is big enough, save it
                    chunks.append(_chunk(separator.join(current_units), separator))
                    current_units = []
                    current_tokens = 0

                # Split the oversized unit: on single newlines first when it
                # has several lines (one-paragraph-per-line files), otherwise
                # into sentences. Sub-chunks carry the inner separator as
                # join_before, so reassembly restores line breaks for the
                # former and keeps the paragraph whole for the latter.
                lines = self.split_paragraph_into_lines(unit)
                if len(lines) > 1:
                    sub_units, sub_separator = lines, "\n"
                else:
                    sub_units, sub_separator = self.split_paragraph_into_sentences(unit), " "
                if len(sub_units) > 1:
                    sentence_chunks = self._chunk_units(sub_units, separator=sub_separator)

                    # Prepend small prefix to first sentence chunk if exists
                    if prefix_units and sentence_chunks:
                        prefix_text = separator.join(prefix_units)
                        prefix_tokens = self.count_tokens(prefix_text)
                        first_chunk_tokens = self.count_tokens(sentence_chunks[0]["text"])

                        # Only merge if combined size is reasonable
                        if prefix_tokens + first_chunk_tokens <= self.max_tokens:
                            sentence_chunks[0]["text"] = (
                                prefix_text + separator + sentence_chunks[0]["text"]
                            )
                        else:
                            # Prefix too big, save it separately
                            chunks.append(_chunk(prefix_text, separator))
                    elif prefix_units:
                        # No sentence chunks but have prefix
                        chunks.append(_chunk(separator.join(prefix_units), separator))

                    if sentence_chunks:
                        # The run starts a new paragraph relative to whatever
                        # preceded it; only chunks 2..n are continuations.
                        sentence_chunks[0]["join_before"] = separator

                    chunks.extend(sentence_chunks)
                else:
                    # Can't split further, prepend prefix if any
                    if prefix_units:
                        chunks.append(
                            _chunk(separator.join(prefix_units) + separator + unit, separator)
                        )
                    else:
                        chunks.append(_chunk(unit, separator))
                continue

            # Check if adding this unit would exceed limits
            potential_tokens = current_tokens + unit_tokens
            if current_units:
                # Account for separator
                potential_tokens += self.count_tokens(separator)

            # If we're past soft limit, check if we should start a new chunk
            if current_tokens >= self.soft_limit and potential_tokens > self.max_tokens:
                # Save current chunk and start new one
                chunks.append(_chunk(separator.join(current_units), separator))
                current_units = [unit]
                current_tokens = unit_tokens
            elif potential_tokens > self.max_tokens:
                # Would exceed hard limit, start new chunk
                if current_units:
                    chunks.append(_chunk(separator.join(current_units), separator))
                current_units = [unit]
                current_tokens = unit_tokens
            else:
                # Add to current chunk
                current_units.append(unit)
                current_tokens = potential_tokens

        # Don't forget the last chunk
        if current_units:
            chunks.append(_chunk(separator.join(current_units), separator))

        return chunks

    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into chunks with context preservation.

        Main algorithm:
        1. Split into paragraphs
        2. Accumulate until soft_limit (~80%)
        3. If next paragraph would exceed max_tokens, finalize chunk
        4. If single paragraph > max_tokens, split into sentences
        5. Return chunks with context_before/main_content/context_after

        Args:
            text: Input text to chunk

        Returns:
            List of chunk dictionaries with keys:
            - context_before: Last paragraph of previous chunk (for context)
            - main_content: Main content to translate
            - context_after: First paragraph of next chunk (for context)
            - join_with: Separator that re-attaches this chunk to the previous
              one on reassembly ("" for the first chunk)
        """
        if not text or not text.strip():
            return []

        # Split into paragraphs
        paragraphs = self.split_into_paragraphs(text)

        if not paragraphs:
            return []

        # Chunk paragraphs
        raw_chunks = self._chunk_units(paragraphs, separator="\n\n")

        if not raw_chunks:
            return []

        # Build structured chunks with context
        structured_chunks = []

        for i, raw_chunk in enumerate(raw_chunks):
            chunk_content = raw_chunk["text"]

            # Context before: last part of previous chunk
            if i > 0:
                context_before = self._edge_context(raw_chunks[i - 1]["text"], last=True)
            else:
                context_before = ""

            # Context after: first part of next chunk
            if i < len(raw_chunks) - 1:
                context_after = self._edge_context(raw_chunks[i + 1]["text"], last=False)
            else:
                context_after = ""

            structured_chunks.append({
                "context_before": context_before,
                "main_content": chunk_content,
                "context_after": context_after,
                "join_with": "" if i == 0 else raw_chunk["join_before"]
            })

        return structured_chunks

    def get_stats(self, chunks: List[Dict[str, str]]) -> Dict:
        """
        Get statistics about the chunked text.

        Args:
            chunks: List of chunk dictionaries from chunk_text()

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "chunks_in_range": 0,
                "compliance_rate": 0.0
            }

        token_counts = [self.count_tokens(c["main_content"]) for c in chunks]

        # Calculate how many are within acceptable range (soft_limit to max_tokens)
        in_range = sum(1 for t in token_counts if t <= self.max_tokens)

        return {
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "chunks_in_range": in_range,
            "compliance_rate": in_range / len(chunks) * 100
        }
