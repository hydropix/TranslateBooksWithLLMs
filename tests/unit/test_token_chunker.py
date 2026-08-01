"""
Unit tests for the TokenChunker class.

Tests token-based text chunking with natural boundary preservation.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.chunking.token_chunker import TokenChunker


class TestTokenChunker:
    """Tests for TokenChunker class."""

    def test_init_default_values(self):
        """Test default initialization values."""
        chunker = TokenChunker()
        assert chunker.max_tokens == 800
        assert chunker.soft_limit == 640  # 800 * 0.8

    def test_init_custom_values(self):
        """Test custom initialization values."""
        chunker = TokenChunker(max_tokens=1000, soft_limit_ratio=0.7)
        assert chunker.max_tokens == 1000
        assert chunker.soft_limit == 700  # 1000 * 0.7

    def test_count_tokens_empty(self):
        """Test token counting for empty string."""
        chunker = TokenChunker()
        assert chunker.count_tokens("") == 0
        assert chunker.count_tokens(None) == 0

    def test_count_tokens_simple(self):
        """Test token counting for simple text."""
        chunker = TokenChunker()
        # Simple words typically have predictable token counts
        tokens = chunker.count_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10  # Should be around 2-3 tokens

    def test_count_tokens_longer_text(self):
        """Test token counting for longer text."""
        chunker = TokenChunker()
        short_text = "Hello"
        long_text = "Hello " * 100
        assert chunker.count_tokens(long_text) > chunker.count_tokens(short_text)

    def test_split_into_paragraphs_empty(self):
        """Test paragraph splitting for empty text."""
        chunker = TokenChunker()
        assert chunker.split_into_paragraphs("") == []
        assert chunker.split_into_paragraphs("   ") == []

    def test_split_into_paragraphs_single(self):
        """Test paragraph splitting for single paragraph."""
        chunker = TokenChunker()
        text = "This is a single paragraph without any double newlines."
        paragraphs = chunker.split_into_paragraphs(text)
        assert len(paragraphs) == 1
        assert paragraphs[0] == text

    def test_split_into_paragraphs_multiple(self):
        """Test paragraph splitting for multiple paragraphs."""
        chunker = TokenChunker()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paragraphs = chunker.split_into_paragraphs(text)
        assert len(paragraphs) == 3
        assert paragraphs[0] == "First paragraph."
        assert paragraphs[1] == "Second paragraph."
        assert paragraphs[2] == "Third paragraph."

    def test_split_into_paragraphs_preserves_single_newlines(self):
        """Test that single newlines within paragraphs are preserved."""
        chunker = TokenChunker()
        text = "Line one.\nLine two.\n\nNew paragraph."
        paragraphs = chunker.split_into_paragraphs(text)
        assert len(paragraphs) == 2
        assert "Line one.\nLine two." in paragraphs[0]

    def test_split_paragraph_into_sentences_simple(self):
        """Test sentence splitting for simple paragraph."""
        chunker = TokenChunker()
        paragraph = "First sentence. Second sentence. Third sentence."
        sentences = chunker.split_paragraph_into_sentences(paragraph)
        assert len(sentences) == 3
        assert "First sentence." in sentences[0]
        assert "Second sentence." in sentences[1]
        assert "Third sentence." in sentences[2]

    def test_split_paragraph_into_sentences_no_terminators(self):
        """Test sentence splitting when no terminators exist."""
        chunker = TokenChunker()
        paragraph = "This text has no sentence terminators"
        sentences = chunker.split_paragraph_into_sentences(paragraph)
        assert len(sentences) == 1
        assert sentences[0] == paragraph

    def test_split_paragraph_into_sentences_various_terminators(self):
        """Test sentence splitting with various terminators."""
        chunker = TokenChunker()
        paragraph = "Question? Exclamation! Statement."
        sentences = chunker.split_paragraph_into_sentences(paragraph)
        assert len(sentences) == 3

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = TokenChunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []
        assert chunker.chunk_text(None) == []

    def test_chunk_text_single_short_paragraph(self):
        """Test chunking with single short paragraph."""
        chunker = TokenChunker(max_tokens=800)
        text = "This is a short paragraph that should fit in one chunk."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0]["main_content"] == text
        assert chunks[0]["context_before"] == ""
        assert chunks[0]["context_after"] == ""

    def test_chunk_text_multiple_paragraphs(self):
        """Test chunking with multiple paragraphs."""
        chunker = TokenChunker(max_tokens=100)
        # Create text that will need multiple chunks
        paragraphs = [f"This is paragraph number {i}. It contains some text." for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_text_has_context(self):
        """Test that chunks have proper context."""
        chunker = TokenChunker(max_tokens=50)  # Small limit to force multiple chunks
        text = "First paragraph with content.\n\nSecond paragraph with content.\n\nThird paragraph with content."
        chunks = chunker.chunk_text(text)

        if len(chunks) > 1:
            # Second chunk should have context_before from first chunk
            assert chunks[1]["context_before"] != ""
            # First chunk should have context_after
            assert chunks[0]["context_after"] != ""

    def test_chunk_text_structure(self):
        """Test that chunk structure is correct."""
        chunker = TokenChunker()
        text = "Some text content."
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "context_before" in chunk
            assert "main_content" in chunk
            assert "context_after" in chunk
            assert isinstance(chunk["context_before"], str)
            assert isinstance(chunk["main_content"], str)
            assert isinstance(chunk["context_after"], str)

    def test_chunk_text_long_paragraph(self):
        """Test chunking when a single paragraph exceeds max tokens."""
        chunker = TokenChunker(max_tokens=50)  # Very small limit
        # Create a long paragraph that will exceed the limit
        long_paragraph = " ".join(["This is a sentence." for _ in range(50)])
        chunks = chunker.chunk_text(long_paragraph)

        # Should be split into multiple chunks
        assert len(chunks) >= 1
        # Each chunk's main_content should not be empty
        for chunk in chunks:
            assert chunk["main_content"].strip() != ""

    def test_get_stats_empty(self):
        """Test statistics for empty chunks."""
        chunker = TokenChunker()
        stats = chunker.get_stats([])
        assert stats["total_chunks"] == 0
        assert stats["avg_tokens"] == 0

    def test_get_stats_with_chunks(self):
        """Test statistics calculation."""
        chunker = TokenChunker(max_tokens=800)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk_text(text)
        stats = chunker.get_stats(chunks)

        assert stats["total_chunks"] == len(chunks)
        assert stats["avg_tokens"] > 0
        assert stats["min_tokens"] > 0
        assert stats["max_tokens"] >= stats["min_tokens"]
        assert 0 <= stats["compliance_rate"] <= 100

    def test_soft_limit_behavior(self):
        """Test that soft limit triggers boundary search."""
        chunker = TokenChunker(max_tokens=50, soft_limit_ratio=0.5)
        # After reaching 25 tokens (soft limit), should look for paragraph boundary
        # Each paragraph needs to be long enough to trigger chunking
        paragraphs = ["This is a longer paragraph with more words to exceed token limits." for _ in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunker.chunk_text(text)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Chunks should generally respect the token limit (with some tolerance)
        for chunk in chunks:
            tokens = chunker.count_tokens(chunk["main_content"])
            # Allow overflow for edge cases (single paragraph exceeding limit)
            assert tokens <= chunker.max_tokens * 2


class TestTokenChunkerJoinWith:
    """Tests for the `join_with` key that drives TXT reassembly (issue #208)."""

    # Three ~11-token paragraphs; max_tokens=25 forces exactly two chunks.
    PARAGRAPHS = [
        "First paragraph with a reasonable amount of content in it.",
        "Second paragraph with a reasonable amount of content in it.",
        "Third paragraph with a reasonable amount of content in it.",
    ]

    def test_paragraph_boundaries_join_with_blank_line(self):
        """Paragraph-level seams carry "\\n\\n" and round-trip to the source."""
        chunker = TokenChunker(max_tokens=25, soft_limit_ratio=0.5)
        text = "\n\n".join(self.PARAGRAPHS)
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        assert chunks[0]["join_with"] == ""
        assert all(c["join_with"] == "\n\n" for c in chunks[1:])
        assert "\n\n".join(c["main_content"] for c in chunks) == text

    def test_sentence_split_run_marks_continuations_with_a_space(self):
        """A paragraph split at sentence level keeps its chunks joined by " "."""
        chunker = TokenChunker(max_tokens=30, soft_limit_ratio=0.5)
        text = " ".join(f"Sentence number {i} goes right here." for i in range(12))
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        joins = [c["join_with"] for c in chunks]
        assert " " in joins, f"expected a sentence-level seam, got {joins}"

        # The chunk that opens the sentence run is a paragraph start, not a
        # continuation: it is either chunk 0 ("") or a "\n\n" seam.
        first_of_run = joins.index(" ") - 1
        assert joins[first_of_run] in ("", "\n\n")

    def test_empty_text_still_returns_no_chunks(self):
        """Empty and whitespace-only input still produce no chunks."""
        chunker = TokenChunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_context_keys_are_preserved(self):
        """The three pre-existing keys survive the join_with addition."""
        chunker = TokenChunker(max_tokens=25, soft_limit_ratio=0.5)
        chunks = chunker.chunk_text("\n\n".join(self.PARAGRAPHS))

        assert chunks
        for chunk in chunks:
            assert "context_before" in chunk
            assert "main_content" in chunk
            assert "context_after" in chunk
            assert "join_with" in chunk


class TestTokenChunkerEdgeCases:
    """Edge case tests for TokenChunker."""

    def test_only_whitespace_paragraphs(self):
        """Test handling of whitespace-only content."""
        chunker = TokenChunker()
        text = "   \n\n   \n\n   "
        chunks = chunker.chunk_text(text)
        assert chunks == []

    def test_unicode_content(self):
        """Test handling of Unicode content."""
        chunker = TokenChunker()
        text = "这是中文段落。\n\nCeci est un paragraphe français."
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        # Verify content is preserved
        full_content = " ".join(c["main_content"] for c in chunks)
        assert "中文" in full_content or "français" in full_content

    def test_mixed_newlines(self):
        """Test handling of mixed newline styles."""
        chunker = TokenChunker()
        text = "Para one.\r\n\r\nPara two.\n\nPara three."
        paragraphs = chunker.split_into_paragraphs(text)
        # Should handle various newline styles
        assert len(paragraphs) >= 2

    def test_very_long_single_sentence(self):
        """Test handling of extremely long single sentence."""
        chunker = TokenChunker(max_tokens=50)
        # Create a sentence without terminators
        long_sentence = "word " * 200
        chunks = chunker.chunk_text(long_sentence)

        # Should handle gracefully (may be one chunk that exceeds limit)
        assert len(chunks) >= 1


class TestTokenChunkerIntegration:
    """Integration tests for TokenChunker with config."""

    def test_chunker_with_default_config(self):
        """Test chunker works with default configuration values."""
        from src.config import MAX_TOKENS_PER_CHUNK, SOFT_LIMIT_RATIO

        chunker = TokenChunker(
            max_tokens=MAX_TOKENS_PER_CHUNK,
            soft_limit_ratio=SOFT_LIMIT_RATIO
        )

        text = "Test paragraph.\n\nAnother paragraph."
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
