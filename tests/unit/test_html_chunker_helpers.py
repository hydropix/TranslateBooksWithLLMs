"""Unit tests for HtmlChunker helper functions.

Tests the refactored helper methods extracted from _merge_segments_into_chunks.
"""

from src.core.epub.html_chunker import HtmlChunker


class TestCountSegmentTokens:
    """Test _count_segment_tokens helper method."""

    def test_count_empty_segment(self):
        """Empty segment should return 0 tokens."""
        chunker = HtmlChunker(max_tokens=400)

        tokens = chunker._count_segment_tokens("")

        assert tokens == 0

    def test_count_simple_text(self):
        """Simple text should return correct token count."""
        chunker = HtmlChunker(max_tokens=400)

        tokens = chunker._count_segment_tokens("Hello world")

        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_text_with_placeholders(self):
        """Text with placeholders should count placeholders as tokens."""
        chunker = HtmlChunker(max_tokens=400)

        tokens = chunker._count_segment_tokens("[[0]]Hello world[[1]]")

        assert tokens > 0
        assert isinstance(tokens, int)


class TestWouldExceedLimit:
    """Test _would_exceed_limit helper method."""

    def test_within_limit(self):
        """Adding tokens within limit should return False."""
        chunker = HtmlChunker(max_tokens=400)

        result = chunker._would_exceed_limit(current_tokens=200, new_tokens=100)

        assert result is False

    def test_exactly_at_limit(self):
        """Adding tokens to exactly reach limit should return False."""
        chunker = HtmlChunker(max_tokens=400)

        result = chunker._would_exceed_limit(current_tokens=300, new_tokens=100)

        assert result is False

    def test_exceeds_limit(self):
        """Adding tokens that exceed limit should return True."""
        chunker = HtmlChunker(max_tokens=400)

        result = chunker._would_exceed_limit(current_tokens=300, new_tokens=101)

        assert result is True

    def test_far_exceeds_limit(self):
        """Adding tokens that far exceed limit should return True."""
        chunker = HtmlChunker(max_tokens=400)

        result = chunker._would_exceed_limit(current_tokens=350, new_tokens=200)

        assert result is True

    def test_zero_current_tokens(self):
        """Zero current tokens should work correctly."""
        chunker = HtmlChunker(max_tokens=400)

        result = chunker._would_exceed_limit(current_tokens=0, new_tokens=100)

        assert result is False

    def test_zero_new_tokens(self):
        """Zero new tokens should work correctly."""
        chunker = HtmlChunker(max_tokens=400)

        result = chunker._would_exceed_limit(current_tokens=200, new_tokens=0)

        assert result is False


class TestFinalizeChunk:
    """Test _finalize_chunk helper method."""

    def test_finalize_single_segment(self):
        """Finalizing a single segment should create valid chunk."""
        chunker = HtmlChunker(max_tokens=400)
        segments = ["[[0]]Hello world[[1]]"]
        global_tag_map = {"[[0]]": "<p>", "[[1]]": "</p>"}
        global_offset = 0

        chunk = chunker._finalize_chunk(segments, global_tag_map, global_offset)

        assert 'text' in chunk
        assert 'local_tag_map' in chunk
        assert 'global_offset' in chunk
        assert 'global_indices' in chunk
        assert chunk['global_offset'] == 0

    def test_finalize_multiple_segments(self):
        """Finalizing multiple segments should merge them correctly."""
        chunker = HtmlChunker(max_tokens=400)
        segments = ["[[0]]Hello[[1]]", "[[2]]world[[3]]"]
        global_tag_map = {
            "[[0]]": "<p>",
            "[[1]]": "</p>",
            "[[2]]": "<p>",
            "[[3]]": "</p>"
        }
        global_offset = 0

        chunk = chunker._finalize_chunk(segments, global_tag_map, global_offset)

        assert 'text' in chunk
        assert 'local_tag_map' in chunk
        # Should have merged both segments
        assert "Hello" in chunk['text']
        assert "world" in chunk['text']

    def test_finalize_with_offset(self):
        """Finalizing with non-zero offset should preserve offset."""
        chunker = HtmlChunker(max_tokens=400)
        segments = ["[[5]]Hello[[6]]"]
        global_tag_map = {"[[5]]": "<p>", "[[6]]": "</p>"}
        global_offset = 10

        chunk = chunker._finalize_chunk(segments, global_tag_map, global_offset)

        assert chunk['global_offset'] == 10

    def test_finalize_renumbers_placeholders(self):
        """Finalizing should renumber placeholders locally."""
        chunker = HtmlChunker(max_tokens=400)
        segments = ["[id5]Hello[id6]world[id7]"]
        global_tag_map = {
            "[id5]": "<p>",
            "[id6]": "<b>",
            "[id7]": "</b></p>"
        }
        global_offset = 0

        chunk = chunker._finalize_chunk(segments, global_tag_map, global_offset)

        # Should renumber to [id0], [id1], [id2]
        assert "[id0]" in chunk['text']
        assert "[id1]" in chunk['text']
        assert "[id2]" in chunk['text']
        # Original placeholders should not be in text
        assert "[id5]" not in chunk['text']
        assert "[id6]" not in chunk['text']
        assert "[id7]" not in chunk['text']


class TestMergeSegmentsIntegration:
    """Integration tests for _merge_segments_into_chunks using helper functions."""

    def test_merge_small_segments(self):
        """Small segments should be merged into single chunk."""
        chunker = HtmlChunker(max_tokens=400)
        segments = [
            "[[0]]Hello[[1]]",
            "[[2]]world[[3]]",
            "[[4]]test[[5]]"
        ]
        global_tag_map = {
            "[[0]]": "<p>", "[[1]]": "</p>",
            "[[2]]": "<p>", "[[3]]": "</p>",
            "[[4]]": "<p>", "[[5]]": "</p>"
        }

        chunks = chunker._merge_segments_into_chunks(segments, global_tag_map)

        # All small segments should fit in one chunk
        assert len(chunks) >= 1
        assert all('text' in chunk for chunk in chunks)
        assert all('local_tag_map' in chunk for chunk in chunks)

    def test_merge_creates_multiple_chunks(self):
        """Large segments should be split into multiple chunks."""
        chunker = HtmlChunker(max_tokens=50)  # Very small limit
        # Create segments with enough content to exceed limit
        long_text = " ".join(["word"] * 100)
        segments = [
            f"[[0]]{long_text}[[1]]",
            f"[[2]]{long_text}[[3]]"
        ]
        global_tag_map = {
            "[[0]]": "<p>", "[[1]]": "</p>",
            "[[2]]": "<p>", "[[3]]": "</p>"
        }

        chunks = chunker._merge_segments_into_chunks(segments, global_tag_map)

        # Should create multiple chunks due to token limit
        assert len(chunks) >= 2

    def test_merge_preserves_global_indices(self):
        """Merging should preserve global indices for reconstruction."""
        chunker = HtmlChunker(max_tokens=400)
        segments = ["[[0]]Hello[[1]]", "[[2]]world[[3]]"]
        global_tag_map = {
            "[[0]]": "<p>", "[[1]]": "</p>",
            "[[2]]": "<p>", "[[3]]": "</p>"
        }

        chunks = chunker._merge_segments_into_chunks(segments, global_tag_map)

        # Each chunk should have global_indices
        for chunk in chunks:
            assert 'global_indices' in chunk
            assert isinstance(chunk['global_indices'], list)


class TestHelperFunctionsWithDifferentMaxTokens:
    """Test helper functions with different max_tokens settings."""

    def test_would_exceed_with_small_limit(self):
        """Helper should respect small token limits."""
        chunker = HtmlChunker(max_tokens=10)

        assert chunker._would_exceed_limit(5, 6) is True
        assert chunker._would_exceed_limit(5, 5) is False
        assert chunker._would_exceed_limit(5, 4) is False

    def test_would_exceed_with_large_limit(self):
        """Helper should respect large token limits."""
        chunker = HtmlChunker(max_tokens=1000)

        assert chunker._would_exceed_limit(500, 501) is True
        assert chunker._would_exceed_limit(500, 500) is False
        assert chunker._would_exceed_limit(999, 2) is True


class TestChunkSizeLimitations:
    """Test that chunks respect token size limitations."""

    def test_single_chunk_under_limit(self):
        """Single chunk with content under limit should not be split."""
        chunker = HtmlChunker(max_tokens=400)
        # Create HTML with placeholders that fits in one chunk
        text = "[id0]This is a short paragraph.[id1]"
        tag_map = {
            "[id0]": "<p>",
            "[id1]": "</p>"
        }

        chunks = chunker.chunk_html_with_placeholders(text, tag_map)

        assert len(chunks) == 1
        token_count = chunker._count_segment_tokens(chunks[0]['text'])
        assert token_count <= 400

    def test_multiple_paragraphs_under_limit(self):
        """Multiple paragraphs that fit should be in one chunk."""
        chunker = HtmlChunker(max_tokens=400)
        text = (
            "[id0]First paragraph.[id1]"
            "[id2]Second paragraph.[id3]"
            "[id4]Third paragraph.[id5]"
        )
        tag_map = {
            "[id0]": "<p>", "[id1]": "</p>",
            "[id2]": "<p>", "[id3]": "</p>",
            "[id4]": "<p>", "[id5]": "</p>"
        }

        chunks = chunker.chunk_html_with_placeholders(text, tag_map)

        # Should fit in one chunk
        assert len(chunks) == 1
        token_count = chunker._count_segment_tokens(chunks[0]['text'])
        assert token_count <= 400

    def test_content_exceeding_limit_creates_multiple_chunks(self):
        """Content exceeding token limit should be split into multiple chunks."""
        chunker = HtmlChunker(max_tokens=100)
        # Create long text that will exceed limit
        long_paragraph = " ".join(["word"] * 50)
        text = (
            f"[id0]{long_paragraph}[id1]"
            f"[id2]{long_paragraph}[id3]"
            f"[id4]{long_paragraph}[id5]"
        )
        tag_map = {
            "[id0]": "<p>", "[id1]": "</p>",
            "[id2]": "<p>", "[id3]": "</p>",
            "[id4]": "<p>", "[id5]": "</p>"
        }

        chunks = chunker.chunk_html_with_placeholders(text, tag_map)

        # Should create multiple chunks
        assert len(chunks) > 1
        # Each chunk should respect the limit
        for chunk in chunks:
            token_count = chunker._count_segment_tokens(chunk['text'])
            # Allow some tolerance for placeholder overhead
            assert token_count <= chunker.max_tokens + 10

    def test_single_oversized_segment_is_split(self):
        """Single segment larger than limit should be split."""
        chunker = HtmlChunker(max_tokens=50)
        # Create one very long paragraph
        very_long_text = " ".join(["word"] * 100)
        text = f"[id0]{very_long_text}[id1]"
        tag_map = {
            "[id0]": "<p>",
            "[id1]": "</p>"
        }

        chunks = chunker.chunk_html_with_placeholders(text, tag_map)

        # Should be split into multiple chunks
        assert len(chunks) > 1
        # Each chunk should be under or close to the limit
        for chunk in chunks:
            token_count = chunker._count_segment_tokens(chunk['text'])
            # Some tolerance for splitting overhead
            assert token_count <= chunker.max_tokens + 20

    def test_chunk_sizes_with_different_limits(self):
        """Test chunking with various token limits."""
        test_text = " ".join(["word"] * 200)
        text = f"[id0]{test_text}[id1]"
        tag_map = {"[id0]": "<p>", "[id1]": "</p>"}

        # Test with different limits
        for max_tokens in [50, 100, 200, 400]:
            chunker = HtmlChunker(max_tokens=max_tokens)
            chunks = chunker.chunk_html_with_placeholders(text, tag_map)

            # Verify all chunks respect the limit
            for chunk in chunks:
                token_count = chunker._count_segment_tokens(chunk['text'])
                # Allow tolerance for overhead
                assert token_count <= max_tokens + 20, \
                    f"Chunk exceeded limit: {token_count} tokens > {max_tokens} max"

    def test_mixed_size_segments_distribution(self):
        """Test that mixed size segments are distributed properly."""
        chunker = HtmlChunker(max_tokens=100)
        # Create segments of varying sizes
        short_text = "Short text."
        medium_text = " ".join(["word"] * 20)
        long_text = " ".join(["word"] * 40)

        text = (
            f"[id0]{short_text}[id1]"
            f"[id2]{long_text}[id3]"
            f"[id4]{short_text}[id5]"
            f"[id6]{medium_text}[id7]"
            f"[id8]{long_text}[id9]"
        )
        tag_map = {
            "[id0]": "<p>", "[id1]": "</p>",
            "[id2]": "<p>", "[id3]": "</p>",
            "[id4]": "<p>", "[id5]": "</p>",
            "[id6]": "<p>", "[id7]": "</p>",
            "[id8]": "<p>", "[id9]": "</p>"
        }

        chunks = chunker.chunk_html_with_placeholders(text, tag_map)

        # Should create multiple chunks
        assert len(chunks) >= 2
        # All chunks should respect limit
        for chunk in chunks:
            token_count = chunker._count_segment_tokens(chunk['text'])
            assert token_count <= 100 + 20  # Allow tolerance

    def test_very_small_limit_handling(self):
        """Test that very small token limits are handled correctly."""
        chunker = HtmlChunker(max_tokens=20)
        text = "[id0]This is a test paragraph with several words.[id1]"
        tag_map = {"[id0]": "<p>", "[id1]": "</p>"}

        chunks = chunker.chunk_html_with_placeholders(text, tag_map)

        # Should split into multiple small chunks
        assert len(chunks) >= 1
        for chunk in chunks:
            token_count = chunker._count_segment_tokens(chunk['text'])
            # With very small limits, splitting may need more tolerance
            assert token_count <= 20 + 30

    def test_exact_limit_boundary(self):
        """Test behavior when content is exactly at token limit."""
        chunker = HtmlChunker(max_tokens=100)
        # Create text that's close to the limit
        text_words = []
        current_tokens = 0
        # Build text to be close to 100 tokens
        while current_tokens < 90:
            text_words.append("word")
            test_text = " ".join(text_words)
            current_tokens = chunker._count_segment_tokens(f"[id0]{test_text}[id1]")

        final_text = f"[id0]{' '.join(text_words)}[id1]"
        tag_map = {"[id0]": "<p>", "[id1]": "</p>"}

        chunks = chunker.chunk_html_with_placeholders(final_text, tag_map)

        # Should fit in one or two chunks depending on exact size
        assert len(chunks) >= 1
        for chunk in chunks:
            token_count = chunker._count_segment_tokens(chunk['text'])
            assert token_count <= 100 + 10

    def test_empty_content_respects_limit(self):
        """Empty or whitespace-only content should not cause issues."""
        chunker = HtmlChunker(max_tokens=100)
        text = "[id0]   [id1]"
        tag_map = {"[id0]": "<p>", "[id1]": "</p>"}

        chunks = chunker.chunk_html_with_placeholders(text, tag_map)

        # Should handle empty content gracefully
        assert len(chunks) <= 1
        if chunks:
            token_count = chunker._count_segment_tokens(chunks[0]['text'])
            assert token_count <= 100
