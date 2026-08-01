"""
Unit tests for TXT reassembly.

Regression coverage for issue #208: rejoining translated chunks with a plain
"\\n" collapsed every paragraph break at a chunk seam. Chunks now carry the
separator they were split on (`join_with`), so both the translate path
(`TxtAdapter.reconstruct_output`) and the refine path (`refine_txt_file`)
restore the source paragraph structure — without turning a sentence-level split
inside a single paragraph into a fake paragraph break.
"""
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.adapters.txt_adapter import TxtAdapter
from src.core.chunking.reassembly import DEFAULT_JOINER, join_translated_chunks
from src.core.text_processor import split_text_into_chunks


# Three ~11-token paragraphs; max_tokens=25 splits them into exactly two chunks.
PARAGRAPHS = [
    "First paragraph with a reasonable amount of content in it.",
    "Second paragraph with a reasonable amount of content in it.",
    "Third paragraph with a reasonable amount of content in it.",
]
MULTI_PARAGRAPH_TEXT = "\n\n".join(PARAGRAPHS)

# A single paragraph long enough to be re-chunked at sentence level.
SENTENCE_SPLIT_TEXT = " ".join(
    f"Sentence number {i} goes right here." for i in range(12)
)

MULTI_PARAGRAPH_CONFIG = {"max_tokens_per_chunk": 25, "soft_limit_ratio": 0.5}
SENTENCE_SPLIT_CONFIG = {"max_tokens_per_chunk": 30, "soft_limit_ratio": 0.5}


def _make_adapter(tmp_path, text, config):
    """Write `text` to a temp file and return a TxtAdapter over it."""
    input_path = tmp_path / "input.txt"
    input_path.write_text(text, encoding="utf-8")
    return TxtAdapter(str(input_path), str(tmp_path / "output.txt"), config)


class TestJoinTranslatedChunks:
    """Direct tests for the shared reassembly helper."""

    def test_missing_join_with_falls_back_to_blank_line(self):
        """Chunk dicts from older checkpoints have no join_with key."""
        assert join_translated_chunks(["a", "b"], [{}, {}]) == "a\n\nb"

    def test_empty_join_with_falls_back_to_blank_line(self):
        """An empty join_with after index 0 is treated as missing."""
        chunks = [{"join_with": ""}, {"join_with": ""}]
        assert join_translated_chunks(["a", "b"], chunks) == "a\n\nb"

    def test_short_chunks_list_falls_back_to_blank_line(self):
        """More parts than chunks still joins on the default separator."""
        assert join_translated_chunks(["a", "b"], [{}]) == "a\n\nb"
        assert DEFAULT_JOINER == "\n\n"

    def test_sentence_level_join_is_a_single_space(self):
        """A continuation chunk is re-attached with its own separator."""
        chunks = [{"join_with": ""}, {"join_with": " "}]
        assert join_translated_chunks(["a", "b"], chunks) == "a b"

    def test_empty_and_single_inputs(self):
        """No parts yields an empty string; one part yields it stripped."""
        assert join_translated_chunks([], []) == ""
        assert join_translated_chunks(["  only  "], [{}]) == "only"

    def test_falsy_part_uses_the_fallback(self):
        """An untranslated chunk falls back to its source content."""
        result = join_translated_chunks(
            ["translated", None], [{}, {}], fallbacks=["src-a", "src-b"]
        )
        assert result == "translated\n\nsrc-b"

    def test_falsy_part_without_fallbacks_is_empty(self):
        """Without fallbacks, a falsy part contributes an empty string."""
        assert join_translated_chunks([None, "b"], [{}, {}]) == "\n\nb"


class TestTxtAdapterReconstruct:
    """Round-trip tests for the translate path."""

    @pytest.mark.asyncio
    async def test_round_trip_identity_multi_paragraph(self, tmp_path):
        """Untranslated chunks reassemble byte-identically. Issue #208."""
        adapter = _make_adapter(
            tmp_path, MULTI_PARAGRAPH_TEXT, MULTI_PARAGRAPH_CONFIG
        )
        assert await adapter.prepare_for_translation()
        assert len(adapter.chunks) > 1, "fixture must span several chunks"

        adapter.translated_chunks = [c["main_content"] for c in adapter.chunks]

        output = (await adapter.reconstruct_output()).decode("utf-8")
        assert output == MULTI_PARAGRAPH_TEXT

    @pytest.mark.asyncio
    async def test_round_trip_identity_sentence_split(self, tmp_path):
        """A sentence-level split never becomes a paragraph break."""
        adapter = _make_adapter(
            tmp_path, SENTENCE_SPLIT_TEXT, SENTENCE_SPLIT_CONFIG
        )
        assert await adapter.prepare_for_translation()
        assert len(adapter.chunks) > 1, "fixture must be force-split"
        assert any(c["join_with"] == " " for c in adapter.chunks[1:])

        adapter.translated_chunks = [c["main_content"] for c in adapter.chunks]

        output = (await adapter.reconstruct_output()).decode("utf-8")
        assert output == SENTENCE_SPLIT_TEXT
        assert "\n\n" not in output, "no blank line existed in the source"

    @pytest.mark.asyncio
    async def test_untranslated_chunk_falls_back_to_source(self, tmp_path):
        """A chunk that was never translated keeps its original text."""
        adapter = _make_adapter(
            tmp_path, MULTI_PARAGRAPH_TEXT, MULTI_PARAGRAPH_CONFIG
        )
        assert await adapter.prepare_for_translation()

        output = (await adapter.reconstruct_output()).decode("utf-8")
        assert output == MULTI_PARAGRAPH_TEXT

    @pytest.mark.asyncio
    async def test_no_triple_newline_when_chunks_end_with_newline(self, tmp_path):
        """Trailing whitespace on a part never compounds into a blank line."""
        adapter = _make_adapter(
            tmp_path, MULTI_PARAGRAPH_TEXT, MULTI_PARAGRAPH_CONFIG
        )
        assert await adapter.prepare_for_translation()

        adapter.translated_chunks = [
            c["main_content"] + "\n" for c in adapter.chunks
        ]

        output = (await adapter.reconstruct_output()).decode("utf-8")
        assert "\n\n\n" not in output
        assert output == MULTI_PARAGRAPH_TEXT

    @pytest.mark.asyncio
    async def test_bilingual_output_is_unchanged(self, tmp_path):
        """The bilingual layout must stay byte-for-byte what it was."""
        adapter = _make_adapter(
            tmp_path, MULTI_PARAGRAPH_TEXT, MULTI_PARAGRAPH_CONFIG
        )
        assert await adapter.prepare_for_translation()
        assert len(adapter.chunks) == 2, "expected string assumes two chunks"

        adapter.translated_chunks = ["TRANSLATED ONE", "TRANSLATED TWO"]

        expected = (
            "First paragraph with a reasonable amount of content in it.\n\n"
            "Second paragraph with a reasonable amount of content in it.\n\n"
            "TRANSLATED ONE\n\n"
            "──────────"
            "──────────"
            "──────────"
            "──────────\n\n"
            "Third paragraph with a reasonable amount of content in it.\n\n"
            "TRANSLATED TWO"
        )

        output = (await adapter.reconstruct_output(bilingual=True)).decode("utf-8")
        assert output == expected


class TestRefinePathReassembly:
    """The refine path must rebuild the same structure as the translate path."""

    def test_refine_join_matches_the_translate_path(self):
        """refine_txt_file's join reproduces the source paragraph structure."""
        structured_chunks = split_text_into_chunks(
            MULTI_PARAGRAPH_TEXT,
            max_tokens_per_chunk=25,
            soft_limit_ratio=0.5,
        )
        assert len(structured_chunks) > 1

        refined_parts = [c["main_content"] for c in structured_chunks]

        assert join_translated_chunks(refined_parts, structured_chunks) == \
            MULTI_PARAGRAPH_TEXT

    def test_refine_join_keeps_sentence_runs_on_one_line(self):
        """A force-split paragraph stays a single paragraph after refining."""
        structured_chunks = split_text_into_chunks(
            SENTENCE_SPLIT_TEXT,
            max_tokens_per_chunk=30,
            soft_limit_ratio=0.5,
        )
        assert len(structured_chunks) > 1

        refined_parts = [c["main_content"] for c in structured_chunks]
        output = join_translated_chunks(refined_parts, structured_chunks)

        assert output == SENTENCE_SPLIT_TEXT
        assert "\n\n" not in output

    def test_refine_single_block_fallback_has_no_join_with(self):
        """The single-block fallback dict works without a join_with key."""
        fallback_chunks = [{
            "context_before": "",
            "main_content": MULTI_PARAGRAPH_TEXT,
            "context_after": "",
        }]
        result = join_translated_chunks([MULTI_PARAGRAPH_TEXT], fallback_chunks)
        assert result == MULTI_PARAGRAPH_TEXT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
