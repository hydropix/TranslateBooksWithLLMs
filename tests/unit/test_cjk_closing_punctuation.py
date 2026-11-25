"""Regression tests: full-width closing punctuation must stay in its own block.

With Japanese and Chinese targets, "。" / "？" moved to the START of the next
paragraph ("<p>こんにちは</p><p>。お元気ですか？</p>"):

* the token-alignment fallback (Phase 2) snapped each placeholder to the
  nearest "word boundary" and inserted it BEFORE that character; CJK text has
  no spaces, so the nearest boundary was the "。" itself and the closing
  </p><p> placeholder landed in front of it;
* a model can do the same on its own, answering "こんにちは[id1]。" in Phase 1
  or "こんにちは\\n\\n。" in Plain Text Mode.

All tests use fakes; no LLM is called.
"""
import pytest
from lxml import etree

import src.core.common.plain_text_pipeline as plain_pipeline
from src.common.cjk_punctuation import (
    CJK_CLOSING_PUNCTUATION,
    PARAGRAPH_BREAK_PATTERN,
    reattach_closing_punctuation,
    skip_closing_punctuation,
)
from src.core.epub import chunk_fallback
from src.core.epub.chunk_fallback import reattach_block_closing_punctuation
from src.core.epub.html_utils import extract_text_and_positions, reinsert_placeholders
from src.core.epub.token_alignment_fallback import TokenAlignmentFallback
from tests.epub_engine_helpers import translate_xhtml

XHTML_NS = "http://www.w3.org/1999/xhtml"
SOURCE_TWO_PARAGRAPHS = "[id0]Hello there.[id1]How are you?[id2]"
PLACEHOLDERS = ["[id0]", "[id1]", "[id2]"]


# ---------------------------------------------------------------------------
# The shared set
# ---------------------------------------------------------------------------

def test_set_holds_closing_marks_only():
    for char in "。．？！、，：；」』）】〕〉》":
        assert char in CJK_CLOSING_PUNCTUATION
    # Opening marks belong to the next block; ellipsis and wave dash can
    # legitimately open a paragraph ("……はい。", "〜序章〜").
    for char in "「『（【〔〈《…‥〜～.?!":
        assert char not in CJK_CLOSING_PUNCTUATION


def test_skip_closing_punctuation():
    assert skip_closing_punctuation("はい。」次", 2) == 4
    assert skip_closing_punctuation("はい。「次", 2) == 3
    assert skip_closing_punctuation("abc", 1) == 1


# ---------------------------------------------------------------------------
# Phase 2: token alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source, translated, expected", [
    (SOURCE_TWO_PARAGRAPHS, "こんにちは。お元気ですか？",
     "[id0]こんにちは。[id1]お元気ですか？[id2]"),
    ("[id0]Hello there, friend.[id1]How are you today?[id2]", "你好，朋友。你今天好吗？",
     "[id0]你好，朋友。[id1]你今天好吗？[id2]"),
    ("[id0]He said: hello.[id1]She left.[id2]", "「こんにちは。」彼女は去った。",
     "[id0]「こんにちは。」[id1]彼女は去った。[id2]"),
])
def test_token_alignment_keeps_closing_punctuation_before_the_placeholder(
        source, translated, expected):
    result = TokenAlignmentFallback().align_and_insert_placeholders(
        source, translated, PLACEHOLDERS)
    assert result == expected


def test_token_alignment_latin_output_unchanged():
    result = TokenAlignmentFallback().align_and_insert_placeholders(
        SOURCE_TWO_PARAGRAPHS, "Bonjour. Comment vas-tu ?", PLACEHOLDERS)
    assert result == "[id0]Bonjour.[id1] Comment vas-tu ?[id2]"


def test_proportional_reinsertion_keeps_closing_punctuation_before_the_placeholder():
    _, positions = extract_text_and_positions(SOURCE_TWO_PARAGRAPHS)
    result = reinsert_placeholders("こんにちは。お元気ですか？", positions)
    assert result == "[id0]こんにちは。[id1]お元気ですか？[id2]"


# ---------------------------------------------------------------------------
# Phase 1: the model puts the mark after a block-closing placeholder
# ---------------------------------------------------------------------------

TAG_MAP = {"[id0]": "<p>", "[id1]": "</p><p>", "[id2]": "</p>"}


@pytest.mark.parametrize("translated, expected", [
    ("[id0]こんにちは[id1]。お元気ですか[id2]？",
     "[id0]こんにちは。[id1]お元気ですか？[id2]"),
    ("[id0]你好[id1]。你好吗[id2]？",
     "[id0]你好。[id1]你好吗？[id2]"),
    ("[id0]「こんにちは[id1]。」さようなら。[id2]",
     "[id0]「こんにちは。」[id1]さようなら。[id2]"),
])
def test_reply_punctuation_after_block_end_is_moved_back(translated, expected):
    assert reattach_block_closing_punctuation(
        SOURCE_TWO_PARAGRAPHS, translated, TAG_MAP) == expected


@pytest.mark.parametrize("translated", [
    "[id0]こんにちは。[id1]「お元気ですか？」[id2]",   # opening bracket stays
    "[id0]こんにちは。[id1]……はい。[id2]",            # leading ellipsis stays
])
def test_legitimate_block_starts_are_untouched(translated):
    assert reattach_block_closing_punctuation(
        SOURCE_TWO_PARAGRAPHS, translated, TAG_MAP) == translated


def test_inline_placeholder_is_not_a_block_boundary():
    tag_map = {"[id0]": "<p>", "[id1]": "<em>", "[id2]": "</em>", "[id3]": "</p>"}
    translated = "[id0]これは[id1]重要[id2]、本当に。[id3]"
    assert reattach_block_closing_punctuation(
        "[id0]This is[id1]important[id2], really.[id3]", translated, tag_map) == translated


def test_source_block_starting_with_closing_mark_disarms_the_repair():
    source = "[id0]前の文[id1]。次の文[id2]"
    translated = "[id0]前の文[id1]。次の文[id2]"
    assert reattach_block_closing_punctuation(source, translated, TAG_MAP) == translated


# ---------------------------------------------------------------------------
# End to end through the XHTML pipeline
# ---------------------------------------------------------------------------

def _paragraph_texts(doc):
    return ["".join(p.itertext()) for p in doc.iter(f"{{{XHTML_NS}}}p")]


def _doc():
    return etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><head><title>t</title></head>'
        "<body><p>Hello there.</p><p>How are you?</p></body></html>".encode("utf-8")
    )


@pytest.mark.asyncio
async def test_xhtml_phase2_japanese_keeps_punctuation_before_line_break(monkeypatch):
    """Phase 2 on a single block: the mark stays before the <br/>."""
    async def drops_placeholders(main_content, *args, **kwargs):
        return "こんにちは。お元気ですか？"

    monkeypatch.setattr(chunk_fallback, "generate_translation_request", drops_placeholders)
    doc = etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><head><title>t</title></head>'
        "<body><p>Hello there.<br/>How are you?</p></body></html>".encode("utf-8"))
    success, stats = await translate_xhtml(doc, "English", "Japanese", "fake",
                                           llm_client=object(), max_retries=1)
    assert success is True
    assert stats.token_alignment_used == 1
    br = next(doc.iter(f"{{{XHTML_NS}}}br"))
    assert br.getparent().text == "こんにちは。"
    assert br.tail == "お元気ですか？"


@pytest.mark.asyncio
async def test_xhtml_lost_placeholders_keep_punctuation_in_its_paragraph(monkeypatch):
    """Two paragraphs whose placeholders are dropped are translated apart."""
    replies = {"Hello there.": "こんにちは。", "How are you?": "お元気ですか？"}

    async def drops_placeholders(main_content, *args, **kwargs):
        if "[id" in main_content:
            return "こんにちは。お元気ですか？"
        return replies[main_content.strip()]

    monkeypatch.setattr(chunk_fallback, "generate_translation_request", drops_placeholders)
    doc = _doc()
    success, stats = await translate_xhtml(doc, "English", "Japanese", "fake",
                                           llm_client=object(), max_retries=1)
    assert success is True
    assert stats.token_alignment_used == 0
    assert _paragraph_texts(doc) == ["こんにちは。", "お元気ですか？"]


@pytest.mark.asyncio
async def test_xhtml_phase1_chinese_reply_after_closing_tag(monkeypatch):
    async def misplaced(main_content, *args, **kwargs):
        # Keeps every placeholder but writes the marks after the block ends.
        assert main_content.count("[id") == 3
        return "[id0]你好[id1]。你好吗[id2]？"

    monkeypatch.setattr(chunk_fallback, "generate_translation_request", misplaced)
    doc = _doc()
    success, stats = await translate_xhtml(doc, "English", "Chinese", "fake",
                                           llm_client=object(), max_retries=1)
    assert success is True
    assert stats.token_alignment_used == 0
    assert _paragraph_texts(doc) == ["你好。", "你好吗？"]


# ---------------------------------------------------------------------------
# Plain Text Mode
# ---------------------------------------------------------------------------

def test_reattach_across_paragraph_break():
    assert reattach_closing_punctuation(
        "こんにちは\n\n。お元気ですか？", PARAGRAPH_BREAK_PATTERN, "Hi.\n\nHow?"
    ) == "こんにちは。\n\nお元気ですか？"


@pytest.mark.asyncio
async def test_plain_text_mode_keeps_punctuation_in_its_paragraph(monkeypatch):
    async def fake_request(*, main_content, **kwargs):
        return "こんにちは\n\n。お元気ですか\n\n？"

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake_request)
    out, stats, interrupted = await plain_pipeline.translate_paragraphs_plain(
        paragraphs=["Hello there.", "How are you?"],
        source_language="English",
        target_language="Japanese",
        model_name="m",
        llm_client=object(),
        max_tokens_per_chunk=1000,
        parallel_workers=1,
        log_callback=None,
    )
    assert not interrupted
    assert out == ["こんにちは。", "お元気ですか？"]
    assert stats.paragraph_count_mismatches == 0
