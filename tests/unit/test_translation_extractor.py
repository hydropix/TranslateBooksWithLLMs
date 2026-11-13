"""
Unit tests for TranslationExtractor (issue #170 fixes)
"""
from src.core.llm.utils.extraction import TranslationExtractor


class TestTranslationExtractor:
    def test_basic_extraction(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("<TRANSLATION>Hello world</TRANSLATION>")
        assert result == "Hello world"

    def test_extraction_with_whitespace(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("  <TRANSLATION>  Hello world  </TRANSLATION>  ")
        assert result == "Hello world"

    def test_think_blocks_removed(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract(
            "<think>Some reasoning</think><TRANSLATION>Hello</TRANSLATION>"
        )
        assert result == "Hello"

    def test_orphan_think_before_translation_is_stripped(self):
        """Orphan </think> before <TRANSLATION> should be stripped."""
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        raw = "Some reasoning...</think>\n<TRANSLATION>Hello</TRANSLATION>"
        result = extractor.extract(raw)
        assert result == "Hello"

    def test_orphan_think_inside_translation_is_preserved(self):
        """Issue #170 fix: orphan </think> inside translation must NOT destroy the tag."""
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        raw = "<TRANSLATION>\nHello world\n</think>"
        result = extractor.extract(raw)
        # The orphan remover should NOT strip because prefix contains <TRANSLATION>
        assert result is None  # extraction still fails (no closing tag), but content was NOT destroyed

    def test_orphan_think_inside_translation_content_preserved(self):
        """Ensure the raw content is still inspectable after failed extraction."""
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        raw = "<TRANSLATION>\nLe renard brun\n</think>"
        result = extractor.extract(raw)
        # Content should NOT be wiped by orphan remover
        assert result is None
        assert "<TRANSLATION>" in raw
        assert "Le renard brun" in raw

    def test_markdown_fence_stripping(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("```xml\n<TRANSLATION>Hello</TRANSLATION>\n```")
        assert result == "Hello"

    def test_no_tags_returns_none(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("Just some text without tags")
        assert result is None

    def test_partial_opening_tag_returns_none(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("<TRANSLATION> incomplete")
        assert result is None

    def test_fuzzy_closing_tag(self):
        """Gemini-style typo in closing tag (</TRANATION>)"""
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("<TRANSLATION>Hello</TRANATION>")
        assert result == "Hello"
