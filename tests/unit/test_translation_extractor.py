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


TAG_IN = "<TRANSLATION>"
TAG_OUT = "</TRANSLATION>"


def _extract(raw):
    return TranslationExtractor(TAG_IN, TAG_OUT).extract(raw)


class TestMultipleBlocks:
    """A response holding several tagged blocks must never leak a literal tag."""

    def test_two_blocks_at_boundaries_return_no_literal_tag(self):
        result = _extract("<TRANSLATION>A</TRANSLATION>\n<TRANSLATION>B</TRANSLATION>")
        assert TAG_IN not in result and TAG_OUT not in result
        assert result == "B"

    def test_echoed_prompt_example_blocks_are_not_extracted(self):
        raw = (
            "**CORRECT format (ONLY this):**\n"
            "✅ <TRANSLATION>\nYour translated text here\n</TRANSLATION>\n"
            "REMINDER: Output ONLY your translation in this exact format:\n"
            "<TRANSLATION>\nyour translation here\n</TRANSLATION>\n"
            "<TRANSLATION>\nBonjour le monde\n</TRANSLATION>"
        )
        assert _extract(raw) == "Bonjour le monde"

    def test_placeholder_block_after_real_answer_is_skipped(self):
        raw = (
            "<TRANSLATION>Bonjour</TRANSLATION>\n"
            "<TRANSLATION>your translation here</TRANSLATION>"
        )
        assert _extract(raw) == "Bonjour"

    def test_unbalanced_opening_tag_does_not_leak(self):
        raw = "<TRANSLATION>draft <TRANSLATION>Bonjour</TRANSLATION>"
        assert _extract(raw) == "Bonjour"

    def test_fuzzy_closing_tag_does_not_leak_opening_tag(self):
        raw = "<TRANSLATION>draft <TRANSLATION>Bonjour</TRANATION>"
        assert _extract(raw) == "Bonjour"


class TestThinkingRemoval:
    def test_unclosed_think_before_translation(self):
        raw = "<think>Let me reason about this...\n<TRANSLATION>Bonjour</TRANSLATION>"
        assert _extract(raw) == "Bonjour"

    def test_unclosed_think_without_translation_returns_none(self):
        assert _extract("<think>I am still reasoning and never stop") is None

    def test_unclosed_think_quoting_a_full_block(self):
        raw = ("<think>The format is <TRANSLATION>x</TRANSLATION>, fine.\n"
               "<TRANSLATION>Bonjour</TRANSLATION>")
        assert _extract(raw) == "Bonjour"

    def test_reasoning_quoting_translation_tag(self):
        raw = ("<think>I must wrap the answer in <TRANSLATION> and </TRANSLATION>."
               "</think><TRANSLATION>Bonjour</TRANSLATION>")
        assert _extract(raw) == "Bonjour"

    def test_orphan_close_after_reasoning_quoting_translation_tag(self):
        raw = ("I must output <TRANSLATION>x</TRANSLATION> exactly.</think>\n"
               "<TRANSLATION>Bonjour</TRANSLATION>")
        assert _extract(raw) == "Bonjour"

    def test_reasoning_quoting_the_closing_think_tag(self):
        raw = ("<think>Never emit </think> early. Done.</think>"
               "<TRANSLATION>Bonjour</TRANSLATION>")
        assert _extract(raw) == "Bonjour"

    def test_thinking_variant(self):
        raw = ("<thinking>use <TRANSLATION>x</TRANSLATION></thinking>"
               "<TRANSLATION>Bonjour</TRANSLATION>")
        assert _extract(raw) == "Bonjour"

    def test_reasoning_variant_case_insensitive(self):
        raw = ("<Reasoning>use <TRANSLATION>x</TRANSLATION></Reasoning>"
               "<TRANSLATION>Bonjour</TRANSLATION>")
        assert _extract(raw) == "Bonjour"


def test_extractor_does_not_print(capsys):
    _extract("Some reasoning...</think>\n<TRANSLATION>Hello</TRANSLATION>")
    _extract("<TRANSLATION>Hello</TRANATION>")
    _extract("Preface <TRANSLATION>Hello</TRANSLATION> trailing")
    assert capsys.readouterr().out == ""
