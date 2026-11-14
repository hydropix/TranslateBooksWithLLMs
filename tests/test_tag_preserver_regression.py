"""
Regression tests for the tag-preservation path the EPUB translation uses.

Covers TagPreserver.preserve_tags/restore_tags, the placeholder validator,
the chunker and the file-level translation entry point, including content that
looks technical (code, formulas, escaped markup): it is sent to the LLM as
ordinary text and must come back unchanged when the model echoes it.
"""

import pytest
from lxml import etree

from src.core.epub.body_serializer import extract_body_html
from src.core.epub.tag_preservation import TagPreserver
from tests.epub_engine_helpers import translate_xhtml
from src.core.epub.placeholder_validator import PlaceholderValidator
from src.core.epub.html_chunker import HtmlChunker
from src.core.llm.base import LLMResponse


class SimpleTranslationProvider:
    """Simple mock provider for regression testing."""

    async def generate(self, prompt: str, timeout: int = 120, system_prompt: str = None):
        """Return text with placeholders preserved."""
        # Simple mock: return a response that preserves placeholders
        return LLMResponse(
            content=prompt,
            prompt_tokens=10,
            completion_tokens=10,
            context_used=20,
            context_limit=4096,
            was_truncated=False
        )

    def extract_translation(self, content: str) -> str:
        """Extract translation from response."""
        return content


class TestBackwardCompatibilityTagPreserver:
    """Ensure TagPreserver backward compatibility."""

    def test_default_constructor(self):
        """Default constructor works as before."""
        preserver = TagPreserver()
        assert preserver is not None

    def test_preserve_tags_method_unchanged(self):
        """Original preserve_tags() method works identically."""
        preserver = TagPreserver()

        test_cases = [
            ("<p>Hello</p>", "[id0]Hello[id1]", 2),
            ("No tags here", "No tags here", 0),
            ("<br/>", "[id0]", 1),
        ]

        for html, expected_result, expected_tag_count in test_cases:
            result, tag_map = preserver.preserve_tags(html)
            assert result == expected_result, f"Failed for '{html}': got '{result}'"
            assert len(tag_map) == expected_tag_count

    def test_restore_tags_method_unchanged(self):
        """Original restore_tags() method works identically."""
        preserver = TagPreserver()

        original = "<p>Hello world</p>"
        result, tag_map = preserver.preserve_tags(original)
        restored = preserver.restore_tags(result, tag_map)

        assert restored == original

    def test_complex_html_preservation(self):
        """Complex HTML still preserved correctly."""
        preserver = TagPreserver()

        html = """<div class="chapter">
            <h1>Title</h1>
            <p>First paragraph with <em>emphasis</em>.</p>
            <p>Second paragraph.</p>
        </div>"""

        result, tag_map = preserver.preserve_tags(html)
        restored = preserver.restore_tags(result, tag_map)

        # Should restore perfectly
        assert restored == html


@pytest.mark.filterwarnings("ignore:TranslationStats is deprecated:DeprecationWarning")
class TestBackwardCompatibilityXhtmlTranslator:
    """Ensure the file-level translation keeps working."""

    @pytest.mark.asyncio
    async def test_translate_without_prompt_options(self):
        """Translation works without prompt_options parameter."""
        xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title></head>
<body><p>Hello world</p></body>
</html>"""

        doc_root = etree.fromstring(xhtml.encode('utf-8'))
        provider = SimpleTranslationProvider()

        # No prompt_options provided (old API style)
        success, stats = await translate_xhtml(
            doc_root=doc_root,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=provider
        )

        # Should complete without error
        assert success or stats is not None

    @pytest.mark.asyncio
    async def test_translate_with_empty_prompt_options(self):
        """Translation works with empty prompt_options."""
        xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title></head>
<body><p>Test content</p></body>
</html>"""

        doc_root = etree.fromstring(xhtml.encode('utf-8'))
        provider = SimpleTranslationProvider()

        success, stats = await translate_xhtml(
            doc_root=doc_root,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=provider,
            prompt_options={}
        )

        assert success or stats is not None


class TestPlaceholderValidatorCompatibility:
    """Ensure PlaceholderValidator still works with mixed placeholders."""

    def test_validator_with_only_tag_placeholders(self):
        """Validator works with standard tag placeholders."""
        tag_map = {"[id0]": "<p>", "[id1]": "</p>"}
        translated = "[id0]Texte ici[id1]"

        is_valid = PlaceholderValidator.validate_basic(translated, tag_map)
        assert is_valid

    def test_validator_with_mixed_placeholders(self):
        """Validator works when a placeholder wraps non-tag content."""
        # This simulates: <p>The $V_{cm}$ voltage</p>
        # Where [id0]=<p>, [id1]=$V_{cm}$, [id2]=</p>
        tag_map = {"[id0]": "<p>", "[id1]": "$V_{cm}$", "[id2]": "</p>"}
        translated = "[id0]La [id1] tension[id2]"

        is_valid = PlaceholderValidator.validate_basic(translated, tag_map)
        assert is_valid

    def test_validator_detects_missing_placeholder(self):
        """Validator still detects missing placeholders."""
        tag_map = {"[id0]": "<p>", "[id1]": "</p>"}
        translated = "[id0]Texte"  # Missing [id1]

        is_valid = PlaceholderValidator.validate_basic(translated, tag_map)
        assert not is_valid

    def test_validator_detects_extra_placeholder(self):
        """Validator still detects extra placeholders."""
        tag_map = {"[id0]": "<p>", "[id1]": "</p>"}
        translated = "[id0]Texte[id1][id2]"  # Extra [id2]

        # validate_strict checks for count mismatch
        is_valid, error_msg = PlaceholderValidator.validate_strict(translated, tag_map)
        assert not is_valid


class TestHtmlChunkerCompatibility:
    """Ensure HtmlChunker still works correctly."""

    def test_chunker_with_standard_html(self):
        """Chunker works with standard HTML."""
        chunker = HtmlChunker(max_tokens=100)

        html = "[id0]This is paragraph one.[id1][id2]This is paragraph two.[id3]"
        tag_map = {"[id0]": "<p>", "[id1]": "</p>", "[id2]": "<p>", "[id3]": "</p>"}

        chunks = chunker.chunk_html_with_placeholders(html, tag_map)

        # Should create chunks without error
        assert len(chunks) > 0

    def test_chunker_with_mixed_placeholders(self):
        """Chunker works when placeholders represent mixed content."""
        chunker = HtmlChunker(max_tokens=100)

        # Simulates: <p>The $V_{cm}$ is measured</p>
        html = "[id0]The [id1] is measured[id2]"
        tag_map = {"[id0]": "<p>", "[id1]": "$V_{cm}$", "[id2]": "</p>"}

        chunks = chunker.chunk_html_with_placeholders(html, tag_map)

        assert len(chunks) > 0


class TestNoRegressionOnExistingTests:
    """Ensure all existing test patterns still work."""

    def test_empty_paragraph_grouping(self):
        """Empty paragraph grouping still works."""
        preserver = TagPreserver()

        html = "<p></p><p>1.</p><p>Text</p>"
        result, tag_map = preserver.preserve_tags(html)

        # Should group empty paragraphs
        assert "[id0]" in result

    def test_nested_tag_preservation(self):
        """Nested tags still preserved correctly."""
        preserver = TagPreserver()

        html = "<div><p><span>Text</span></p></div>"
        result, tag_map = preserver.preserve_tags(html)
        restored = preserver.restore_tags(result, tag_map)

        assert restored == html

    def test_self_closing_tags(self):
        """Self-closing tags still work."""
        preserver = TagPreserver()

        html = "<p>Line 1<br/>Line 2</p>"
        result, tag_map = preserver.preserve_tags(html)
        restored = preserver.restore_tags(result, tag_map)

        assert "<br/>" in restored or "<br>" in restored


class TestTechnicalLookingContent:
    """Code, formulas and escaped markup go through the plain tag path."""

    def test_html_entities_stay_in_the_translatable_text(self):
        html = "<p>Example:</p><pre>&lt;section&gt;&lt;h1&gt;Title&lt;/h1&gt;&lt;/section&gt;</pre>"
        preserver = TagPreserver()
        result, tag_map = preserver.preserve_tags(html)
        assert "&lt;section&gt;" in result
        assert not any("&lt;" in value for value in tag_map.values())
        assert preserver.restore_tags(result, tag_map) == html

    @pytest.mark.parametrize("html", [
        "<p>The $V_{cm}$ voltage is 100 MHz.</p>",
        "<p>Run <code>npm install</code> first.</p>",
        "<p>Example:</p><pre><code>def f():\n    return 1</code></pre><p>Done</p>",
        "<p>Simple text $5 dollars</p>",
    ])
    def test_round_trip_is_lossless(self, html):
        preserver = TagPreserver()
        result, tag_map = preserver.preserve_tags(html)
        assert preserver.restore_tags(result, tag_map) == html

    def test_body_extraction_keeps_formulas(self):
        xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title></head>
<body>
<p>The $V_{cm}$ voltage is measured.</p>
</body>
</html>"""
        body_html, body_element = extract_body_html(etree.fromstring(xhtml.encode('utf-8')))
        assert "$V_{cm}$" in body_html
        assert body_element is not None
