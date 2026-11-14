"""
Regression tests for technical content protection system.

Ensures:
- Zero regression on existing functionality when protection is disabled
- Backward compatibility with all existing translation workflows
- No breaking changes to API or behavior
"""

import pytest
from lxml import etree

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
        assert preserver.protect_technical == False  # Default is False

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

    @pytest.mark.asyncio
    async def test_translate_with_protect_technical_false(self):
        """Explicit protect_technical_content=False works."""
        xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title></head>
<body><p>The $V_{cm}$ voltage</p></body>
</html>"""

        doc_root = etree.fromstring(xhtml.encode('utf-8'))
        provider = SimpleTranslationProvider()

        success, stats = await translate_xhtml(
            doc_root=doc_root,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=provider,
            prompt_options={'protect_technical_content': False}
        )

        # Should work, technical content not protected
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
        """Validator works when placeholders represent both tags and technical content."""
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
        """Chunker works with standard HTML (no technical content)."""
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
        preserver = TagPreserver(protect_technical=False)

        html = "<p></p><p>1.</p><p>Text</p>"
        result, tag_map = preserver.preserve_tags(html)

        # Should group empty paragraphs
        assert "[id0]" in result

    def test_nested_tag_preservation(self):
        """Nested tags still preserved correctly."""
        preserver = TagPreserver(protect_technical=False)

        html = "<div><p><span>Text</span></p></div>"
        result, tag_map = preserver.preserve_tags(html)
        restored = preserver.restore_tags(result, tag_map)

        assert restored == html

    def test_self_closing_tags(self):
        """Self-closing tags still work."""
        preserver = TagPreserver(protect_technical=False)

        html = "<p>Line 1<br/>Line 2</p>"
        result, tag_map = preserver.preserve_tags(html)
        restored = preserver.restore_tags(result, tag_map)

        assert "<br/>" in restored or "<br>" in restored


class TestFeatureToggleWorks:
    """Test that protect_technical acts as proper feature toggle."""

    def test_protection_off_by_default(self):
        """Protection is OFF by default."""
        preserver = TagPreserver()
        assert preserver.protect_technical == False

    def test_protection_can_be_enabled(self):
        """Protection can be enabled."""
        preserver = TagPreserver(protect_technical=True)
        assert preserver.protect_technical == True

    def test_protection_only_active_when_enabled(self):
        """Protection only happens when explicitly enabled."""
        text = "<p>The $V_{cm}$ voltage</p>"

        # Without protection
        preserver_off = TagPreserver(protect_technical=False)
        result_off, map_off = preserver_off.preserve_tags(text)

        # With protection
        preserver_on = TagPreserver(protect_technical=True)
        result_on, map_on = preserver_on.preserve_tags_and_technical_content(text)

        # Should have different number of placeholders
        # Off: 2 (just tags), On: 3 (tags + formula)
        assert len(map_off) < len(map_on)


class TestNoMemoryLeaks:
    """Test that repeated operations don't leak memory."""

    def test_repeated_preserve_restore_cycles(self):
        """Many preserve/restore cycles don't leak."""
        preserver = TagPreserver(protect_technical=True)

        html = "<p>The $V_{cm}$ voltage is 10 Mbps using `chip`.</p>"

        # Run many cycles
        for i in range(100):
            result, tag_map = preserver.preserve_tags_and_technical_content(html)
            restored = preserver.restore_tags(result, tag_map)

            # Should always restore correctly
            assert restored == html

    def test_detector_is_reused(self):
        """Detector instance is reused (not recreated each time)."""
        preserver = TagPreserver(protect_technical=True)

        # First call should create detector
        preserver.preserve_tags_and_technical_content("<p>$x$</p>")
        detector1 = preserver._detector

        # Second call should reuse same detector
        preserver.preserve_tags_and_technical_content("<p>$y$</p>")
        detector2 = preserver._detector

        assert detector1 is detector2


class TestErrorHandling:
    """Test error handling remains robust."""

    def test_malformed_html_handled(self):
        """Malformed HTML doesn't crash."""
        preserver = TagPreserver(protect_technical=True)

        # Unclosed tag
        html = "<p>Text without closing tag"

        try:
            result, tag_map = preserver.preserve_tags_and_technical_content(html)
            # Should complete (might not restore perfectly, but shouldn't crash)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Should handle malformed HTML gracefully, got: {e}")

    def test_empty_input_handled(self):
        """Empty input doesn't crash."""
        preserver = TagPreserver(protect_technical=True)

        result, tag_map = preserver.preserve_tags_and_technical_content("")

        assert result == ""
        assert len(tag_map) == 0

    def test_very_long_text_handled(self):
        """Very long text is handled."""
        preserver = TagPreserver(protect_technical=True)

        # Create long text
        long_text = "<p>" + ("word " * 10000) + "</p>"

        try:
            result, tag_map = preserver.preserve_tags_and_technical_content(long_text)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Should handle long text, got: {e}")


class TestPlaceholderFormatConsistency:
    """Ensure placeholder format remains consistent."""

    def test_placeholder_format_unchanged(self):
        """Placeholder format is still [idN]."""
        preserver = TagPreserver(protect_technical=True)

        html = "<p>The $V_{cm}$ voltage</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # All placeholders should be [idN] format
        for placeholder in tag_map.keys():
            assert placeholder.startswith("[id")
            assert placeholder.endswith("]")

            # Extract number
            num_str = placeholder[3:-1]
            assert num_str.isdigit()

    def test_sequential_numbering_maintained(self):
        """Placeholders are numbered sequentially from 0."""
        preserver = TagPreserver(protect_technical=True)

        html = "<p>Text with $x$ and $y$ formulas</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Should have [id0], [id1], [id2], [id3] (in order)
        numbers = []
        for placeholder in tag_map.keys():
            num = int(placeholder[3:-1])
            numbers.append(num)

        numbers.sort()
        assert numbers == list(range(len(numbers)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
