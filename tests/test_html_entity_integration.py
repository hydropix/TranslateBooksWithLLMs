"""
Integration test for HTML entity protection in EPUB translation

This test ensures that HTML entities in EPUB content are properly protected
during the entire translation pipeline (tag preservation + technical protection).
"""

import pytest
from src.core.epub.tag_preservation import TagPreserver


class TestHTMLEntityIntegration:
    """Test HTML entity protection in full translation pipeline."""

    def test_html_entities_protected_with_technical_flag(self):
        """Test that HTML entities are protected when protect_technical=True."""
        # This simulates the exact scenario from the bug report
        text = (
            "<p>Here is an example:</p>"
            "<pre>&lt;section epub:type=\"part\"&gt;\n"
            "    &lt;h1&gt;Part I&lt;/h1&gt;\n"
            "\n"
            "    &lt;section epub:type=\"chapter\"&gt;\n"
            "        &lt;h2&gt;Chapter 1&lt;/h2&gt;\n"
            "        …\n"
            "    &lt;/section&gt;\n"
            "&lt;/section&gt;</pre>"
            "<p>This shows the structure.</p>"
        )

        preserver = TagPreserver(protect_technical=True)
        processed_text, tag_map = preserver.preserve_tags_and_technical_content(text)

        # The HTML entities should be in placeholders
        print(f"\nProcessed text: {processed_text}")
        print(f"\nTag map: {tag_map}")

        # Check that HTML entities are NOT in the processed text
        assert "&lt;section" not in processed_text
        assert "&gt;" not in processed_text
        assert "&lt;h1&gt;" not in processed_text

        # Check that HTML entities ARE in the tag map (protected)
        entity_placeholders = [
            placeholder for placeholder, content in tag_map.items()
            if "&lt;" in content or "&gt;" in content
        ]
        assert len(entity_placeholders) > 0, "HTML entities should be protected in placeholders"

        # Verify we can restore correctly
        restored_text = preserver.restore_tags(processed_text, tag_map)
        assert "&lt;section" in restored_text
        assert "&lt;h1&gt;" in restored_text

    def test_html_entities_not_protected_without_technical_flag(self):
        """Test that HTML entities are NOT protected when protect_technical=False."""
        text = (
            "<p>Example: &lt;section&gt;&lt;h1&gt;Title&lt;/h1&gt;&lt;/section&gt;</p>"
        )

        preserver = TagPreserver(protect_technical=False)
        processed_text, tag_map = preserver.preserve_tags(text)

        # Without technical protection, HTML entities should remain in text
        # (only HTML tags are protected)
        assert "&lt;" in processed_text
        assert "&gt;" in processed_text

    def test_mixed_html_tags_and_entities(self):
        """Test correct handling of real HTML tags vs HTML entities."""
        text = (
            "<p>This is a <strong>real tag</strong>, "
            "but this is escaped: &lt;fake&gt;tag&lt;/fake&gt;</p>"
        )

        preserver = TagPreserver(protect_technical=True)
        processed_text, tag_map = preserver.preserve_tags_and_technical_content(text)

        print(f"\nProcessed text: {processed_text}")
        print(f"\nTag map: {tag_map}")

        # Real tags should be in placeholders
        assert "<p>" not in processed_text
        assert "<strong>" not in processed_text

        # Escaped entities should also be in placeholders
        assert "&lt;fake&gt;" not in processed_text

        # Both should be in tag map
        real_tag_placeholders = [
            placeholder for placeholder, content in tag_map.items()
            if "<p>" in content or "<strong>" in content
        ]
        entity_placeholders = [
            placeholder for placeholder, content in tag_map.items()
            if "&lt;fake&gt;" in content
        ]

        assert len(real_tag_placeholders) > 0
        assert len(entity_placeholders) > 0

        # Restore and verify
        restored_text = preserver.restore_tags(processed_text, tag_map)
        assert restored_text == text

    def test_bug_report_exact_scenario(self):
        """Test the exact scenario from the bug report."""
        # The problematic part (before fix) would include:
        problem_source = (
            "&lt;section epub:type=\"part\"&gt;\n"
            "    &lt;h1&gt;Part I&lt;/h1&gt;\n"
            "\n"
            "    &lt;section epub:type=\"chapter\"&gt;\n"
            "        &lt;h2&gt;Chapter 1&lt;/h2&gt;\n"
            "        …\n"
            "    &lt;/section&gt;\n"
            "&lt;/section&gt;"
        )

        # With the fix, these entities should be protected
        preserver = TagPreserver(protect_technical=True)
        processed_text, tag_map = preserver.preserve_tags_and_technical_content(problem_source)

        print(f"\nProblem source: {problem_source}")
        print(f"\nProcessed text: {processed_text}")
        print(f"\nTag map keys: {list(tag_map.keys())}")

        # After fix: entities should NOT be in processed text
        assert "&lt;section" not in processed_text
        assert "&gt;" not in processed_text

        # Entities should be protected in tag map
        has_entities = any("&lt;" in content or "&gt;" in content for content in tag_map.values())
        assert has_entities, "HTML entities should be protected in tag map"

    def test_inline_code_with_entities(self):
        """Test that inline code tags are protected (entities within may remain if < 3)."""
        text = "<p>Use the <code>&lt;section&gt;</code> element.</p>"

        preserver = TagPreserver(protect_technical=True)
        processed_text, tag_map = preserver.preserve_tags_and_technical_content(text)

        print(f"\nProcessed text: {processed_text}")
        print(f"\nTag map: {tag_map}")

        # HTML tags should be protected
        assert "<p>" not in processed_text
        assert "<code>" not in processed_text

        # Note: Single pairs of entities (&lt;section&gt; = 2 entities) won't be detected
        # as HTML entity blocks because we require minimum 3 entities to avoid false positives.
        # This is acceptable - the semantic context (being inside <code>) protects them anyway.

        # Only translatable text should remain (or entities if too few)
        assert "Use the" in processed_text
        assert "element." in processed_text

        # Restore and verify
        restored_text = preserver.restore_tags(processed_text, tag_map)
        assert restored_text == text

    def test_inline_code_with_multiple_entities(self):
        """Test that inline code with many entities (3+) gets protected."""
        text = "<p>Use <code>&lt;section&gt;&lt;h1&gt;Title&lt;/h1&gt;&lt;/section&gt;</code> structure.</p>"

        preserver = TagPreserver(protect_technical=True)
        processed_text, tag_map = preserver.preserve_tags_and_technical_content(text)

        print(f"\nProcessed text: {processed_text}")
        print(f"\nTag map: {tag_map}")

        # HTML tags should be protected
        assert "<p>" not in processed_text
        assert "<code>" not in processed_text

        # With 3+ entities, the block should be detected and protected
        assert "&lt;section" not in processed_text or len([p for p in tag_map.values() if "&lt;section" in p]) > 0

        # Translatable text should remain
        assert "Use" in processed_text
        assert "structure." in processed_text

        # Restore and verify
        restored_text = preserver.restore_tags(processed_text, tag_map)
        assert restored_text == text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
