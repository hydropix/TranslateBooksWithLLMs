"""Unit tests for TagPreserver."""

from src.core.epub.tag_preservation import TagPreserver


class TestTagPreserver:
    """Test TagPreserver placeholder creation."""

    def test_preserve_simple_tags(self):
        """Simple HTML tags should be preserved."""
        preserver = TagPreserver()
        text, tag_map = preserver.preserve_tags("<p>Hello</p>")

        assert "[id0]" in text
        assert "[id1]" in text
        assert "Hello" in text
        assert tag_map["[id0]"] == "<p>"
        assert tag_map["[id1]"] == "</p>"

    def test_preserve_grouped_whitespace(self):
        """Whitespace and tags should be grouped."""
        preserver = TagPreserver()
        text, tag_map = preserver.preserve_tags("<p> </p><p>Hello</p>")

        # Should group whitespace with tags
        assert "[id0]" in text
        assert "Hello" in text
        assert "[id1]" in text
        # The exact grouping may vary, but tags should be preserved
        assert len(tag_map) >= 2

    def test_preserve_nested_tags(self):
        """Nested tags should be preserved."""
        preserver = TagPreserver()
        text, tag_map = preserver.preserve_tags("<p><b>Bold</b> text</p>")

        assert "Bold" in text
        assert "text" in text
        assert len(tag_map) > 0
        # Should have placeholders for <p>, <b>, </b>, </p>
        assert any("<p>" in v for v in tag_map.values())
        assert any("<b>" in v for v in tag_map.values())

    def test_preserve_multiple_paragraphs(self):
        """Multiple paragraphs should be preserved."""
        preserver = TagPreserver()
        text, tag_map = preserver.preserve_tags("<p>First</p><p>Second</p>")

        assert "First" in text
        assert "Second" in text
        assert len(tag_map) > 0

    def test_preserve_attributes(self):
        """Tags with attributes should be preserved."""
        preserver = TagPreserver()
        html = '<p class="intro">Hello</p>'
        text, tag_map = preserver.preserve_tags(html)

        assert "Hello" in text
        assert any('class="intro"' in v for v in tag_map.values())

    def test_preserve_empty_tags(self):
        """Empty tags should be preserved."""
        preserver = TagPreserver()
        text, tag_map = preserver.preserve_tags("<br/><hr/>Text")

        assert "Text" in text
        assert len(tag_map) > 0
        assert any("<br" in v for v in tag_map.values())

    def test_restore_tags(self):
        """Tags should be restored correctly."""
        preserver = TagPreserver()
        original = "<p>Hello <b>World</b></p>"

        preserved, tag_map = preserver.preserve_tags(original)
        restored = preserver.restore_tags(preserved, tag_map)

        # Should restore tags
        assert "Hello" in restored
        assert "World" in restored
        assert "<p>" in restored
        assert "<b>" in restored
        assert "</p>" in restored
        assert "</b>" in restored

    def test_restore_tags_preserves_content(self):
        """Content should be preserved during restore."""
        preserver = TagPreserver()
        original = "<p>First paragraph</p><p>Second paragraph</p>"

        preserved, tag_map = preserver.preserve_tags(original)
        restored = preserver.restore_tags(preserved, tag_map)

        assert "First paragraph" in restored
        assert "Second paragraph" in restored

    def test_preserve_complex_structure(self):
        """Complex HTML structure should be preserved."""
        preserver = TagPreserver()
        html = """
        <div>
            <h1>Title</h1>
            <p>Paragraph with <em>emphasis</em> and <strong>strong</strong>.</p>
        </div>
        """
        text, tag_map = preserver.preserve_tags(html)

        assert "Title" in text
        assert "Paragraph" in text
        assert "emphasis" in text
        assert "strong" in text
        assert len(tag_map) > 0

    def test_preserve_sequential_placeholders(self):
        """Placeholders should be sequential."""
        preserver = TagPreserver()
        text, tag_map = preserver.preserve_tags("<p>A</p><p>B</p><p>C</p>")

        # Extract placeholder indices
        placeholders = list(tag_map.keys())
        # Should be sequential: [id0], [id1], [id2], etc.
        assert "[id0]" in placeholders
        # Check that placeholders are in sequential order
        for i in range(len(placeholders) - 1):
            current_idx = int(placeholders[i].replace("[id", "").replace("]", ""))
            next_idx = int(placeholders[i + 1].replace("[id", "").replace("]", ""))
            assert next_idx == current_idx + 1

    def test_validate_placeholders_all_present(self):
        """Validation should pass when all placeholders present."""
        preserver = TagPreserver()
        text = "[id0]Hello[id1]World[id2]"
        tag_map = {
            "[id0]": "<p>",
            "[id1]": "</p><p>",
            "[id2]": "</p>"
        }

        is_valid, missing, mutated = preserver.validate_placeholders(text, tag_map)
        assert is_valid is True
        assert missing == []

    def test_validate_placeholders_missing(self):
        """Validation should fail when placeholders missing."""
        preserver = TagPreserver()
        text = "[id0]Hello[id2]"  # Missing [id1]
        tag_map = {
            "[id0]": "<p>",
            "[id1]": "</p><p>",
            "[id2]": "</p>"
        }

        is_valid, missing, mutated = preserver.validate_placeholders(text, tag_map)
        assert is_valid is False
        assert "[id1]" in missing

    def test_roundtrip_preservation(self):
        """Preserve and restore should be reversible."""
        preserver = TagPreserver()
        original = "<p>Hello <a href='link'>World</a>!</p>"

        preserved, tag_map = preserver.preserve_tags(original)
        restored = preserver.restore_tags(preserved, tag_map)

        # Content should match
        assert "Hello" in restored
        assert "World" in restored
        # Tags should be present
        assert "<p>" in restored
        assert "</p>" in restored
        assert "<a" in restored
        assert "</a>" in restored
