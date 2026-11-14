"""Unit tests for PlaceholderValidator."""

from src.core.epub.placeholder_validator import PlaceholderValidator


class TestPlaceholderValidatorBasic:
    """Test basic validation."""

    def test_validate_basic_all_present(self):
        """All placeholders present should return True."""
        text = "[[0]]Hello[[1]]World[[2]]"
        tag_map = {
            "[[0]]": "<p>",
            "[[1]]": "</p><p>",
            "[[2]]": "</p>"
        }

        assert PlaceholderValidator.validate_basic(text, tag_map) is True

    def test_validate_basic_missing_placeholder(self):
        """Missing placeholder should return False."""
        text = "[[0]]Hello[[2]]"  # Missing [[1]]
        tag_map = {
            "[[0]]": "<p>",
            "[[1]]": "</p><p>",
            "[[2]]": "</p>"
        }

        assert PlaceholderValidator.validate_basic(text, tag_map) is False

    def test_validate_basic_empty_map(self):
        """Empty tag map should return True."""
        assert PlaceholderValidator.validate_basic("Hello", {}) is True

    def test_validate_basic_empty_text(self):
        """Empty text with empty map should return True."""
        assert PlaceholderValidator.validate_basic("", {}) is True

    def test_validate_basic_empty_text_with_map(self):
        """Empty text with placeholders expected should return False."""
        tag_map = {"[[0]]": "<p>"}
        assert PlaceholderValidator.validate_basic("", tag_map) is False


class TestPlaceholderValidatorStrict:
    """Test strict validation."""

    def test_validate_strict_perfect(self):
        """Perfect validation should succeed."""
        text = "[[0]]Hello[[1]]World[[2]]"
        tag_map = {
            "[[0]]": "<p>",
            "[[1]]": "</p><p>",
            "[[2]]": "</p>"
        }

        is_valid, error_msg = PlaceholderValidator.validate_strict(text, tag_map)

        assert is_valid is True
        assert error_msg == ""

    def test_validate_strict_count_mismatch(self):
        """Count mismatch should fail with message."""
        text = "[[0]]Hello[[1]]"  # Only 2 placeholders
        tag_map = {
            "[[0]]": "<p>",
            "[[1]]": "</p><p>",
            "[[2]]": "</p>"  # Expects 3
        }

        is_valid, error_msg = PlaceholderValidator.validate_strict(text, tag_map)

        assert is_valid is False
        assert "count mismatch" in error_msg.lower()
        assert "expected 3" in error_msg
        assert "found 2" in error_msg

    def test_validate_strict_non_sequential(self):
        """Non-sequential indices should fail."""
        text = "[[0]]Hello[[2]]World[[3]]"  # Skips 1
        tag_map = {
            "[[0]]": "<p>",
            "[[2]]": "</p><p>",
            "[[3]]": "</p>"
        }

        is_valid, error_msg = PlaceholderValidator.validate_strict(text, tag_map)

        assert is_valid is False
        assert "sequential" in error_msg.lower() or "missing" in error_msg.lower()

    def test_validate_strict_different_formats(self):
        """Should work with different placeholder formats."""
        formats = [
            ("[[0]]Hello[[1]]", {"[[0]]": "<p>", "[[1]]": "</p>"}),
            ("[id0]Hello[id1]", {"[id0]": "<p>", "[id1]": "</p>"}),
            ("/0/Hello/1/", {"/0/": "<p>", "/1/": "</p>"}),
        ]

        for text, tag_map in formats:
            is_valid, error_msg = PlaceholderValidator.validate_strict(text, tag_map)
            assert is_valid is True, f"Failed for format: {text}, error: {error_msg}"

    def test_validate_strict_empty_map(self):
        """Empty map should always succeed."""
        is_valid, error_msg = PlaceholderValidator.validate_strict("Hello", {})
        assert is_valid is True
        assert error_msg == ""

    def test_validate_strict_extra_placeholders(self):
        """Extra placeholders in text should fail."""
        text = "[[0]]Hello[[1]][[2]][[3]]"  # 4 placeholders
        tag_map = {
            "[[0]]": "<p>",
            "[[1]]": "</p>",
        }  # Expects only 2

        is_valid, error_msg = PlaceholderValidator.validate_strict(text, tag_map)

        assert is_valid is False
        assert "count mismatch" in error_msg.lower()

    def test_validate_strict_duplicate_placeholders(self):
        """Duplicate placeholders should be counted correctly."""
        text = "[[0]]Hello[[1]]World[[1]]"  # [[1]] appears twice
        tag_map = {
            "[[0]]": "<p>",
            "[[1]]": "</p>"
        }

        is_valid, error_msg = PlaceholderValidator.validate_strict(text, tag_map)

        # Should detect the count mismatch (3 found vs 2 expected)
        assert is_valid is False


class TestGetMissingPlaceholders:
    """Test missing placeholder detection."""

    def test_no_missing(self):
        """No missing placeholders."""
        text = "[[0]]Hello[[1]]"
        tag_map = {"[[0]]": "<p>", "[[1]]": "</p>"}

        missing = PlaceholderValidator.get_missing_placeholders(text, tag_map)

        assert missing == []

    def test_some_missing(self):
        """Some placeholders missing."""
        text = "[[0]]Hello"  # Missing [[1]]
        tag_map = {"[[0]]": "<p>", "[[1]]": "</p>"}

        missing = PlaceholderValidator.get_missing_placeholders(text, tag_map)

        assert "[[1]]" in missing
        assert len(missing) == 1

    def test_all_missing(self):
        """All placeholders missing."""
        text = "Hello World"
        tag_map = {"[[0]]": "<p>", "[[1]]": "</p>", "[[2]]": "<br>"}

        missing = PlaceholderValidator.get_missing_placeholders(text, tag_map)

        assert len(missing) == 3
        assert "[[0]]" in missing
        assert "[[1]]" in missing
        assert "[[2]]" in missing

    def test_empty_map_no_missing(self):
        """Empty map means no missing placeholders."""
        text = "Hello World"
        tag_map = {}

        missing = PlaceholderValidator.get_missing_placeholders(text, tag_map)

        assert missing == []

    def test_partial_missing(self):
        """Multiple missing placeholders."""
        text = "[[0]]Hello[[2]]"  # Missing [[1]] and [[3]]
        tag_map = {
            "[[0]]": "<p>",
            "[[1]]": "</p>",
            "[[2]]": "<br>",
            "[[3]]": "</p>"
        }

        missing = PlaceholderValidator.get_missing_placeholders(text, tag_map)

        assert len(missing) == 2
        assert "[[1]]" in missing
        assert "[[3]]" in missing
