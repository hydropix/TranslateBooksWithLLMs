"""Unit tests for custom exceptions."""

import pytest
from src.core.epub.exceptions import (
    EpubTranslationError,
    XmlParsingError,
    BodyExtractionError,
)


class TestEpubTranslationError:
    """Test base exception class."""

    def test_base_exception_message(self):
        """Base exception should store message."""
        error = EpubTranslationError("Test error")
        assert str(error) == "Test error"

    def test_base_exception_inheritance(self):
        """Base exception should inherit from Exception."""
        error = EpubTranslationError("Test")
        assert isinstance(error, Exception)


class TestXmlParsingError:
    """Test XML parsing error."""

    def test_xml_parsing_error_with_preview(self):
        """XmlParsingError should store content preview."""
        original_error = ValueError("Invalid XML")
        error = XmlParsingError(
            "Parse failed",
            original_error=original_error,
            content_preview="<p>Invalid HTML..."
        )

        assert str(error) == "Parse failed"
        assert error.original_error == original_error
        assert "<p>" in error.content_preview
        assert "Invalid HTML" in error.content_preview

    def test_xml_parsing_error_defaults(self):
        """XmlParsingError should have default values."""
        error = XmlParsingError("Parse failed")

        assert str(error) == "Parse failed"
        assert error.original_error is None
        assert error.content_preview is None

    def test_xml_parsing_error_inheritance(self):
        """XmlParsingError should inherit from EpubTranslationError."""
        error = XmlParsingError("Test")
        assert isinstance(error, EpubTranslationError)


class TestBodyExtractionError:
    """Test body extraction error."""

    def test_body_extraction_error(self):
        """BodyExtractionError should work as basic exception."""
        error = BodyExtractionError("Failed to extract body")
        assert str(error) == "Failed to extract body"

    def test_body_extraction_error_inheritance(self):
        """BodyExtractionError should inherit from EpubTranslationError."""
        error = BodyExtractionError("Test")
        assert isinstance(error, EpubTranslationError)


class TestExceptionRaising:
    """Test that exceptions can be raised and caught correctly."""

    def test_catch_base_exception(self):
        """Specific exceptions should be catchable as base EpubTranslationError."""
        with pytest.raises(EpubTranslationError):
            raise XmlParsingError("Test")

        with pytest.raises(EpubTranslationError):
            raise BodyExtractionError("Test")

    def test_catch_specific_exception(self):
        """Should be able to catch specific exception types."""
        try:
            raise XmlParsingError("Test", content_preview="<p>")
        except XmlParsingError as e:
            assert e.content_preview == "<p>"
        except EpubTranslationError:
            pytest.fail("Should have caught XmlParsingError specifically")
