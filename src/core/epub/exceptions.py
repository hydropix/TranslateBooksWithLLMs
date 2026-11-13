"""
Custom exceptions for EPUB translation pipeline.

This module defines specific exception types for different failure scenarios,
enabling better error handling and debugging.
"""


class EpubTranslationError(Exception):
    """Base exception for all EPUB translation errors."""
    pass


class XmlParsingError(EpubTranslationError):
    """Raised when XML/HTML parsing fails after all fallback attempts.

    Attributes:
        original_error: The underlying parsing error
        content_preview: First 200 chars of problematic content
    """
    def __init__(self, message: str, original_error: Exception = None, content_preview: str = None):
        super().__init__(message)
        self.original_error = original_error
        self.content_preview = content_preview


class BodyExtractionError(EpubTranslationError):
    """Raised when body extraction from XHTML fails."""
    pass
