"""
Exceptions raised by the translation adapter entry points.

Only the exceptions actually raised or caught by the codebase live here.
LLM-specific errors (``ContextOverflowError``, ``RepetitionLoopError``, ...)
live in ``src.core.llm.exceptions``; EPUB-specific errors live in
``src.core.epub.exceptions``.
"""

from typing import Optional, Dict, Any


class TranslationError(Exception):
    """Base exception for all translation-related errors.

    Attributes:
        message: Human-readable error message
        context: Additional context about the error
        recoverable: Whether the error can potentially be recovered from
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.recoverable = recoverable

    def __str__(self) -> str:
        base = f"{self.__class__.__name__}: {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base += f" (context: {context_str})"
        return base


class FileFormatError(TranslationError):
    """Base exception for file format errors."""
    pass


class UnsupportedFormatError(FileFormatError):
    """Raised when file format is not supported."""

    def __init__(
        self,
        message: str,
        file_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        ctx = context or {}
        if file_type is not None:
            ctx['file_type'] = file_type
        super().__init__(message, ctx, recoverable=False)
