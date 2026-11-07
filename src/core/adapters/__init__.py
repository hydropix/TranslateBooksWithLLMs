"""
Adapter pattern implementation for generic file translation.

This module provides a unified interface for translating different file formats
(TXT, SRT, PGN, EPUB, PDF) using the adapter pattern. Each format implements the
FormatAdapter interface to handle format-specific operations.

New in Phase 5: Comprehensive error handling system with:
- Custom exception hierarchy
- Retry logic with exponential backoff
- Error recovery strategies
- Comprehensive logging
"""

from .translation_unit import TranslationUnit
from .format_adapter import FormatAdapter
from .generic_translator import GenericTranslator
from .txt_adapter import TxtAdapter
from .srt_adapter import SrtAdapter
from .pgn_adapter import PgnAdapter
from .epub_adapter import EpubAdapter

# Unified translation entry point (Phase 6)
from .translate_file import translate_file, get_file_type_from_path, build_translated_output

# Refine-only entry point
from .refine_file import refine_file

# Term-fix entry point (correct terms in an already-translated file)
from .term_fix_file import term_fix_file, scan_file

# Error handling system (Phase 5)
from .exceptions import (
    TranslationError,
    AdapterError,
    AdapterInitializationError,
    AdapterPreparationError,
    AdapterReconstructionError,
    TranslationUnitError,
    UnitTranslationError,
    LLMError,
    ContextOverflowError,
    RepetitionLoopError,
    LLMConnectionError,
    LLMRateLimitError,
    CheckpointError,
    FileFormatError,
    UnsupportedFormatError,
    RetryExhaustedError,
)
from .retry_manager import RetryManager, RetryConfig, RetryStrategy, with_retry
from .error_recovery import ErrorRecoveryManager, RecoveryResult, ContentSplitter, GracefulDegradation
from .error_logger import ErrorLogger, ErrorLoggerContext, ErrorSeverity, ErrorRecord
from .error_handler import ErrorHandler, with_error_handling

__all__ = [
    # Core adapter components
    'TranslationUnit',
    'FormatAdapter',
    'GenericTranslator',
    'TxtAdapter',
    'SrtAdapter',
    'PgnAdapter',
    'EpubAdapter',

    # Unified translation entry point (Phase 6)
    'translate_file',
    'get_file_type_from_path',
    'build_translated_output',

    # Refine-only entry point
    'refine_file',

    # Term-fix entry point
    'term_fix_file',
    'scan_file',

    # Error handling - Exceptions
    'TranslationError',
    'AdapterError',
    'AdapterInitializationError',
    'AdapterPreparationError',
    'AdapterReconstructionError',
    'TranslationUnitError',
    'UnitTranslationError',
    'LLMError',
    'ContextOverflowError',
    'RepetitionLoopError',
    'LLMConnectionError',
    'LLMRateLimitError',
    'CheckpointError',
    'FileFormatError',
    'UnsupportedFormatError',
    'RetryExhaustedError',

    # Error handling - Retry
    'RetryManager',
    'RetryConfig',
    'RetryStrategy',
    'with_retry',

    # Error handling - Recovery
    'ErrorRecoveryManager',
    'RecoveryResult',
    'ContentSplitter',
    'GracefulDegradation',

    # Error handling - Logging
    'ErrorLogger',
    'ErrorLoggerContext',
    'ErrorSeverity',
    'ErrorRecord',

    # Error handling - Unified handler
    'ErrorHandler',
    'with_error_handling',
]
