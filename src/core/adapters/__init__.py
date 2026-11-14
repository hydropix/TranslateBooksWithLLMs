"""
Adapter pattern implementation for generic file translation.

This module provides a unified interface for translating different file formats
(TXT, SRT, PGN) using the adapter pattern. Each format implements the
FormatAdapter interface to handle format-specific operations. EPUB, DOCX and
PDF are routed by translate_file() to their dedicated pipelines.
"""

from .translation_unit import TranslationUnit
from .format_adapter import FormatAdapter
from .generic_translator import GenericTranslator
from .txt_adapter import TxtAdapter
from .srt_adapter import SrtAdapter
from .pgn_adapter import PgnAdapter

# Unified translation entry point
from .translate_file import translate_file, get_file_type_from_path

# Refine-only entry point
from .refine_file import refine_file

# Term-fix entry point (correct terms in an already-translated file)
from .term_fix_file import term_fix_file, scan_file

from .exceptions import (
    TranslationError,
    FileFormatError,
    UnsupportedFormatError,
)

__all__ = [
    # Core adapter components
    'TranslationUnit',
    'FormatAdapter',
    'GenericTranslator',
    'TxtAdapter',
    'SrtAdapter',
    'PgnAdapter',

    # Unified translation entry point
    'translate_file',
    'get_file_type_from_path',

    # Refine-only entry point
    'refine_file',

    # Term-fix entry point
    'term_fix_file',
    'scan_file',

    # Exceptions
    'TranslationError',
    'FileFormatError',
    'UnsupportedFormatError',
]
