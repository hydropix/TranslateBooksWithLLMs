"""
EPUB translation module

This module provides specialized translation functionality for EPUB files,
using the simplified translation approach with smart chunking and tag preservation.

Main entry points:
    translate_epub_file() - Translate an EPUB file using LLM
    translate_document() - Translate a single XHTML document
    prepare_document() - Extract and chunk a single XHTML document

Components:
    - translator: Main translation orchestration
    - document_translation: File-level translation (body-based, smart chunking)
    - chunk_loop / chunk_fallback: Checkpointed chunk loop and per-chunk fallback ladder
    - body_reassembly: Chunk reassembly, bilingual interleave, body replacement
    - tag_preservation: HTML/XML tag preservation during translation
    - html_chunker: HTML-aware text chunking with placeholder management
    - body_serializer: Body extraction and replacement for XHTML documents
    - constants: Configuration constants
"""

from .translator import translate_epub_file
from .tag_preservation import TagPreserver
from .document_translation import prepare_document, translate_document
from .html_chunker import HtmlChunker
from .translation_metrics import TranslationMetrics
from .body_serializer import extract_body_html, replace_body_content
from .constants import (
    MIN_CONTEXT_LINES,
    MIN_CONTEXT_WORDS,
    MAX_CONTEXT_LINES,
    MAX_CONTEXT_BLOCKS,
    PLACEHOLDER_PATTERN
)

__all__ = [
    # Main translation functions
    'translate_epub_file',
    'translate_document',
    'prepare_document',

    # Tag preservation
    'TagPreserver',

    # HTML chunking
    'HtmlChunker',

    # Translation metrics
    'TranslationMetrics',

    # Body serialization
    'extract_body_html',
    'replace_body_content',

    # Constants
    'MIN_CONTEXT_LINES',
    'MIN_CONTEXT_WORDS',
    'MAX_CONTEXT_LINES',
    'MAX_CONTEXT_BLOCKS',
    'PLACEHOLDER_PATTERN',
]
