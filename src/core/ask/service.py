"""
Build (and cache) a BookIndex for an uploaded file.

Indexes are cached in memory keyed by path, size, mtime and embedding model, so
asking several questions about the same book only extracts and embeds it once.
"""
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from src.core.document_text import extract_plain_text
from src.utils.language_detector import LanguageDetector

from .passages import split_into_passages
from .retrieval import BookIndex

SUPPORTED_FILE_TYPES = ("txt", "epub", "docx", "pdf")
_CACHE_SIZE = 4


@dataclass
class IndexedBook:
    index: BookIndex
    language: str
    chars: int


_cache: "OrderedDict[tuple, IndexedBook]" = OrderedDict()
_lock = threading.Lock()


def _detect_language(text: str) -> str:
    language, _confidence = LanguageDetector.detect_language_from_text(text[:20000])
    return language or "English"


def load_book(file_path: str, file_type: str, embedding_model: Optional[str] = None,
              embedding_endpoint: Optional[str] = None) -> IndexedBook:
    """Extract, split and index a book, reusing a cached index when the file is unchanged."""
    file_type = file_type.lower()
    if file_type not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Ask the Book does not support {file_type!r} files")
    stat = os.stat(file_path)
    key = (os.path.abspath(file_path), stat.st_size, stat.st_mtime_ns, embedding_model or "")
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

    text = extract_plain_text(file_path, file_type)
    if not text or not text.strip():
        raise ValueError("No readable text found in this file")
    passages = split_into_passages(text)
    book = IndexedBook(
        index=BookIndex.build(passages, embedding_model or None, embedding_endpoint),
        language=_detect_language(text),
        chars=len(text),
    )
    with _lock:
        _cache[key] = book
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return book
