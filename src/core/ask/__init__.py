"""
Ask the Book: answer questions about a document using only its own text.

Public API:
    - split_into_passages / Passage: document -> retrievable passages
    - BookIndex: BM25 (+ optional Ollama embeddings) passage search
    - BookQA / AskResult: grounded answering with out-of-scope detection
    - load_book: cached extraction + indexing of an uploaded file
"""
from .answerer import AskResult, BookQA
from .passages import Passage, split_into_passages
from .retrieval import BookIndex
from .service import SUPPORTED_FILE_TYPES, IndexedBook, load_book

__all__ = [
    "AskResult",
    "BookIndex",
    "BookQA",
    "IndexedBook",
    "Passage",
    "SUPPORTED_FILE_TYPES",
    "load_book",
    "split_into_passages",
]
