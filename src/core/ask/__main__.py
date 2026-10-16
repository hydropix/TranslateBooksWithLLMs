"""
Interactive terminal mode for Ask the Book.

    python -m src.core.ask book.epub --provider ollama -m gemma3:4b
    python -m src.core.ask book.pdf --embedding-model bge-m3 --answer-language Hinglish

Commands inside the prompt: /sources (passages behind the last answer), /reset, /quit
"""
import argparse
import asyncio
import sys
from typing import List, Tuple

from src.api.api_keys import provider_env_var, resolve_api_key
from src.config import (
    API_ENDPOINT, ASK_EMBEDDING_MODEL, ASK_MIN_SIMILARITY, ASK_TOP_K,
    DEFAULT_MODEL, LLM_PROVIDER, OLLAMA_NUM_CTX,
)
from src.core.llm.factory import create_llm_provider
from src.utils.file_detector import detect_file_type

from .answerer import AskResult, BookQA
from .service import load_book

OUT_OF_SCOPE_MESSAGE = "This question is outside the scope of the book: the book does not cover it."
_DIM, _BOLD, _RESET = "\033[2m", "\033[1m", "\033[0m"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.core.ask",
                                     description="Ask questions about a book; answers come only from its text.")
    parser.add_argument("file", help="Book to ask about (TXT, EPUB, DOCX or PDF).")
    parser.add_argument("--provider", default=LLM_PROVIDER, help=f"LLM provider (default: {LLM_PROVIDER}).")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--api_endpoint", default=API_ENDPOINT, help="Endpoint for Ollama or OpenAI-compatible servers.")
    parser.add_argument("--embedding-model", default=ASK_EMBEDDING_MODEL,
                        help="Ollama embedding model for semantic search (default: ASK_EMBEDDING_MODEL, empty = BM25 only).")
    parser.add_argument("--embedding-endpoint", default=None,
                        help="Ollama endpoint for embeddings (default: --api_endpoint).")
    parser.add_argument("--answer-language", default="auto",
                        help='Language for answers, e.g. "Hinglish" (default: same as the question).')
    parser.add_argument("--min-similarity", type=float, default=ASK_MIN_SIMILARITY)
    parser.add_argument("--top-k", type=int, default=ASK_TOP_K)
    parser.add_argument("--debug", action="store_true", help="Show search query and retrieval scores.")
    return parser.parse_args(argv)


def print_result(result: AskResult, debug: bool) -> None:
    if debug:
        sim = "n/a" if result.top_similarity is None else f"{result.top_similarity:.3f}"
        print(f"{_DIM}[query: {result.search_query!r}  similarity: {sim}  bm25: {result.top_lexical:.2f}  "
              f"in_scope: {result.in_scope}  reason: {result.reason}]{_RESET}")
    if not result.in_scope:
        print(f"\n{OUT_OF_SCOPE_MESSAGE}\n")
        return
    print(f"\n{result.answer}\n")
    for c in result.citations:
        print(f"{_DIM}  [P{c.passage_id}] {c.section or '(untitled section)'}{_RESET}")
    print()


async def run(args) -> int:
    file_type = detect_file_type(args.file)
    embed_endpoint = args.embedding_endpoint or (args.api_endpoint if args.provider == "ollama" else None)
    print(f"Indexing {args.file} ...", file=sys.stderr)
    book = load_book(args.file, file_type, args.embedding_model or None, embed_endpoint)
    print(f"{len(book.index.passages)} passages, language: {book.language}, "
          f"search: {'BM25 + embeddings' if book.index.has_embeddings else 'BM25'}", file=sys.stderr)

    kwargs = {"model": args.model, "context_window": OLLAMA_NUM_CTX}
    env_var = provider_env_var(args.provider)
    api_key = resolve_api_key("__USE_ENV__", env_var) if env_var else None
    if api_key:
        kwargs["api_key"] = api_key
    if args.api_endpoint and args.provider in ("ollama", "openai"):
        kwargs["api_endpoint"] = args.api_endpoint
    provider = create_llm_provider(args.provider, **kwargs)

    qa = BookQA(book.index, provider, book.language, top_k=args.top_k, min_similarity=args.min_similarity)
    history: List[Tuple[str, str]] = []
    last: AskResult = None
    print("Ask a question about the book (/sources, /reset, /quit).\n")
    try:
        while True:
            try:
                question = input(f"{_BOLD}> {_RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not question:
                continue
            if question in ("/quit", "/exit"):
                return 0
            if question == "/reset":
                history.clear()
                print("Conversation cleared.\n")
                continue
            if question == "/sources":
                for c in (last.citations if last else []):
                    print(f"\n[P{c.passage_id}] {c.section}\n{c.text}")
                print()
                continue
            last = await qa.ask(question, history, args.answer_language)
            print_result(last, args.debug)
            if last.in_scope:
                history.append((question, last.answer))
    finally:
        await provider.close()


def main(argv=None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
