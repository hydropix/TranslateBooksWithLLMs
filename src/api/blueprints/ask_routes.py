"""
Ask the Book routes.

Endpoints:
    POST /api/ask/index  - extract + index an uploaded book (cached), return its stats
    POST /api/ask/query  - answer one question from the book's own text, or report it out of scope
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Blueprint, jsonify, request

from src.api.api_keys import provider_env_var, resolve_api_key
from src.api.services.path_validator import PathValidator
from src.config import (
    ASK_EMBEDDING_MODEL, ASK_MIN_SIMILARITY, ASK_TOP_K,
    OLLAMA_API_ENDPOINT, OLLAMA_NUM_CTX,
)
from src.core.ask import SUPPORTED_FILE_TYPES, BookQA, load_book
from src.core.llm.factory import create_llm_provider
from src.utils.file_detector import detect_file_type

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
MAX_HISTORY_TURNS = 6


def _run_async(coro_factory):
    """Run a coroutine on a private event loop (Flask handlers are synchronous)."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro_factory())
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def _parse_history(raw: Any) -> List[Tuple[str, str]]:
    """Accept [[question, answer], ...] or [{question, answer}, ...]; keep the last few turns."""
    turns: List[Tuple[str, str]] = []
    for item in raw or []:
        if isinstance(item, dict):
            q, a = item.get("question"), item.get("answer")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            q, a = item
        else:
            continue
        if isinstance(q, str) and isinstance(a, str):
            turns.append((q[:MAX_QUESTION_CHARS], a[:4000]))
    return turns[-MAX_HISTORY_TURNS:]


def _instantiate_provider(data: Dict[str, Any]):
    provider = (data.get("provider") or "ollama").lower()
    env_var = provider_env_var(provider)
    api_key = resolve_api_key(data.get("api_key"), env_var) if env_var else None
    kwargs: Dict[str, Any] = {
        "model": data.get("model"),
        "context_window": int(data.get("context_window") or OLLAMA_NUM_CTX),
    }
    if api_key:
        kwargs["api_key"] = api_key
    if data.get("api_endpoint"):
        kwargs["api_endpoint"] = data["api_endpoint"]
    return create_llm_provider(provider, **kwargs)


def create_ask_blueprint(output_dir="."):
    bp = Blueprint("ask", __name__)
    uploads_dir = Path(output_dir) / "uploads"

    def _load(data: Dict[str, Any]):
        """Validate the client path (confined to uploads/) and return (book, error_response)."""
        safe_path, path_error = PathValidator.validate_upload_path(data.get("file_path"), uploads_dir)
        if path_error is not None:
            status = 400 if path_error.startswith("Missing") else 403
            return None, (jsonify({"error": path_error}), status)
        try:
            file_type = detect_file_type(str(safe_path))
        except Exception as exc:
            return None, (jsonify({"error": f"Cannot detect file type: {exc}"}), 400)
        if file_type not in SUPPORTED_FILE_TYPES:
            return None, (jsonify({"error": "unsupported_file_type", "file_type": file_type}), 400)
        try:
            return load_book(str(safe_path), file_type, ASK_EMBEDDING_MODEL or None, OLLAMA_API_ENDPOINT), None
        except ValueError as exc:
            return None, (jsonify({"error": str(exc)}), 400)
        except Exception as exc:
            logger.exception("Ask the Book: indexing failed")
            return None, (jsonify({"error": f"Indexing failed: {exc}"}), 500)

    @bp.route("/api/ask/index", methods=["POST"])
    def index_book():
        book, error = _load(request.get_json(silent=True) or {})
        if error:
            return error
        return jsonify({
            "passages": len(book.index.passages),
            "language": book.language,
            "chars": book.chars,
            "search": "hybrid" if book.index.has_embeddings else "bm25",
        })

    @bp.route("/api/ask/query", methods=["POST"])
    def ask_question():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Missing question"}), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify({"error": "Question too long"}), 400
        if not data.get("model"):
            return jsonify({"error": "Missing model"}), 400

        book, error = _load(data)
        if error:
            return error

        async def _answer():
            provider = _instantiate_provider(data)
            try:
                qa = BookQA(book.index, provider, book.language,
                            top_k=ASK_TOP_K, min_similarity=ASK_MIN_SIMILARITY)
                return await qa.ask(question, _parse_history(data.get("history")),
                                    data.get("answer_language") or "auto")
            finally:
                await provider.close()

        try:
            result = _run_async(_answer)
        except Exception as exc:
            logger.exception("Ask the Book: answering failed")
            return jsonify({"error": f"LLM request failed: {exc}"}), 502
        return jsonify(result.to_dict())

    return bp
