"""State objects of one EPUB translation run.

``EpubRun`` is what every file of a book shares: the job, its live hooks, the
LLM client and the chunking settings. ``FileTask`` is what one XHTML file
adds: its href, its resume state, the chunks prepared for it ahead of time and
the stats hook that turns its progress into book progress. ``PreparedDocument``
is the result of extracting a body, replacing its tags with placeholders and
chunking it, computed once per file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config import DEFAULT_MODEL, MAX_TRANSLATION_ATTEMPTS

from ..job import JobConfig, JobContext


@dataclass(frozen=True)
class EpubRun:
    """Settings and hooks shared by every file of one book.

    Attributes:
        job: The job being run.
        ctx: Its live hooks (log, stats, interruption, checkpoint manager).
        llm_client: The provider every request goes through.
        context_manager: Adaptive context manager, or None.
        max_tokens_per_chunk: The job's chunk size.
        prompt_options: The engine's private copy of the prompt options
            (``bilingual`` set when the job asks for bilingual output).
        parallel_workers: Concurrent chunk requests (already resolved
            against the provider).
        max_retries: Phase 1 attempts per chunk.
    """

    job: JobConfig
    ctx: JobContext
    llm_client: Any
    context_manager: Any = None
    max_tokens_per_chunk: int = 450
    prompt_options: Optional[Dict[str, Any]] = None
    parallel_workers: int = 1
    max_retries: int = MAX_TRANSLATION_ATTEMPTS

    @property
    def source_language(self) -> str:
        return self.job.source_language

    @property
    def target_language(self) -> str:
        return self.job.target_language

    @property
    def model_name(self) -> str:
        return self.job.model or DEFAULT_MODEL

    @property
    def bilingual(self) -> bool:
        return bool((self.prompt_options or {}).get('bilingual', False))

    @property
    def plain_text_mode(self) -> bool:
        return bool((self.prompt_options or {}).get('plain_text_mode'))

    @property
    def retry_token_aligned(self) -> bool:
        """Explicit retry of the token-aligned chunks (never automatic, D3)."""
        return bool(self.job.retry_token_aligned)

    @property
    def log_callback(self) -> Optional[Callable[..., Any]]:
        return self.ctx.log_callback

    @property
    def checkpoint_manager(self) -> Any:
        return self.ctx.checkpoint_manager

    @property
    def translation_id(self) -> Optional[str]:
        return self.ctx.translation_id

    @property
    def check_interruption(self) -> Optional[Callable[[], bool]]:
        return self.ctx.check_interruption

    @property
    def can_checkpoint(self) -> bool:
        return bool(self.ctx.checkpoint_manager and self.ctx.translation_id)

    def log(self, key: str, message: str = "", data: Optional[dict] = None) -> None:
        self.ctx.log(key, message, data)

    def interrupted(self) -> bool:
        return self.ctx.interrupted()


@dataclass(frozen=True)
class PreparedDocument:
    """A body turned into translatable chunks.

    Attributes:
        chunks: HtmlChunker output (``text`` with local placeholders,
            ``local_tag_map``, ``global_indices``).
        global_tag_map: Document-level placeholder -> markup map.
        placeholder_format: ``(prefix, suffix)`` of the placeholders.
        is_empty: The body holds nothing to translate.
    """

    chunks: List[Dict[str, Any]]
    global_tag_map: Dict[str, str]
    placeholder_format: Tuple[str, str]
    is_empty: bool = False

    @property
    def chunk_count(self) -> int:
        return 0 if self.is_empty else len(self.chunks)


@dataclass
class FileTask:
    """One XHTML file of the book, as the file loop hands it to the translator.

    Attributes:
        href: Manifest href; the key of the file's partial state.
        file_path: Absolute path of the file in the extracted book.
        stats_callback: Receives the file's metrics dict on every update.
        resume_state: The file's ``XHTMLTranslationState`` when it resumes.
        prepared: Chunks prepared from the source file ahead of time (the
            chunk precount), or None to prepare them now.
        global_total_chunks: Chunks of the whole book.
        global_completed_chunks: Chunks finished in earlier files.
    """

    href: str
    file_path: str
    stats_callback: Optional[Callable[[Dict[str, Any]], Any]] = None
    resume_state: Any = None
    prepared: Optional[PreparedDocument] = None
    global_total_chunks: Optional[int] = None
    global_completed_chunks: Optional[int] = None
