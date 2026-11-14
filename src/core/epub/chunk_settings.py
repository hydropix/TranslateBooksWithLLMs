"""Settings of one placeholder-chunk translation pass.

``ChunkLoopSettings`` is everything the chunk fallback ladder and the
checkpointed chunk loop need from their caller. The EPUB engine builds it from
an ``EpubRun`` and a ``FileTask`` (see ``document_translation``); the DOCX
adapter, which has no ``EpubRun``, fills it directly. It replaces the long
parameter lists those functions used to take.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class ChunkLoopSettings:
    """What translating the chunks of one document needs.

    Attributes:
        source_language: Source language name.
        target_language: Target language name.
        model_name: Model name (prompt logs and persisted state).
        llm_client: The provider every request goes through.
        max_retries: Phase 1 attempts per chunk.
        context_manager: Adaptive context manager, or None.
        prompt_options: Prompt customization options, or None.
        log_callback: The job's log callback, or None.
        stats_callback: Receives the metrics dict on every update, or None.
        check_interruption: Returns True when the job must stop, or None.
        parallel_workers: Concurrent chunk requests (already resolved
            against the provider).
        max_tokens_per_chunk: Chunk size used to build the chunks; None
            reads MAX_TOKENS_PER_CHUNK at call time.
        bilingual: Bilingual output (source then translation).
        retry_token_aligned: Explicit, user-initiated retry of the chunks
            token alignment translated with approximate tag positions. Never
            set by the automatic path (design decision D3).
        protect_technical: The chunks were built with technical content
            protection. Only recorded in the partial state.
        checkpoint_manager: Saves the partial state, or None.
        translation_id: The job id the partial state is saved under.
        file_href: The document's key in the partial state.
        file_path: The document's path, recorded in the partial state
            (``file_href`` when None).
        global_total_chunks: Chunks of the whole book (EPUB), or None.
        global_completed_chunks: Chunks finished in earlier files, or None.
    """

    source_language: str
    target_language: str
    model_name: str
    llm_client: Any
    max_retries: int = 1
    context_manager: Any = None
    prompt_options: Optional[Dict[str, Any]] = None
    log_callback: Optional[Callable[..., Any]] = None
    stats_callback: Optional[Callable[[Dict[str, Any]], Any]] = None
    check_interruption: Optional[Callable[[], bool]] = None
    parallel_workers: int = 1
    max_tokens_per_chunk: Optional[int] = None
    bilingual: bool = False
    retry_token_aligned: bool = False
    protect_technical: bool = False
    checkpoint_manager: Any = None
    translation_id: Optional[str] = None
    file_href: Optional[str] = None
    file_path: Optional[str] = None
    global_total_chunks: Optional[int] = None
    global_completed_chunks: Optional[int] = None

    @property
    def can_checkpoint(self) -> bool:
        return bool(self.checkpoint_manager and self.translation_id and self.file_href)

    @property
    def chunk_size(self) -> int:
        """``max_tokens_per_chunk``, or the configured default."""
        if self.max_tokens_per_chunk is not None:
            return self.max_tokens_per_chunk
        from src.config import MAX_TOKENS_PER_CHUNK
        return MAX_TOKENS_PER_CHUNK

    @property
    def plain_text_mode(self) -> bool:
        return bool((self.prompt_options or {}).get('plain_text_mode'))

    def log(self, event_name: str, message: str) -> None:
        if self.log_callback:
            self.log_callback(event_name, message)
