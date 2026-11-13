"""
Unified file translation function using the adapter pattern.

This module provides a single entry point for translating files of any supported
format (TXT, SRT, EPUB, DOCX, PDF) using format-specific adapters.

File type detection supports:
1. Known extensions (.txt, .epub, .srt, .docx, .pdf)
2. Content-based detection for unknown extensions (e.g., .log, .md, .text -> txt)
3. Automatic routing to appropriate processors
"""

import os
from typing import Optional, Dict, Any

from .generic_translator import GenericTranslator
from .txt_adapter import TxtAdapter
from .srt_adapter import SrtAdapter
from .pgn_adapter import PgnAdapter
from .exceptions import UnsupportedFormatError
from src.core.job import JobConfig, JobContext, JobResult
from src.utils.file_detector import detect_file_type, detect_file_type_by_content


def _ensure_checkpoint_job(
    checkpoint_manager: Any,
    translation_id: Optional[str],
    file_type: str,
    config: Dict[str, Any],
    input_file_path: str
) -> None:
    """
    Guarantee a job row exists before the legacy EPUB/DOCX pipelines run.

    Those pipelines write checkpoint chunks but never create the parent job:
    the web path relies on the API handler having created it first. The CLI
    calls translate_file() directly, so nothing created it there and every
    chunk row was silently orphaned. Foreign-key enforcement now rejects such
    writes outright, so the row has to exist before the first chunk is saved.
    """
    if not (checkpoint_manager and translation_id):
        return

    try:
        if checkpoint_manager.db.get_job(translation_id) is not None:
            return
        checkpoint_manager.start_job(
            translation_id=translation_id,
            file_type=file_type,
            config=config,
            input_file_path=input_file_path
        )
    except Exception as e:
        # Checkpointing is best effort; never block a translation over it.
        print(f"Could not create checkpoint job {translation_id}: {e}")


def _detect_type(input_filepath: str, ctx: JobContext) -> str:
    """Detect the format, by extension first, then by content."""
    _, ext = os.path.splitext(input_filepath.lower())

    # Use content-based detection for file type determination
    # This handles files with non-standard extensions (e.g., .log, .text, .md)
    try:
        detected_type = detect_file_type(input_filepath)
    except ValueError:
        # If detection fails, try content-based detection as fallback
        detected_type = detect_file_type_by_content(input_filepath)
        if detected_type is None:
            raise UnsupportedFormatError(
                f"Cannot determine file type for: {ext}. "
                f"The file does not appear to be a supported format."
            )

    # Log detected type if different from extension
    if detected_type != ext.lstrip('.'):
        ctx.log("file_type_detected",
                f"📄 File with extension '{ext}' detected as '{detected_type.upper()}' format")
    return detected_type


def _resolve_workers(job: JobConfig, ctx: JobContext) -> int:
    """Resolve concurrent workers once; local providers are forced to 1."""
    from src.config import resolve_parallel_workers, is_local_provider
    llm_provider = job.llm_provider
    requested_workers = job.parallel_workers
    parallel_workers = resolve_parallel_workers(llm_provider, requested_workers)
    if parallel_workers > 1:
        ctx.log("parallel_enabled",
                f"⚡ Parallel translation: {parallel_workers} concurrent chunks ({llm_provider})")
    elif (requested_workers or 1) > 1 and is_local_provider(llm_provider):
        ctx.log("parallel_disabled_local",
                f"ℹ️ Parallel translation requested ({requested_workers}) but '{llm_provider}' is a "
                f"local provider — forced to 1 (a single local instance serializes requests).")
    return parallel_workers


def _dict_result(result: Dict[str, Any], stats: Dict[str, Any], job: JobConfig,
                 ctx: JobContext) -> JobResult:
    """JobResult of a DOCX/PDF run (their engines return a result dict)."""
    return JobResult.from_run(
        stats, job.output_path,
        engine_ok=bool(result.get('success', False)),
        interrupted=ctx.interrupted(),
    )


async def translate_file(job: JobConfig, ctx: JobContext) -> JobResult:
    """
    Translate a file using the adapter pattern.

    This is the unified entry point for all file format translations (TXT,
    SRT, PGN, EPUB, DOCX, PDF), used by both the web API and the CLI.

    Args:
        job: The job. ``input_path`` is read, ``output_path`` written; the
            provider, languages, chunking and context parameters,
            ``prompt_options``, ``bilingual_output``, ``parallel_workers``
            and ``retry_token_aligned`` (EPUB only: explicit, user-initiated
            retry of the chunks token alignment translated with approximate
            tag positions, D3) come from it. ``prompt_options['refine']``
            still runs the in-engine EPUB/DOCX refine pass; the job runner,
            not the engine, avoids a double refinement.
        ctx: Log/stats/interruption hooks, checkpoint manager, job id and
            ``resume_from_index``.

    Returns:
        The run's :class:`JobResult`, its status from the merged stats the
        engine emitted (``JobResult.from_run``).

    Raises:
        UnsupportedFormatError: If the file format is not supported
        RateLimitError: propagated unchanged so the caller can pause.

    Example::

        job = JobConfig(input_path="book.epub", output_path="book_fr.epub",
                        file_type="epub", source_language="English",
                        target_language="French",
                        provider=ProviderConfig(provider="ollama", model="llama3.2"))
        ctx = JobContext(translation_id="job_123", checkpoint_manager=CheckpointManager())
        result = await translate_file(job, ctx)
    """
    # Resolve max_tokens_per_chunk: when the job carries none, read the
    # current .env-backed default. Doing it here (not at import time) means a
    # reload_config() between calls is honoured for subsequent runs.
    if job.max_tokens_per_chunk is None:
        from src.config import MAX_TOKENS_PER_CHUNK as _DEFAULT_MAX_TOKENS
        job = job.with_changes(max_tokens_per_chunk=_DEFAULT_MAX_TOKENS)

    # Every downstream pipeline re-resolves idempotently.
    parallel_workers = _resolve_workers(job, ctx)
    job = job.with_changes(parallel_workers=parallel_workers)

    input_filepath = job.input_path
    detected_type = _detect_type(input_filepath, ctx)

    # The EPUB/DOCX/PDF pipelines save checkpoint chunks without ever creating
    # the parent job row, which only worked because the web handler creates
    # it beforehand. Make the invariant hold on every entry point.
    if detected_type in ('epub', 'docx', 'pdf'):
        _ensure_checkpoint_job(
            checkpoint_manager=ctx.checkpoint_manager,
            translation_id=ctx.translation_id,
            file_type=detected_type,
            config={
                'input_file_path': input_filepath,
                'output_file_path': job.output_path,
                'source_language': job.source_language,
                'target_language': job.target_language,
                'model_name': job.model,
                'llm_provider': job.llm_provider,
            },
            input_file_path=input_filepath
        )

    ctx, seen_stats = ctx.tap_stats()

    # EPUB keeps its dedicated pipeline (HTML chunking, tag preservation,
    # technical content protection) and returns a JobResult itself.
    if detected_type == 'epub':
        from src.core.epub.translator import translate_epub_file
        return await translate_epub_file(job, ctx)

    # DOCX reuses the EPUB-style pipeline (HTML chunking, tag preservation).
    if detected_type == 'docx':
        from src.core.docx.translator import translate_docx_file
        result = await translate_docx_file(job, ctx, max_retries=1,
                                           parallel_workers=parallel_workers)
        return _dict_result(result, seen_stats, job, ctx)

    # PDF: extractor + plain-text pipeline + reflowed PDF builder.
    if detected_type == 'pdf':
        from src.core.pdf.translator import translate_pdf_file
        result = await translate_pdf_file(job, ctx, max_retries=1,
                                          parallel_workers=parallel_workers)
        return _dict_result(result, seen_stats, job, ctx)

    # Map detected file types to adapters
    adapter_map = {
        'txt': TxtAdapter,
        'srt': SrtAdapter,
        'pgn': PgnAdapter,
    }

    adapter_class = adapter_map.get(detected_type)
    if not adapter_class:
        supported = ', '.join(['txt', 'srt', 'pgn', 'epub', 'docx', 'pdf'])
        raise UnsupportedFormatError(
            f"Unsupported file format: {detected_type}. Supported formats: {supported}"
        )

    # Adapter configuration (format-specific settings)
    adapter_config = {
        'context_window': job.context_window,
        'auto_adjust_context': job.auto_adjust_context,
        'min_chunk_size': job.min_chunk_size,
        'max_tokens_per_chunk': job.max_tokens_per_chunk,
        'prompt_options': job.engine_prompt_options(),
    }

    adapter = adapter_class(
        input_file_path=input_filepath,
        output_file_path=job.output_path,
        config=adapter_config
    )

    translator = GenericTranslator(
        adapter=adapter,
        checkpoint_manager=ctx.checkpoint_manager,
        translation_id=ctx.translation_id
    )

    ok = await translator.translate(job, ctx)
    return JobResult.from_run(seen_stats, job.output_path, engine_ok=ok,
                              interrupted=ctx.interrupted())


def get_file_type_from_path(filepath: str) -> str:
    """
    Get the file type identifier from a file path.

    Args:
        filepath: Path to the file

    Returns:
        File type string: 'txt', 'srt', 'pgn', 'epub', 'docx', 'pdf', or 'unknown'
    """
    _, ext = os.path.splitext(filepath.lower())

    type_map = {
        '.txt': 'txt',
        '.srt': 'srt',
        '.pgn': 'pgn',
        '.epub': 'epub',
        '.docx': 'docx',
        '.pdf': 'pdf',
    }

    return type_map.get(ext, 'unknown')
