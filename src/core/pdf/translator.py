"""
PDF translation entry point.

Wires the PDF extractor, the shared plain-text pipeline and the PDF builder:

    input.pdf -> extract_pdf_paragraphs -> translate_paragraphs_plain -> build_pdf -> output.pdf

This mirrors ``DocxTranslationAdapter._translate_plain_text`` without the
``GenericTranslationOrchestrator`` layer: PDF has no placeholder path, so the
orchestrator would be dead weight. Segment-level checkpointing reuses the
plain-text partial-state contract unchanged, with the DOCX ``file_href``
convention (the input file's basename).
"""

import os
from typing import Any, Callable, Dict, Optional

from src.core.common.plain_text_checkpoint import (
    build_plain_checkpoint_hook,
    delete_plain_checkpoint,
    resume_plain_segments,
)
from src.core.common.plain_text_pipeline import translate_paragraphs_plain
from src.core.job import JobConfig, JobContext
from src.core.pdf.builder import build_pdf
from src.core.pdf.content import PdfError
from src.core.pdf.extractor import extract_pdf_paragraphs


async def translate_pdf_file(
    job: JobConfig,
    ctx: JobContext,
    *,
    llm_client: Optional[Any] = None,
    context_manager: Optional[Any] = None,
    max_retries: int = 1,
    parallel_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Translate a text PDF into a new, reflowed PDF.

    Steps:
    1. Load a plain-text partial state for this file when checkpointing is on
    2. Extract paragraphs, styles and images (PyMuPDF)
    3. Translate the paragraphs through the shared plain-text pipeline,
       resuming at the first unattempted segment when a state exists
    4. Rebuild a PDF from the translated paragraphs and drop the checkpoint

    Args:
        job: The job: ``input_path``, ``output_path``, languages, provider,
            ``max_tokens_per_chunk`` (the current ``MAX_TOKENS_PER_CHUNK``
            when unset), ``prompt_options`` (``bilingual`` is honoured by the
            builder) and ``parallel_workers``.
        ctx: Log/stats/interruption hooks, checkpoint manager and job id.
        llm_client: LLM client to use. When None, one is built from the job
            (``create_llm_provider(job.provider_for(...))``) and closed on exit.
        context_manager: Adaptive context manager (optional)
        max_retries: Max translation retries (recorded in the checkpoint)
        parallel_workers: Concurrent chunks; None resolves
            ``job.parallel_workers`` against the provider.

    Returns:
        Dict with success, stats, output_path (and ``interrupted`` when the
        translation was interrupted; no output file is written in that case
        and the checkpoint is kept for a later resume).

    Raises:
        PdfError: A subclass (PdfOpenError, PdfEncryptedError,
            PdfNoTextLayerError) when the input cannot be extracted, or
            PdfBuildError when the output cannot be generated. No output file
            is written on an extraction failure.
    """
    from src.config import resolve_parallel_workers
    from src.core.epub.translator import _job_max_tokens_per_chunk

    if parallel_workers is None:
        parallel_workers = resolve_parallel_workers(job.llm_provider, job.parallel_workers)
    owns_client = llm_client is None
    if owns_client:
        from src.core.llm.factory import create_llm_provider
        llm_client = create_llm_provider(job.provider_for(log_callback=ctx.log_callback))

    try:
        return await _translate_pdf(
            input_filepath=job.input_path,
            output_filepath=job.output_path,
            source_language=job.source_language,
            target_language=job.target_language,
            model_name=job.model,
            llm_client=llm_client,
            max_tokens_per_chunk=_job_max_tokens_per_chunk(job),
            log_callback=ctx.log_callback,
            stats_callback=ctx.stats_callback,
            prompt_options=job.engine_prompt_options(),
            max_retries=max_retries,
            context_manager=context_manager,
            check_interruption_callback=ctx.check_interruption,
            checkpoint_manager=ctx.checkpoint_manager,
            translation_id=ctx.translation_id,
            parallel_workers=parallel_workers,
        )
    finally:
        if owns_client and hasattr(llm_client, 'close'):
            try:
                await llm_client.close()
            except Exception:
                pass


async def _translate_pdf(
    *,
    input_filepath: str,
    output_filepath: str,
    source_language: str,
    target_language: str,
    model_name: str,
    llm_client: Any,
    max_tokens_per_chunk: int,
    log_callback: Optional[Callable],
    stats_callback: Optional[Callable],
    prompt_options: Optional[Dict],
    max_retries: int,
    context_manager: Optional[Any],
    check_interruption_callback: Optional[Callable],
    checkpoint_manager: Optional[Any],
    translation_id: Optional[str],
    parallel_workers: int,
) -> Dict[str, Any]:
    # Use filename as file_href for checkpointing (DOCX convention)
    file_href = os.path.basename(input_filepath)
    bilingual = bool(prompt_options.get('bilingual')) if prompt_options else False

    # Check for resume state
    resume_state = None
    if checkpoint_manager and translation_id:
        resume_state = checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)

    try:
        content = extract_pdf_paragraphs(input_filepath)
    except PdfError as exc:
        if log_callback:
            log_callback("pdf_extraction_failed", f"❌ {exc}")
        raise

    paragraph_count = len(content.paragraphs_text)
    n_images = sum(len(images) for images in content.images_by_paragraph.values())

    if log_callback:
        log_callback(
            "pdf_extracted",
            f"📄 PDF: {content.page_count} pages, {paragraph_count} paragraphs, "
            f"{len(content.tables)} tables, {n_images} images"
        )

    resume_segments, resume_translated = resume_plain_segments(
        resume_state, paragraph_count, log_callback
    )
    checkpoint_hook = build_plain_checkpoint_hook(
        checkpoint_manager=checkpoint_manager,
        translation_id=translation_id,
        file_href=file_href,
        source_language=source_language,
        target_language=target_language,
        model_name=model_name,
        max_tokens_per_chunk=max_tokens_per_chunk,
        max_retries=max_retries,
        paragraph_count=paragraph_count,
        prompt_options=prompt_options,
        bilingual=bilingual,
    )

    translated, stats, was_interrupted = await translate_paragraphs_plain(
        paragraphs=content.paragraphs_text,
        source_language=source_language,
        target_language=target_language,
        model_name=model_name,
        llm_client=llm_client,
        max_tokens_per_chunk=max_tokens_per_chunk,
        log_callback=log_callback,
        stats_callback=stats_callback,
        context_manager=context_manager,
        check_interruption_callback=check_interruption_callback,
        prompt_options=prompt_options,
        parallel_workers=parallel_workers,
        resume_segments=resume_segments,
        resume_translated=resume_translated,
        checkpoint_hook=checkpoint_hook,
    )

    if was_interrupted:
        if log_callback:
            log_callback("pdf_interrupted", "⏸️ PDF translation interrupted - state saved")
        return {
            'success': False,
            'stats': stats.to_dict(),
            'output_path': None,
            'interrupted': True,
        }

    build_pdf(
        translated,
        content,
        output_filepath,
        target_language,
        source_language,
        bilingual,
    )

    if log_callback:
        log_callback("pdf_rebuilt", f"📄 PDF rebuilt ({os.path.getsize(output_filepath)} bytes)")

    # The document is complete: drop the segment checkpoint so a later
    # resume of the same job does not replay a stale state.
    delete_plain_checkpoint(checkpoint_manager, translation_id, file_href, log_callback)

    return {
        'success': True,
        'stats': stats.to_dict(),
        'output_path': output_filepath,
    }
