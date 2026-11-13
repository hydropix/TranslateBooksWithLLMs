"""
DOCX translation using generic orchestrator.

Phase 2 implementation: Uses the unified GenericTranslationOrchestrator
with DocxTranslationAdapter for clean, reusable architecture.
"""

from typing import Optional, Dict, Any
from ..common.translation_orchestrator import GenericTranslationOrchestrator
from ..job import JobConfig, JobContext
from .docx_translation_adapter import DocxTranslationAdapter


async def translate_docx_file(
    job: JobConfig,
    ctx: JobContext,
    *,
    llm_client: Optional[Any] = None,
    context_manager: Optional[Any] = None,
    max_retries: int = 1,
    parallel_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Translate a complete DOCX file using the generic orchestrator.

    This implementation uses the unified translation pipeline:
    1. Extract content (DOCX → HTML via mammoth)
    2. Preserve structure (HTML tags → placeholders)
    3. Chunk intelligently (HTML-aware chunking)
    4. Translate chunks (with 3-phase fallback)
    5. Optional: Refine translation
    6. Reconstruct content (restore HTML tags)
    7. Finalize output (HTML → DOCX via python-docx)

    Args:
        job: The job: ``input_path``, ``output_path``, languages, provider,
            ``max_tokens_per_chunk`` (the current ``MAX_TOKENS_PER_CHUNK``
            when unset), ``prompt_options`` and ``parallel_workers``.
        ctx: Log/stats/interruption hooks, checkpoint manager and job id.
        llm_client: LLM client to use. When None, one is built from the job
            (``create_llm_provider(job.provider_for(...))``) and closed on exit.
        context_manager: Adaptive context manager (optional)
        max_retries: Max translation retries
        parallel_workers: Concurrent chunks; None resolves
            ``job.parallel_workers`` against the provider.

    Returns:
        Dict with success, stats, output_path
    """
    from src.config import resolve_parallel_workers
    from src.core.epub.translator import _job_max_tokens_per_chunk

    input_filepath = job.input_path
    output_filepath = job.output_path
    model_name = job.model
    log_callback = ctx.log_callback
    checkpoint_manager = ctx.checkpoint_manager
    translation_id = ctx.translation_id
    if parallel_workers is None:
        parallel_workers = resolve_parallel_workers(job.llm_provider, job.parallel_workers)

    owns_client = llm_client is None
    if owns_client:
        from src.core.llm.factory import create_llm_provider
        llm_client = create_llm_provider(job.provider_for(log_callback=log_callback))

    try:
        return await _translate_docx(
            input_filepath, output_filepath, job.source_language, job.target_language,
            model_name, llm_client, _job_max_tokens_per_chunk(job), log_callback,
            ctx.stats_callback, job.engine_prompt_options(), max_retries, context_manager,
            ctx.check_interruption, checkpoint_manager, translation_id, parallel_workers,
        )
    finally:
        if owns_client and hasattr(llm_client, 'close'):
            try:
                await llm_client.close()
            except Exception:
                pass


async def _translate_docx(
    input_filepath, output_filepath, source_language, target_language, model_name,
    llm_client, max_tokens_per_chunk, log_callback, stats_callback, prompt_options,
    max_retries, context_manager, check_interruption_callback, checkpoint_manager,
    translation_id, parallel_workers,
) -> Dict[str, Any]:
    import os

    # Create adapter and orchestrator
    adapter = DocxTranslationAdapter()
    orchestrator = GenericTranslationOrchestrator(adapter)

    # Use filename as file_href for checkpointing
    file_href = os.path.basename(input_filepath)

    # Check for resume state
    resume_state = None
    if checkpoint_manager and translation_id:
        resume_state = checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
        if resume_state and log_callback:
            log_callback("docx_resume_detected", f"Found checkpoint at chunk {resume_state.current_chunk_index}")

    # Translate using generic pipeline with checkpoint support
    docx_bytes, stats = await orchestrator.translate(
        source=input_filepath,
        source_language=source_language,
        target_language=target_language,
        model_name=model_name,
        llm_client=llm_client,
        max_tokens_per_chunk=max_tokens_per_chunk,
        log_callback=log_callback,
        context_manager=context_manager,
        max_retries=max_retries,
        prompt_options=prompt_options,
        stats_callback=stats_callback,
        check_interruption_callback=check_interruption_callback,
        checkpoint_manager=checkpoint_manager,
        translation_id=translation_id,
        file_href=file_href,
        resume_state=resume_state,
        parallel_workers=parallel_workers
    )

    # Check if translation was interrupted
    if not docx_bytes:
        if log_callback:
            log_callback("docx_incomplete", "DOCX translation interrupted - partial state saved")
        return {
            'success': False,
            'stats': stats.to_dict(),
            'output_path': None,
            'interrupted': True
        }

    # Save to output file
    with open(output_filepath, 'wb') as f:
        f.write(docx_bytes)

    if log_callback:
        log_callback("file_saved", f"DOCX saved to {output_filepath}")

    return {
        'success': True,
        'stats': stats.to_dict(),
        'output_path': output_filepath
    }
