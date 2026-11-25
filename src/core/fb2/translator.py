"""
FB2 translation through the placeholder pipeline.

Pipeline (same machinery as DOCX and EPUB):
1. Parse the FB2 file and collect its translatable leaf blocks
   (:mod:`src.core.fb2.document`).
2. Join the blocks into an HTML-like string, replace its tags with
   placeholders (TagPreserver) and chunk it (HtmlChunker).
3. Translate the chunks with the checkpointed EPUB chunk loop (resume,
   interruption, progress, parallel workers, fallbacks).
4. Restore the tags, write each translated block back into the XML tree, set
   ``<lang>`` to the target language and save UTF-8.
"""

import os
from typing import Any, Callable, Dict, Optional

from ..job import JobConfig, JobContext
from .document import (
    Fb2Error, apply_translations, build_translation_html, is_block_boundary,
    parse_fb2, serialize_fb2, set_document_language, split_translated_html,
)


async def translate_fb2_file(
    job: JobConfig,
    ctx: JobContext,
    *,
    llm_client: Optional[Any] = None,
    context_manager: Optional[Any] = None,
    max_retries: int = 1,
    parallel_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Translate a complete FB2 file.

    Refinement is not done here (``--refine`` is reported as unsupported for
    FB2 input). Plain Text Mode is ignored: FB2 always goes through
    placeholders, which is what keeps its inline markup and note links.

    Args:
        job: The job (paths, languages, provider, chunk size, prompt options,
            ``bilingual_output``, ``parallel_workers``).
        ctx: Log/stats/interruption hooks, checkpoint manager and job id.
        llm_client: LLM client to use; one is built from the job (and closed
            on exit) when None.
        context_manager: Adaptive context manager (optional).
        max_retries: Max translation retries per chunk.
        parallel_workers: Concurrent chunks; None resolves
            ``job.parallel_workers`` against the provider.

    Returns:
        Dict with success, stats, output_path (and interrupted when stopped).
    """
    from src.config import resolve_parallel_workers

    if parallel_workers is None:
        parallel_workers = resolve_parallel_workers(job.llm_provider, job.parallel_workers)

    owns_client = llm_client is None
    if owns_client:
        from src.core.llm.factory import create_llm_provider
        llm_client = create_llm_provider(job.provider_for(log_callback=ctx.log_callback))

    try:
        return await _translate_fb2(job, ctx, llm_client, context_manager,
                                    max_retries, parallel_workers)
    finally:
        if owns_client and hasattr(llm_client, 'close'):
            try:
                await llm_client.close()
            except Exception:
                pass


def _target_language_code(target_language: str) -> str:
    from src.core.epub.lang_support import get_language_code
    return get_language_code(target_language) or target_language


def _load_resume_state(checkpoint_manager, translation_id, file_href,
                       chunks, global_tag_map, log: Callable[[str, str], None]):
    """The saved partial state of this file, when it still describes it.

    The fresh chunking is deterministic, so a state whose chunks or tag map
    differ was written for another file (or another chunk size) and is not
    replayed.
    """
    if not (checkpoint_manager and translation_id):
        return None
    state = checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
    if state is None:
        return None
    from src.core.common.plain_text_checkpoint import is_plain_text_state
    same_chunks = [c.get('text') for c in state.chunks or []] == [c.get('text') for c in chunks]
    if is_plain_text_state(state) or not same_chunks or state.global_tag_map != global_tag_map:
        log("fb2_resume_ignored",
            "⚠️ Checkpoint does not match this FB2 file; starting over")
        return None
    log("fb2_resume_detected",
        f"📂 Resuming FB2 translation from chunk {state.current_chunk_index}/{len(state.chunks)}")
    return state


async def _translate_fb2(job: JobConfig, ctx: JobContext, llm_client, context_manager,
                         max_retries: int, parallel_workers: int) -> Dict[str, Any]:
    from src.core.epub.chunk_loop import translate_chunks
    from src.core.epub.chunk_settings import ChunkLoopSettings
    from src.core.epub.container import TranslationContainer
    from src.core.epub.llm_setup import job_max_tokens_per_chunk
    from src.core.epub.translation_metrics import TranslationMetrics

    log_callback = ctx.log_callback

    def log(event: str, message: str) -> None:
        if log_callback:
            log_callback(event, message)

    with open(job.input_path, 'rb') as f:
        raw = f.read()
    try:
        document = parse_fb2(raw)
    except Fb2Error as e:
        raise RuntimeError(f"Could not read the FB2 file: {e}") from e
    log("fb2_parsed", f"FB2 ({document.source_encoding}): "
                      f"{len(document.blocks)} text blocks to translate")

    bilingual = bool(job.bilingual_output)
    file_href = os.path.basename(job.input_path)
    max_tokens = job_max_tokens_per_chunk(job)
    container = TranslationContainer()
    preserver = container.tag_preserver

    html = build_translation_html(document.blocks)
    text, global_tag_map = preserver.preserve_tags(html) if document.blocks else ('', {})
    placeholder_format = (preserver.placeholder_format.prefix,
                          preserver.placeholder_format.suffix)
    chunks = container.chunker_for(max_tokens).chunk_html_with_placeholders(
        text, global_tag_map)
    log("chunks_created", f"Created {len(chunks)} chunks")

    translated_chunks, start_index, chunk_statuses = [], 0, None
    stats = TranslationMetrics()
    resume_state = _load_resume_state(ctx.checkpoint_manager, ctx.translation_id,
                                      file_href, chunks, global_tag_map, log)
    if resume_state is not None:
        translated_chunks = list(resume_state.translated_chunks)
        start_index = resume_state.current_chunk_index
        chunk_statuses = resume_state.chunk_statuses
        if resume_state.stats:
            stats = TranslationMetrics.from_dict(resume_state.stats)
    stats.total_chunks = len(chunks)

    if chunks:
        settings = ChunkLoopSettings(
            source_language=job.source_language,
            target_language=job.target_language,
            model_name=job.model,
            llm_client=llm_client,
            max_retries=max_retries,
            context_manager=context_manager,
            prompt_options=_placeholder_prompt_options(job.engine_prompt_options()),
            log_callback=log_callback,
            stats_callback=ctx.stats_callback,
            check_interruption=ctx.check_interruption,
            parallel_workers=parallel_workers,
            max_tokens_per_chunk=max_tokens,
            checkpoint_manager=ctx.checkpoint_manager,
            translation_id=ctx.translation_id,
            file_href=file_href,
            file_path=job.input_path,
            global_total_chunks=len(chunks),
            global_completed_chunks=len(translated_chunks),
            block_boundary=is_block_boundary,
        )
        translated_chunks, stats, interrupted = await translate_chunks(
            chunks, settings, placeholder_format,
            global_tag_map=global_tag_map,
            start_chunk_index=start_index,
            translated_chunks=translated_chunks,
            chunk_statuses=chunk_statuses,
            stats=stats,
        )
        if interrupted:
            log("fb2_interrupted", "FB2 translation interrupted - state saved")
            return {'success': False, 'stats': stats.to_dict(), 'output_path': None,
                    'interrupted': True}

        restored = preserver.restore_tags(''.join(translated_chunks), global_tag_map)
        failed = apply_translations(document, split_translated_html(restored),
                                    bilingual=bilingual)
        if failed:
            log("fb2_blocks_kept",
                f"⚠️ {len(failed)} of {len(document.blocks)} FB2 blocks could not be "
                f"rebuilt from the translation and keep their source text")
    else:
        log("no_content", "No content to translate")

    if not set_document_language(document.root, _target_language_code(job.target_language)):
        log("fb2_no_lang", "⚠️ No <title-info><lang> element; language not updated")

    with open(job.output_path, 'wb') as f:
        f.write(serialize_fb2(document))
    log("file_saved", f"FB2 saved to {job.output_path}")

    _drop_finished_state(ctx.checkpoint_manager, ctx.translation_id, file_href)

    return {'success': True, 'stats': stats.to_dict(), 'output_path': job.output_path}


def _drop_finished_state(checkpoint_manager, translation_id, file_href) -> None:
    """Delete the partial state once every chunk is translated.

    A state that still lists untranslated chunks is kept: it is what lets a
    ``--resume`` retry them instead of starting over.
    """
    if not (checkpoint_manager and translation_id):
        return
    from src.core.epub.xhtml_translation_state import unfinished_chunk_indices
    try:
        state = checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
        if state is None or unfinished_chunk_indices(state.chunk_statuses):
            return
        checkpoint_manager.delete_xhtml_partial_state(translation_id, file_href)
    except Exception:
        pass


def _placeholder_prompt_options(prompt_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Prompt options with Plain Text Mode turned off (FB2 needs placeholders)."""
    options = dict(prompt_options or {})
    options.pop('plain_text_mode', None)
    return options

