"""File-level entry points of the EPUB engine.

``prepare_document`` extracts a body, replaces its tags with placeholders and
chunks it; the book layer calls it once per file for the chunk precount and
hands the result to ``translate_document`` through ``FileTask.prepared``.
``translate_document`` translates one parsed XHTML document in place:

1. locate the body - extracting it once (boilerplate stripped, ruby folded),
   or reusing ``task.prepared`` when the file does not resume;
2. finalize an empty body as is;
3. Plain Text Mode: hand the body to ``plain_text_document``;
4. otherwise prepare (or restore from the resume state) the chunks, translate
   them, reassemble the body and write it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from lxml import etree

from .body_reassembly import mark_file_untranslated, reconstruct_html, replace_body
from .body_serializer import extract_body_html, find_body, replace_body_content
from .chunk_loop import translate_chunks
from .chunk_settings import ChunkLoopSettings
from .container import TranslationContainer
from .document_chunking import (
    chunk_body_html,
    tag_preserver_for,
    tags_preserved_message,
)
from .exceptions import BodyExtractionError, XmlParsingError
from .run_context import EpubRun, FileTask, PreparedDocument
from .tag_preservation import TagPreserver
from .translation_metrics import TranslationMetrics
from .xhtml_translation_state import CHUNK_PENDING, CHUNK_UNTRANSLATED


def prepare_document(
    doc_root: etree._Element,
    max_tokens_per_chunk: int,
    container: Optional[TranslationContainer] = None,
) -> Optional[PreparedDocument]:
    """Extract, tag-preserve and chunk ``doc_root``'s body.

    Mutates ``doc_root`` (boilerplate removal, ruby folding): call it on a
    tree that is not written back. Returns None when there is no body.
    """
    body_html, body_element = extract_body_html(doc_root)
    if body_element is None:
        return None
    return chunk_body_html(body_html, max_tokens_per_chunk, container=container)


def settings_for_file(run: EpubRun, task: FileTask) -> ChunkLoopSettings:
    """The chunk-loop settings of one file of the book."""
    return ChunkLoopSettings(
        source_language=run.source_language,
        target_language=run.target_language,
        model_name=run.model_name,
        llm_client=run.llm_client,
        max_retries=run.max_retries,
        context_manager=run.context_manager,
        prompt_options=run.prompt_options,
        log_callback=run.log_callback,
        stats_callback=task.stats_callback,
        check_interruption=run.check_interruption,
        parallel_workers=run.parallel_workers,
        max_tokens_per_chunk=run.max_tokens_per_chunk,
        bilingual=run.bilingual,
        retry_token_aligned=run.retry_token_aligned,
        checkpoint_manager=run.checkpoint_manager,
        translation_id=run.translation_id,
        file_href=task.href,
        # The partial state has always recorded the href as its file path.
        file_path=task.href,
        global_total_chunks=task.global_total_chunks,
        global_completed_chunks=task.global_completed_chunks,
        emphasis_totals=run.emphasis_totals,
    )


async def translate_document(
    doc_root: etree._Element,
    run: EpubRun,
    task: FileTask,
) -> Tuple[bool, Any]:
    """Translate one XHTML document in place; ``(success, TranslationMetrics)``.

    ``task.prepared`` is reused when the file has no resume state, so the
    body is then never extracted again. Raises BodyExtractionError when the
    document has no <body>.
    """
    return await translate_xhtml_document(
        doc_root, settings_for_file(run, task),
        resume_state=task.resume_state, prepared=task.prepared,
    )


async def translate_xhtml_document(
    doc_root: etree._Element,
    settings: ChunkLoopSettings,
    *,
    resume_state: Optional[Any] = None,
    prepared: Optional[PreparedDocument] = None,
    container: Optional[TranslationContainer] = None,
) -> Tuple[bool, TranslationMetrics]:
    """Translate one XHTML document in place; ``(success, stats)``.

    See the module docstring for the steps. ``prepared`` is used only for a
    placeholder-mode file without ``resume_state``.
    """
    container = container or TranslationContainer()
    if prepared is not None and resume_state is None and not settings.plain_text_mode:
        body = find_body(doc_root)
        if body is None:
            raise BodyExtractionError("No <body> element found")
        if prepared.is_empty:
            return _finalize_empty_body(body, settings)
        return await _translate_with_placeholders(body, settings, None, prepared, container)

    body_html, body = extract_body_html(doc_root, log_callback=settings.log_callback)
    if body is None:
        raise BodyExtractionError("No <body> element found")
    settings.log("body_extracted", f"Extracted {len(body_html)} chars from XHTML body")
    if not body_html.strip():
        return _finalize_empty_body(body, settings)
    if settings.plain_text_mode:
        # The body was extracted above (boilerplate stripped, ruby folded)
        # before the plain extraction captures it: that order is what the
        # Plain Text Mode output and its checkpoint's source body depend on.
        from .plain_text_document import translate_plain_text
        return await translate_plain_text(body, settings, resume_state)
    if not _resumes_placeholder_state(resume_state, settings):
        resume_state = None
        prepared = chunk_body_html(body_html, settings.chunk_size, container=container)
    return await _translate_with_placeholders(body, settings, resume_state, prepared, container)


def _finalize_empty_body(body: etree._Element, settings: ChunkLoopSettings) -> Tuple[bool, TranslationMetrics]:
    """A body holding nothing is written back as is (no LLM call)."""
    settings.log("no_content", "No content to translate")
    try:
        replace_body_content(body, "", log_callback=settings.log_callback)
        settings.log("body_replaced", "Body content replaced successfully")
        success = True
    except (XmlParsingError, BodyExtractionError) as e:
        settings.log("replace_body_error", str(e))
        success = False
    return success, TranslationMetrics()


def _resumes_placeholder_state(resume_state: Optional[Any], settings: ChunkLoopSettings) -> bool:
    """True when ``resume_state`` is a placeholder-mode state to resume.

    A Plain Text Mode checkpoint carries no tag map and an empty placeholder
    format, so replaying it here would restore a prefix under a placeholder
    scheme that never produced it. This is the mirror of the guard the plain
    pipeline applies to placeholder states: each mode resumes only its own.
    """
    if not resume_state:
        return False
    from src.core.common.plain_text_checkpoint import is_plain_text_state
    if is_plain_text_state(resume_state):
        settings.log(
            "xhtml_resume_ignored",
            "⚠️ Checkpoint was written in Plain Text Mode; restarting this "
            "file with placeholder translation"
        )
        return False
    return True


@dataclass
class _ChunkPlan:
    """The chunks of a file and where their translation stands."""

    chunks: List[Dict]
    global_tag_map: Dict[str, str]
    placeholder_format: Tuple[str, str]
    tag_preserver: TagPreserver
    stats: TranslationMetrics
    translated_chunks: List[str]
    start_chunk_index: int = 0
    chunk_statuses: Optional[List[str]] = None
    original_chunks: Optional[List[Dict]] = None


def _plan_from_resume(resume_state: Any, settings: ChunkLoopSettings,
                      container: TranslationContainer) -> _ChunkPlan:
    """Restore the chunks and progress of a resumed file."""
    # Per-chunk statuses tell the two kinds of remaining work apart: a chunk
    # left in the source language by the Phase 3 fallback (retried, even
    # though it sits below the resume pointer) and a chunk never attempted
    # (issue #261).
    chunk_statuses = resume_state.chunk_statuses
    retry_indices = [index for index, status in enumerate(chunk_statuses or [])
                     if status == CHUNK_UNTRANSLATED]
    never_attempted = sum(1 for status in (chunk_statuses or []) if status == CHUNK_PENDING)
    settings.log("xhtml_resume_partial",
        f"📂 Resuming XHTML translation from chunk {resume_state.current_chunk_index}/{len(resume_state.chunks)} "
        f"- retrying {len(retry_indices)} untranslated chunk(s) {retry_indices}, "
        f"{never_attempted} never attempted")

    placeholder_format = resume_state.placeholder_format
    return _ChunkPlan(
        chunks=resume_state.chunks,
        global_tag_map=resume_state.global_tag_map,
        placeholder_format=placeholder_format,
        # Needed for the final reconstruction.
        tag_preserver=tag_preserver_for(placeholder_format, container),
        stats=(TranslationMetrics.from_dict(resume_state.stats) if resume_state.stats
               else TranslationMetrics()),
        translated_chunks=resume_state.translated_chunks.copy(),  # never mutate the state
        start_chunk_index=resume_state.current_chunk_index,
        chunk_statuses=chunk_statuses,
        original_chunks=resume_state.original_chunks if resume_state.bilingual else None,
    )


def _fresh_plan(prepared: PreparedDocument, settings: ChunkLoopSettings,
                container: TranslationContainer) -> _ChunkPlan:
    """A first pass over freshly prepared chunks (every chunk pending)."""
    settings.log("tags_preserved", tags_preserved_message(
        len(prepared.global_tag_map), prepared.placeholder_format))
    settings.log("chunks_created", f"Created {len(prepared.chunks)} chunks")

    stats = TranslationMetrics()
    stats.total_chunks = len(prepared.chunks)
    return _ChunkPlan(
        chunks=prepared.chunks,
        global_tag_map=prepared.global_tag_map,
        placeholder_format=prepared.placeholder_format,
        tag_preserver=tag_preserver_for(prepared.placeholder_format, container),
        stats=stats,
        translated_chunks=[],
        original_chunks=prepared.chunks.copy() if settings.bilingual else None,
    )


async def _translate_with_placeholders(
    body: etree._Element,
    settings: ChunkLoopSettings,
    resume_state: Optional[Any],
    prepared: Optional[PreparedDocument],
    container: TranslationContainer,
) -> Tuple[bool, TranslationMetrics]:
    """Translate the chunks, reassemble them and write the body back."""
    plan = (_plan_from_resume(resume_state, settings, container) if resume_state
            else _fresh_plan(prepared, settings, container))

    translated_chunks, stats, was_interrupted = await translate_chunks(
        plan.chunks, settings, plan.placeholder_format,
        global_tag_map=plan.global_tag_map,
        start_chunk_index=plan.start_chunk_index,
        translated_chunks=plan.translated_chunks,
        chunk_statuses=plan.chunk_statuses,
        stats=plan.stats,
        original_chunks=plan.original_chunks,
    )
    if was_interrupted:
        settings.log("xhtml_interrupted_saved", "⏸️ Translation interrupted - state saved for resume")
        return False, stats  # incomplete

    final_html = reconstruct_html(
        translated_chunks, plan.global_tag_map, plan.tag_preserver,
        original_chunks=plan.chunks if settings.bilingual else None,
        bilingual=settings.bilingual,
    )
    xml_success = replace_body(body, final_html, settings.log_callback)
    if not xml_success:
        # The body kept its source text, so every chunk of this file is still
        # owed. Say so in the persisted statuses: otherwise they all read
        # 'translated' and the file would look clean to the retry map.
        mark_file_untranslated(stats, len(plan.chunks), settings.checkpoint_manager,
                               settings.translation_id, settings.file_href, settings.log_callback)

    # Partial state deletion is handled by the book layer after the file is
    # saved: the state is deleted ONLY after the file is safely on disk.
    settings.log("translation_complete", "Body translation complete")
    return xml_success, stats
