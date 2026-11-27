"""Plain Text Mode translation of one XHTML document.

Plain Text Mode skips placeholders entirely: the body becomes a list of plain
paragraphs (images anchored to their paragraph), the paragraphs go through
the common plain-text pipeline, and the body is rewritten with a flat
structure (block tags and their attributes preserved, images reattached after
their parent paragraph, inline formatting dropped).

Segment-level checkpointing reuses the XHTML partial-state machinery: a rate
limit or an interruption leaves a resumable checkpoint behind, and a resume
state restarts at the first unattempted segment and retries every segment
that kept its source text (issue #285). The EPUB translator deletes the state
once the file has been saved, unless a segment is still untranslated: then
the state, which carries the source body, is what the next pass re-enters the
file with.

With the ``plain_text_keep_emphasis`` option (issue #352), <i>/<em> spans
travel as literal markers in the paragraph text and are rebuilt as real
elements when the translated markers pass validation; the rebuild's counters
are logged per document and summed for the book.
"""

from typing import Any, Dict, Optional, Tuple

from lxml import etree

from .chunk_settings import ChunkLoopSettings
from .translation_metrics import TranslationMetrics


def _restore_source_body(
    body: etree._Element,
    original_body_html: str,
    settings: ChunkLoopSettings,
) -> Optional[etree._Element]:
    """Swap `body` for the source body a Plain Text Mode state carries.

    Returns the restored source body, or None when the stored markup cannot
    be parsed or has nowhere to go - in which case `body` is left untouched.
    """
    parent = body.getparent()
    source_body = None
    if parent is not None:
        try:
            parser = etree.XMLParser(encoding='utf-8', recover=True, remove_blank_text=False)
            source_body = etree.fromstring(original_body_html.encode('utf-8'), parser)
        except etree.XMLSyntaxError:
            source_body = None
    if source_body is None:
        settings.log(
            "plain_text_source_body_unusable",
            f"⚠️ {settings.file_href or 'document'}: the checkpoint's source body could not "
            "be restored - keeping this file as it is and its retry for later"
        )
        return None
    parent.replace(body, source_body)
    return source_body


async def _translate_paragraphs(paragraphs_text, original_body_html: str,
                                settings: ChunkLoopSettings, resume_state: Optional[Any]):
    """Run the plain-text pipeline over the paragraphs, with segment
    checkpoints and resume; ``(translated, stats, was_interrupted)``."""
    from src.core.common.plain_text_pipeline import translate_paragraphs_plain
    from src.core.common.plain_text_checkpoint import (
        build_plain_checkpoint_hook,
        resume_plain_segments,
    )

    keep_emphasis = settings.plain_text_keep_emphasis
    resume_segments, resume_translated = resume_plain_segments(
        resume_state, len(paragraphs_text), settings.log_callback,
        keep_emphasis=keep_emphasis,
    )
    checkpoint_hook = build_plain_checkpoint_hook(
        checkpoint_manager=settings.checkpoint_manager,
        translation_id=settings.translation_id,
        file_href=settings.file_href,
        source_language=settings.source_language,
        target_language=settings.target_language,
        model_name=settings.model_name,
        max_tokens_per_chunk=settings.chunk_size,
        max_retries=settings.max_retries,
        paragraph_count=len(paragraphs_text),
        prompt_options=settings.prompt_options,
        bilingual=settings.bilingual,
        original_body_html=original_body_html,
        keep_emphasis=keep_emphasis,
    )

    # The statuses and counters only mean something next to the prefix they
    # were written with, so they follow it when the resume guard accepts it.
    resumed = resume_segments is not None
    return await translate_paragraphs_plain(
        paragraphs=paragraphs_text,
        source_language=settings.source_language,
        target_language=settings.target_language,
        model_name=settings.model_name,
        llm_client=settings.llm_client,
        max_tokens_per_chunk=settings.chunk_size,
        log_callback=settings.log_callback,
        stats_callback=settings.stats_callback,
        context_manager=settings.context_manager,
        check_interruption_callback=settings.check_interruption,
        prompt_options=settings.prompt_options,
        parallel_workers=settings.parallel_workers,
        resume_segments=resume_segments,
        resume_translated=resume_translated,
        resume_statuses=resume_state.chunk_statuses if resumed else None,
        resume_stats=resume_state.stats if resumed else None,
        checkpoint_hook=checkpoint_hook,
        # A failed segment is persisted as CHUNK_UNTRANSLATED and retried by
        # the next pass, which makes it a fallback, not a terminal failure.
        count_failures_as_fallback=True,
        keep_emphasis=keep_emphasis,
    )


def _report_emphasis(emphasis: Optional[Dict[str, int]], settings: ChunkLoopSettings) -> None:
    """Log one document's italics counters and add them to the book totals."""
    if not emphasis or not emphasis.get("source_spans"):
        return
    recovered = emphasis.get("recovered_spans", 0)
    settings.log(
        "plain_text_emphasis",
        f"Italics: {emphasis['kept_spans']}/{emphasis['source_spans']} spans kept, "
        + (f"{recovered} recovered from Markdown *...*, " if recovered else "")
        + f"{emphasis['rejected_paragraphs']} paragraph(s) fell back to plain"
    )
    if settings.emphasis_totals is not None:
        for key, value in emphasis.items():
            settings.emphasis_totals[key] = settings.emphasis_totals.get(key, 0) + value


async def translate_plain_text(
    body: etree._Element,
    settings: ChunkLoopSettings,
    resume_state: Optional[Any] = None,
) -> Tuple[bool, TranslationMetrics]:
    """Translate ``body`` (already extracted: boilerplate stripped, ruby
    folded) in Plain Text Mode, in place; ``(success, stats)``."""
    from .plain_extractor import extract_plain_paragraphs, replace_body_with_paragraphs
    from src.core.common.plain_text_checkpoint import is_plain_text_state

    # A file re-entered to retry its untranslated segments was already saved,
    # so the copy on disk is its translated rebuild: re-extracting that would
    # yield other paragraphs (bilingual twins, image wrappers, flattened tags)
    # and the resume guard below would restart the file on translated text.
    # The state carries the source body for exactly this case (issue #285); a
    # state without one (a file paused mid-way, whose copy on disk is still
    # the source) leaves the body as it is.
    disk_body = None
    if is_plain_text_state(resume_state) and resume_state.original_body_html:
        disk_body = body
        body = _restore_source_body(body, resume_state.original_body_html, settings)
        if body is None:
            # Translating the copy on disk would translate translated text.
            # Failing the file keeps that copy and the state, so the retry is
            # still owed on the next pass.
            return False, TranslationMetrics()

    # Captured before extraction, which folds <ruby> in place.
    original_body_html = etree.tostring(body, encoding='unicode')

    keep_emphasis = settings.plain_text_keep_emphasis
    paragraphs_text, paragraphs_tag, images_by_paragraph, paragraphs_attrib = extract_plain_paragraphs(
        body, keep_emphasis=keep_emphasis)
    settings.log(
        "plain_text_extracted",
        f"📝 Plain Text Mode: {len(paragraphs_text)} paragraphs, "
        f"{sum(len(v) for v in images_by_paragraph.values())} images anchored"
    )

    translated, stats, was_interrupted = await _translate_paragraphs(
        paragraphs_text, original_body_html, settings, resume_state)

    if was_interrupted:
        # The caller (EPUB translator) treats a failed translation as keeping
        # the original; the body is left untouched so the partial output keeps
        # the source text. A swapped-in source body is not "untouched": the
        # caller writes this document into the partial EPUB, so the translated
        # copy the pass started from goes back in place. What the pass
        # repaired is in the state and is not lost.
        if disk_body is not None:
            body.getparent().replace(body, disk_body)
        return False, stats

    emphasis = replace_body_with_paragraphs(
        body_element=body,
        translated_paragraphs=translated,
        paragraphs_tag=paragraphs_tag,
        images_by_paragraph=images_by_paragraph,
        bilingual=settings.bilingual,
        # Always passed: outside bilingual mode the source text is what an
        # empty translation falls back to, so the paragraph survives instead
        # of being dropped. The bilingual flag only drives the interleaving.
        source_paragraphs=paragraphs_text,
        # Carry each source block's attributes (class, id, xml:lang, ...)
        # onto the rebuilt element instead of emitting a bare tag.
        paragraphs_attrib=paragraphs_attrib,
        keep_emphasis=keep_emphasis,
    )
    _report_emphasis(emphasis, settings)
    return True, stats
