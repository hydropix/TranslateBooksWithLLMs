"""Checkpoint plumbing shared by the EPUB and DOCX Plain Text Mode adapters.

Plain Text Mode has no persistence format of its own: it reuses the
``XHTMLTranslationState`` partial-state machinery that the placeholder pipeline
already writes through ``CheckpointManager``. The two adapters only differ in
where the paragraph count and the file identifier come from, so the resume
guard, the state builder and the completion delete live here rather than being
duplicated on both sides.

The ``plain_text_mode`` marker written into ``doc_metadata`` is what keeps the
two persistence users apart: a placeholder-mode state (whose chunks carry
``local_tag_map`` / ``global_indices`` and real placeholders) must never be
replayed through the plain pipeline, and vice versa.

``plain_text_keep_emphasis`` records whether the paragraphs carried emphasis
markers (EPUB, issue #352). A state written with the other setting holds
translations of differently extracted text and is not reused; a state that
predates the key was written without markers.
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core.epub.xhtml_translation_state import XHTMLTranslationState


# doc_metadata key of the emphasis-marker setting a state was written with.
KEEP_EMPHASIS_METADATA_KEY = 'plain_text_keep_emphasis'


def _utc_now_iso() -> str:
    """Timestamp in the exact format used by the XHTML checkpoint writer."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + 'Z'


def _as_persistable_chunks(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add the two inert keys ``XHTMLTranslationState.validate()`` requires.

    Plain segments are ``{'indices', 'text', 'partial'}``; the placeholder
    pipeline's chunks additionally carry ``local_tag_map`` and
    ``global_indices``, and ``validate()`` rejects any chunk missing them. A
    rejected state is silently dropped by ``load_xhtml_partial_state``, which
    would make the checkpoint unreadable. The two added keys stay empty and are
    never read back by the plain pipeline.
    """
    return [
        {'local_tag_map': {}, 'global_indices': [], **segment}
        for segment in segments
    ]


def is_plain_text_state(resume_state: Optional[Any]) -> bool:
    """True when a partial state was written by Plain Text Mode.

    The single source of truth for telling the two persistence users apart.
    The plain pipeline uses it to refuse a placeholder state, and the
    placeholder pipelines use it to refuse a plain one: replaying a plain state
    there would restore a prefix under an empty placeholder scheme.
    """
    metadata = getattr(resume_state, 'doc_metadata', None)
    return isinstance(metadata, dict) and metadata.get('plain_text_mode') is True


def resume_plain_segments(
    resume_state: Optional[Any],
    paragraph_count: int,
    log_callback: Optional[Callable] = None,
    keep_emphasis: bool = False,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[str]]]:
    """Validate a partial state for a plain-text resume.

    Returns ``(segments, translated_prefix)`` when the state was written by this
    pipeline for this exact document, and ``(None, None)`` otherwise so the
    caller starts fresh. Both rejection paths log and never raise.

    Args:
        resume_state: XHTMLTranslationState loaded from the checkpoint, or None.
        paragraph_count: number of source paragraphs extracted for this run.
        log_callback: optional (key, message) logger.
        keep_emphasis: whether this run extracts emphasis markers; a state
            written with the other setting is not reused.
    """
    if resume_state is None:
        return None, None

    if not is_plain_text_state(resume_state):
        if log_callback:
            log_callback(
                "plain_text_resume_ignored",
                "⚠️ Checkpoint was not written by Plain Text Mode, restarting this file"
            )
        return None, None

    stored_count = resume_state.doc_metadata.get('paragraph_count')
    if stored_count != paragraph_count:
        if log_callback:
            log_callback(
                "plain_text_resume_ignored",
                f"⚠️ Checkpoint covers {stored_count} paragraphs but the source has "
                f"{paragraph_count}, restarting this file"
            )
        return None, None

    stored_emphasis = resume_state.doc_metadata.get(KEEP_EMPHASIS_METADATA_KEY) is True
    if stored_emphasis != bool(keep_emphasis):
        if log_callback:
            log_callback(
                "plain_text_resume_ignored",
                "⚠️ Checkpoint was written with the italics option "
                f"{'on' if stored_emphasis else 'off'}, restarting this file"
            )
        return None, None

    return list(resume_state.chunks or []), list(resume_state.translated_chunks or [])


def build_plain_checkpoint_hook(
    *,
    checkpoint_manager: Optional[Any],
    translation_id: Optional[str],
    file_href: Optional[str],
    source_language: str,
    target_language: str,
    model_name: str,
    max_tokens_per_chunk: int,
    max_retries: Optional[int],
    paragraph_count: int,
    prompt_options: Optional[Dict[str, Any]] = None,
    bilingual: bool = False,
    original_body_html: str = "",
    keep_emphasis: bool = False,
) -> Optional[Callable[..., None]]:
    """Build the ``checkpoint_hook`` for ``translate_paragraphs_plain``.

    Returns None when there is nothing to write against (no manager, no job id
    or no file identifier); the pipeline then skips checkpointing entirely.

    ``original_body_html`` is the serialized SOURCE ``<body>``. The EPUB adapter
    passes it because a finished file that still holds untranslated segments
    keeps its state, and the next pass re-enters it from a copy on disk that
    has already been replaced by its translated rebuild (issue #285): the state
    is then the only place the source structure (block tags, attributes,
    anchored images) can be read back from.
    """
    if not (checkpoint_manager and translation_id and file_href):
        return None

    def _save_state(
        segments: List[Dict[str, Any]],
        prefix: List[str],
        next_index: int,
        stats_dict: Dict[str, Any],
        chunk_statuses: Optional[List[str]] = None,
    ) -> None:
        """Persist the contiguous translated prefix; resume restarts at next_index.

        ``chunk_statuses`` records which segments of the prefix fell back to
        their source text, so a later pass retries them (issue #285). None
        leaves the field unset, which loads as "whole prefix translated".
        """
        now = _utc_now_iso()
        state = XHTMLTranslationState(
            file_path=file_href or '',
            translation_id=translation_id,
            file_href=file_href,
            source_language=source_language,
            target_language=target_language,
            model_name=model_name,
            max_tokens_per_chunk=max_tokens_per_chunk,
            max_retries=max_retries or 1,
            chunks=_as_persistable_chunks(segments),
            global_tag_map={},
            placeholder_format=("", ""),
            translated_chunks=list(prefix),
            current_chunk_index=next_index,
            original_body_html=original_body_html or "",
            doc_metadata={
                'plain_text_mode': True,
                'paragraph_count': paragraph_count,
                KEEP_EMPHASIS_METADATA_KEY: bool(keep_emphasis),
            },
            stats=stats_dict,
            created_at=now,
            updated_at=now,
            global_stats=None,
            prompt_options=prompt_options,
            bilingual=bilingual,
            original_chunks=None,
            chunk_statuses=list(chunk_statuses) if chunk_statuses is not None else None,
        )
        checkpoint_manager.save_xhtml_partial_state(translation_id, file_href, state)

    return _save_state


def delete_plain_checkpoint(
    checkpoint_manager: Optional[Any],
    translation_id: Optional[str],
    file_href: Optional[str],
    log_callback: Optional[Callable] = None,
) -> None:
    """Drop the plain-text partial state once the output has been rebuilt.

    EPUB gets this for free from ``_save_translated_file``; DOCX has no
    equivalent seam, so it calls this explicitly. Failing to delete must never
    fail the translation.
    """
    if not (checkpoint_manager and translation_id and file_href):
        return
    try:
        checkpoint_manager.delete_xhtml_partial_state(translation_id, file_href)
    except Exception as exc:  # noqa: BLE001 - cleanup must not break the result
        if log_callback:
            log_callback(
                "plain_text_checkpoint_delete_failed",
                f"⚠️ Could not delete the plain-text checkpoint: {exc}"
            )
