"""File-level entry points of the EPUB engine.

``prepare_document`` extracts a body, replaces its tags with placeholders and
chunks it; the book layer calls it once per file for the chunk precount and
hands the result to ``translate_document`` through ``FileTask.prepared``.
``translate_document`` translates one parsed XHTML document in place.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from lxml import etree

from .body_serializer import extract_body_html
from .container import TranslationContainer
from .run_context import EpubRun, FileTask, PreparedDocument


def prepare_document(
    doc_root: etree._Element,
    max_tokens_per_chunk: int,
    container: Optional[TranslationContainer] = None,
) -> Optional[PreparedDocument]:
    """Extract, tag-preserve and chunk ``doc_root``'s body.

    Mutates ``doc_root`` (boilerplate removal, ruby folding): call it on a
    tree that is not written back. Returns None when there is no body.
    """
    container = container or TranslationContainer()
    preserver = container.tag_preserver
    placeholder_format = (preserver.placeholder_format.prefix,
                          preserver.placeholder_format.suffix)
    body_html, body_element = extract_body_html(doc_root)
    if body_element is None:
        return None
    if not body_html or not body_html.strip():
        return PreparedDocument([], {}, placeholder_format, is_empty=True)
    text, tag_map = preserver.preserve_tags(body_html)
    chunks = container.chunker_for(max_tokens_per_chunk).chunk_html_with_placeholders(text, tag_map)
    return PreparedDocument(chunks, tag_map, placeholder_format)


async def translate_document(
    doc_root: etree._Element,
    run: EpubRun,
    task: FileTask,
) -> Tuple[bool, Any]:
    """Translate one XHTML document in place; ``(success, TranslationMetrics)``.

    Transitional implementation over ``EpubTranslationAdapter``; it ignores
    ``task.prepared``.
    """
    from .epub_translation_adapter import EpubTranslationAdapter

    return await EpubTranslationAdapter().translate_document(
        doc_root,
        log_callback=run.log_callback,
        source_language=run.source_language,
        target_language=run.target_language,
        model_name=run.model_name,
        llm_client=run.llm_client,
        max_tokens_per_chunk=run.max_tokens_per_chunk,
        context_manager=run.context_manager,
        max_retries=run.max_retries,
        prompt_options=run.prompt_options,
        stats_callback=task.stats_callback,
        checkpoint_manager=run.checkpoint_manager,
        translation_id=run.translation_id,
        file_href=task.href,
        check_interruption_callback=run.check_interruption,
        resume_state=task.resume_state,
        retry_token_aligned=run.retry_token_aligned,
        global_total_chunks=task.global_total_chunks,
        global_completed_chunks=task.global_completed_chunks,
        parallel_workers=run.parallel_workers,
    )
