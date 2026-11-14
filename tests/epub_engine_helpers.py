"""Call the file-level EPUB engine from tests with loose keyword arguments.

The engine takes a ``ChunkLoopSettings``; these helpers build one from the
keyword arguments the tests pass (``check_interruption_callback`` is the
settings' ``check_interruption``), with test-friendly defaults.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.core.epub.chunk_settings import ChunkLoopSettings


def make_settings(
    source_language: str = "English",
    target_language: str = "French",
    model_name: str = "test-model",
    llm_client: Any = None,
    *,
    check_interruption_callback=None,
    **fields: Any,
) -> ChunkLoopSettings:
    """A ChunkLoopSettings; ``file_path`` defaults to ``file_href``."""
    if check_interruption_callback is not None:
        fields['check_interruption'] = check_interruption_callback
    fields.setdefault('file_path', fields.get('file_href'))
    return ChunkLoopSettings(
        source_language=source_language,
        target_language=target_language,
        model_name=model_name,
        llm_client=llm_client if llm_client is not None else object(),
        **fields,
    )


async def translate_xhtml(
    doc_root,
    source_language: str = "English",
    target_language: str = "French",
    model_name: str = "test-model",
    llm_client: Any = None,
    *,
    resume_state: Any = None,
    prepared: Any = None,
    container: Any = None,
    **fields: Any,
):
    """``document_translation.translate_xhtml_document`` with loose settings."""
    from src.core.epub.document_translation import translate_xhtml_document

    settings = make_settings(source_language, target_language, model_name, llm_client, **fields)
    return await translate_xhtml_document(
        doc_root, settings, resume_state=resume_state, prepared=prepared, container=container)


async def run_chunk_loop(
    chunks: List[Dict],
    placeholder_format: Tuple[str, str] = ("[id", "]"),
    *,
    global_tag_map: Optional[Dict[str, str]] = None,
    start_chunk_index: int = 0,
    translated_chunks: Optional[List[str]] = None,
    chunk_statuses: Optional[List[str]] = None,
    stats: Any = None,
    original_chunks: Optional[List[Dict]] = None,
    **fields: Any,
):
    """``chunk_loop.translate_chunks`` with loose settings."""
    from src.core.epub.chunk_loop import translate_chunks

    return await translate_chunks(
        chunks, make_settings(**fields), placeholder_format,
        global_tag_map=global_tag_map,
        start_chunk_index=start_chunk_index,
        translated_chunks=translated_chunks,
        chunk_statuses=chunk_statuses,
        stats=stats,
        original_chunks=original_chunks,
    )


async def translate_chunk(
    chunk_text: str,
    local_tag_map: Dict[str, str],
    global_indices: List[int],
    stats: Any,
    *,
    placeholder_format: Optional[Tuple[str, str]] = None,
    chunk_index: Optional[int] = None,
    **fields: Any,
):
    """``chunk_fallback.translate_chunk_with_fallback`` with loose settings."""
    from src.core.epub.chunk_fallback import translate_chunk_with_fallback

    return await translate_chunk_with_fallback(
        chunk_text, local_tag_map, global_indices, make_settings(**fields), stats,
        placeholder_format, chunk_index=chunk_index)
