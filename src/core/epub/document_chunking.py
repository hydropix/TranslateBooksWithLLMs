"""Turning a body into placeholder chunks.

One implementation for every caller that chunks an XHTML body: the
translation (``document_translation``), the chunk precount
(``prepare_document``) and the EPUB refine-only mode, so the refiner chunks
exactly like the translation did.
"""

from typing import Callable, Dict, List, Optional, Tuple

from lxml import etree

from .body_serializer import extract_body_html
from .container import TranslationContainer
from .run_context import PreparedDocument
from .tag_preservation import TagPreserver


def preserve_tags(
    body_html: str,
    tag_preserver: TagPreserver,
    log_callback: Optional[Callable] = None,
) -> Tuple[str, Dict[str, str], Tuple[str, str]]:
    """Replace HTML tags with placeholders.

    Returns:
        Tuple of (text_with_placeholders, global_tag_map, placeholder_format)
    """
    text_with_placeholders, global_tag_map = tag_preserver.preserve_tags(body_html)

    placeholder_format = (tag_preserver.placeholder_format.prefix, tag_preserver.placeholder_format.suffix)
    if log_callback:
        log_callback("tags_preserved", tags_preserved_message(
            len(global_tag_map), placeholder_format))
    return text_with_placeholders, global_tag_map, placeholder_format


def tags_preserved_message(tag_count: int, placeholder_format: Tuple[str, str]) -> str:
    format_info = f" using format {placeholder_format[0]}N{placeholder_format[1]}"
    return f"Preserved {tag_count} tag groups{format_info}"


def create_chunks(
    text: str,
    tag_map: Dict[str, str],
    max_tokens: int,
    log_callback: Optional[Callable] = None,
    container: Optional[TranslationContainer] = None
) -> List[Dict]:
    """Chunk placeholder text into translatable segments.

    The caller's ``max_tokens`` always wins over the container's default size.
    """
    if container is not None:
        chunker = container.chunker_for(max_tokens)
    else:
        from .html_chunker import HtmlChunker
        chunker = HtmlChunker(max_tokens=max_tokens)
    chunks = chunker.chunk_html_with_placeholders(text, tag_map)
    if log_callback:
        log_callback("chunks_created", f"Created {len(chunks)} chunks")
    return chunks


def chunk_body_html(
    body_html: str,
    max_tokens_per_chunk: int,
    *,
    container: Optional[TranslationContainer] = None,
    log_callback: Optional[Callable] = None,
) -> PreparedDocument:
    """Tag-preserve and chunk an extracted body.

    An empty body gives an empty ``PreparedDocument``.
    """
    container = container or TranslationContainer()
    preserver = container.tag_preserver
    if not body_html or not body_html.strip():
        return PreparedDocument([], {}, (preserver.placeholder_format.prefix,
                                         preserver.placeholder_format.suffix), is_empty=True)
    text, tag_map, placeholder_format = preserve_tags(body_html, preserver, log_callback)
    chunks = create_chunks(text, tag_map, max_tokens_per_chunk, log_callback, container)
    return PreparedDocument(chunks, tag_map, placeholder_format)


def chunk_document(
    doc_root: etree._Element,
    max_tokens_per_chunk: int,
    *,
    container: Optional[TranslationContainer] = None,
    log_callback: Optional[Callable] = None,
) -> Tuple[Optional[etree._Element], Optional[PreparedDocument]]:
    """Extract ``doc_root``'s body and chunk it like the translation does.

    Mutates the body (boilerplate removal, ruby folding). Returns
    ``(body, prepared)``; ``(None, None)`` when there is no body.
    """
    body_html, body = extract_body_html(doc_root, log_callback=log_callback)
    if body is None:
        return None, None
    return body, chunk_body_html(body_html, max_tokens_per_chunk, container=container,
                                 log_callback=log_callback)


def tag_preserver_for(placeholder_format: Tuple[str, str],
                      container: Optional[TranslationContainer] = None) -> TagPreserver:
    """A TagPreserver restoring placeholders of ``placeholder_format``."""
    preserver = container.tag_preserver if container is not None else TagPreserver()
    preserver.placeholder_format.prefix = placeholder_format[0]
    preserver.placeholder_format.suffix = placeholder_format[1]
    return preserver
