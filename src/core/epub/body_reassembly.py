"""Turning translated chunks back into a body.

Reassembly joins the chunks (or, in bilingual mode, interleaves the full
source and the full translation paragraph by paragraph), escapes the stray
markup characters an LLM may emit, restores the tags and writes the result
into the document's <body>.
"""
import copy as _copy
import re
from html.entities import html5 as _HTML5_ENTITIES
from typing import Any, Callable, Dict, List, Optional

from lxml import etree

from .body_serializer import replace_body_content
from .engine_log import report_error
from .exceptions import BodyExtractionError, XmlParsingError
from .placeholder_renumberer import globalize_chunks
from .tag_preservation import TagPreserver
from .translation_metrics import TranslationMetrics
from .xhtml_translation_state import CHUNK_PENDING, CHUNK_UNTRANSLATED


def reconstruct_html(
    translated_chunks: List[str],
    global_tag_map: Dict[str, str],
    tag_preserver: TagPreserver,
    original_chunks: Optional[List[Dict]] = None,
    bilingual: bool = False
) -> str:
    """Reconstruct the full body HTML from translated chunks.

    Args:
        translated_chunks: Translated chunk texts (global placeholders)
        global_tag_map: Global tag map
        tag_preserver: TagPreserver used to restore the tags
        original_chunks: Source chunks (required for bilingual mode)
        bilingual: If True, interleave source and translated content

    Returns:
        Reconstructed HTML string
    """
    if bilingual and original_chunks:
        return _reconstruct_bilingual(translated_chunks, global_tag_map, tag_preserver,
                                      original_chunks)
    # Standard mode: just join translated chunks.
    # Escape stray < and > in LLM output before restoring placeholders.
    # By this point the joined text should contain only placeholders [idN]
    # and plain translated text; all real HTML tags came from the source
    # and live in global_tag_map, ready to be injected by restore_tags().
    # If the source used literal angle brackets as stylistic markers
    # (common in Korean webnovels: <SkillName>, <ItemName>, status windows)
    # the LLM passes them through as raw < and >. Left unescaped, they
    # become phantom HTML elements at replace_body_content() time and
    # corrupt the document. Escaping before restore_tags() keeps real
    # tags (from the tag map) intact while turning stray brackets into
    # the entities that render as literal "<...>" in the EPUB reader.
    full_translated_text = escape_stray_markup(''.join(translated_chunks))
    return tag_preserver.restore_tags(full_translated_text, global_tag_map)


def _reconstruct_bilingual(
    translated_chunks: List[str],
    global_tag_map: Dict[str, str],
    tag_preserver: TagPreserver,
    original_chunks: List[Dict],
) -> str:
    """Bilingual reconstruction: source paragraph, then its translation.

    Reconstruct the FULL source and FULL translation as two complete,
    well-formed documents, then interleave them at the paragraph level. We
    must NOT reconstruct per chunk: a chapter is often wrapped in a container
    element (e.g. <div class="Section13">) that spans many chunks, so a single
    chunk's restored fragment has unbalanced nesting. Reparsing such fragments
    in isolation corrupted the document and truncated most of the chapter
    (discussion #199).
    """
    # Rebuild the full original by joining every chunk's source (each chunk
    # holds LOCAL placeholders + the global_indices needed to map them back),
    # then restoring tags once over the whole body.
    orig_joined = ''.join(globalize_chunks(original_chunks))
    # Escape stray < and > before tag restoration so literal angle brackets
    # in the text (LLM passthrough, Korean webnovel <Skill> markers, ...)
    # don't corrupt the XML when the body is reinjected.
    orig_full_html = tag_preserver.restore_tags(escape_stray_markup(orig_joined), global_tag_map)
    trans_full_html = tag_preserver.restore_tags(
        escape_stray_markup(''.join(translated_chunks)), global_tag_map
    )

    interleaved = interleave_bilingual(orig_full_html, trans_full_html)
    if interleaved is not None:
        return interleaved

    # Lossless fallback: if the two documents' container structures don't
    # align (e.g. a placeholder-recovery fallback reshaped the translation),
    # emit the full source then the full translation. The layout is coarser
    # but no text is dropped.
    return (
        f'<div class="bilingual-original" style="{_BILINGUAL_ORIGINAL_STYLE}">{orig_full_html}</div>'
        f'<div class="bilingual-translation" style="{_BILINGUAL_TRANSLATION_STYLE}">{trans_full_html}</div>'
    )


# XML predefines exactly five named entities; every other name (&nbsp;,
# &hellip;, ...) is undefined without a DTD and gets DROPPED by lxml's
# recover-mode parser at body reinjection, taking surrounding text with it.
_XML_PREDEFINED_ENTITIES = frozenset({'amp', 'lt', 'gt', 'quot', 'apos'})

# Matches '&' optionally followed by a well-formed reference body: a numeric
# character reference (decimal or hex) or a named entity, semicolon included.
# A bare '&' (group 1 is None) is not part of any reference.
_AMP_OR_ENTITY_RE = re.compile(r'&(#(?:[0-9]+|[xX][0-9a-fA-F]+);|[a-zA-Z][a-zA-Z0-9]*;)?')


def _is_valid_xml_codepoint(cp: int) -> bool:
    return (cp in (0x9, 0xA, 0xD)
            or 0x20 <= cp <= 0xD7FF
            or 0xE000 <= cp <= 0xFFFD
            or 0x10000 <= cp <= 0x10FFFF)


def _escape_stray_ampersands(text: str) -> str:
    """Neutralize every '&' that the XML parser would not accept (issue #202).

    etree.XMLParser(recover=True) silently DELETES malformed or undefined
    entity references together with adjacent text ('AT&T' -> 'AT'), so any
    '&' surviving to replace_body_content() must be part of a reference the
    parser understands:

    - the five predefined XML entities and valid numeric references: kept;
    - known HTML named entities (&nbsp;, &hellip;, ...): replaced by their
      literal character(s), since they are undefined in DTD-less XHTML;
    - everything else (bare '&', unknown names, out-of-range numerics):
      escaped to '&amp;' so the original text renders verbatim.
    """
    def _fix(match: re.Match) -> str:
        ref = match.group(1)
        if ref is None:
            return '&amp;'
        if ref.startswith('#'):
            try:
                cp = int(ref[2:-1], 16) if ref[1] in 'xX' else int(ref[1:-1])
            except ValueError:
                cp = -1
            return match.group(0) if _is_valid_xml_codepoint(cp) else '&amp;' + ref
        if ref[:-1] in _XML_PREDEFINED_ENTITIES:
            return match.group(0)
        decoded = _HTML5_ENTITIES.get(ref)
        if decoded is not None:
            return (decoded.replace('&', '&amp;')
                           .replace('<', '&lt;')
                           .replace('>', '&gt;'))
        return '&amp;' + ref

    return _AMP_OR_ENTITY_RE.sub(_fix, text)


def escape_stray_markup(text: str) -> str:
    """Escape stray markup characters so the reinjected body parses cleanly.

    Public: the refine passes (EPUB and DOCX) run it over refined text before
    restoring the tags, exactly like the translation reassembly does.

    Ampersands are sanitized first (see _escape_stray_ampersands), then every
    remaining raw < and > becomes an entity. Placeholders [idN] use square
    brackets, so they are untouched, and entities already present in the text
    (&lt;, &amp;, ...) survive because the ampersand pass recognizes them."""
    return _escape_stray_ampersands(text).replace('<', '&lt;').replace('>', '&gt;')


# Inline styles for bilingual output (kept inline so the EPUB renders without
# requiring an extra stylesheet).
_BILINGUAL_ORIGINAL_STYLE = "color: #666; font-size: 0.9em;"
_BILINGUAL_TRANSLATION_STYLE = "margin-bottom: 1em; padding-bottom: 0.5em; border-bottom: 1px dashed #ccc;"

# Block-level container tags: bilingual interleaving recurses INTO these and
# rebuilds them once, pairing their children. Everything else (p, h1-6, span…)
# is treated as a leaf "paragraph" unit emitted as source-then-translation.
# This keeps container structure (e.g. <div class="Section13"> wrapping a whole
# chapter) intact instead of duplicating or fragmenting it.
_BILINGUAL_CONTAINER_TAGS = frozenset({
    'body', 'div', 'section', 'article', 'aside', 'nav', 'header', 'footer',
    'main', 'blockquote', 'ul', 'ol', 'li', 'dl', 'dd', 'dt', 'table',
    'tbody', 'thead', 'tfoot', 'tr', 'td', 'th', 'figure', 'figcaption',
})


def _local_tag(el) -> Optional[str]:
    """Return an element's lowercase local tag name, or None for comments/PIs."""
    tag = el.tag
    if not isinstance(tag, str):
        return None
    return tag.split('}')[-1].lower()


def _element_children(el) -> List:
    """Real element children (skipping comments and processing instructions)."""
    return [c for c in el if isinstance(c.tag, str)]


def _has_translatable_text(el) -> bool:
    """True if the element subtree contains any non-whitespace text."""
    return bool("".join(el.itertext()).strip())


def _same_bilingual_structure(orig_el, trans_el) -> bool:
    """Verify two elements share the structure we rely on for interleaving.

    We only need the CONTAINER skeleton to match (same container tags, same
    number of children at each container level), because we pair children by
    index when recursing into containers. Leaf paragraphs may differ inside
    (the LLM can restructure inline tags), so we don't recurse into them.
    """
    if _local_tag(orig_el) in _BILINGUAL_CONTAINER_TAGS:
        oc, tc = _element_children(orig_el), _element_children(trans_el)
        if len(oc) != len(tc):
            return False
        return all(_same_bilingual_structure(o, t) for o, t in zip(oc, tc))
    return True


def _merge_bilingual(orig_el, trans_el, out_parent) -> None:
    """Append interleaved source/translation nodes for one element pair.

    Containers are rebuilt once (preserving tag, attributes and text) with
    their children interleaved. Leaf paragraphs are emitted as a styled
    source copy followed by a styled translation copy; blank/separator leaves
    (no translatable text) are emitted once to avoid duplicating empty lines.
    """
    if _local_tag(orig_el) in _BILINGUAL_CONTAINER_TAGS:
        # Copy-then-clear rather than SubElement + set(): the recovering
        # parser keeps attributes with undeclared prefixes (e.g. epub:type,
        # whose xmlns lives on the <html> root that is not part of these
        # fragments) under their literal colon names, which set() rejects.
        new = _copy.deepcopy(orig_el)
        del new[:]
        new.tail = None
        out_parent.append(new)
        orig_children = _element_children(orig_el)
        trans_children = _element_children(trans_el)
        for oc, tc in zip(orig_children, trans_children):
            before = len(new)
            _merge_bilingual(oc, tc, new)
            # Preserve inter-element whitespace/newlines from the source.
            if len(new) > before and oc.tail:
                new[-1].tail = oc.tail
        return

    # Leaf paragraph
    if _has_translatable_text(orig_el):
        od = etree.SubElement(out_parent, 'div')
        od.set('class', 'bilingual-original')
        od.set('style', _BILINGUAL_ORIGINAL_STYLE)
        oc = _copy.deepcopy(orig_el)
        oc.tail = None
        od.append(oc)

        td = etree.SubElement(out_parent, 'div')
        td.set('class', 'bilingual-translation')
        td.set('style', _BILINGUAL_TRANSLATION_STYLE)
        tc = _copy.deepcopy(trans_el)
        tc.tail = None
        td.append(tc)
    else:
        # Blank/separator block: keep the original once for spacing.
        c = _copy.deepcopy(orig_el)
        c.tail = None
        out_parent.append(c)


def interleave_bilingual(orig_html: str, trans_html: str) -> Optional[str]:
    """Interleave full source and translation HTML at the paragraph level.

    Both inputs are complete, well-formed reconstructions of the SAME body
    (identical tag skeleton, since tags come from the same placeholder map and
    the LLM only translates the text between them). Operating on the full
    documents — not per chunk — is essential: a chapter is often wrapped in a
    container element that spans many chunks, so a single chunk's restored
    fragment has unbalanced nesting and cannot be safely reparsed in isolation
    (this caused catastrophic truncation/data loss, discussion #199).

    Returns interleaved inner HTML, or None if the documents cannot be parsed
    or their container structures don't align (caller falls back to a coarse
    but lossless source-then-translation layout).
    """
    parser = etree.XMLParser(recover=True, huge_tree=True, remove_blank_text=False)
    try:
        o_root = etree.fromstring(f"<temp>{orig_html}</temp>".encode('utf-8'), parser)
        t_root = etree.fromstring(f"<temp>{trans_html}</temp>".encode('utf-8'), parser)
    except (etree.XMLSyntaxError, ValueError):
        return None
    if o_root is None or t_root is None:
        return None

    o_children = _element_children(o_root)
    t_children = _element_children(t_root)
    if len(o_children) != len(t_children):
        return None
    if not all(_same_bilingual_structure(o, t) for o, t in zip(o_children, t_children)):
        return None

    # Merge + serialize under the same guard: lxml validates names on element
    # construction and serialization, so quirks the recovering parser let
    # through can still raise here. Falling back (None) keeps the file
    # translated with the coarse lossless layout instead of failing it.
    try:
        out = etree.Element('temp')
        out.text = o_root.text
        for oc, tc in zip(o_children, t_children):
            before = len(out)
            _merge_bilingual(oc, tc, out)
            if len(out) > before and oc.tail:
                out[-1].tail = oc.tail

        parts = [out.text] if out.text else []
        for child in out:
            parts.append(etree.tostring(child, encoding='unicode', method='xml'))
        return "".join(parts)
    except (etree.XMLSyntaxError, ValueError):
        return None


def _warn_unreplaced_placeholders(new_html: str, log_callback: Optional[Callable]) -> None:
    """Report placeholders the tag restoration left in the HTML."""
    # Only the placeholder format the system actually uses is checked.
    from src.common.placeholder_format import PlaceholderFormat
    fmt = PlaceholderFormat.from_config()

    matches = re.findall(fmt.pattern, new_html)
    if matches:
        # Rebuild the full placeholder strings (the pattern captures the number).
        remaining_placeholders = [fmt.create(int(num)) for num in matches]
        report_error(log_callback, "unreplaced_placeholders_warning",
                     f"⚠️ WARNING: {len(remaining_placeholders)} unreplaced placeholders found in reconstructed HTML: {remaining_placeholders[:10]}")


def _log_html_preview(new_html: str, log_callback: Optional[Callable]) -> None:
    if log_callback:
        preview = new_html[:500] if len(new_html) > 500 else new_html
        log_callback("replace_body_html_preview", f"HTML preview: {preview}")


def replace_body(
    body_element: etree._Element,
    new_html: str,
    log_callback: Optional[Callable] = None
) -> bool:
    """Replace the body content with the reconstructed HTML.

    A parse failure leaves the body untouched and returns False; a rate
    limit propagates (auto-pause); any other error returns False too, except
    in DEBUG_MODE where it is re-raised to fail fast.

    Args:
        body_element: Body element to update
        new_html: New HTML content
        log_callback: The job's log callback

    Returns:
        True if successful, False otherwise
    """
    _warn_unreplaced_placeholders(new_html, log_callback)
    try:
        replace_body_content(body_element, new_html, log_callback=log_callback)
        return True
    except (XmlParsingError, BodyExtractionError) as e:
        # Expected XML/parsing errors - handled gracefully.
        report_error(log_callback, "replace_body_error", f"Failed to replace body content: {str(e)}")
        _log_html_preview(new_html, log_callback)
    except Exception as e:
        # Re-raise RateLimitError to trigger auto-pause
        from src.core.llm.exceptions import RateLimitError as _RLE
        if isinstance(e, _RLE):
            raise

        # Unexpected error - log the full traceback for debugging.
        import traceback
        report_error(log_callback, "replace_body_unexpected_error",
                     f"⚠️ UNEXPECTED ERROR in replace_body_content: {type(e).__name__}: {str(e)}")
        if log_callback:
            log_callback("replace_body_traceback", f"Full traceback:\n{traceback.format_exc()}")
        _log_html_preview(new_html, log_callback)

        # In debug mode, re-raise unexpected errors to fail fast
        from src.config import DEBUG_MODE
        if DEBUG_MODE:
            raise
    return False


def mark_file_untranslated(
    stats: TranslationMetrics,
    chunk_count: int,
    checkpoint_manager: Optional[Any],
    translation_id: Optional[str],
    file_href: Optional[str],
    log_callback: Optional[Callable] = None,
) -> None:
    """Record that every chunk of a file fell back to its source text.

    Used when the translated chunks could not be written back into the body
    (the body is left untouched, i.e. in the source language). The in-memory
    outcomes and the persisted partial state are both updated, so the EPUB file
    loop lists the file in `unfinished_units` and a later pass retries it.
    """
    for index in range(chunk_count):
        stats.record_chunk_outcome(index, CHUNK_UNTRANSLATED)

    if not (checkpoint_manager and translation_id and file_href):
        return
    try:
        state = checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
    except Exception:
        state = None
    if state is None or not state.chunk_statuses:
        return
    state.chunk_statuses = [
        status if status == CHUNK_PENDING else CHUNK_UNTRANSLATED
        for status in state.chunk_statuses
    ]
    try:
        checkpoint_manager.save_xhtml_partial_state(translation_id, file_href, state)
    except Exception as e:
        report_error(log_callback, "replace_body_state_error",
                     f"Could not record the failed body replacement in the checkpoint: {e}")
