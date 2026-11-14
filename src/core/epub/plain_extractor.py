"""
Plain-text extraction and rebuild for Plain Text Mode (EPUB).

Walks an XHTML <body> in DOM order, collecting block-level paragraphs as plain
strings and anchoring any <img> they contain to their parent paragraph index.
The LLM never sees inline markup or images — only the textual content of each
block. The one exception is <ruby>, whose reading would otherwise be glued to
its base: it is folded into base（reading） first (see ruby_annotations).

At rebuild time, the body is wiped and reconstructed as a flat sequence of
block elements (<p>, <h1..h6>, <li>, <blockquote>, <pre>, and a container
such as <div> that held its text directly, see _collect_blocks) plus, after each
block that originally contained images, an extra <p class="plain-text-images"> wrapper
with the original <img> elements unchanged. Each block element is rebuilt with
its original tag *and the source attributes that still apply to a flattened
block* (class, id, style, dir, title, epub:type), so semantic attributes
survive translation even though the tag's inline children are flattened to
text — see CARRIED_ATTRIBUTES for what is copied and why. Void blocks (<hr>)
carry no text and no images: they keep their own slot in the paragraph list and
are re-emitted as a bare element at the same position, without ever reaching
the LLM. The one case where the body is left alone is a page that yielded no
block at all (everything inside a DROP_TAGS subtree, e.g. an SVG-wrapped
cover): its source markup is kept verbatim.
"""
from typing import Dict, List, Optional, Tuple

from lxml import etree

from .ruby_annotations import fold_ruby_annotations


# Block-level tags we preserve at rebuild time (li flattens to p later — see replace_body_with_paragraphs).
BLOCK_TAGS = ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre")
# Block-level tags with no text of their own. They are collected so they keep a
# slot in the paragraph list and are re-emitted verbatim at rebuild time; their
# empty text means build_plain_segments never sends them to the LLM.
VOID_BLOCK_TAGS = ("hr",)
# Containers we descend into looking for blocks
CONTAINER_TAGS = ("div", "section", "article", "main", "header", "footer", "aside", "nav")
# Subtrees never sent to the LLM in Plain Text Mode. Tables, figures and
# pictures are NOT dropped: their text is extracted and their <img> elements
# anchored, otherwise that content would be silently deleted from the output.
DROP_TAGS = ("svg", "video", "audio", "iframe", "form", "script", "style")
# Elements whose text would glue to the next sibling's without an explicit
# separator once tags are stripped (adjacent table cells, rows, caption).
SPACED_TAGS = ("td", "th", "tr", "caption")
# Table cell-level elements emitted as individual paragraphs
TABLE_CELL_TAGS = ("td", "th", "caption")
# List wrappers we descend into (the inner <li> items become individual blocks)
LIST_WRAPPER_TAGS = ("ul", "ol")
# Phrasing-level tags that belong to the same line of text as the loose text
# around them. Inside a container, a run of these plus loose text is gathered
# into a single paragraph (see _collect_blocks). <img> and DROP_TAGS are
# handled as part of a run too; any tag not listed anywhere is treated as a
# block of its own, as before.
INLINE_TAGS = (
    "a", "abbr", "b", "bdi", "bdo", "big", "br", "cite", "code", "data", "del",
    "dfn", "em", "font", "i", "ins", "kbd", "label", "mark", "q", "rb", "rp",
    "rt", "rtc", "ruby", "s", "samp", "small", "span", "strike", "strong",
    "sub", "sup", "time", "tt", "u", "var", "wbr",
)
EPUB_TYPE_ATTR = "{http://www.idpf.org/2007/ops}type"
# Source attributes copied onto a rebuilt block. The list is a whitelist rather
# than "everything the source block had" for two reasons:
#   * Plain Text Mode flattens <li>, <td>, <th> and unknown blocks into <p>, so
#     a tag-specific attribute (colspan, rowspan, scope, headers, value) would
#     land on an element that cannot legally carry it and epubcheck rejects it.
#   * `lang` / `xml:lang` describe the SOURCE language. Copied onto translated
#     text they would override the target language that lang_support writes on
#     <html>, and a reading system would hand a French paragraph to an English
#     speech engine. Dropping them lets the block inherit the correct root lang.
CARRIED_ATTRIBUTES = ("class", "id", "style", "dir", "title", EPUB_TYPE_ATTR)


def local_name(elem: etree._Element) -> str:
    """Return the lowercase local tag name, stripping XHTML namespace."""
    tag = elem.tag
    if isinstance(tag, str) and tag.startswith("{"):
        tag = tag.split("}", 1)[1]
    return tag.lower() if isinstance(tag, str) else ""


def _extract_text_keep_inline(elem: etree._Element, image_sink: List[etree._Element]) -> str:
    """
    Flatten an element's textual content, ignoring inline tags.

    Adds any <img> encountered to image_sink (preserves DOM order).
    Returns whitespace-normalized text.
    """
    return _extract_run_text(elem.text, list(elem), image_sink)


def _extract_run_text(
    lead_text: str, nodes: List[etree._Element], image_sink: List[etree._Element]
) -> str:
    """
    Flatten a run of sibling nodes, preceded by lead_text, into one string.

    Each node contributes its own content and its tail. This is the body of
    _extract_text_keep_inline, split out so a container can flatten a slice of
    its children (a run of inline content between two blocks) and not only an
    element as a whole.
    """
    out: List[str] = []

    def walk(node: etree._Element, include_tail: bool):
        if not isinstance(node.tag, str):
            # Comment or processing instruction: its .text is markup, never
            # book text, and must not reach the LLM. Its tail is book text.
            if include_tail and node.tail:
                out.append(node.tail)
            return
        name = local_name(node)
        if name in DROP_TAGS:
            # Skip subtree entirely. Still pick up its tail since it sits at
            # the parent's level.
            if include_tail and node.tail:
                out.append(node.tail)
            return
        if name == "img":
            image_sink.append(_clone_img(node))
            if include_tail and node.tail:
                out.append(node.tail)
            return
        if name == "br":
            out.append(" ")
            if include_tail and node.tail:
                out.append(node.tail)
            return
        if node.text:
            out.append(node.text)
        for child in node:
            walk(child, include_tail=True)
        if name in SPACED_TAGS:
            out.append(" ")
        if include_tail and node.tail:
            out.append(node.tail)

    if lead_text:
        out.append(lead_text)
    for node in nodes:
        walk(node, include_tail=True)

    text = "".join(out)
    return " ".join(text.split())


def _clone_img(img: etree._Element) -> etree._Element:
    """Create a standalone copy of an <img> with its attributes, no namespace."""
    new = etree.Element("img")
    for k, v in img.attrib.items():
        if isinstance(k, str) and k.startswith("{"):
            k = k.split("}", 1)[1]
        new.set(k, v)
    return new


def _set_attributes(
    elem: etree._Element, attrib: Dict[str, str], with_id: bool = True
) -> None:
    """Copy the carried source attributes (see CARRIED_ATTRIBUTES) onto a rebuilt element.

    with_id=False leaves `id` behind: bilingual mode emits two elements for one
    source block, and duplicating the id there would produce invalid XHTML and
    an ambiguous target for every TOC anchor pointing at it.
    """
    for key in CARRIED_ATTRIBUTES:
        if key == "id" and not with_id:
            continue
        value = (attrib or {}).get(key)
        if value is not None:
            elem.set(key, value)


def _add_class(elem: etree._Element, cls: str) -> None:
    """Append a plain-text marker class to an element, preserving any source class."""
    existing = elem.get("class")
    elem.set("class", f"{existing} {cls}".strip() if existing else cls)


def _enclosing_table(elem: etree._Element) -> etree._Element:
    """Return the nearest <table> ancestor, or None."""
    parent = elem.getparent()
    while parent is not None:
        if local_name(parent) == "table":
            return parent
        parent = parent.getparent()
    return None


def _collect_table_blocks(
    table: etree._Element,
    paragraphs_text: List[str],
    paragraphs_tag: List[str],
    paragraphs_attrib: List[Dict[str, str]],
    images_by_paragraph: Dict[int, List[etree._Element]],
) -> None:
    """
    Emit each cell (td/th) and the caption of a table as its own <p> block.

    Plain Text Mode cannot represent tabular layout, but the cell text must
    survive translation instead of being deleted. Cells of nested tables are
    flattened into their outer cell's text rather than emitted twice.
    """
    for elem in table.iter():
        if local_name(elem) not in TABLE_CELL_TAGS:
            continue
        if _enclosing_table(elem) is not table:
            continue

        images: List[etree._Element] = []
        text = _extract_text_keep_inline(elem, images)
        if text.strip() or images:
            idx = len(paragraphs_text)
            paragraphs_text.append(text)
            paragraphs_tag.append("p")
            paragraphs_attrib.append(dict(elem.attrib))
            if images:
                images_by_paragraph[idx] = images


def _is_inline(node: etree._Element) -> bool:
    """True for a node that sits on the same line as the loose text around it."""
    if not isinstance(node.tag, str):
        # Comments and processing instructions: invisible, but their tail is
        # loose text of the enclosing container.
        return True
    name = local_name(node)
    return name in INLINE_TAGS or name in DROP_TAGS or name == "img"


def _has_loose_text(lead_text: str, nodes: List[etree._Element]) -> bool:
    """True when a run carries text that belongs to no element of its own."""
    if lead_text and lead_text.strip():
        return True
    return any(node.tail and node.tail.strip() for node in nodes)


def _collect_blocks(
    root: etree._Element,
    paragraphs_text: List[str],
    paragraphs_tag: List[str],
    paragraphs_attrib: List[Dict[str, str]],
    images_by_paragraph: Dict[int, List[etree._Element]],
    is_container: bool = False,
) -> None:
    """
    DOM-walk a container, emitting one entry per block-level element found.

    For lists, we descend into <li> items individually (each is its own block).
    For containers (div, section, ...), we recurse.

    Text can also sit loose in the container itself: before its first child,
    or in the tail of a child (issue #288, calibre's one-<div>-per-paragraph
    output being the common case). The children are therefore split into runs
    of inline content separated by block children. A run that carries loose
    text becomes one paragraph, with its inline elements flattened into it; a
    run without any is dispatched element by element exactly as before, so a
    body that lost no text extracts the same paragraphs as it always did.

    is_container: root is one of CONTAINER_TAGS. When such a container holds no
    block child at all, it IS the paragraph, so the rebuilt block keeps its tag
    and its attributes (a <div class="calibre8"> stays one). Loose text next to
    block children, or directly in <body> or a list wrapper, becomes an
    anonymous <p>: those tags cannot stand in for the paragraph.
    """
    segments: List[tuple] = []
    run_lead, run_nodes = root.text, []
    for child in root:
        if _is_inline(child):
            run_nodes.append(child)
            continue
        segments.append(("run", run_lead, run_nodes))
        segments.append(("block", child))
        run_lead, run_nodes = child.tail, []
    segments.append(("run", run_lead, run_nodes))
    only_inline = len(segments) == 1

    for segment in segments:
        if segment[0] == "block":
            _collect_block(
                segment[1], paragraphs_text, paragraphs_tag, paragraphs_attrib, images_by_paragraph
            )
            continue

        _, lead_text, nodes = segment
        if not _has_loose_text(lead_text, nodes):
            for node in nodes:
                _collect_block(
                    node, paragraphs_text, paragraphs_tag, paragraphs_attrib, images_by_paragraph
                )
            continue

        images: List[etree._Element] = []
        text = _extract_run_text(lead_text, nodes, images)
        idx = len(paragraphs_text)
        paragraphs_text.append(text)
        if only_inline and is_container:
            paragraphs_tag.append(local_name(root))
            paragraphs_attrib.append(dict(root.attrib))
        else:
            paragraphs_tag.append("p")
            paragraphs_attrib.append({})
        if images:
            images_by_paragraph[idx] = images


def _append_paragraph(
    paragraphs_text: List[str],
    paragraphs_tag: List[str],
    paragraphs_attrib: List[Dict[str, str]],
    images_by_paragraph: Dict[int, List[etree._Element]],
    text: str,
    tag: str,
    attrib: Dict[str, str],
    images: Optional[List[etree._Element]] = None,
) -> None:
    """Append one paragraph entry to the parallel lists, anchoring its images."""
    idx = len(paragraphs_text)
    paragraphs_text.append(text)
    paragraphs_tag.append(tag)
    paragraphs_attrib.append(attrib)
    if images:
        images_by_paragraph[idx] = images


def _anchor_standalone_image(
    child: etree._Element,
    paragraphs_text: List[str],
    paragraphs_tag: List[str],
    paragraphs_attrib: List[Dict[str, str]],
    images_by_paragraph: Dict[int, List[etree._Element]],
) -> None:
    """Anchor a body-level <img> to the previous block, or to a synthetic one."""
    img_copy = _clone_img(child)
    if paragraphs_text:
        anchor = len(paragraphs_text) - 1
        images_by_paragraph.setdefault(anchor, []).append(img_copy)
    else:
        paragraphs_text.append("")
        paragraphs_tag.append("p")
        paragraphs_attrib.append({})
        images_by_paragraph[0] = [img_copy]


def _collect_block(
    child: etree._Element,
    paragraphs_text: List[str],
    paragraphs_tag: List[str],
    paragraphs_attrib: List[Dict[str, str]],
    images_by_paragraph: Dict[int, List[etree._Element]],
) -> None:
    """Emit the entries for one child of a container (see _collect_blocks)."""
    if not isinstance(child.tag, str):
        # Comment or processing instruction: nothing to translate.
        return

    name = local_name(child)
    lists = (paragraphs_text, paragraphs_tag, paragraphs_attrib, images_by_paragraph)

    if name in DROP_TAGS:
        return

    if name in VOID_BLOCK_TAGS:
        _append_paragraph(*lists, "", name, dict(child.attrib))
        return

    if name == "table":
        _collect_table_blocks(child, *lists)
        return

    if name in LIST_WRAPPER_TAGS:
        _collect_blocks(child, *lists)
        return

    if name in CONTAINER_TAGS:
        _collect_blocks(child, *lists, is_container=True)
        return

    if name in BLOCK_TAGS:
        images: List[etree._Element] = []
        if name == "pre":
            # Preserve code/pre verbatim — but skip <img> inside (rare)
            text = "".join(child.itertext())
        else:
            text = _extract_text_keep_inline(child, images)
        _append_paragraph(*lists, text, name, dict(child.attrib), images)
        return

    if name == "img":
        # Standalone <img> at body level — anchor to the previous block,
        # or create a synthetic anchor if it's first.
        _anchor_standalone_image(child, *lists)
        return

    # Anything else: try to extract textual content as a generic paragraph
    images: List[etree._Element] = []
    text = _extract_text_keep_inline(child, images)
    if text.strip() or images:
        _append_paragraph(*lists, text, "p", dict(child.attrib), images)


def extract_plain_paragraphs(
    body_element: etree._Element,
) -> Tuple[List[str], List[str], Dict[int, List[etree._Element]], List[Dict[str, str]]]:
    """
    Extract the body as a flat list of (text, tag) pairs plus an image anchor map.

    Args:
        body_element: <body> element from a parsed XHTML doc.

    Returns:
        paragraphs_text:        list of plain-text strings, one per block
        paragraphs_tag:         parallel list of tag names ("p", "h1", "li", ...)
        images_by_paragraph:    {paragraph_index: [<img> elements]}
        paragraphs_attrib:      parallel list of attribute dicts, one per block
    """
    paragraphs_text: List[str] = []
    paragraphs_tag: List[str] = []
    paragraphs_attrib: List[Dict[str, str]] = []
    images_by_paragraph: Dict[int, List[etree._Element]] = {}

    if body_element is None:
        return paragraphs_text, paragraphs_tag, images_by_paragraph, paragraphs_attrib

    # Fold <ruby> annotations first, otherwise flattening would glue the base
    # and its reading into a word that does not exist (issue #242).
    fold_ruby_annotations(body_element)

    _collect_blocks(
        body_element, paragraphs_text, paragraphs_tag, paragraphs_attrib, images_by_paragraph
    )
    return paragraphs_text, paragraphs_tag, images_by_paragraph, paragraphs_attrib


def _target_emission(
    text: str,
    source_text: str,
    source_emitted: bool,
    has_images: bool,
) -> Tuple[bool, bool]:
    """Decide whether the target block is emitted and whether it is untranslated.

    Returns ``(emit_target, untranslated)``.
    """
    # A block is flagged untranslated whenever its output text IS its source
    # text. Keying on text identity rather than on the empty-slot branch
    # below is what keeps the marker working now that the plain-text
    # pipeline substitutes source text itself when a paragraph-level repair
    # fails (issue #253); it also finally marks whole chunks that fell back
    # to source, which were invisible before. A paragraph whose correct
    # translation happens to equal its source (a bare name, a numeral) is
    # flagged too — a cosmetic class on a correct block is a cheaper price
    # than losing the marker on the blocks that matter.
    untranslated = bool(source_text) and text == source_text

    # An empty translation must never delete the block.
    if text:
        emit_target = True
    elif source_emitted:
        # Bilingual: the plain-text-source block above already carries this
        # text, so falling back to it here would print the source twice.
        emit_target = False
    elif source_text:
        # Untranslated but not deleted: the paragraph survives carrying its
        # source text, and the class below is what actually makes that
        # failure findable in the output (issue #253).
        emit_target = True
        untranslated = True
    else:
        # Nothing to say at all. Keep an empty block so the source's spacer
        # <p></p> survives and the output block count matches the input —
        # unless images are anchored here, in which case the images wrapper
        # below already stands in for the block (a source <p><img/></p> must
        # come out as one <p>, not two).
        emit_target = not has_images

    return emit_target, untranslated


def _emit_target_block(
    body_element: etree._Element,
    tag: str,
    attrib: Dict[str, str],
    text: str,
    source_text: str,
    source_emitted: bool,
    untranslated: bool,
    bilingual: bool,
) -> None:
    """Append the translated (or source-fallback) block with its marker class."""
    block = etree.SubElement(body_element, tag)
    # The bilingual source twin, when there is one, already claimed the
    # source id: it comes first in reading order, so an anchor aimed at
    # this block still lands at the top of the right section.
    _set_attributes(block, attrib, with_id=not source_emitted)
    if untranslated:
        # One marker per block, untranslated over bilingual-target when
        # both apply; either is appended to the source class, which the
        # _set_attributes call above already restored.
        _add_class(block, "plain-text-untranslated")
    elif bilingual:
        _add_class(block, "plain-text-target")
    block.text = text or source_text


def _emit_paragraph(
    body_element: etree._Element,
    i: int,
    translated_paragraphs: List[str],
    paragraphs_tag: List[str],
    images_by_paragraph: Dict[int, List[etree._Element]],
    bilingual: bool,
    source_paragraphs: Optional[List[str]],
    paragraphs_attrib: Optional[List[Dict[str, str]]],
) -> None:
    """Append everything paragraph `i` contributes to the rebuilt body."""
    text = (translated_paragraphs[i] or "").strip()
    source_text = ""
    if source_paragraphs and i < len(source_paragraphs):
        source_text = (source_paragraphs[i] or "").strip()
    raw_tag = paragraphs_tag[i] if i < len(paragraphs_tag) else "p"
    # <li> outside <ul>/<ol> is not valid XHTML — flatten to <p> in Plain Text Mode.
    tag = "p" if raw_tag == "li" else raw_tag
    attrib = paragraphs_attrib[i] if paragraphs_attrib and i < len(paragraphs_attrib) else {}

    if raw_tag in VOID_BLOCK_TAGS:
        block = etree.SubElement(body_element, raw_tag)
        _set_attributes(block, attrib)
        return

    # Bilingual: emit source first when we have it
    source_emitted = False
    if bilingual and source_text:
        src_block = etree.SubElement(body_element, tag)
        _set_attributes(src_block, attrib)
        _add_class(src_block, "plain-text-source")
        src_block.text = source_text
        source_emitted = True

    emit_target, untranslated = _target_emission(
        text, source_text, source_emitted, bool(images_by_paragraph.get(i)))

    if emit_target:
        _emit_target_block(body_element, tag, attrib, text, source_text,
                           source_emitted, untranslated, bilingual)

    # Emit anchored images right after
    if i in images_by_paragraph and images_by_paragraph[i]:
        img_wrapper = etree.SubElement(body_element, "p")
        img_wrapper.set("class", "plain-text-images")
        for img in images_by_paragraph[i]:
            img_wrapper.append(img)


def replace_body_with_paragraphs(
    body_element: etree._Element,
    translated_paragraphs: List[str],
    paragraphs_tag: List[str],
    images_by_paragraph: Dict[int, List[etree._Element]],
    bilingual: bool = False,
    source_paragraphs: List[str] = None,
    paragraphs_attrib: List[Dict[str, str]] = None,
) -> None:
    """
    Wipe body_element and refill it from the translated paragraphs.

    Args:
        body_element: target <body> to overwrite
        translated_paragraphs: same length as paragraphs_tag
        paragraphs_tag: tag name per paragraph ("p", "h1", "li", ...)
        images_by_paragraph: anchored images per paragraph index
        bilingual: when True, emit a <p class="src"> with the source text
                   right before each translated block.
        source_paragraphs: source text per paragraph. Required when bilingual
                   is True; otherwise optional but strongly recommended — it is
                   what an empty translation falls back to instead of the block
                   being dropped. Legacy callers that omit it still get an
                   empty block rather than a deletion.
        paragraphs_attrib: attribute dict per paragraph, captured from the source
                   block at extraction time. When provided, each rebuilt block
                   keeps the subset of them that still applies once the block is
                   flattened (CARRIED_ATTRIBUTES); marker classes are appended
                   to, never replacing, a source class, and `id` is emitted once
                   per source block even in bilingual mode.
    """
    count = len(translated_paragraphs)

    # Never replace a populated body with nothing. Every iteration below emits
    # at least one element (the block itself, its bilingual source twin, or its
    # anchored-images wrapper), so the rebuild can only come out empty when
    # extraction genuinely found no block at all — which is what happens to a
    # page whose whole content sits inside a DROP_TAGS subtree, the calibre-style
    # <div><svg><image/></svg></div> cover being the canonical case. There is
    # nothing to translate on such a page, so keeping its source markup verbatim
    # is exactly right; deleting it never is. This is only a backstop: the
    # empty-translation fallback below is what keeps individual blocks alive.
    if count == 0 and (len(body_element) or (body_element.text or "").strip()):
        return

    # Clear body
    body_element.text = None
    for child in list(body_element):
        body_element.remove(child)

    for i in range(count):
        _emit_paragraph(body_element, i, translated_paragraphs, paragraphs_tag,
                        images_by_paragraph, bilingual, source_paragraphs,
                        paragraphs_attrib)
