"""
Plain-text extraction and rebuild for Plain Text Mode (EPUB).

Walks an XHTML <body> in DOM order, collecting block-level paragraphs as plain
strings and anchoring any <img> they contain to their parent paragraph index.
The LLM never sees inline markup or images — only the textual content of each
block. The one exception is <ruby>, whose reading would otherwise be glued to
its base: it is folded into base（reading） first (see ruby_annotations).

At rebuild time, the body is wiped and reconstructed as a flat sequence of
block elements (<p>, <h1..h6>, <li>, <blockquote>, <pre>) plus, after each
block that originally contained images, an extra <p class="plain-text-images"> wrapper
with the original <img> elements unchanged. Each block element is rebuilt with
its original tag *and its original attributes* (class, id, xml:lang, epub:type,
title, ...) carried across, so semantic attributes survive translation even
though the tag's inline children are flattened to text. Void blocks (<hr>)
carry no text and no images: they keep their own slot in the paragraph list and
are re-emitted as a bare element at the same position, without ever reaching
the LLM. The one case where the body is left alone is a page that yielded no
block at all (everything inside a DROP_TAGS subtree, e.g. an SVG-wrapped
cover): its source markup is kept verbatim.
"""
from typing import Dict, List, Tuple

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


def _local_name(elem: etree._Element) -> str:
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
    out: List[str] = []

    def walk(node: etree._Element, include_tail: bool):
        name = _local_name(node)
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

    if elem.text:
        out.append(elem.text)
    for child in elem:
        walk(child, include_tail=True)

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


def _set_attributes(elem: etree._Element, attrib: Dict[str, str]) -> None:
    """Copy recorded source attributes (including namespaced epub:type, xml:lang) onto a rebuilt element."""
    for k, v in (attrib or {}).items():
        elem.set(k, v)


def _add_class(elem: etree._Element, cls: str) -> None:
    """Append a plain-text marker class to an element, preserving any source class."""
    existing = elem.get("class")
    elem.set("class", f"{existing} {cls}".strip() if existing else cls)


def _enclosing_table(elem: etree._Element) -> etree._Element:
    """Return the nearest <table> ancestor, or None."""
    parent = elem.getparent()
    while parent is not None:
        if _local_name(parent) == "table":
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
        if _local_name(elem) not in TABLE_CELL_TAGS:
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


def _collect_blocks(
    root: etree._Element,
    paragraphs_text: List[str],
    paragraphs_tag: List[str],
    paragraphs_attrib: List[Dict[str, str]],
    images_by_paragraph: Dict[int, List[etree._Element]],
) -> None:
    """
    DOM-walk a container, emitting one entry per block-level element found.

    For lists, we descend into <li> items individually (each is its own block).
    For containers (div, section, ...), we recurse.
    """
    for child in root:
        name = _local_name(child)

        if name in DROP_TAGS:
            continue

        if name in VOID_BLOCK_TAGS:
            paragraphs_text.append("")
            paragraphs_tag.append(name)
            paragraphs_attrib.append(dict(child.attrib))
            continue

        if name == "table":
            _collect_table_blocks(
                child, paragraphs_text, paragraphs_tag, paragraphs_attrib, images_by_paragraph
            )
            continue

        if name in LIST_WRAPPER_TAGS:
            _collect_blocks(child, paragraphs_text, paragraphs_tag, paragraphs_attrib, images_by_paragraph)
            continue

        if name in CONTAINER_TAGS:
            _collect_blocks(child, paragraphs_text, paragraphs_tag, paragraphs_attrib, images_by_paragraph)
            continue

        if name in BLOCK_TAGS:
            images: List[etree._Element] = []
            if name == "pre":
                # Preserve code/pre verbatim — but skip <img> inside (rare)
                text = "".join(child.itertext())
            else:
                text = _extract_text_keep_inline(child, images)

            idx = len(paragraphs_text)
            paragraphs_text.append(text)
            paragraphs_tag.append(name)
            paragraphs_attrib.append(dict(child.attrib))
            if images:
                images_by_paragraph[idx] = images
            continue

        if name == "img":
            # Standalone <img> at body level — anchor to the previous block,
            # or create a synthetic anchor if it's first.
            img_copy = _clone_img(child)
            if paragraphs_text:
                anchor = len(paragraphs_text) - 1
                images_by_paragraph.setdefault(anchor, []).append(img_copy)
            else:
                paragraphs_text.append("")
                paragraphs_tag.append("p")
                paragraphs_attrib.append({})
                images_by_paragraph[0] = [img_copy]
            continue

        # Anything else: try to extract textual content as a generic paragraph
        images: List[etree._Element] = []
        text = _extract_text_keep_inline(child, images)
        if text.strip() or images:
            idx = len(paragraphs_text)
            paragraphs_text.append(text)
            paragraphs_tag.append("p")
            paragraphs_attrib.append(dict(child.attrib))
            if images:
                images_by_paragraph[idx] = images


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
                   keeps its source attributes (class, id, xml:lang, epub:type,
                   ...); marker classes are appended to, never replacing, a
                   source class.
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
            continue

        # Bilingual: emit source first when we have it
        source_emitted = False
        if bilingual and source_text:
            src_block = etree.SubElement(body_element, tag)
            _set_attributes(src_block, attrib)
            _add_class(src_block, "plain-text-source")
            src_block.text = source_text
            source_emitted = True

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
            emit_target = not images_by_paragraph.get(i)

        if emit_target:
            block = etree.SubElement(body_element, tag)
            _set_attributes(block, attrib)
            if untranslated:
                # One marker per block, untranslated over bilingual-target when
                # both apply; either is appended to the source class, which the
                # _set_attributes call above already restored.
                _add_class(block, "plain-text-untranslated")
            elif bilingual:
                _add_class(block, "plain-text-target")
            block.text = text or source_text

        # Emit anchored images right after
        if i in images_by_paragraph and images_by_paragraph[i]:
            img_wrapper = etree.SubElement(body_element, "p")
            img_wrapper.set("class", "plain-text-images")
            for img in images_by_paragraph[i]:
                img_wrapper.append(img)
