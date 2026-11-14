"""Table-of-contents labels of a translated EPUB.

Body translation never touches the reader's side-panel TOC: EPUB2 keeps its
labels in the NCX (``navLabel/text``), EPUB3 in the navigation document
(``<nav epub:type="toc">``). These passes map every TOC target back to the
translated XHTML document and copy its heading text into the label, leaving
the link targets untouched.
"""

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import unquote

from lxml import etree

from src.config import NAMESPACES


def update_ncx_toc_labels_from_translated_docs(
    opf_dir: str,
    parsed_xhtml_docs: Dict[str, etree._Element],
    log_callback: Optional[Callable] = None
) -> Dict[str, int]:
    """
    Update EPUB2 NCX TOC labels using translated XHTML headings.

    The NCX side-panel TOC stores display labels separately from the XHTML body
    in ``navLabel/text`` nodes. Body translation does not touch those labels,
    so this helper maps each NCX ``content src`` target back to the already
    translated XHTML document and copies the translated heading text into the
    NCX label. The ``content src`` attribute is never modified, preserving
    reader navigation.
    """
    stats = {"updated": 0, "unchanged": 0, "errors": 0}
    opf_dir_path = Path(opf_dir)
    ncx_paths = list(opf_dir_path.glob("*.ncx"))
    if not ncx_paths:
        return stats

    docs_by_path = {
        os.path.normcase(os.path.abspath(path)): doc
        for path, doc in parsed_xhtml_docs.items()
    }
    ns = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}

    for ncx_path in ncx_paths:
        try:
            parser = etree.XMLParser(encoding="utf-8", recover=True, remove_blank_text=False)
            tree = etree.parse(str(ncx_path), parser)
            changed = False

            for nav_point in tree.findall(".//ncx:navPoint", namespaces=ns):
                text_el = nav_point.find("./ncx:navLabel/ncx:text", namespaces=ns)
                content_el = nav_point.find("./ncx:content", namespaces=ns)
                if text_el is None or content_el is None:
                    stats["unchanged"] += 1
                    continue

                src = content_el.get("src")
                if not src:
                    stats["unchanged"] += 1
                    continue

                translated_title = _get_translated_title_for_src(
                    src=src,
                    base_dir=opf_dir,
                    docs_by_path=docs_by_path
                )
                if not translated_title:
                    stats["unchanged"] += 1
                    continue

                if text_el.text != translated_title:
                    text_el.text = translated_title
                    changed = True
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1

            if changed:
                tree.write(
                    str(ncx_path),
                    encoding="utf-8",
                    xml_declaration=True,
                    pretty_print=True
                )
        except Exception as exc:
            stats["errors"] += 1
            if log_callback:
                log_callback("epub_ncx_toc_error", f"Could not update NCX TOC '{ncx_path}': {exc}")

    if log_callback and (stats["updated"] or stats["errors"]):
        log_callback(
            "epub_ncx_toc_updated",
            f"📚 NCX TOC labels updated: {stats['updated']} updated, "
            f"{stats['unchanged']} unchanged, {stats['errors']} errors"
        )

    return stats


def _get_translated_title_for_src(
    src: str,
    base_dir: str,
    docs_by_path: Dict[str, etree._Element]
) -> Optional[str]:
    """Resolve a TOC ``src``/``href`` to the translated heading text.

    ``base_dir`` is the directory the link is relative to (the NCX file's
    directory for EPUB2, the nav document's directory for EPUB3).
    """
    href, fragment = _split_ncx_src(src)
    if not href:
        return None

    file_path = os.path.normcase(os.path.abspath(os.path.join(base_dir, href)))
    doc_root = docs_by_path.get(file_path)
    if doc_root is None:
        return None

    if fragment:
        # A fragment names one place in the file: resolve it or keep the
        # original label. Falling back to the file's first heading would give
        # every section of a multi-section file the same label.
        anchor = _find_element_by_id_or_name(doc_root, fragment)
        if anchor is None:
            return None
        return _extract_heading_text_near_anchor(anchor)

    return _extract_first_heading_text(doc_root)


def _find_nav_doc_href(opf_tree: etree._ElementTree) -> Optional[str]:
    """Return the href of the EPUB3 navigation document, or None.

    The nav document is the manifest item carrying ``properties="nav"``.
    """
    opf_root = opf_tree.getroot()
    manifest = opf_root.find('.//opf:manifest', namespaces=NAMESPACES)
    if manifest is None:
        return None
    for item in manifest.findall('.//opf:item', namespaces=NAMESPACES):
        props = (item.get("properties") or "").split()
        if "nav" in props:
            return item.get("href")
    return None


def _set_anchor_text(anchor: etree._Element, title: str) -> None:
    """Replace an ``<a>`` element's visible text with ``title``.

    TOC links are normally plain text, but some carry inline markup (e.g. a
    numbering ``<span>``). Clearing children and setting ``text`` guarantees
    the link displays exactly the translated heading.
    """
    for child in list(anchor):
        anchor.remove(child)
    anchor.text = title


def update_nav_toc_labels_from_translated_docs(
    opf_dir: str,
    opf_tree: etree._ElementTree,
    parsed_xhtml_docs: Dict[str, etree._Element],
    log_callback: Optional[Callable] = None
) -> Dict[str, int]:
    """
    Update EPUB3 nav document TOC links using translated XHTML headings.

    EPUB3 readers build their table of contents from the navigation document
    (``<nav epub:type="toc">``) rather than the legacy NCX. The link labels in
    that document are stored separately from the chapter bodies, so body
    translation never touches them. This helper maps each TOC ``<a href>``
    target back to the already translated XHTML heading and copies the
    translated text into the link, leaving ``href`` untouched so navigation
    keeps working. Non-TOC navs (``landmarks``, ``page-list``) are skipped.
    """
    stats = {"updated": 0, "unchanged": 0, "errors": 0}

    nav_href = _find_nav_doc_href(opf_tree)
    if not nav_href:
        return stats

    nav_path = os.path.join(opf_dir, unquote(nav_href))
    if not os.path.exists(nav_path):
        return stats

    # nav links are relative to the nav document's own directory.
    nav_dir = os.path.dirname(nav_path)
    docs_by_path = {
        os.path.normcase(os.path.abspath(path)): doc
        for path, doc in parsed_xhtml_docs.items()
    }
    epub_type_attr = f"{{{NAMESPACES['epub']}}}type"
    skip_types = {"landmarks", "page-list"}

    try:
        parser = etree.XMLParser(encoding="utf-8", recover=True, remove_blank_text=False)
        tree = etree.parse(str(nav_path), parser)
        changed = False

        for nav_el in tree.iter():
            if _local_name(nav_el) != "nav":
                continue
            if (nav_el.get(epub_type_attr) or "").strip() in skip_types:
                continue

            for anchor in nav_el.iter():
                if _local_name(anchor) != "a":
                    continue
                src = anchor.get("href")
                if not src:
                    continue

                translated_title = _get_translated_title_for_src(
                    src=src,
                    base_dir=nav_dir,
                    docs_by_path=docs_by_path
                )
                if not translated_title:
                    stats["unchanged"] += 1
                    continue

                if _normalized_element_text(anchor) != translated_title:
                    _set_anchor_text(anchor, translated_title)
                    changed = True
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1

        if changed:
            tree.write(
                str(nav_path),
                encoding="utf-8",
                xml_declaration=True,
                pretty_print=False
            )
    except Exception as exc:
        stats["errors"] += 1
        if log_callback:
            log_callback("epub_nav_toc_error", f"Could not update nav TOC '{nav_path}': {exc}")

    if log_callback and (stats["updated"] or stats["errors"]):
        log_callback(
            "epub_nav_toc_updated",
            f"📚 EPUB3 nav TOC labels updated: {stats['updated']} updated, "
            f"{stats['unchanged']} unchanged, {stats['errors']} errors"
        )

    return stats


def _split_ncx_src(src: str) -> Tuple[str, Optional[str]]:
    href, _, fragment = src.partition("#")
    return unquote(href), unquote(fragment) if fragment else None


def _find_element_by_id_or_name(doc_root: etree._Element, fragment: str) -> Optional[etree._Element]:
    for element in doc_root.iter():
        if element.get("id") == fragment or element.get("name") == fragment:
            return element
    return None


_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


def _extract_heading_text_near_anchor(anchor: etree._Element) -> Optional[str]:
    """Heading text a TOC fragment points at, or None when unsure.

    Resolution order:
    1. the anchor is a heading, or sits inside one: that heading;
    2. the anchor wraps a heading (``<section id="ch2"><h2>``): the first
       heading it contains;
    3. the anchor holds no text (``<a id="ch2"/><h2>``): the next heading in
       document order, provided no other text comes before it;
    otherwise None, so the caller keeps the original label instead of using a
    whole section's text or an unrelated heading.
    """
    current = anchor
    while current is not None:
        if _local_name(current) in _HEADING_TAGS:
            return _normalized_element_text(current)
        current = current.getparent()

    for element in anchor.iterdescendants():
        if _local_name(element) in _HEADING_TAGS:
            return _normalized_element_text(element)

    if _normalized_element_text(anchor):
        return None

    return _next_heading_text_after(anchor)


def _document_events(element: etree._Element):
    """Yield ("start", el) and ("text", str) events in document order."""
    if not isinstance(element.tag, str):
        return
    yield ("start", element)
    if element.text:
        yield ("text", element.text)
    for child in element:
        yield from _document_events(child)
        if child.tail:
            yield ("text", child.tail)


def _next_heading_text_after(anchor: etree._Element) -> Optional[str]:
    """Text of the first heading after an empty anchor, if no text precedes it."""
    root = anchor.getroottree().getroot()
    seen_anchor = False
    for kind, value in _document_events(root):
        if not seen_anchor:
            seen_anchor = kind == "start" and value is anchor
            continue
        if kind == "text":
            if value.strip():
                return None
        elif _local_name(value) in _HEADING_TAGS:
            return _normalized_element_text(value)
    return None


def _extract_first_heading_text(doc_root: etree._Element) -> Optional[str]:
    for element in doc_root.iter():
        if _local_name(element) in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            title = _normalized_element_text(element)
            if title:
                return title
    return None


def _normalized_element_text(element: etree._Element) -> Optional[str]:
    text = " ".join("".join(element.itertext()).split())
    return text or None


def _local_name(element: etree._Element) -> str:
    try:
        return etree.QName(element).localname.lower()
    except ValueError:
        return str(element.tag).split("}", 1)[-1].lower()
