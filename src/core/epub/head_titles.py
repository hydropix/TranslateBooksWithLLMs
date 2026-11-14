"""``<head><title>`` of translated XHTML documents.

Only the ``<body>`` of a document is sent to the LLM, so its ``<title>`` (what
some readers show in their header, and what TTS and accessibility tools
announce) stays in the source language. This pass localizes it without an
extra LLM call, reusing text that is already translated, but only when the
original title is provably the same text:

1. the original title equals the original first heading (h1..h6) of the body:
   it becomes the translated first heading;
2. otherwise, it equals the original label of a TOC entry pointing at the
   file: it becomes that entry's translated label, i.e. exactly what the TOC
   label passes (``toc_labels``) write into the NCX / nav document.

Equality is exact after whitespace normalization (runs of whitespace collapse
to one space, leading/trailing whitespace is ignored); case is significant.
Any other title is left unchanged.

The source side has to be read before anything overwrites the extracted
files: a resumed job restores its finished documents from the checkpoint over
the extracted ones, so :func:`capture_source_titles` runs right after the
manifest is parsed, and :func:`localize_head_titles` runs on the translated
trees just before they are written back.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote

from lxml import etree

from src.config import NAMESPACES
from src.utils.security import is_within_directory

from .toc_labels import (
    _extract_first_heading_text,
    _find_nav_doc_href,
    _get_translated_title_for_src,
    _local_name,
    _normalized_element_text,
    _split_ncx_src,
)

_NCX_NS = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}
_SKIPPED_NAV_TYPES = frozenset({"landmarks", "page-list"})


@dataclass(frozen=True)
class TocEntry:
    """One TOC link: its target, the directory it is relative to, its label."""
    src: str
    base_dir: str
    label: str


@dataclass
class SourceTitles:
    """What the source book says about each document's title.

    ``documents`` maps a document key (see :func:`_doc_key`) to its original
    ``(title, first heading)``, both whitespace-normalized (None when absent).
    ``toc`` maps a document key to the TOC entries that point at it.
    """
    documents: Dict[str, Tuple[Optional[str], Optional[str]]] = field(default_factory=dict)
    toc: Dict[str, List[TocEntry]] = field(default_factory=dict)


def _doc_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _normalize(text: Optional[str]) -> Optional[str]:
    normalized = " ".join((text or "").split())
    return normalized or None


def _find_head_title(doc_root: etree._Element) -> Optional[etree._Element]:
    for element in doc_root.iter():
        if not isinstance(element.tag, str) or _local_name(element) != "title":
            continue
        parent = element.getparent()
        if parent is not None and _local_name(parent) == "head":
            return element
    return None


def _parse(path: str) -> etree._ElementTree:
    parser = etree.XMLParser(encoding="utf-8", recover=True, remove_blank_text=False)
    return etree.parse(str(path), parser)


def capture_source_titles(
    content_paths: Iterable[str],
    opf_dir: str,
    opf_tree: etree._ElementTree,
    book_root: str,
) -> SourceTitles:
    """Read the original titles, first headings and TOC labels from disk.

    Must run while the extracted files are still the source ones (before a
    checkpoint restore and before the translated files are written back).
    Unreadable files are skipped: their title is simply left unchanged.
    ``book_root`` is the extracted book's directory: a navigation document the
    manifest places outside it is not read.
    """
    source = SourceTitles()
    for path in content_paths:
        try:
            root = _parse(path).getroot()
        except (OSError, etree.XMLSyntaxError):
            continue
        if root is None:
            continue
        title_el = _find_head_title(root)
        title = _normalized_element_text(title_el) if title_el is not None else None
        source.documents[_doc_key(path)] = (title, _extract_first_heading_text(root))

    for entry in _ncx_entries(opf_dir) + _nav_entries(opf_dir, opf_tree, book_root):
        href, _fragment = _split_ncx_src(entry.src)
        if not href:
            continue
        key = _doc_key(os.path.join(entry.base_dir, href))
        source.toc.setdefault(key, []).append(entry)

    return source


def _ncx_entries(opf_dir: str) -> List[TocEntry]:
    """Same NCX lookup as ``update_ncx_toc_labels_from_translated_docs``."""
    entries: List[TocEntry] = []
    for ncx_path in Path(opf_dir).glob("*.ncx"):
        try:
            tree = _parse(ncx_path)
        except (OSError, etree.XMLSyntaxError):
            continue
        for nav_point in tree.findall(".//ncx:navPoint", namespaces=_NCX_NS):
            text_el = nav_point.find("./ncx:navLabel/ncx:text", namespaces=_NCX_NS)
            content_el = nav_point.find("./ncx:content", namespaces=_NCX_NS)
            if text_el is None or content_el is None or not content_el.get("src"):
                continue
            label = _normalize(text_el.text)
            if label:
                entries.append(TocEntry(content_el.get("src"), opf_dir, label))
    return entries


def _nav_entries(opf_dir: str, opf_tree: etree._ElementTree,
                 book_root: str) -> List[TocEntry]:
    """Same nav lookup as ``update_nav_toc_labels_from_translated_docs``."""
    nav_href = _find_nav_doc_href(opf_tree)
    if not nav_href:
        return []
    nav_path = os.path.join(opf_dir, unquote(nav_href))
    if not is_within_directory(nav_path, book_root) or not os.path.exists(nav_path):
        return []
    try:
        tree = _parse(nav_path)
    except (OSError, etree.XMLSyntaxError):
        return []

    nav_dir = os.path.dirname(nav_path)
    epub_type_attr = f"{{{NAMESPACES['epub']}}}type"
    entries: List[TocEntry] = []
    for nav_el in tree.iter():
        if not isinstance(nav_el.tag, str) or _local_name(nav_el) != "nav":
            continue
        if (nav_el.get(epub_type_attr) or "").strip() in _SKIPPED_NAV_TYPES:
            continue
        for anchor in nav_el.iter():
            if not isinstance(anchor.tag, str) or _local_name(anchor) != "a":
                continue
            label = _normalized_element_text(anchor)
            if anchor.get("href") and label:
                entries.append(TocEntry(anchor.get("href"), nav_dir, label))
    return entries


def localize_head_titles(
    docs: Dict[str, etree._Element],
    source: SourceTitles,
    log_callback: Optional[Callable] = None,
) -> int:
    """Set each translated document's ``<title>`` per the module rule.

    ``docs`` are the translated trees (failed documents excluded, as for the
    TOC labels). Returns how many titles were changed.
    """
    docs_by_key = {_doc_key(path): doc for path, doc in docs.items()}
    updated = 0
    for key, doc_root in docs_by_key.items():
        original_title, original_heading = source.documents.get(key, (None, None))
        if not original_title:
            continue
        title_el = _find_head_title(doc_root)
        if title_el is None:
            continue

        new_title = None
        if original_title == original_heading:
            new_title = _extract_first_heading_text(doc_root)
        if not new_title:
            for entry in source.toc.get(key, ()):
                if entry.label != original_title:
                    continue
                new_title = _get_translated_title_for_src(entry.src, entry.base_dir, docs_by_key)
                if new_title:
                    break

        if new_title and new_title != _normalized_element_text(title_el):
            for child in list(title_el):
                title_el.remove(child)
            title_el.text = new_title
            updated += 1

    if log_callback and updated:
        log_callback("epub_head_titles_updated",
                     f"🏷️ Document titles localized: {updated} updated")
    return updated
