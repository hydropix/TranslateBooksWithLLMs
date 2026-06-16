"""
Web-scraping boilerplate removal for EPUB content documents.

Many EPUBs are built by scraping web pages (WordPress/Jetpack blogs, web-novel
sites, ...). Their chapter bodies carry non-content artifacts that the original
page rendered as chrome but that survive into the XHTML: social share bars
("Share on X (Opens in new window)"), related-post widgets, prev/next
navigation, "Loading..." spinners, screen-reader-only labels, etc.

These artifacts are not prose. Left in place they get chunked, sent to the LLM,
and translated literally (issue #239), polluting the output and wasting tokens.

This module strips them from the body DOM *before* serialization, so they never
reach the translator and never appear in the output. It is deliberately
conservative: it targets structural signals (semantic tags, well-known
class/id tokens, hidden attributes) rather than guessing from visible text,
and it preserves the EPUB3 navigation document's TOC.
"""
import re
from typing import Callable, Optional

from lxml import etree

# EPUB Open Packaging Structure namespace (carries epub:type).
_OPS_NS = "http://www.idpf.org/2007/ops"

# Local tag names that are non-content chrome in a chapter body.
# <nav> is handled specially below so the EPUB3 TOC is preserved.
_BOILERPLATE_TAGS = {"footer", "nav"}

# epub:type values that mark a *legitimate* navigation document. A <nav> with
# one of these is the book's real TOC/landmarks/page-list and must survive
# (its labels are localized separately, see _update_nav_toc_labels_*).
_PROTECTED_NAV_TYPES = {"toc", "landmarks", "page-list", "lot", "loi", "lov", "lot"}

# class/id tokens that mark scraped widgets. Matched per whitespace-separated
# token (and against the raw id), case-insensitively. Kept tight to avoid
# nuking real prose containers.
_BOILERPLATE_TOKEN_RE = re.compile(
    r"""^(?:
        sharedaddy | sd-sharing | sd-social | sd-block | sd-content |
        share-?buttons? | sharebox | share-?bar | social-?share |
        social-?links? | social-?media | social-?icons? |
        addtoany | addthis | sharethis | jp-sharing |
        jp-relatedposts | related-?posts? | related-?articles? | yarpp(?:-.*)? |
        post-?navigation | post-?nav | nav-?links? | nav-?previous | nav-?next |
        pagination | page-?links? | prev-?next | post-?pagination |
        comment-?respond | comments?-area | comments?-list | jp-post-flair |
        screen-?reader-?text | sr-only | visually-?hidden | a11y-hidden
    )$""",
    re.IGNORECASE | re.VERBOSE,
)


def _local_name(element: etree._Element) -> str:
    """Return the namespace-stripped lowercased tag name (or '' for comments)."""
    tag = element.tag
    if not isinstance(tag, str):  # comments, PIs
        return ""
    return etree.QName(tag).localname.lower()


def _is_protected_nav(element: etree._Element) -> bool:
    """True if this <nav> is the EPUB3 TOC / landmarks / page-list."""
    epub_type = element.get(f"{{{_OPS_NS}}}type", "") or element.get("epub:type", "")
    if any(t in _PROTECTED_NAV_TYPES for t in epub_type.lower().split()):
        return True
    # ARIA fallback used by some toolchains.
    return element.get("role", "").lower() == "doc-toc"


def _matches_boilerplate_attr(element: etree._Element) -> bool:
    """True if the element's class tokens or id match a known boilerplate token."""
    class_attr = element.get("class", "")
    if class_attr:
        for token in class_attr.split():
            if _BOILERPLATE_TOKEN_RE.match(token):
                return True
    element_id = element.get("id", "")
    if element_id and _BOILERPLATE_TOKEN_RE.match(element_id.strip()):
        return True
    return False


def _is_hidden(element: etree._Element) -> bool:
    """True if the element is explicitly hidden (so it carries no visible prose)."""
    if element.get("hidden") is not None:
        return True
    if element.get("aria-hidden", "").lower() == "true":
        return True
    style = element.get("style", "")
    if style and re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", style, re.IGNORECASE):
        return True
    return False


def _should_strip(element: etree._Element) -> bool:
    name = _local_name(element)
    if name == "nav":
        return not _is_protected_nav(element)
    if name in _BOILERPLATE_TAGS:
        return True
    if _matches_boilerplate_attr(element):
        return True
    if _is_hidden(element):
        return True
    return False


def _remove_preserving_tail(element: etree._Element) -> None:
    """Detach an element while keeping any text that trailed it in the parent.

    lxml drops an element's ``tail`` when it is removed; for boilerplate that
    tail is whitespace, but we move it to the nearest text anchor anyway so we
    never silently drop adjacent prose.
    """
    parent = element.getparent()
    if parent is None:
        return
    tail = element.tail
    if tail:
        prev = element.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(element)


def strip_web_boilerplate(
    body: Optional[etree._Element],
    log_callback: Optional[Callable] = None,
) -> int:
    """Remove scraped web boilerplate from a body element, in place.

    Walks the body once, collecting the topmost matching elements (a matched
    ancestor short-circuits its descendants), then detaches them. Returns the
    number of elements removed.

    The EPUB3 navigation document's TOC (<nav epub:type="toc">) is preserved.
    """
    if body is None:
        return 0

    to_remove = []
    # iter() is document order (parents before children); skipping the subtree
    # of an already-matched element avoids double-counting nested widgets.
    skip_until_outside = None
    for element in body.iter():
        if not isinstance(element.tag, str):
            continue
        if skip_until_outside is not None:
            # Still inside a subtree already marked for removal?
            ancestor = element.getparent()
            inside = False
            while ancestor is not None:
                if ancestor is skip_until_outside:
                    inside = True
                    break
                ancestor = ancestor.getparent()
            if inside:
                continue
            skip_until_outside = None
        if element is body:
            continue
        if _should_strip(element):
            to_remove.append(element)
            skip_until_outside = element

    for element in to_remove:
        _remove_preserving_tail(element)

    if to_remove and log_callback:
        log_callback(
            "boilerplate_stripped",
            f"🧹 Removed {len(to_remove)} web-scraping boilerplate "
            f"element(s) (share bars, related posts, nav, hidden) before translation",
        )

    return len(to_remove)
