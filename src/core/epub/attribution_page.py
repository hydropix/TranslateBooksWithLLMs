"""
Attribution page for translated EPUBs.

Today the EPUB pipeline signs the OPF metadata only (dc:contributor,
dc:description), which a reader never sees while turning pages. This module
builds a small, self-contained XHTML document that is appended to the end of
the spine so the attribution is actually visible, without touching any
author content and without ever being listed in the table of contents.

This module owns:
  - `build_attribution_xhtml(epub_version)` — a pure function producing the
    XHTML bytes, branching on EPUB2 vs EPUB3 markup rules.
  - `add_attribution_page(opf_tree, opf_dir, log_callback)` — writes the file
    next to the OPF and registers it in the in-memory manifest and spine.

`add_attribution_page` never writes the OPF itself: the caller's existing
metadata-write step persists these in-memory mutations in the same pass, so
there is exactly one OPF write per translation (see the EPUB attribution
plan). It also never raises — attribution is a nice-to-have, never a reason
to fail a translation.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

from lxml import etree

from src.config import (
    ATTRIBUTION_ENABLED,
    ATTRIBUTION_PAGE_ENABLED,
    GENERATOR_NAME,
    GENERATOR_SOURCE,
    NAMESPACES,
)

ATTRIBUTION_ID = "tbl-attribution"          # manifest item @id
ATTRIBUTION_FILENAME = "tbl-attribution.xhtml"
XHTML_MEDIA_TYPE = "application/xhtml+xml"

# Split once so suffixed variants ("tbl-attribution-1.xhtml") stay derived from
# the filename above instead of repeating its stem.
_FILENAME_STEM, _FILENAME_EXT = os.path.splitext(ATTRIBUTION_FILENAME)

# Suffixes tried, in order, when the id or filename above is already taken by
# a foreign manifest item. "" first, then "-1".."-49".
_MAX_SUFFIX_ATTEMPTS = 50


def build_attribution_xhtml(epub_version: str) -> bytes:
    """Build the attribution page as a complete, standalone XHTML document.

    EPUB3 documents wrap the mention in a <section epub:type="colophon">,
    which is the semantic vocabulary readers use to recognize a colophon.
    EPUB2 documents use a plain <div>: <section> and epub:type are not valid
    under the XHTML 1.1 DTD that epubcheck applies to EPUB2 content, and
    epub:type would trip validation on a book that is otherwise EPUB2-valid.

    Pure function: no filesystem access, deterministic for a given input.
    """
    is_epub3 = epub_version.startswith("3")

    if is_epub3:
        html_open = (
            '<html xmlns="http://www.w3.org/1999/xhtml" '
            'xmlns:epub="http://www.idpf.org/2007/ops">'
        )
        wrapper_open = '<section epub:type="colophon" class="tbl-attribution">'
        wrapper_close = '</section>'
    else:
        html_open = '<html xmlns="http://www.w3.org/1999/xhtml">'
        wrapper_open = '<div class="tbl-attribution">'
        wrapper_close = '</div>'

    xhtml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'{html_open}\n'
        '<head>\n'
        '<title>Translation</title>\n'
        '<style type="text/css">\n'
        '.tbl-attribution {\n'
        '  margin-top: 3em;\n'
        '  text-align: center;\n'
        '  font-size: 0.85em;\n'
        '  line-height: 1.6;\n'
        '}\n'
        '.tbl-attribution a {\n'
        '  text-decoration: none;\n'
        '}\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        f'{wrapper_open}\n'
        f'<p><strong>Translated using {GENERATOR_NAME}</strong></p>\n'
        f'<p><a href="{GENERATOR_SOURCE}">{GENERATOR_SOURCE}</a></p>\n'
        f'{wrapper_close}\n'
        '</body>\n'
        '</html>\n'
    )
    return xhtml.encode("utf-8")


def _already_added(manifest_items: List[etree._Element]) -> bool:
    """True when this tool's attribution page is already in the manifest."""
    # Idempotence: a page this tool already added has both our id and
    # our filename on the *same* item. A foreign item can own one
    # without the other (see the suffix search below), so this must be
    # an AND on a single item, not an OR across the manifest.
    for item in manifest_items:
        href = item.get("href") or ""
        if item.get("id") == ATTRIBUTION_ID and os.path.basename(href) == ATTRIBUTION_FILENAME:
            return True
    return False


def _find_free_slot(
    manifest_items: List[etree._Element],
    opf_dir: str,
) -> Optional[Tuple[str, str]]:
    """Return a free ``(item_id, filename)`` pair, or None when none is left."""
    # A foreign item may own our id or our filename without owning both,
    # so probe for a free suffix that clears both checks at once.
    used_ids = {item.get("id") for item in manifest_items}
    for i in range(_MAX_SUFFIX_ATTEMPTS):
        suffix = "" if i == 0 else f"-{i}"
        candidate_id = f"{ATTRIBUTION_ID}{suffix}"
        candidate_filename = f"{_FILENAME_STEM}{suffix}{_FILENAME_EXT}"
        if candidate_id in used_ids:
            continue
        if os.path.exists(os.path.join(opf_dir, candidate_filename)):
            continue
        return candidate_id, candidate_filename
    return None


def _register_page(manifest: etree._Element, spine: etree._Element,
                   item_id: str, filename: str) -> None:
    """Add the page to the manifest and as the last spine item."""
    item_el = etree.SubElement(manifest, '{%s}item' % NAMESPACES['opf'])
    item_el.set('id', item_id)
    item_el.set('href', filename)
    item_el.set('media-type', XHTML_MEDIA_TYPE)

    itemref_el = etree.SubElement(spine, '{%s}itemref' % NAMESPACES['opf'])
    itemref_el.set('idref', item_id)


def add_attribution_page(
    opf_tree: etree._ElementTree,
    opf_dir: str,
    log_callback: Optional[Callable] = None,
) -> Optional[str]:
    """Append an attribution page to the spine of an in-memory OPF tree.

    Writes the XHTML file into `opf_dir` and registers it in `<manifest>`
    and as the last `<spine>` item, or does nothing at all. Never writes the
    OPF (the caller's metadata step performs the single OPF write) and never
    raises.

    Returns the href of the new page relative to `opf_dir`, or None if the
    page was disabled, already present, or could not be added.
    """
    try:
        if not (ATTRIBUTION_ENABLED and ATTRIBUTION_PAGE_ENABLED):
            return None

        opf_root = opf_tree.getroot()
        manifest = opf_root.find('.//opf:manifest', namespaces=NAMESPACES)
        spine = opf_root.find('.//opf:spine', namespaces=NAMESPACES)
        if manifest is None or spine is None:
            if log_callback:
                log_callback(
                    "epub_attribution_page_skipped",
                    "⚠️ manifest or spine missing, attribution page skipped",
                )
            return None

        manifest_items = manifest.findall('.//opf:item', namespaces=NAMESPACES)
        if _already_added(manifest_items):
            return None

        slot = _find_free_slot(manifest_items, opf_dir)
        if slot is None:
            if log_callback:
                log_callback(
                    "epub_attribution_page_skipped",
                    "⚠️ no free filename for attribution page",
                )
            return None
        item_id, filename = slot

        epub_version = opf_root.get('version') or ""
        xhtml_bytes = build_attribution_xhtml(epub_version)

        # Write the file first: the tree is only mutated once it has
        # succeeded, so a failure here can never leave an orphan manifest
        # entry pointing at a missing file.
        path = os.path.join(opf_dir, filename)
        with open(path, 'wb') as f:
            f.write(xhtml_bytes)

        _register_page(manifest, spine, item_id, filename)

        if log_callback:
            log_callback(
                "epub_attribution_page_added",
                f"📄 Attribution page added: {filename}",
            )
        return filename
    except Exception as e:
        if log_callback:
            log_callback(
                "epub_attribution_page_failed",
                f"⚠️ Could not add attribution page: {e}",
            )
        return None
