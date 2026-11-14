"""Ordering test for the attribution page in the EPUB translation pipeline.

Reproduces steps 4.7 (add_attribution_page) -> 5 (_update_epub_metadata) ->
6.5 (apply_target_language_to_xhtml_directory) on a synthetic extracted-EPUB
tree, without any LLM or network. This is what proves D7 (the module never
writes the OPF; step 5's single write persists the in-memory edits) and D10
(the page is written before RTL/lang so it inherits them like any other
document).
"""
from pathlib import Path

from lxml import etree

from src.core.epub.attribution_page import ATTRIBUTION_ID, add_attribution_page
from src.core.epub.opf_metadata import update_epub_metadata
from src.core.epub.lang_support import apply_target_language_to_xhtml_directory
from src.core.epub.rtl_support import apply_rtl_to_epub_directory
from src.config import NAMESPACES


XML_NS = "http://www.w3.org/XML/1998/namespace"


def _enable_attribution(monkeypatch):
    """Force both flags on regardless of the developer's local .env.

    `_update_epub_metadata` reads ATTRIBUTION_ENABLED from the translator
    module's own namespace, not attribution_page's, so it must be patched
    there too for the dc:contributor assertion to be deterministic.
    """
    monkeypatch.setattr("src.core.epub.attribution_page.ATTRIBUTION_ENABLED", True)
    monkeypatch.setattr("src.core.epub.attribution_page.ATTRIBUTION_PAGE_ENABLED", True)
    monkeypatch.setattr("src.core.epub.opf_metadata.ATTRIBUTION_ENABLED", True)


def _build_minimal_opf(opf_dir: Path) -> Path:
    """Write a minimal EPUB3 OPF (one manifest item, one itemref, dc:language)."""
    opf_path = opf_dir / "content.opf"
    opf_path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="{NAMESPACES['opf']}" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
  </spine>
</package>""",
        encoding="utf-8",
    )
    (opf_dir / "chapter1.xhtml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body><p>Dummy chapter content.</p></body>
</html>""",
        encoding="utf-8",
    )
    return opf_path


def test_attribution_page_survives_metadata_write_and_inherits_target_lang(tmp_path: Path, monkeypatch):
    _enable_attribution(monkeypatch)
    opf_path = _build_minimal_opf(tmp_path)

    # Step 2 (parse manifest) equivalent.
    opf_tree = etree.parse(str(opf_path))

    # Step 4.7.
    href = add_attribution_page(opf_tree, str(tmp_path))
    assert href is not None

    # Step 5. The single OPF write of the whole pipeline.
    update_epub_metadata(opf_tree, str(opf_path), "French")

    # Step 6.5.
    apply_target_language_to_xhtml_directory(str(tmp_path), "French")

    # D7: the OPF on disk carries the manifest item and spine itemref even
    # though add_attribution_page never wrote the file itself.
    on_disk_tree = etree.parse(str(opf_path))
    manifest_items = on_disk_tree.getroot().findall('.//opf:manifest/opf:item', namespaces=NAMESPACES)
    attribution_items = [i for i in manifest_items if i.get("id") == ATTRIBUTION_ID]
    assert len(attribution_items) == 1
    assert attribution_items[0].get("href") == href

    spine_itemrefs = on_disk_tree.getroot().findall('.//opf:spine/opf:itemref', namespaces=NAMESPACES)
    assert spine_itemrefs[-1].get("idref") == ATTRIBUTION_ID

    # step 5 did not lose its existing metadata behavior.
    metadata = on_disk_tree.getroot().find('.//opf:metadata', namespaces=NAMESPACES)
    contributors = metadata.findall('.//{http://purl.org/dc/elements/1.1/}contributor')
    assert len(contributors) == 1
    # The auxiliary per-install dc:identifier was removed: step 5 must never
    # emit one again, whatever the attribution settings say.
    render_uid = metadata.find('.//{http://purl.org/dc/elements/1.1/}identifier[@id="render-uid"]')
    assert render_uid is None

    # D10: the attribution page written at 4.7 was walked by step 6.5 just
    # like any other document, so it carries the target language.
    attribution_path = tmp_path / href
    attribution_tree = etree.parse(str(attribution_path))
    attribution_root = attribution_tree.getroot()
    assert attribution_root.get("lang") == "fr"
    assert attribution_root.get(f"{{{XML_NS}}}lang") == "fr"


def test_attribution_page_receives_rtl_css_like_any_other_document(tmp_path: Path, monkeypatch):
    _enable_attribution(monkeypatch)
    opf_path = _build_minimal_opf(tmp_path)

    opf_tree = etree.parse(str(opf_path))
    href = add_attribution_page(opf_tree, str(tmp_path))
    assert href is not None

    update_epub_metadata(opf_tree, str(opf_path), "Arabic")

    attribution_path = tmp_path / href
    before_bytes = attribution_path.read_bytes()

    apply_rtl_to_epub_directory(str(tmp_path), "Arabic", "English")

    after_bytes = attribution_path.read_bytes()
    assert after_bytes != before_bytes
