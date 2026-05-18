from pathlib import Path

from lxml import etree

from src.core.epub.translator import _update_ncx_toc_labels_from_translated_docs


NCX_NS = "http://www.daisy.org/z3986/2005/ncx/"


def _xhtml_with_heading(anchor_id: str, heading_text: str) -> etree._Element:
    return etree.fromstring(
        f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <h2><a id="{anchor_id}"/>{heading_text}</h2>
    <p>Translated body.</p>
  </body>
</html>""".encode("utf-8")
    )


def test_updates_ncx_nav_labels_from_translated_xhtml_without_changing_jump_targets(tmp_path: Path):
    opf_dir = tmp_path / "OEBPS"
    opf_dir.mkdir()
    chapter_path = opf_dir / "chapter1.xhtml"
    chapter_path.write_text("<html/>", encoding="utf-8")

    ncx_path = opf_dir / "toc.ncx"
    ncx_path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="{NCX_NS}" version="2005-1">
  <navMap>
    <navPoint id="nav1" playOrder="1">
      <navLabel><text>Chapter 1: Original Title</text></navLabel>
      <content src="chapter1.xhtml#chapter-one"/>
    </navPoint>
  </navMap>
</ncx>""",
        encoding="utf-8",
    )

    parsed_docs = {
        str(chapter_path): _xhtml_with_heading(
            "chapter-one",
            "第一章：译后标题",
        )
    }

    result = _update_ncx_toc_labels_from_translated_docs(
        opf_dir=str(opf_dir),
        parsed_xhtml_docs=parsed_docs,
    )

    tree = etree.parse(str(ncx_path))
    ns = {"ncx": NCX_NS}
    assert result == {"updated": 1, "unchanged": 0, "errors": 0}
    assert tree.findtext(".//ncx:navLabel/ncx:text", namespaces=ns) == "第一章：译后标题"
    assert tree.find(".//ncx:content", namespaces=ns).get("src") == "chapter1.xhtml#chapter-one"


def test_uses_first_heading_when_ncx_target_has_no_fragment(tmp_path: Path):
    opf_dir = tmp_path / "OEBPS"
    opf_dir.mkdir()
    chapter_path = opf_dir / "chapter2.xhtml"
    chapter_path.write_text("<html/>", encoding="utf-8")

    ncx_path = opf_dir / "toc.ncx"
    ncx_path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="{NCX_NS}" version="2005-1">
  <navMap>
    <navPoint id="nav2" playOrder="2">
      <navLabel><text>Chapter 2: Original Title</text></navLabel>
      <content src="chapter2.xhtml"/>
    </navPoint>
  </navMap>
</ncx>""",
        encoding="utf-8",
    )

    parsed_docs = {
        str(chapter_path): _xhtml_with_heading(
            "unused-anchor",
            "第二章：无片段目标",
        )
    }

    result = _update_ncx_toc_labels_from_translated_docs(
        opf_dir=str(opf_dir),
        parsed_xhtml_docs=parsed_docs,
    )

    tree = etree.parse(str(ncx_path))
    assert result["updated"] == 1
    assert tree.findtext(".//{http://www.daisy.org/z3986/2005/ncx/}text") == "第二章：无片段目标"
