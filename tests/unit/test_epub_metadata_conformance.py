"""EPUB 3 conformance for the translation attribution metadata.

The attribution branch used to write ``opf:role`` on ``dc:contributor``. EPUB
3.4 rejects that attribute (EPUBCheck RSC-005). The canonical representation is
a ``meta`` element that refines the contributor with the MARC relator code.
"""
from __future__ import annotations

from lxml import etree

from src.core.epub import translator

OPF = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">urn:uuid:12345678-1234-1234-1234-123456789abc</dc:identifier>
    <dc:title>Test</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest/>
  <spine/>
</package>
"""

NAMESPACES = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def test_contributor_role_uses_epub3_refinement(tmp_path, monkeypatch):
    monkeypatch.setattr(translator, "ATTRIBUTION_ENABLED", True)
    monkeypatch.setattr(translator, "GENERATOR_NAME", "TranslateBook with LLM (TBL)")
    monkeypatch.setattr(
        translator,
        "GENERATOR_SOURCE",
        "https://github.com/hydropix/TranslateBookWithLLM",
    )

    opf_path = tmp_path / "content.opf"
    opf_path.write_text(OPF, encoding="utf-8")
    tree = etree.parse(str(opf_path))

    translator._update_epub_metadata(tree, str(opf_path), "Portuguese (Brazil)")

    root = etree.parse(str(opf_path)).getroot()
    contributors = root.findall(".//dc:contributor", NAMESPACES)
    assert contributors, "attribution contributor was not written"
    contributor = contributors[-1]
    assert contributor.get("id") == "tbl-translator"
    assert contributor.get("{http://www.idpf.org/2007/opf}role") is None

    role = root.find(".//opf:meta[@refines='#tbl-translator']", NAMESPACES)
    assert role is not None, "canonical role refinement was not written"
    assert role.get("property") == "role"
    assert role.get("scheme") == "marc:relators"
    assert (role.text or "").strip() == "trl"

    serialized = opf_path.read_text(encoding="utf-8")
    assert "ns0:role" not in serialized
    assert "opf:role" not in serialized
