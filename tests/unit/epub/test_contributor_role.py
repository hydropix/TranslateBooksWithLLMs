"""Translator role on the attribution dc:contributor (issue #332).

EPUB 3 forbids ``opf:role`` on ``dc:contributor`` (EPUBCheck RSC-005) and
expects a ``meta refines`` element instead. EPUB 2 only knows the attribute.
"""
from pathlib import Path

import pytest
from lxml import etree

from src.config import NAMESPACES
from src.core.epub.translator import _update_epub_metadata

OPF_ROLE = "{http://www.idpf.org/2007/opf}role"


def _write_opf(tmp_path: Path, version: str, extra_metadata: str = "") -> Path:
    opf_path = tmp_path / "content.opf"
    opf_path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="{NAMESPACES['opf']}" version="{version}" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="{NAMESPACES['opf']}">
    <dc:identifier id="bookid">urn:uuid:00000000-0000-0000-0000-000000000000</dc:identifier>
    <dc:title>Test Book</dc:title>
    <dc:language>en</dc:language>
    {extra_metadata}
  </metadata>
  <manifest/>
  <spine/>
</package>""",
        encoding="utf-8",
    )
    return opf_path


def _run(opf_path: Path) -> etree._Element:
    _update_epub_metadata(etree.parse(str(opf_path)), str(opf_path), "French")
    return etree.parse(str(opf_path)).getroot()


def _contributors(root):
    return root.findall(".//dc:contributor", namespaces=NAMESPACES)


@pytest.fixture
def attribution_on(monkeypatch):
    monkeypatch.setattr("src.core.epub.translator.ATTRIBUTION_ENABLED", True)


@pytest.mark.parametrize("version", ["3.0", "3.3"])
def test_epub3_uses_meta_refinement(tmp_path, attribution_on, version):
    root = _run(_write_opf(tmp_path, version))

    contributor = _contributors(root)[-1]
    assert contributor.get(OPF_ROLE) is None
    contributor_id = contributor.get("id")
    assert contributor_id

    role = root.find(f".//opf:meta[@refines='#{contributor_id}']", namespaces=NAMESPACES)
    assert role is not None
    assert role.get("property") == "role"
    assert role.get("scheme") == "marc:relators"
    assert role.text == "trl"


def test_epub2_keeps_opf_role_attribute(tmp_path, attribution_on):
    root = _run(_write_opf(tmp_path, "2.0"))

    contributor = _contributors(root)[-1]
    assert contributor.get(OPF_ROLE) == "trl"
    assert root.find(".//opf:meta[@refines]", namespaces=NAMESPACES) is None


def test_epub3_id_does_not_collide_with_existing_one(tmp_path, attribution_on):
    existing = '<dc:contributor id="tbl-translator">Someone</dc:contributor>'
    root = _run(_write_opf(tmp_path, "3.0", existing))

    ids = root.xpath("//@id")
    assert len(ids) == len(set(ids))
    new_contributor = _contributors(root)[-1]
    assert new_contributor.get("id") == "tbl-translator-2"
    assert root.find(".//opf:meta[@refines='#tbl-translator-2']", namespaces=NAMESPACES) is not None


def test_attribution_disabled_adds_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.epub.translator.ATTRIBUTION_ENABLED", False)
    root = _run(_write_opf(tmp_path, "3.0"))

    assert _contributors(root) == []
    assert root.find(".//opf:meta[@refines]", namespaces=NAMESPACES) is None
