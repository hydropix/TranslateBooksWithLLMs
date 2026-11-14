"""Regression tests: RTL injection/removal must keep XHTML well-formed XML.

Both functions used to parse XHTML with etree.HTMLParser and serialize with
method='html', leaving <br>/<img> unclosed and dropping the XML declaration.
The file is later re-parsed as XML with recovery, which then nested every
following sibling inside the void element.
"""
import pytest
from lxml import etree

from src.core.epub.rtl_support import inject_rtl_css_to_html, remove_rtl_from_html

XHTML_NS = "http://www.w3.org/1999/xhtml"

SOURCE = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<!DOCTYPE html>\n'
    f'<html xmlns="{XHTML_NS}" xmlns:epub="http://www.idpf.org/2007/ops">'
    '<head><title>Chapter</title></head>'
    '<body><p>First line<br/>Second line</p>'
    '<img src="images/a.png" alt="A"/>'
    '<p>After the image</p></body></html>'
)


def _strict_parse(text: str) -> etree._Element:
    # No recovery: the output must be well-formed XML on its own.
    return etree.fromstring(text.encode("utf-8"), etree.XMLParser(recover=False))


def _local(el) -> str:
    return etree.QName(el).localname


@pytest.mark.parametrize("transform", [
    lambda html: inject_rtl_css_to_html(html, "Arabic"),
    remove_rtl_from_html,
    lambda html: remove_rtl_from_html(inject_rtl_css_to_html(html, "Hebrew")),
])
def test_roundtrip_keeps_xml_structure(transform):
    result = transform(SOURCE)

    assert result.startswith("<?xml")
    assert "<!DOCTYPE html>" in result
    root = _strict_parse(result)
    assert root.tag == f"{{{XHTML_NS}}}html"

    body = root.find(f"{{{XHTML_NS}}}body")
    # Void elements stay empty and their following siblings stay siblings.
    assert [_local(child) for child in body] == ["p", "img", "p"]
    img = body[1]
    assert len(img) == 0 and not (img.text or "").strip()
    br = body[0].find(f"{{{XHTML_NS}}}br")
    assert br is not None and len(br) == 0
    assert br.tail == "Second line"
    assert body[2].text == "After the image"


def test_injected_style_lives_in_the_xhtml_namespace():
    root = _strict_parse(inject_rtl_css_to_html(SOURCE, "Arabic"))
    assert root.get("dir") == "rtl"
    styles = root.findall(f"{{{XHTML_NS}}}head/{{{XHTML_NS}}}style")
    assert len(styles) == 1
    assert "direction: rtl" in styles[0].text


def test_remove_rtl_strips_namespaced_rtl_style():
    injected = inject_rtl_css_to_html(SOURCE, "Arabic")
    root = _strict_parse(remove_rtl_from_html(injected))
    assert root.get("dir") == "ltr"
    styles = root.findall(f"{{{XHTML_NS}}}head/{{{XHTML_NS}}}style")
    assert len(styles) == 1
    assert "LTR Reset" in styles[0].text
