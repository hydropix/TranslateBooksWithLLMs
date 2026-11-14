"""The translated-body re-parse must not need recovery for epub:type.

The body is serialized out of its document, so the ``epub`` prefix declared
on <html> is not in scope when it is parsed back. Without a declaration on
the wrapper every ``epub:type`` attribute was a namespace error: every EPUB 3
body was parsed in recover mode, logged "recovered from N errors", and was
held to the tighter text-loss bound meant for broken markup.
"""
from lxml import etree

from src.core.epub.body_serializer import _parse_translated_body, replace_body_content

XHTML_NS = "http://www.w3.org/1999/xhtml"
OPS_NS = "http://www.idpf.org/2007/ops"

BODY = (
    '<section epub:type="chapter"><h1>Chapter</h1>'
    '<p>Text<a epub:type="noteref" href="#fn1">1</a></p>'
    '<aside epub:type="footnote" id="fn1"><p>Note</p></aside></section>'
)


def test_epub_type_body_parses_without_recovery():
    temp, method, warnings = _parse_translated_body(BODY)
    assert method == "XML"
    assert warnings == []
    section = temp.find(f"{{{XHTML_NS}}}section")
    assert section.get(f"{{{OPS_NS}}}type") == "chapter"


def test_replaced_body_round_trips_with_namespaced_epub_type():
    doc = etree.fromstring(
        f'<html xmlns="{XHTML_NS}" xmlns:epub="{OPS_NS}"><head><title>t</title>'
        f'</head><body><p>old</p></body></html>'.encode("utf-8")
    )
    body = doc.find(f".//{{{XHTML_NS}}}body")
    replace_body_content(body, BODY)

    serialized = etree.tostring(doc, encoding="unicode")
    assert 'epub:type="chapter"' in serialized
    reparsed = etree.fromstring(serialized.encode("utf-8"))
    types = [el.get(f"{{{OPS_NS}}}type") for el in reparsed.iter() if el.get(f"{{{OPS_NS}}}type")]
    assert types == ["chapter", "noteref", "footnote"]
