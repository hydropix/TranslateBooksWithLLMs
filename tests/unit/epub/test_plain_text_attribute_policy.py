"""
Which source attributes Plain Text Mode carries onto a rebuilt block.

Plain Text Mode wipes the body and re-emits one element per collected block, so
every attribute the output has was put there deliberately. Copying the source
block's whole attribute dict is not safe:

  * `<li>`, `<td>`, `<th>` and unknown blocks are flattened to `<p>`, so a
    tag-specific attribute (colspan, scope, value, ...) would end up on an
    element that cannot legally carry it and epubcheck rejects the file.
  * `lang` / `xml:lang` name the SOURCE language. On translated text they
    override the target language `lang_support` writes on `<html>`, which is
    how a French paragraph ends up read aloud by an English speech engine.
  * Bilingual mode emits two elements for one source block: a duplicated `id`
    is invalid XHTML and makes every TOC anchor pointing at it ambiguous.

`plain_extractor.CARRIED_ATTRIBUTES` is the whitelist; these tests pin it.
"""
from lxml import etree

from src.core.epub.plain_extractor import (
    EPUB_TYPE_ATTR,
    extract_plain_paragraphs,
    replace_body_with_paragraphs,
)

XML_LANG_ATTR = "{http://www.w3.org/XML/1998/namespace}lang"
NS = 'xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"'


def _rebuild(body_inner: str, translations, **kwargs):
    """Extract then rebuild a body, returning its block children."""
    root = etree.fromstring(f"<html {NS}><body>{body_inner}</body></html>".encode())
    body = root[0]
    paragraphs, tags, images, attrib = extract_plain_paragraphs(body)
    replace_body_with_paragraphs(
        body, translations, tags, images,
        source_paragraphs=paragraphs, paragraphs_attrib=attrib, **kwargs
    )
    return body


def _local(elem) -> str:
    return elem.tag.split("}")[-1]


def test_semantic_attributes_survive_the_rebuild():
    body = _rebuild(
        '<h1 id="ch1" class="head" style="text-indent:0" title="Part one" '
        'dir="ltr" epub:type="title">Chapter One</h1>',
        ["Chapitre Un"],
    )

    heading = body[0]
    assert _local(heading) == "h1"
    assert heading.get("id") == "ch1"
    assert heading.get("class") == "head"
    assert heading.get("style") == "text-indent:0"
    assert heading.get("title") == "Part one"
    assert heading.get("dir") == "ltr"
    assert heading.get(EPUB_TYPE_ATTR) == "title"


def test_source_language_attributes_are_not_carried_over():
    """The block must inherit the target lang set on <html>, not keep the source's."""
    body = _rebuild(
        '<p xml:lang="en" lang="en" class="body">Hello there.</p>', ["Bonjour."]
    )

    paragraph = body[0]
    assert paragraph.get(XML_LANG_ATTR) is None
    assert paragraph.get("lang") is None
    # The rest of the attributes are untouched by the exclusion.
    assert paragraph.get("class") == "body"


def test_table_cell_attributes_do_not_follow_the_cell_into_a_paragraph():
    body = _rebuild(
        '<table><tr><td colspan="2" rowspan="3" scope="col" headers="h1" '
        'class="cell">Cell</td></tr></table>',
        ["Cellule"],
    )

    paragraph = body[0]
    assert _local(paragraph) == "p"
    assert paragraph.get("class") == "cell"
    for attr in ("colspan", "rowspan", "scope", "headers"):
        assert paragraph.get(attr) is None, f"{attr} is not valid on <p>"


def test_list_item_attributes_do_not_follow_the_item_into_a_paragraph():
    body = _rebuild('<ol><li value="3" class="item">Item</li></ol>', ["Article"])

    paragraph = body[0]
    assert _local(paragraph) == "p"
    assert paragraph.get("class") == "item"
    assert paragraph.get("value") is None, "value is not valid on <p>"


def test_bilingual_emits_the_source_id_exactly_once():
    body = _rebuild(
        '<h2 id="ch1" class="head">Chapter 1</h2>', ["Chapitre 1"], bilingual=True
    )

    source_block, target_block = body[0], body[1]
    assert [b.get("id") for b in body] == ["ch1", None], (
        "a duplicated id is invalid XHTML and makes the TOC anchor ambiguous"
    )
    # The id goes to the block that comes first in reading order, so an anchor
    # still lands at the top of the right section.
    assert source_block.get("class") == "head plain-text-source"
    assert target_block.get("class") == "head plain-text-target"
