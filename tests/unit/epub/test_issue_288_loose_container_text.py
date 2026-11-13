"""
Regression tests for issue #288: Plain Text Mode silently deleted text that sat
loose inside a container (EPUB).

`_collect_blocks` treated every CONTAINER_TAGS element (div, section, ...) as a
pure wrapper: it recursed into the element's children and never read the
element's own `.text`, nor the `.tail` of any of its children. Calibre writes
one `<div class="calibreN">` per paragraph with the text directly inside it, so
on such a book whole chapters vanished. Only inline children survived, each
emitted as a paragraph of its own (`<div>Hello <i>two</i>.</div>` came out as
the lone word "two"). And because `replace_body_with_paragraphs` wipes the body
before refilling it, the lost text was deleted from the output, not merely left
untranslated. The same blind spot dropped text sitting directly in `<body>`,
the tail of a block (`<p>A</p>tail`) and the tail of a dropped `<svg>`.

The fix splits a container's children into runs of inline content separated
by block children. A run carrying loose text becomes one paragraph; a run
without any is dispatched element by element as before, so a body that lost
no text extracts exactly the same paragraphs as it always did.

Also covered here: XML comments. Their `.text` is markup, but the flattener
read it as book text, so `<!-- page 12 -->` was sent to the LLM and printed in
the translated book.
"""
import zipfile
from pathlib import Path

import pytest
from lxml import etree

from src.core.epub.plain_extractor import (
    extract_plain_paragraphs,
    replace_body_with_paragraphs,
)

import src.core.epub.translator as translator_module
from src.core.epub.translator import translate_epub_file

from tests.unit.epub.conftest import (
    REAL_CSS,
    _build_cjk_epub_dir,
    _disable_attribution,
    _echo_llm_client,
    _write,
    _zip_dir_as_epub,
)
from tests.job_helpers import make_ctx, make_job


XHTML_NS = "http://www.w3.org/1999/xhtml"


def _parse_body(body_inner: str) -> etree._Element:
    doc = f"""<html xmlns="{XHTML_NS}"><body>{body_inner}</body></html>"""
    root = etree.fromstring(doc.encode("utf-8"))
    return root.find(f"{{{XHTML_NS}}}body")


def _extract(body_inner: str):
    return extract_plain_paragraphs(_parse_body(body_inner))


# ---------------------------------------------------------------------------
# Loose text is extracted
# ---------------------------------------------------------------------------

def test_calibre_div_paragraphs_are_extracted():
    """The exact shape reported in the issue."""
    paragraphs, tags, images, attrib = _extract(
        '<div class="class8">“This won’t kill him!” Seiya scoffed.</div>\n'
        '<div class="class8">“You see?” Seiya pointed out.</div>\n'
    )

    assert paragraphs == [
        "“This won’t kill him!” Seiya scoffed.",
        "“You see?” Seiya pointed out.",
    ]
    assert tags == ["div", "div"]
    assert attrib == [{"class": "class8"}, {"class": "class8"}]
    assert images == {}


def test_inline_children_join_the_loose_text_of_their_div():
    paragraphs, tags, _images, _attrib = _extract(
        '<div>Hello <i>two</i>, <span class="x">three</span>.</div>'
    )

    assert paragraphs == ["Hello two, three."]
    assert tags == ["div"]


def test_loose_text_around_blocks_keeps_its_position():
    paragraphs, tags, _images, attrib = _extract(
        '<div class="chapter">Lead text<p>Para</p>tail <b>bold</b> text<p>Last</p></div>'
    )

    assert paragraphs == ["Lead text", "Para", "tail bold text", "Last"]
    # Loose text next to block children is an anonymous paragraph: the
    # container's own tag and attributes cannot stand in for a slice of it.
    assert tags == ["p", "p", "p", "p"]
    assert attrib == [{}, {}, {}, {}]


def test_loose_text_directly_in_body_is_extracted():
    paragraphs, tags, _images, attrib = _extract("Bare body text<p>Para</p>after")

    assert paragraphs == ["Bare body text", "Para", "after"]
    assert tags == ["p", "p", "p"]
    assert attrib == [{}, {}, {}]


def test_tail_of_a_dropped_subtree_is_extracted():
    paragraphs, tags, _images, _attrib = _extract(
        '<div><svg xmlns="http://www.w3.org/2000/svg"><text>ignored</text></svg>Caption</div>'
    )

    assert paragraphs == ["Caption"]
    assert tags == ["div"]


def test_nested_containers_with_loose_text():
    paragraphs, tags, _images, attrib = _extract(
        '<section id="s1"><div class="a">One</div><div class="b">Two</div></section>'
    )

    assert paragraphs == ["One", "Two"]
    assert tags == ["div", "div"]
    assert attrib == [{"class": "a"}, {"class": "b"}]


def test_image_inside_a_loose_text_div_is_anchored_to_it():
    paragraphs, tags, images, _attrib = _extract(
        '<p>Before</p><div class="c">Text <img src="a.png" alt=""/> more</div>'
    )

    assert paragraphs == ["Before", "Text more"]
    assert tags == ["p", "div"]
    assert list(images) == [1]
    assert images[1][0].get("src") == "a.png"


# ---------------------------------------------------------------------------
# Comments are never book text
# ---------------------------------------------------------------------------

def test_comment_content_is_never_extracted():
    paragraphs, _tags, _images, _attrib = _extract(
        "<!-- page 12 --><p>A <!-- note --> B</p>"
    )

    assert paragraphs == ["A B"]


def test_comment_tail_is_kept_as_loose_text():
    paragraphs, tags, _images, _attrib = _extract("<div><!-- page 12 -->Text</div>")

    assert paragraphs == ["Text"]
    assert tags == ["div"]


# ---------------------------------------------------------------------------
# Bodies that lost no text extract exactly as before
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "body_inner, expected",
    [
        # Wrapper div: recursed into, whitespace between blocks ignored.
        (
            '<div class="chapter">\n  <p>A</p>\n  <p>B</p>\n</div>',
            (["A", "B"], ["p", "p"], [{}, {}]),
        ),
        # A lone inline element keeps the generic-paragraph shape.
        (
            '<div class="c"><span class="s">T</span></div>',
            (["T"], ["p"], [{"class": "s"}]),
        ),
        # Adjacent inline elements with no loose text stay separate blocks.
        (
            "<div><b>A</b><i>B</i></div>",
            (["A", "B"], ["p", "p"], [{}, {}]),
        ),
    ],
)
def test_extraction_is_unchanged_without_loose_text(body_inner, expected):
    paragraphs, tags, _images, attrib = _extract(body_inner)

    assert (paragraphs, tags, attrib) == expected


def test_div_wrapped_image_still_anchors_to_the_previous_block():
    paragraphs, tags, images, _attrib = _extract(
        '<p>A</p><div class="img"><img src="a.png" alt=""/></div><p>B</p>'
    )

    assert paragraphs == ["A", "B"]
    assert tags == ["p", "p"]
    assert list(images) == [0]


def test_svg_cover_page_still_yields_no_block():
    """The calibre cover must keep extracting to nothing (body left verbatim)."""
    paragraphs, _tags, images, _attrib = _extract(
        '<div><svg xmlns="http://www.w3.org/2000/svg"><image width="1" height="1"/></svg></div>'
    )

    assert paragraphs == []
    assert images == {}


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------

def test_rebuild_keeps_the_div_and_its_attributes():
    body = _parse_body('<div class="class8" id="p1">Hello <i>world</i>.</div>')
    paragraphs, tags, images, attrib = extract_plain_paragraphs(body)

    replace_body_with_paragraphs(
        body, ["Bonjour le monde."], tags, images,
        source_paragraphs=paragraphs, paragraphs_attrib=attrib,
    )

    assert len(body) == 1
    block = body[0]
    assert block.tag.split("}")[-1] == "div"
    assert block.get("class") == "class8"
    assert block.get("id") == "p1"
    assert block.text == "Bonjour le monde."


def test_bilingual_rebuild_emits_the_id_once():
    body = _parse_body('<div class="class8" id="p1">Hello.</div>')
    paragraphs, tags, images, attrib = extract_plain_paragraphs(body)

    replace_body_with_paragraphs(
        body, ["Bonjour."], tags, images, bilingual=True,
        source_paragraphs=paragraphs, paragraphs_attrib=attrib,
    )

    assert [child.tag.split("}")[-1] for child in body] == ["div", "div"]
    assert [child.get("id") for child in body] == ["p1", None]
    assert [child.text for child in body] == ["Hello.", "Bonjour."]


# ---------------------------------------------------------------------------
# End-to-end: the real EPUB adapter path
# ---------------------------------------------------------------------------

CALIBRE_CHAPTER_XHTML = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>part0010</title></head>'
    '<body class="calibre">\n'
    '<div class="class8">Biino spoke up to defend the flailing mascot.</div>\n'
    '<div class="class8">“This won’t kill him!” Seiya <i>scoffed</i>.</div>\n'
    '<div class="class8">“You see?” Seiya pointed out.</div>\n'
    '</body></html>\n'
)

CALIBRE_SENTENCES = (
    "Biino spoke up to defend the flailing mascot.",
    "“This won’t kill him!” Seiya scoffed.",
    "“You see?” Seiya pointed out.",
)


@pytest.fixture
def calibre_chapter_epub(tmp_path: Path) -> Path:
    root = _build_cjk_epub_dir(tmp_path / "src_epub", REAL_CSS.read_text(encoding="utf-8"))
    _write(root / "OEBPS" / "Text" / "intro.xhtml", CALIBRE_CHAPTER_XHTML)
    return _zip_dir_as_epub(root, tmp_path / "input.epub")


@pytest.mark.asyncio
async def test_calibre_div_text_survives_the_full_plain_text_pipeline(
    calibre_chapter_epub, tmp_path, monkeypatch
):
    """With an echo LLM, every sentence of the chapter must be in the output."""
    monkeypatch.setattr(
        translator_module, "_create_llm_client", lambda *args, **kwargs: _echo_llm_client()
    )
    _disable_attribution(monkeypatch)

    output_epub = tmp_path / "output_plain.epub"
    await translate_epub_file(
        make_job(
            input_path=str(calibre_chapter_epub),
            output_path=str(output_epub),
            source_language="English",
            target_language="French",
            prompt_options={"plain_text_mode": True},
        ),
        make_ctx(),
    )

    with zipfile.ZipFile(output_epub) as archive:
        output_text = archive.read("OEBPS/Text/intro.xhtml").decode("utf-8")

    parser = etree.XMLParser(recover=True)
    output_root = etree.fromstring(output_text.encode("utf-8"), parser)
    divs = [
        el for el in output_root.iter()
        if isinstance(el.tag, str) and el.tag.split("}")[-1] == "div"
    ]
    assert ["".join(div.itertext()) for div in divs] == list(CALIBRE_SENTENCES)
    assert all("class8" in (div.get("class") or "") for div in divs)
