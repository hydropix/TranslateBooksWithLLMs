"""
Regression tests for issue #254: Plain Text Mode silently dropped `<hr>` scene
separators (EPUB).

`<hr>` is not in any of the tag sets that drive Plain Text Mode's block
collection (`BLOCK_TAGS`, `CONTAINER_TAGS`, `DROP_TAGS`), so it fell through
the generic tail branch of `_collect_blocks`, which only keeps a child that
has text or images -- neither of which `<hr>` has. It was therefore never
collected, and `replace_body_with_paragraphs` -- which wipes the body and
refills it only from the collected list -- could not re-emit it.

The fix makes `<hr>` a "void block": it is collected into a `VOID_BLOCK_TAGS`
slot with empty text, occupies its own position in the paragraph list, is
never sent to the LLM (an empty/whitespace-only paragraph is already skipped
by `build_plain_segments`), and is re-emitted as a bare `<hr/>` at its
original position during rebuild.
"""
import zipfile
from pathlib import Path
from typing import List

import pytest
from lxml import etree

from src.config import INPUT_TAG_IN, INPUT_TAG_OUT
from src.core.epub.plain_extractor import (
    extract_plain_paragraphs,
    replace_body_with_paragraphs,
)
from src.core.common.plain_text_pipeline import build_plain_segments

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


XHTML_NS = "http://www.w3.org/1999/xhtml"


def _parse_body(body_inner: str) -> etree._Element:
    doc = f"""<html xmlns="{XHTML_NS}"><body>{body_inner}</body></html>"""
    root = etree.fromstring(doc.encode("utf-8"))
    return root.find(f"{{{XHTML_NS}}}body")


def _local_tags(element: etree._Element) -> list:
    return [child.tag.split("}")[-1] for child in element]


def test_hr_is_collected_as_void_block():
    body = _parse_body("<p>A</p><hr/><p>B</p>")
    paragraphs, tags, images, _attrib = extract_plain_paragraphs(body)

    assert (paragraphs, tags, images) == (["A", "", "B"], ["p", "hr", "p"], {})
    assert len(paragraphs) == len(tags)


def test_hr_nested_in_div_is_collected():
    body = _parse_body("<div><p>A</p><hr/><p>B</p></div>")
    paragraphs, tags, images, _attrib = extract_plain_paragraphs(body)

    assert (paragraphs, tags, images) == (["A", "", "B"], ["p", "hr", "p"], {})
    assert len(paragraphs) == len(tags)


def test_hr_is_reemitted_at_its_position():
    body = _parse_body("<p>A</p><hr/><p>B</p>")
    replace_body_with_paragraphs(
        body, ["Traduction A", "", "Traduction B"], ["p", "hr", "p"], {}
    )

    assert len(body) == 3
    assert _local_tags(body) == ["p", "hr", "p"]

    hr = body[1]
    assert not hr.attrib
    assert not (hr.text or "")
    assert len(hr) == 0


def test_hr_is_never_sent_to_the_llm():
    segments = build_plain_segments(["A", "", "B"], 1800)

    for segment in segments:
        assert 1 not in segment["indices"], (
            "the void block's index must never be part of a segment sent to "
            "the LLM"
        )
        assert segment["text"].strip(), (
            "a segment must never carry an empty entry"
        )


def test_hr_survives_bilingual_rebuild():
    body = _parse_body("<p>A</p><hr/><p>B</p>")
    replace_body_with_paragraphs(
        body,
        ["Traduction A", "", "Traduction B"],
        ["p", "hr", "p"],
        {},
        bilingual=True,
        source_paragraphs=["A", "", "B"],
    )

    hr_children = [child for child in body if child.tag.split("}")[-1] == "hr"]
    assert len(hr_children) == 1

    hr = hr_children[0]
    assert hr.get("class") != "plain-text-target"
    assert hr.get("class") != "plain-text-source"

    # No plain-text-source twin was emitted for the void slot itself: A and B
    # each produce exactly one source twin, and the hr contributes none.
    tags_and_classes = [(child.tag.split("}")[-1], child.get("class")) for child in body]
    source_twins = [tc for tc in tags_and_classes if tc[1] == "plain-text-source"]
    assert len(source_twins) == 2, (
        "only A and B should have a plain-text-source twin; the void hr slot "
        f"must contribute none, got {tags_and_classes}"
    )


def test_hr_attributes_are_preserved():
    body = _parse_body('<p>A</p><hr class="scene"/><p>B</p>')
    paragraphs, tags, images, attrib = extract_plain_paragraphs(body)
    assert attrib == [{}, {"class": "scene"}, {}]

    replace_body_with_paragraphs(body, list(paragraphs), tags, images, paragraphs_attrib=attrib)

    hr_children = [child for child in body if child.tag.split("}")[-1] == "hr"]
    assert len(hr_children) == 1
    assert hr_children[0].attrib == {"class": "scene"}


# ---------------------------------------------------------------------------
# End-to-end: the real EPUB adapter path (plan Phase 2)
# ---------------------------------------------------------------------------

# A chapter carrying two scene separators: one at body level, one nested inside
# a <div> (the shape that only works because the void-block branch lives in the
# recursive `_collect_blocks`, see plan decision D6).
CHAPTER_WITH_HR_XHTML = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>第1章</title></head>'
    '<body>'
    '<h3>第1章</h3>\n'
    '<p>归墟，海中无底之谷。</p>\n'
    '<hr/>\n'
    '<p>他站在谷底，抬头看向天空。</p>\n'
    '<div>'
    '<p>换场之后的第一句。</p>\n'
    '<hr class="scene"/>\n'
    '<p>换场之后的第二句。</p>'
    '</div>'
    '</body></html>\n'
)


def _count_hr_elements(xhtml_text: str) -> int:
    """Count <hr> elements in a serialized XHTML document, namespace-agnostic."""
    parser = etree.XMLParser(recover=True, remove_blank_text=False)
    root = etree.fromstring(xhtml_text.encode("utf-8"), parser)
    return sum(
        1 for el in root.iter()
        if isinstance(el.tag, str) and (el.tag == "hr" or el.tag.endswith("}hr"))
    )


def _recording_llm_client(requests: List[str]):
    """Identity-echo stub that records the content of every request it gets.

    The client interface used by the pipeline is `await client.generate(user_prompt,
    system_prompt=...)`, with the translatable payload wrapped between
    INPUT_TAG_IN / INPUT_TAG_OUT inside `user_prompt`. We unwrap it the same way
    the echo stub does, so `requests` holds exactly what was submitted for
    translation -- which is what lets the test prove a void block is never billed.
    """
    client = _echo_llm_client()
    echo_generate = client.generate

    async def generate(user_prompt, system_prompt=None, **kwargs):
        start = user_prompt.find(INPUT_TAG_IN)
        end = user_prompt.find(INPUT_TAG_OUT)
        if start != -1 and end != -1:
            requests.append(user_prompt[start + len(INPUT_TAG_IN):end].strip("\n"))
        else:
            requests.append(user_prompt)
        return await echo_generate(user_prompt, system_prompt=system_prompt, **kwargs)

    client.generate = generate
    return client


@pytest.fixture
def hr_chapter_epub(tmp_path: Path) -> Path:
    """A real .epub whose only chapter carries the two scene separators above."""
    root = _build_cjk_epub_dir(tmp_path / "src_epub", REAL_CSS.read_text(encoding="utf-8"))
    _write(root / "OEBPS" / "Text" / "intro.xhtml", CHAPTER_WITH_HR_XHTML)
    return _zip_dir_as_epub(root, tmp_path / "input.epub")


@pytest.mark.asyncio
async def test_hr_survives_the_full_plain_text_epub_pipeline(hr_chapter_epub, tmp_path, monkeypatch):
    """The separators must survive the real adapter path, not just the helpers.

    Runs `translate_epub_file` with Plain Text Mode and an echo LLM stub, then
    compares the output chapter's <hr> count with the input's, and asserts the
    stub never received an empty request -- void blocks cost zero LLM calls.
    """
    requests: List[str] = []
    monkeypatch.setattr(
        translator_module, "_create_llm_client",
        lambda **kwargs: _recording_llm_client(requests),
    )
    _disable_attribution(monkeypatch)

    output_epub = tmp_path / "output_plain.epub"
    await translate_epub_file(
        input_filepath=str(hr_chapter_epub),
        output_filepath=str(output_epub),
        source_language="Chinese",
        target_language="French",
        prompt_options={"plain_text_mode": True},
    )

    with zipfile.ZipFile(hr_chapter_epub) as archive:
        input_text = archive.read("OEBPS/Text/intro.xhtml").decode("utf-8")
    with zipfile.ZipFile(output_epub) as archive:
        output_text = archive.read("OEBPS/Text/intro.xhtml").decode("utf-8")

    expected_hr = _count_hr_elements(input_text)
    assert expected_hr >= 2, "the fixture must carry at least two separators"
    assert _count_hr_elements(output_text) == expected_hr, (
        f"expected {expected_hr} <hr> in the translated chapter, got "
        f"{_count_hr_elements(output_text)}"
    )

    assert requests, "the stub LLM was never called — the fixture translated nothing"
    assert all(content.strip() for content in requests), (
        "a void block must never be submitted to the LLM; got an empty or "
        f"whitespace-only request among {requests}"
    )
