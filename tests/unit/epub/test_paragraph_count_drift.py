"""
F6 detection test (plan/PLAN_CjkSourceRendering.md, Phase 7, §1.4): translate a
fixture with a stubbed LLM that returns each chunk verbatim, and assert the
output `<p>` count equals the input `<p>` count.

Purpose and limits. With a verbatim echo stub the reassembly must reproduce
every paragraph exactly, so this test makes any *future* reassembly-side
paragraph loss a hard failure. It does NOT attempt to fix the LLM-side merging
of very short lines observed in the real run (5 paragraphs lost across 28
chapters, ~0.2% -- see §1.4 of the plan): that loss happens because a real
model sometimes folds a very short source line into a neighboring one, which
cannot be reproduced by an echo stub that never rewrites anything, and fixing
it would be speculative.

Two paths are covered:

  - The default (placeholder) path. This is a genuine PASS: an empty `<p></p>`
    (a common spacer/formatting block) reduces to a zero-character
    placeholder-only chunk, which Phase 1's `is_text_free_chunk` guard passes
    through verbatim instead of sending it to the LLM, so the tag survives
    untouched. Verified empirically, not assumed.

  - Plain Text Mode (`prompt_options={'plain_text_mode': True}`). This used to
    reproduce a REAL reassembly-side loss with the verbatim stub, and was pinned
    here as a strict xfail: an empty `<p></p>` is extracted as the same kind of
    block as any other paragraph, is skipped when segments are built (nothing to
    send to the LLM), and comes back as an empty string;
    `replace_body_with_paragraphs` then did `if text:` before emitting the block,
    so the empty paragraph was silently dropped from the rebuilt body -- one
    `<p>` fewer in the output than in the input, deterministically, with no LLM
    behaviour involved. That is the mechanism Phase 2 named in
    plan/PLAN_CjkSourceRendering.md §5.1 ("F2 and F6 have the same origin").
    **Fixed** in `replace_body_with_paragraphs`
    (src/core/epub/plain_extractor.py): an empty translation now falls back to
    the source text, and when both are empty an empty block is emitted, so no
    paragraph is ever deleted. The test below is therefore a normal passing
    test guarding that fix. See
    `tests/unit/epub/test_issue_207_content_loss.py` (section D) for the
    unit-level coverage of the same two guards.
"""
import zipfile
from pathlib import Path

import pytest
from lxml import etree

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


# A chapter body mixing ordinary paragraphs, the very-short reaction lines
# called out in F6 ("“嗯？”", "……"), and one
# completely empty `<p></p>` -- a spacer/formatting block, verbatim from how
# CJK web-novel EPUBs commonly separate scenes.
CHAPTER_XHTML = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>第1章</title></head>'
    '<body>'
    '<h3>第1章</h3>\n'
    '<p>归墟，海中无底之谷。</p>\n'
    '<p></p>\n'
    '<p>他站在谷底，抬头看向天空。</p>\n'
    '<p>“嗯？”</p>\n'
    '<p>……</p>'
    '</body></html>\n'
)


def _stub_create_llm_client(*args, **kwargs):
    return _echo_llm_client()


def _count_p_elements(xhtml_text: str) -> int:
    """Count <p> elements in a serialized XHTML document, namespace-agnostic."""
    parser = etree.XMLParser(recover=True, remove_blank_text=False)
    root = etree.fromstring(xhtml_text.encode("utf-8"), parser)
    return sum(
        1 for el in root.iter()
        if isinstance(el.tag, str) and (el.tag == "p" or el.tag.endswith("}p"))
    )


async def _translate(input_epub: Path, output_epub: Path, monkeypatch, **overrides) -> None:
    monkeypatch.setattr(translator_module, "create_llm_client", _stub_create_llm_client)
    _disable_attribution(monkeypatch)
    await translate_epub_file(
        make_job(str(input_epub), str(output_epub),
                 source_language="Chinese", target_language="French", **overrides),
        make_ctx(),
    )


@pytest.fixture
def chapter_epub(tmp_path: Path) -> Path:
    """The reported book's container shape, with the chapter body above."""
    root = _build_cjk_epub_dir(tmp_path / "src_epub", REAL_CSS.read_text(encoding="utf-8"))
    _write(root / "OEBPS" / "Text" / "intro.xhtml", CHAPTER_XHTML)
    return _zip_dir_as_epub(root, tmp_path / "input.epub")


@pytest.mark.asyncio
async def test_default_path_preserves_paragraph_count(chapter_epub, tmp_path, monkeypatch):
    """Placeholder path: a verbatim stub must reproduce every <p>, including
    the empty spacer paragraph (handled by Phase 1's text-free-chunk guard)."""
    output_epub = tmp_path / "output.epub"
    await _translate(chapter_epub, output_epub, monkeypatch)

    with zipfile.ZipFile(chapter_epub) as archive:
        input_text = archive.read("OEBPS/Text/intro.xhtml").decode("utf-8")
    with zipfile.ZipFile(output_epub) as archive:
        output_text = archive.read("OEBPS/Text/intro.xhtml").decode("utf-8")

    assert _count_p_elements(output_text) == _count_p_elements(input_text)


@pytest.mark.asyncio
async def test_plain_text_mode_preserves_the_empty_paragraph(chapter_epub, tmp_path, monkeypatch):
    """Plain Text Mode: the empty spacer <p></p> must survive the rebuild.

    Was a strict xfail while the loss stood; now guards the fix in
    `replace_body_with_paragraphs` (src/core/epub/plain_extractor.py), which no
    longer omits a block whose translation came back empty.
    """
    output_epub = tmp_path / "output_plain.epub"
    await _translate(
        chapter_epub, output_epub, monkeypatch,
        prompt_options={"plain_text_mode": True},
    )

    with zipfile.ZipFile(chapter_epub) as archive:
        input_text = archive.read("OEBPS/Text/intro.xhtml").decode("utf-8")
    with zipfile.ZipFile(output_epub) as archive:
        output_text = archive.read("OEBPS/Text/intro.xhtml").decode("utf-8")

    assert _count_p_elements(output_text) == _count_p_elements(input_text)
