"""A rate limit during an EPUB or DOCX refine pass keeps the work already done.

The TXT and SRT refiners save partial output before letting RateLimitError
propagate to the pause logic. The EPUB and DOCX refiners used to drop it: the
EPUB temp directory was discarded and the DOCX never written.
"""

import pytest

import src.core.epub.chunk_refinement as chunk_refinement
import src.core.refine.docx_refiner as docx_refiner
import src.core.refine.epub_refiner as epub_refiner
from src.core.llm.exceptions import RateLimitError
from src.utils.file_utils import get_partial_output_path
from tests.characterization import fixtures
from tests.job_helpers import make_ctx, make_job


class _Client:
    async def close(self):
        pass


@pytest.mark.asyncio
async def test_refine_chunks_attaches_a_complete_partial_result(monkeypatch):
    async def fake_refine(self, idx, translated_text, chunk_dict):
        if idx == 1:
            raise RateLimitError("429", provider="test")
        return translated_text.upper()

    monkeypatch.setattr(chunk_refinement._ChunkRefiner, "refine", fake_refine)
    drafts = ["one", "two", "three"]
    chunks = [{"local_tag_map": {}, "global_indices": []} for _ in drafts]

    with pytest.raises(RateLimitError) as info:
        await chunk_refinement.refine_chunks(
            drafts, chunks, "French", "m", _Client(), None, ("[id", "]"), None, None)

    assert info.value.partial_result == ["ONE", "two", "three"]


class _Converter:
    def to_html(self, path):
        return "<p>Alpha paragraph.</p><p>Beta paragraph.</p>", {}

    def from_html(self, html, metadata, path, source_docx_path=None):
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)


@pytest.mark.asyncio
async def test_docx_refiner_writes_partial_output_then_reraises(tmp_path, monkeypatch):
    async def fake_refine(*, translated_chunks, **kwargs):
        error = RateLimitError("429", provider="test")
        error.partial_result = [text.replace("Alpha", "ALPHA") for text in translated_chunks]
        raise error

    monkeypatch.setattr(docx_refiner, "build_refine_client", lambda *a, **kw: (_Client(), None))
    monkeypatch.setattr(docx_refiner, "DocxHtmlConverter", _Converter)
    monkeypatch.setattr(docx_refiner, "refine_chunks", fake_refine)
    src = tmp_path / "in.docx"
    src.write_bytes(b"x")
    out = tmp_path / "out.docx"

    with pytest.raises(RateLimitError):
        await docx_refiner.refine_docx_file(
            make_job(str(src), str(out), operation="refine", target_language="French"),
            make_ctx(),
        )

    written = out.read_text(encoding="utf-8")
    assert "ALPHA paragraph." in written
    assert "Beta paragraph." in written


@pytest.mark.asyncio
async def test_epub_refiner_writes_partial_epub_then_reraises(tmp_path, monkeypatch):
    calls = []

    async def fake_refine_one(**kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RateLimitError("429", provider="test")
        return True

    monkeypatch.setattr(epub_refiner, "build_refine_client", lambda *a, **kw: (_Client(), None))
    monkeypatch.setattr(epub_refiner, "_refine_one_xhtml", fake_refine_one)
    src = fixtures.build_epub(tmp_path)
    out = tmp_path / "refined.epub"

    with pytest.raises(RateLimitError):
        await epub_refiner.refine_epub_file(
            make_job(str(src), str(out), operation="refine", target_language="French"),
            make_ctx(),
        )

    assert len(calls) == 2
    assert not out.exists()
    from pathlib import Path
    assert Path(get_partial_output_path(str(out))).exists()
