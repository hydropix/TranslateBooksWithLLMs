"""DOCX refine-only must chunk with the caller's max_tokens_per_chunk.

refine_docx_file built a default TranslationContainer whose chunker used the
MAX_TOKENS_PER_CHUNK frozen at import, ignoring the max_tokens_per_chunk it
was given.
"""

import pytest

import src.core.refine.docx_refiner as docx_refiner

HTML = "".join(
    f"<p>Paragraph {k} has a handful of words so that it weighs a few tokens.</p>"
    for k in range(60)
)


class _Client:
    async def close(self):
        pass


class _Converter:
    def to_html(self, path):
        return HTML, {}

    def from_html(self, html, metadata, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)


async def _chunk_count(tmp_path, monkeypatch, max_tokens):
    seen = {}

    async def fake_refine(*, translated_chunks, chunks, **kwargs):
        seen["count"] = len(chunks)
        return translated_chunks

    monkeypatch.setattr(docx_refiner, "build_refine_client", lambda **kw: (_Client(), None))
    monkeypatch.setattr(docx_refiner, "DocxHtmlConverter", _Converter)
    monkeypatch.setattr(docx_refiner, "_refine_epub_chunks", fake_refine)
    src = tmp_path / "in.docx"
    src.write_bytes(b"x")

    ok = await docx_refiner.refine_docx_file(
        str(src), str(tmp_path / f"out{max_tokens}.docx"), "French",
        max_tokens_per_chunk=max_tokens,
    )
    assert ok is True
    return seen["count"]


@pytest.mark.asyncio
async def test_caller_chunk_size_wins(tmp_path, monkeypatch):
    small = await _chunk_count(tmp_path, monkeypatch, 60)
    large = await _chunk_count(tmp_path, monkeypatch, 5000)
    assert small > large >= 1
