"""The chunk precount prepares each source file once and hands it on.

``precount_chunks`` runs ``prepare_document`` on its own parse of every file;
the result is kept only for the files at or above the resume pointer (a
restored file's copy on disk is a translation), passed to the file's
``FileTask.prepared`` and released once used.
"""
import pytest

import src.core.epub.book_loop as book_loop
from src.core.epub.book_loop import FileResult, precount_chunks, process_content_files
from src.core.epub.book_state import BookRunState
from src.core.epub.run_context import EpubRun

from tests.job_helpers import make_ctx, make_job

DOC = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>
<body><p>{text}</p></body></html>
"""


def _write_book(tmp_path, count=3):
    hrefs = []
    for i in range(count):
        name = f"ch{i}.xhtml"
        (tmp_path / name).write_text(DOC.format(text=f"Paragraph number {i}."), encoding="utf-8")
        hrefs.append(name)
    return hrefs


@pytest.mark.asyncio
async def test_prepared_documents_are_kept_from_the_resume_pointer(tmp_path):
    hrefs = _write_book(tmp_path)
    precount = await precount_chunks(hrefs, str(tmp_path), 400, keep_prepared_from=1)

    assert precount.chunks_per_file == [1, 1, 1]
    assert precount.total_chunks == 3
    assert sorted(precount.prepared) == [1, 2]
    assert precount.prepared[1].chunk_count == 1


@pytest.mark.asyncio
async def test_plain_text_mode_keeps_nothing(tmp_path):
    hrefs = _write_book(tmp_path)
    precount = await precount_chunks(hrefs, str(tmp_path), 400,
                                     plain_text_mode=True, keep_prepared_from=0)
    assert precount.chunks_per_file == [1, 1, 1]
    assert precount.prepared == {}


@pytest.mark.asyncio
async def test_broken_and_missing_files_count_zero(tmp_path, monkeypatch):
    hrefs = _write_book(tmp_path, count=2)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(book_loop, "prepare_document", explode)
    precount = await precount_chunks(hrefs + ["missing.xhtml"], str(tmp_path), 400,
                                     keep_prepared_from=0)
    assert precount.chunks_per_file == [0, 0, 0]
    assert precount.prepared == {}


@pytest.mark.asyncio
async def test_each_file_receives_its_prepared_document_once(tmp_path, monkeypatch):
    hrefs = _write_book(tmp_path)
    precount = await precount_chunks(hrefs, str(tmp_path), 400, keep_prepared_from=0)
    expected = dict(precount.prepared)
    book = BookRunState(hrefs, str(tmp_path), str(tmp_path),
                        total_chunks=precount.total_chunks,
                        chunks_per_file=precount.chunks_per_file,
                        prepared=precount.prepared)
    seen = {}

    async def fake_translate(run, task):
        seen[task.href] = task.prepared
        # Released before the file is translated: memory does not hold it.
        assert hrefs.index(task.href) not in book.prepared
        return FileResult(None, False, None, None)

    monkeypatch.setattr(book_loop, "translate_xhtml_file", fake_translate)
    run = EpubRun(job=make_job(), ctx=make_ctx(), llm_client=object())
    await process_content_files(run, book)

    assert seen == {href: expected[i] for i, href in enumerate(hrefs)}
    assert book.prepared == {}
