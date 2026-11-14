"""Plain Text Mode must not write a truncated answer as the translation.

Placeholder mode rejects a reply that lost most of the text (text-survival
guard), but the plain pipeline shared by EPUB, DOCX and PDF Plain Text Mode
had no such check: an answer keeping 10% of a paragraph was written to the
output and the job reported completed. Pre-existing on main.
"""
import pytest

import src.core.common.plain_text_pipeline as plain_pipeline

LONG = ("Alice was beginning to get very tired of sitting by her sister on the "
        "bank, and of having nothing to do: once or twice she had peeped into "
        "the book her sister was reading.")
SHORT = "A short line."


@pytest.mark.asyncio
async def test_truncated_answer_keeps_source_and_is_counted(monkeypatch):
    async def truncating(*, main_content, **kwargs):
        return "T::" + main_content[: len(main_content) // 10]

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", truncating)
    logs = []

    out, stats, interrupted = await plain_pipeline.translate_paragraphs_plain(
        [LONG], "English", "French", "m", None, 1000, prompt_options={},
        log_callback=lambda kind, msg, data=None: logs.append(kind),
    )

    assert out == [LONG]
    assert stats.failed_chunks == 1
    assert "plain_text_chunk_rejected" in logs
    assert not interrupted


@pytest.mark.asyncio
async def test_normal_and_short_answers_are_accepted(monkeypatch):
    async def translating(*, main_content, **kwargs):
        return "\n\n".join("T::" + p for p in main_content.split("\n\n"))

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", translating)

    out, stats, _ = await plain_pipeline.translate_paragraphs_plain(
        [LONG, SHORT], "English", "French", "m", None, 1000, prompt_options={},
    )

    assert out == ["T::" + LONG, "T::" + SHORT]
    assert stats.failed_chunks == 0
