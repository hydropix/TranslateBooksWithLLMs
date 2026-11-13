"""Plain Text Mode pipeline: rate limit during repair, padding, <pre> blocks.

- A RateLimitError raised by the mismatch retry / bisection repair (called
  from the consumer loop, outside the concurrent scheduler) escaped without
  running the checkpoint hook or setting the partial result.
- _reconcile_paragraph_counts padded missing paragraphs with "", which
  _reassemble kept as empty paragraphs instead of the source text.
- A <pre> paragraph holding blank lines always broke the paragraph-count
  contract and had its blank lines collapsed.
"""

import re

import pytest

import src.core.common.plain_text_pipeline as plain_pipeline
from src.core.llm.exceptions import RateLimitError

PARAGRAPHS = ["Alpha paragraph.", "Beta paragraph.", "Gamma paragraph."]


def test_missing_paragraphs_keep_their_source_text():
    segments = [{'indices': [0, 1, 2], 'text': "\n\n".join(PARAGRAPHS), 'partial': False}]
    out = plain_pipeline._reassemble(segments, ["T::Alpha paragraph."], PARAGRAPHS)
    assert out == ["T::Alpha paragraph.", "Beta paragraph.", "Gamma paragraph."]


@pytest.mark.asyncio
async def test_rate_limit_during_repair_runs_the_checkpoint_hook(monkeypatch):
    async def merging_then_429(*, main_content, **kwargs):
        paragraphs = [p for p in re.split(r"\n{2,}", main_content) if p.strip()]
        if kwargs.get("prompt_options", {}).get(
                plain_pipeline.PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION):
            raise RateLimitError("429", provider="test")
        return "T::" + " ".join(paragraphs)  # wrong count: triggers the retry

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", merging_then_429)
    hooks = []

    with pytest.raises(RateLimitError) as info:
        await plain_pipeline.translate_paragraphs_plain(
            PARAGRAPHS, "English", "French", "m", None, 1000,
            prompt_options={},
            checkpoint_hook=lambda segments, prefix, next_index, stats, **kw: hooks.append(next_index),
        )

    assert hooks == [0]  # nothing committed: the segment is retried on resume
    assert info.value.partial_result == PARAGRAPHS


@pytest.mark.asyncio
async def test_pre_block_with_blank_lines_keeps_them(monkeypatch):
    pre = "def f():\n    return 1\n\n\ndef g():\n    return 2"
    source = ["Intro paragraph.", pre, "Outro paragraph."]

    async def per_paragraph(*, main_content, **kwargs):
        if main_content == pre:
            return main_content  # code comes back verbatim
        paragraphs = [p for p in re.split(r"\n{2,}", main_content) if p.strip()]
        return "\n\n".join("T::" + p for p in paragraphs)

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", per_paragraph)
    logs = []

    out, stats, interrupted = await plain_pipeline.translate_paragraphs_plain(
        source, "English", "French", "m", None, 1000, prompt_options={},
        log_callback=lambda kind, msg, data=None: logs.append(kind),
    )

    assert out == ["T::Intro paragraph.", pre, "T::Outro paragraph."]
    assert stats.paragraph_count_mismatches == 0
    assert "plain_text_paragraph_mismatch" not in logs
