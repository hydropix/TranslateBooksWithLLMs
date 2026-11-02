"""Regression tests for issue #302: SRT refine lost its work on a rate limit.

`refine_srt_file` let `RateLimitError` propagate bare, so a 429 midway through
a refine pass threw away every cue already refined and a resume started over
from the first cue. The refiner now writes the partial file (refined cues up
to the failing block, the rest untouched), records the last finished cue in
the checkpoint, re-raises so the pause still triggers, and a resume reuses the
finished cues instead of sending them to the LLM again.
"""

import re

import pytest

from src.config import INPUT_TAG_IN, INPUT_TAG_OUT, TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.llm.exceptions import RateLimitError
from src.core.srt_processor import SRTProcessor


# Gapped numbering and two-cue blocks: 6 cues -> 3 blocks of 2.
CUES = [
    (str(10 * (i + 1)), f"00:00:0{i},000", f"00:00:0{i},900", f"line {i}")
    for i in range(6)
]
BLOCK_SIZE = 2

_ZERO_WIDTH = re.compile('[​‌‍⁠﻿]')


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLMClient:
    """Uppercases every [N]text line it is sent; 429s on call `fail_at_call`.

    `seen` records the subtitle texts of every block received, so a test can
    tell which cues were actually sent to the LLM.
    """

    def __init__(self, fail_at_call=None):
        self.fail_at_call = fail_at_call
        self.calls = 0
        self.seen = []

    async def make_request(self, prompt, model_name, system_prompt=None):
        call = self.calls
        self.calls += 1
        if call == self.fail_at_call:
            raise RateLimitError("429 Too Many Requests", provider="test")
        match = re.search(
            re.escape(INPUT_TAG_IN) + r"\n(.*?)\n" + re.escape(INPUT_TAG_OUT),
            prompt,
            re.DOTALL,
        )
        assert match, "user prompt does not contain a SOURCE_TEXT block"
        refined_lines = []
        for line in match.group(1).split("\n"):
            marker = re.match(r"^(\[\d+\])(.*)$", line)
            if marker:
                self.seen.append(marker.group(2))
                refined_lines.append(f"{marker.group(1)}{marker.group(2).upper()}")
        body = "\n".join(refined_lines)
        return FakeLLMResponse(f"{TRANSLATE_TAG_IN}\n{body}\n{TRANSLATE_TAG_OUT}")

    def extract_translation(self, content: str):
        match = re.search(
            re.escape(TRANSLATE_TAG_IN) + r"\n(.*?)\n" + re.escape(TRANSLATE_TAG_OUT),
            content,
            re.DOTALL,
        )
        return match.group(1) if match else content

    async def close(self):
        pass


class FakeCheckpointManager:
    def __init__(self):
        self.saved = []

    def save_checkpoint(self, translation_id, chunk_index, **kwargs):
        self.saved.append({'translation_id': translation_id,
                           'chunk_index': chunk_index, **kwargs})
        return True


def _build_srt(cues):
    return "\n".join(f"{n}\n{start} --> {end}\n{text}\n" for n, start, end, text in cues)


def _parse(path):
    subs = SRTProcessor().parse_srt(path.read_text(encoding="utf-8"))
    return [(s['number'], s['start_time'], s['end_time'], _ZERO_WIDTH.sub("", s['text']))
            for s in subs]


async def _refine(tmp_path, monkeypatch, client, **kwargs):
    import src.core.refine.srt_refiner as srt_refiner

    monkeypatch.setattr(srt_refiner, "create_llm_client", lambda *a, **kw: client)
    monkeypatch.setattr(srt_refiner, "SRT_LINES_PER_BLOCK", BLOCK_SIZE)

    input_file = tmp_path / "input.srt"
    output_file = tmp_path / "output.srt"
    input_file.write_text(_build_srt(CUES), encoding="utf-8")
    log_events = []
    result = await srt_refiner.refine_srt_file(
        input_filepath=str(input_file),
        output_filepath=str(output_file),
        target_language="English",
        log_callback=lambda event, message, **kw: log_events.append(event),
        **kwargs,
    )
    return result, output_file, log_events


@pytest.mark.asyncio
async def test_rate_limit_writes_partial_srt_and_re_raises(tmp_path, monkeypatch):
    checkpoint = FakeCheckpointManager()
    client = FakeLLMClient(fail_at_call=1)  # second block hits the 429

    with pytest.raises(RateLimitError):
        await _refine(tmp_path, monkeypatch, client,
                      checkpoint_manager=checkpoint, translation_id="job-1")

    output_file = tmp_path / "output.srt"
    assert output_file.exists()
    expected = [
        (n, start, end, text.upper() if i < BLOCK_SIZE else text)
        for i, (n, start, end, text) in enumerate(CUES)
    ]
    # Refined first block, untouched rest, numbering and timings intact.
    assert _parse(output_file) == expected

    assert len(checkpoint.saved) == 1
    saved = checkpoint.saved[0]
    assert saved['translation_id'] == "job-1"
    assert saved['chunk_index'] == BLOCK_SIZE - 1
    assert saved['original_text'] == CUES[BLOCK_SIZE - 1][3]
    assert saved['translated_text'] == CUES[BLOCK_SIZE - 1][3].upper()
    assert saved['completed_chunks'] == BLOCK_SIZE
    assert saved['total_chunks'] == len(CUES)


@pytest.mark.asyncio
async def test_rate_limit_on_first_block_saves_drafts_without_checkpoint(
    tmp_path, monkeypatch
):
    checkpoint = FakeCheckpointManager()

    with pytest.raises(RateLimitError):
        await _refine(tmp_path, monkeypatch, FakeLLMClient(fail_at_call=0),
                      checkpoint_manager=checkpoint, translation_id="job-1")

    assert _parse(tmp_path / "output.srt") == CUES
    assert checkpoint.saved == []


@pytest.mark.asyncio
async def test_resume_reuses_finished_cues(tmp_path, monkeypatch):
    with pytest.raises(RateLimitError):
        await _refine(tmp_path, monkeypatch, FakeLLMClient(fail_at_call=1),
                      checkpoint_manager=FakeCheckpointManager(),
                      translation_id="job-1")

    stats = []
    client = FakeLLMClient()
    result, output_file, log_events = await _refine(
        tmp_path, monkeypatch, client,
        resume_from_index=BLOCK_SIZE, stats_callback=stats.append)

    assert result is True
    # Only the cues left over by the paused pass reach the LLM.
    assert client.seen == [text for _, _, _, text in CUES[BLOCK_SIZE:]]
    assert _parse(output_file) == [
        (n, start, end, text.upper()) for n, start, end, text in CUES
    ]
    assert "srt_refine_resume" in log_events
    # Progress is reported against the whole file.
    assert stats[0]['completed_chunks'] == BLOCK_SIZE
    assert stats[-1]['completed_chunks'] == len(CUES)
    assert all(s['total_chunks'] == len(CUES) for s in stats)


@pytest.mark.asyncio
async def test_resume_with_mismatched_output_restarts(tmp_path, monkeypatch):
    # A stale output that no longer lines up with the input cue for cue.
    (tmp_path / "output.srt").write_text(_build_srt(CUES[:3]), encoding="utf-8")

    client = FakeLLMClient()
    result, output_file, log_events = await _refine(
        tmp_path, monkeypatch, client, resume_from_index=BLOCK_SIZE)

    assert result is True
    assert "srt_refine_resume_restart" in log_events
    assert client.seen == [text for _, _, _, text in CUES]
    assert _parse(output_file) == [
        (n, start, end, text.upper()) for n, start, end, text in CUES
    ]
