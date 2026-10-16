"""Regression tests for issue #205: SRT refine cues mis-assigned.

The refine-only path used to key translations by the cue number printed in
the file while update_translated_subtitles applies them by list position.
Any SRT not numbered exactly 1..N (gaps, restarts, 0-based numbering) got
its refined text assigned to the wrong cue / timestamp.
"""

import re

import pytest

from src.config import INPUT_TAG_IN, INPUT_TAG_OUT, TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.srt_processor import SRTProcessor


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLMClient:
    """Echoes back the input block with every subtitle text uppercased.

    Parses the [N]text lines between the SOURCE_TEXT tags of the user
    prompt, so the refined output keeps the exact [N] markers the caller
    sent and only changes the casing of the text.
    """

    async def make_request(self, prompt, model_name, system_prompt=None):
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
                refined_lines.append(f"{marker.group(1)}{marker.group(2).upper()}")
            else:
                refined_lines.append(line.upper())
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


def _build_srt(cues):
    """cues: list of (number, start, end, text) tuples."""
    blocks = []
    for number, start, end, text in cues:
        blocks.append(f"{number}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


_ZERO_WIDTH = re.compile('[\u200b\u200c\u200d\u2060\ufeff]')


def _parse_output(content):
    """Map start_time -> text for every cue of a generated SRT file.

    Zero-width characters are stripped: reconstruct_srt may add invisible
    normalization characters to the first cue, which are irrelevant to the
    cue/text assignment under test.
    """
    processor = SRTProcessor()
    return {
        sub["start_time"]: _ZERO_WIDTH.sub("", sub["text"])
        for sub in processor.parse_srt(content)
    }


async def _refine(tmp_path, monkeypatch, srt_content):
    import src.core.refine.srt_refiner as srt_refiner

    monkeypatch.setattr(
        srt_refiner, "create_llm_client", lambda *args, **kwargs: FakeLLMClient()
    )

    input_file = tmp_path / "input.srt"
    output_file = tmp_path / "output.srt"
    input_file.write_text(srt_content, encoding="utf-8")

    ok = await srt_refiner.refine_srt_file(
        input_filepath=str(input_file),
        output_filepath=str(output_file),
        target_language="French",
    )
    assert ok, "refine_srt_file reported failure"
    return output_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_refine_srt_gapped_numbering(tmp_path, monkeypatch):
    """Cue numbers with a gap (1, 5, 6) must keep text under its own timestamp."""
    srt = _build_srt([
        ("1", "00:00:01,000", "00:00:02,000", "alpha one"),
        ("5", "00:00:03,000", "00:00:04,000", "bravo two"),
        ("6", "00:00:05,000", "00:00:06,000", "charlie three"),
    ])

    output = await _refine(tmp_path, monkeypatch, srt)
    by_start = _parse_output(output)

    assert by_start["00:00:01,000"] == "ALPHA ONE"
    assert by_start["00:00:03,000"] == "BRAVO TWO"
    assert by_start["00:00:05,000"] == "CHARLIE THREE"


@pytest.mark.asyncio
async def test_refine_srt_zero_based_numbering(tmp_path, monkeypatch):
    """0-based cue numbers must not shift texts (cue 0 used to land on the last cue)."""
    srt = _build_srt([
        ("0", "00:00:01,000", "00:00:02,000", "alpha one"),
        ("1", "00:00:03,000", "00:00:04,000", "bravo two"),
        ("2", "00:00:05,000", "00:00:06,000", "charlie three"),
    ])

    output = await _refine(tmp_path, monkeypatch, srt)
    by_start = _parse_output(output)

    assert by_start["00:00:01,000"] == "ALPHA ONE"
    assert by_start["00:00:03,000"] == "BRAVO TWO"
    assert by_start["00:00:05,000"] == "CHARLIE THREE"


@pytest.mark.asyncio
async def test_refine_srt_restarted_numbering(tmp_path, monkeypatch):
    """Duplicate cue numbers (restart at 1) must not collapse or overwrite cues."""
    srt = _build_srt([
        ("1", "00:00:01,000", "00:00:02,000", "alpha one"),
        ("2", "00:00:03,000", "00:00:04,000", "bravo two"),
        ("1", "00:01:01,000", "00:01:02,000", "charlie three"),
        ("2", "00:01:03,000", "00:01:04,000", "delta four"),
    ])

    output = await _refine(tmp_path, monkeypatch, srt)
    by_start = _parse_output(output)

    assert by_start["00:00:01,000"] == "ALPHA ONE"
    assert by_start["00:00:03,000"] == "BRAVO TWO"
    assert by_start["00:01:01,000"] == "CHARLIE THREE"
    assert by_start["00:01:03,000"] == "DELTA FOUR"


def test_update_translated_subtitles_ignores_negative_index():
    """A negative index must not silently write to the end of the list."""
    processor = SRTProcessor()
    subtitles = [
        {"number": "0", "start_time": "00:00:01,000", "end_time": "00:00:02,000",
         "text": "first", "original_text": "first"},
        {"number": "1", "start_time": "00:00:03,000", "end_time": "00:00:04,000",
         "text": "last", "original_text": "last"},
    ]

    updated = processor.update_translated_subtitles(subtitles, {-1: "INTRUDER"})

    assert updated[-1]["text"] == "last"


class RateLimitedLLMClient(FakeLLMClient):
    """Refines the first block, then answers every later call with a 429."""

    def __init__(self):
        self.calls = 0

    async def make_request(self, prompt, model_name, system_prompt=None):
        from src.core.llm.exceptions import RateLimitError

        self.calls += 1
        if self.calls > 1:
            raise RateLimitError("429 Too Many Requests", provider="test")
        return await super().make_request(prompt, model_name, system_prompt)


@pytest.mark.asyncio
async def test_refine_srt_rate_limit_saves_refined_cues(tmp_path, monkeypatch):
    """Issue #302: a 429 mid-pass must write the cues already refined (the rest
    unchanged) and still re-raise so auto-pause triggers."""
    import src.core.refine.srt_refiner as srt_refiner
    from src.core.llm.exceptions import RateLimitError

    monkeypatch.setattr(srt_refiner, "SRT_LINES_PER_BLOCK", 1)
    monkeypatch.setattr(
        srt_refiner, "create_llm_client", lambda *args, **kwargs: RateLimitedLLMClient()
    )

    srt = _build_srt([
        ("1", "00:00:01,000", "00:00:02,000", "alpha one"),
        ("2", "00:00:03,000", "00:00:04,000", "bravo two"),
        ("3", "00:00:05,000", "00:00:06,000", "charlie three"),
    ])
    input_file = tmp_path / "input.srt"
    output_file = tmp_path / "output.srt"
    input_file.write_text(srt, encoding="utf-8")

    with pytest.raises(RateLimitError):
        await srt_refiner.refine_srt_file(
            input_filepath=str(input_file),
            output_filepath=str(output_file),
            target_language="French",
        )

    by_start = _parse_output(output_file.read_text(encoding="utf-8"))
    assert by_start == {
        "00:00:01,000": "ALPHA ONE",
        "00:00:03,000": "bravo two",
        "00:00:05,000": "charlie three",
    }
