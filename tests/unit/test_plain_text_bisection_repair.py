"""
Issue #259: the paragraph-count repair must locate the divergence.

The old fallback re-translated every paragraph of a mismatched segment, one
call each (~53 calls on an average segment), even though most mismatches are
off by exactly one paragraph. The bisection repair splits the segment, accepts
every half that comes back with its stated count, and only narrows down on the
half that still mismatches. These tests pin the call counts for a +1, a -1 and
a large mismatch, and that the output stays aligned to the source.
"""
import re

import pytest

import src.core.common.plain_text_pipeline as plain_pipeline
from src.prompts.prompts import PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION


SEGMENT_SIZE = 53
SOURCE = [f"Source paragraph number {i}." for i in range(SEGMENT_SIZE)]
EXPECTED = [f"T::{p}" for p in SOURCE]


def _split(text):
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


class _FakeModel:
    """Translates 1:1 except on multi-paragraph calls that cover a bad paragraph.

    mode='drop'  : the bad paragraph is folded into its neighbour (-1)
    mode='split' : the bad paragraph comes back as two paragraphs (+1)
    A single-paragraph call always translates correctly, so the repair can
    always finish. Every call is recorded.
    """

    def __init__(self, bad, mode):
        self.bad = {SOURCE[i] for i in bad}
        self.mode = mode
        self.range_calls = []
        self.one_calls = []

    def _answer(self, paragraphs):
        out = ["T::" + p for p in paragraphs]
        if len(paragraphs) == 1:
            return out[0]
        # Right to left so earlier positions stay valid while editing.
        for k in reversed([k for k, p in enumerate(paragraphs) if p in self.bad]):
            if len(out) == 1:
                break
            if self.mode == 'split':
                out[k:k + 1] = [out[k] + " (a)", "(b)"]
            elif k > 0:
                out[k - 1] += " " + out.pop(k)
            else:
                out[0] += " " + out.pop(1)
        return "\n\n".join(out)

    async def translate_range(self, text, before, after, options):
        self.range_calls.append({
            'text': text,
            'count': (options or {}).get(PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION),
            'before': before,
            'after': after,
        })
        return self._answer(_split(text))

    async def translate_one(self, text, before, after):
        self.one_calls.append(text)
        return "T::" + text.strip()

    @property
    def total(self):
        return len(self.range_calls) + len(self.one_calls)


async def _repair(model, source=SOURCE):
    segment = {'indices': list(range(len(source))), 'text': "\n\n".join(source), 'partial': False}
    return await plain_pipeline._repair_segment_by_bisection(
        segment,
        source,
        translate_range=model.translate_range,
        translate_one=model.translate_one,
        prompt_options={'plain_text_mode': True},
        log_callback=None,
        segment_index=0,
    )


@pytest.fixture(autouse=True)
def _no_cleanup(monkeypatch):
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s: s)


def _worst_case_bound(n):
    return plain_pipeline.bisection_repair_call_budget(n)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["drop", "split"])
@pytest.mark.parametrize("bad", [0, 20, 26, 27, 52])
async def test_off_by_one_is_repaired_in_a_dozen_calls(mode, bad):
    """A -1 or +1 on 53 paragraphs costs at most 12 calls, not 53."""
    model = _FakeModel(bad=[bad], mode=mode)

    joined, failed, calls = await _repair(model)

    assert joined.split("\n\n") == EXPECTED
    assert failed == 0
    assert calls == model.total
    # Two calls per bisection level (53 -> 27 -> 14 -> 7 -> 4 -> 2) at most,
    # the last two being single-paragraph calls.
    assert calls <= 12
    # Every range call states the count its answer must have.
    assert all(call['count'] == len(_split(call['text'])) for call in model.range_calls)


@pytest.mark.asyncio
async def test_halves_that_obey_the_count_cost_two_calls():
    """When the model behaves on smaller inputs, the repair is two calls."""
    model = _FakeModel(bad=[], mode='drop')

    joined, failed, calls = await _repair(model)

    assert joined.split("\n\n") == EXPECTED
    assert calls == 2
    assert [call['count'] for call in model.range_calls] == [26, 27]
    assert model.one_calls == []


@pytest.mark.asyncio
async def test_halves_carry_their_neighbours_as_context():
    model = _FakeModel(bad=[], mode='drop')

    await _repair(model)

    left, right = model.range_calls
    assert (left['before'], left['after']) == ("", SOURCE[26])
    assert (right['before'], right['after']) == (SOURCE[25], "")


@pytest.mark.asyncio
async def test_large_mismatch_from_one_folded_block_stays_cheap():
    """A front-matter block of ten paragraphs run together (-9): a few calls."""
    block = set(SOURCE[30:40])

    class _BlockFolder(_FakeModel):
        def _answer(self, paragraphs):
            out = ["T::" + p for p in paragraphs]
            if block <= set(paragraphs):
                k = paragraphs.index(SOURCE[30])
                out[k:k + 10] = [" ".join(out[k:k + 10])]
            return "\n\n".join(out)

    model = _BlockFolder(bad=[], mode='drop')

    joined, failed, calls = await _repair(model)

    assert joined.split("\n\n") == EXPECTED
    assert failed == 0
    assert calls <= 6


@pytest.mark.asyncio
async def test_ten_stubborn_paragraphs_still_cost_less_than_the_old_repair():
    """Ten consecutive paragraphs that each break any multi-paragraph call."""
    model = _FakeModel(bad=range(30, 40), mode='drop')

    joined, failed, calls = await _repair(model)

    assert joined.split("\n\n") == EXPECTED
    assert failed == 0
    # Ten paragraphs need their own call each; the rest is resolved in halves.
    assert calls < SEGMENT_SIZE * 2 // 3


@pytest.mark.asyncio
async def test_scattered_mismatches_never_exceed_the_bound():
    """Bad paragraphs everywhere: never worse than the old repair plus 2 per level."""
    model = _FakeModel(bad=range(0, SEGMENT_SIZE, 5), mode='split')

    joined, failed, calls = await _repair(model)

    assert joined.split("\n\n") == EXPECTED
    assert calls <= _worst_case_bound(SEGMENT_SIZE)


@pytest.mark.asyncio
async def test_systematically_failing_model_is_bounded():
    """A model that merges every multi-paragraph call cannot blow up the cost."""
    model = _FakeModel(bad=range(SEGMENT_SIZE), mode='drop')

    joined, failed, calls = await _repair(model)

    assert joined.split("\n\n") == EXPECTED
    # The budget switches to paragraph by paragraph before it can be exceeded.
    assert calls <= _worst_case_bound(SEGMENT_SIZE)
    assert len(model.one_calls) == SEGMENT_SIZE


@pytest.mark.asyncio
async def test_small_segment_goes_straight_to_per_paragraph():
    """Bisecting three paragraphs cannot beat three calls."""
    source = SOURCE[:3]
    model = _FakeModel(bad=[1], mode='drop')

    joined, failed, calls = await _repair(model, source)

    assert joined.split("\n\n") == [f"T::{p}" for p in source]
    assert calls == 3
    assert model.range_calls == []


@pytest.mark.asyncio
async def test_failed_paragraph_keeps_source_and_is_counted():
    source = SOURCE[:8]
    model = _FakeModel(bad=range(8), mode='drop')

    async def failing_one(text, before, after):
        model.one_calls.append(text)
        return None if text == source[5] else "T::" + text

    model.translate_one = failing_one

    joined, failed, _ = await _repair(model, source)

    expected = [f"T::{p}" for p in source]
    expected[5] = source[5]
    assert joined.split("\n\n") == expected
    assert failed == 1


@pytest.mark.asyncio
async def test_pipeline_uses_bisection_and_logs_the_repair_cost(monkeypatch):
    """End to end: one mismatched 53-paragraph segment, a -1 the model repeats."""
    bad = SOURCE[17]
    calls = []

    async def fake_request(*, main_content, prompt_options=None, **kwargs):
        calls.append(main_content)
        paragraphs = _split(main_content)
        if len(paragraphs) > 1 and bad in paragraphs:
            k = paragraphs.index(bad)
            out = ["T::" + p for p in paragraphs]
            if k > 0:
                out[k - 1] += " " + out.pop(k)
            else:
                out[0] += " " + out.pop(1)
            return "\n\n".join(out)
        return "\n\n".join("T::" + p for p in paragraphs)

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake_request)

    logs = []
    out, stats, interrupted = await plain_pipeline.translate_paragraphs_plain(
        paragraphs=SOURCE,
        source_language="English",
        target_language="French",
        model_name="m",
        llm_client=object(),
        max_tokens_per_chunk=100000,
        parallel_workers=1,
        log_callback=lambda key, msg: logs.append((key, msg)),
    )

    assert not interrupted
    assert out == EXPECTED
    assert stats.paragraph_count_mismatches == 1
    assert stats.paragraph_repair_failed == 0
    # First attempt + whole-segment retry + the bisection repair.
    assert len(calls) <= 2 + 12
    done = [msg for key, msg in logs if key == "plain_text_paragraph_repair_done"]
    assert len(done) == 1
    assert f"in {len(calls) - 2} extra call(s)" in done[0]
