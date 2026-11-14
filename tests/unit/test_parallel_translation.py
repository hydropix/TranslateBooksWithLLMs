"""
Unit tests for parallel chunk translation.

Covers:
- resolve_parallel_workers() provider gating and clamping
- the ordered-window helper (gather_window / iter_ordered_windows)
- translate_paragraphs_plain() in sequential vs parallel mode: identical
  ordered output, real concurrency, context chaining only when sequential,
  and graceful failure handling.
"""
import asyncio

import pytest

from src.config import resolve_parallel_workers, MAX_PARALLEL_TRANSLATIONS
from src.core.common.parallel import (
    aclosing,
    gather_window,
    iter_ordered_windows,
    iter_ordered_concurrent,
)
import src.core.common.plain_text_pipeline as plain_pipeline


# ---------------------------------------------------------------------------
# resolve_parallel_workers
# ---------------------------------------------------------------------------
class TestResolveParallelWorkers:
    def test_local_provider_forced_to_one(self):
        assert resolve_parallel_workers("ollama", 8) == 1
        assert resolve_parallel_workers("OLLAMA", 8) == 1

    def test_cloud_provider_honors_request(self):
        assert resolve_parallel_workers("openrouter", 4) == 4

    def test_none_falls_back_to_default(self):
        # Default PARALLEL_TRANSLATIONS is 1 unless overridden via env.
        assert resolve_parallel_workers("openrouter", None) >= 1

    def test_clamped_to_window(self):
        assert resolve_parallel_workers("openrouter", 9999) == MAX_PARALLEL_TRANSLATIONS
        assert resolve_parallel_workers("openrouter", 0) == 1
        assert resolve_parallel_workers("openrouter", -5) == 1

    def test_malformed_value(self):
        assert resolve_parallel_workers("openrouter", "abc") == 1


# ---------------------------------------------------------------------------
# Ordered-window helper
# ---------------------------------------------------------------------------
class TestOrderedWindow:
    @pytest.mark.asyncio
    async def test_gather_window_preserves_order_and_captures_exceptions(self):
        async def worker(x):
            await asyncio.sleep(0)
            if x == 2:
                raise ValueError("boom")
            return x * 10

        results = await gather_window([0, 1, 2, 3], worker)
        assert results[0] == 0
        assert results[1] == 10
        assert isinstance(results[2], ValueError)
        assert results[3] == 30

    @pytest.mark.asyncio
    async def test_iter_ordered_windows_batches(self):
        async def worker(x):
            return x

        seen = []
        async for window in iter_ordered_windows(list(range(5)), 2, worker):
            seen.append([idx for idx, _ in window])
        assert seen == [[0, 1], [2, 3], [4]]

    @pytest.mark.asyncio
    async def test_window_size_one_is_sequential(self):
        order = []

        async def worker(x):
            order.append(x)
            return x

        async for _ in iter_ordered_windows([0, 1, 2], 1, worker):
            pass
        assert order == [0, 1, 2]


# ---------------------------------------------------------------------------
# Continuous-concurrency scheduler
# ---------------------------------------------------------------------------
class TestOrderedConcurrent:
    @pytest.mark.asyncio
    async def test_yields_in_order_despite_out_of_order_completion(self):
        # Item 0 is slowest; it must still be yielded first.
        delays = {0: 0.05, 1: 0.0, 2: 0.0, 3: 0.0}

        async def worker(x):
            await asyncio.sleep(delays[x])
            return x

        seen = []
        async for item, result in iter_ordered_concurrent([0, 1, 2, 3], 4, worker):
            assert item == result
            seen.append(item)
        assert seen == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_keeps_pool_full(self):
        in_flight = 0
        max_in_flight = 0

        async def worker(x):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return x

        async for _ in iter_ordered_concurrent(list(range(12)), 4, worker):
            pass
        assert max_in_flight == 4

    @pytest.mark.asyncio
    async def test_workers_one_is_strictly_sequential(self):
        # With workers=1, the next item must only start after the previous one
        # was yielded (so a caller mutating shared state between yields is safe).
        events = []

        async def worker(x):
            events.append(("start", x))
            await asyncio.sleep(0)
            return x

        async for item, _ in iter_ordered_concurrent([0, 1, 2], 1, worker):
            events.append(("yield", item))

        assert events == [
            ("start", 0), ("yield", 0),
            ("start", 1), ("yield", 1),
            ("start", 2), ("yield", 2),
        ]

    @pytest.mark.asyncio
    async def test_continuous_beats_windowed_with_variable_latency(self):
        import time
        delays = [0.04 if i % 4 == 0 else 0.005 for i in range(16)]

        async def worker(x):
            await asyncio.sleep(delays[x])
            return x

        t0 = time.perf_counter()
        async for _ in iter_ordered_concurrent(list(range(16)), 4, worker):
            pass
        continuous = time.perf_counter() - t0

        t0 = time.perf_counter()
        async for _win in iter_ordered_windows(list(range(16)), 4, worker):
            pass
        windowed = time.perf_counter() - t0

        # Continuous should not be slower than windowed; with this latency
        # profile it is meaningfully faster.
        assert continuous <= windowed + 0.01

    @staticmethod
    def _cancel_tracking_worker(cancelled):
        async def worker(x):
            try:
                await asyncio.sleep(0.05 if x > 0 else 0.0)
                return x
            except asyncio.CancelledError:
                cancelled.append(x)
                raise

        return worker

    @pytest.mark.asyncio
    async def test_break_under_aclosing_cancels_in_flight(self):
        """The documented call pattern: aclosing() cancels deterministically.

        Every production call site wraps the iteration this way, because the
        rate-limit branch breaks out of the loop.
        """
        cancelled = []
        worker = self._cancel_tracking_worker(cancelled)

        async with aclosing(iter_ordered_concurrent([0, 1, 2, 3], 4, worker)) as stream:
            async for item, _ in stream:
                if item == 0:
                    break

        # No sleep needed: aclose() awaits the cancelled tasks before returning.
        assert set(cancelled) == {1, 2, 3}

    @pytest.mark.asyncio
    async def test_bare_break_does_not_cancel_promptly(self):
        """Regression guard for the reason aclosing() is mandatory.

        A bare `break` only suspends the generator; its finally block — and the
        cancellation of in-flight LLM requests — is deferred to garbage
        collection. This test pins that behaviour so a future refactor that
        drops aclosing() at a call site is caught by the test above instead of
        silently leaking billed requests.
        """
        cancelled = []
        worker = self._cancel_tracking_worker(cancelled)

        stream = iter_ordered_concurrent([0, 1, 2, 3], 4, worker)
        async for item, _ in stream:
            if item == 0:
                break

        await asyncio.sleep(0.01)
        assert cancelled == []

        # Explicitly close so the test leaves no orphan tasks behind.
        await stream.aclose()
        assert set(cancelled) == {1, 2, 3}


# ---------------------------------------------------------------------------
# EPUB resume index: interruption after a resume must not double-count
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_epub_resume_then_interrupt_saves_correct_index(monkeypatch):
    """Regression: on a resumed run, the interruption checkpoint index must be
    the absolute count of completed chunks (len(translated_chunks)), NOT
    start_chunk_index + len(...) which double-counts the restored prefix and
    makes resume skip or replay chunks."""
    from src.core.epub import chunk_loop
    from tests.epub_engine_helpers import run_chunk_loop
    from src.core.epub.translation_metrics import TranslationMetrics

    async def fake_translate(*args, **kwargs):
        await asyncio.sleep(0)
        return "<translated/>"

    monkeypatch.setattr(chunk_loop, "translate_chunk_with_fallback", fake_translate)

    # Capture the index persisted by the checkpoint.
    saved = {}

    class FakeCkpt:
        def save_xhtml_partial_state(self, translation_id, file_href, state):
            saved["current_chunk_index"] = state.current_chunk_index

    total = 30
    chunks = [
        {"text": f"[id0]c{i}[id1]", "local_tag_map": {}, "global_indices": []}
        for i in range(total)
    ]

    # Simulate a resume: first 13 chunks already done.
    start = 13
    translated_prefix = [f"done{i}" for i in range(start)]

    stats = TranslationMetrics()
    stats.total_chunks = total

    # Interrupt once 2 more chunks have been translated in this run.
    translated_count = {"n": 0}

    async def fake_translate_counting(*args, **kwargs):
        await asyncio.sleep(0)
        translated_count["n"] += 1
        return "<translated/>"

    monkeypatch.setattr(chunk_loop, "translate_chunk_with_fallback", fake_translate_counting)

    def should_interrupt():
        return translated_count["n"] >= 2

    result_chunks, _, was_interrupted = await run_chunk_loop(
        chunks=chunks,
        source_language="English",
        target_language="French",
        model_name="m",
        llm_client=object(),
        max_retries=1,
        context_manager=None,
        placeholder_format=("[id", "]"),
        checkpoint_manager=FakeCkpt(),
        translation_id="job",
        file_href="ch1.xhtml",
        start_chunk_index=start,
        translated_chunks=list(translated_prefix),
        stats=stats,
        check_interruption_callback=should_interrupt,
        parallel_workers=1,
    )

    assert was_interrupted is True
    # 13 prefix + 2 newly translated = 15 done; next index must be 15, not 28.
    assert len(result_chunks) == 15
    assert saved["current_chunk_index"] == 15


@pytest.mark.asyncio
async def test_epub_interrupt_resume_roundtrip_keeps_advancing(monkeypatch, tmp_path):
    """End-to-end regression: the saved XHTML partial state must pass validate()
    (len(translated_chunks) == current_chunk_index) so load_xhtml_partial_state
    returns it instead of None. A double-counted index makes validation fail,
    silently discarding the state and restarting the file at chunk 0 on every
    resume — the "ça repart toujours très bas" symptom.
    """
    from src.core.epub import chunk_loop
    from tests.epub_engine_helpers import run_chunk_loop
    from src.core.epub.translation_metrics import TranslationMetrics
    from src.persistence.checkpoint_manager import CheckpointManager

    calls = {"n": 0}

    async def fake_translate(*a, **k):
        await asyncio.sleep(0)
        calls["n"] += 1
        return "<x/>"

    monkeypatch.setattr(chunk_loop, "translate_chunk_with_fallback", fake_translate)

    total = 30
    cm = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    tid, href = "job", "ch1.xhtml"
    cm.start_job(tid, "epub", {}, None)

    def chunks():
        return [{"text": f"[id0]c{i}[id1]", "local_tag_map": {}, "global_indices": []}
                for i in range(total)]

    async def run(start_idx, translated, stats, stop_at):
        return await run_chunk_loop(
            chunks=chunks(), source_language="E", target_language="F", model_name="m",
            llm_client=object(), max_retries=1, context_manager=None,
            placeholder_format=("[id", "]"),
            checkpoint_manager=cm, translation_id=tid, file_href=href,
            start_chunk_index=start_idx, translated_chunks=translated, stats=stats,
            check_interruption_callback=lambda: calls["n"] >= stop_at,
            parallel_workers=1, global_total_chunks=total, global_completed_chunks=0,
        )

    # Run 1: fresh, interrupt after 7 translated.
    await run(0, [], None, 7)
    state1 = cm.load_xhtml_partial_state(tid, href)
    assert state1 is not None, "partial state failed validate() and was discarded"
    assert state1.current_chunk_index == 7

    # Run 2: resume from the loaded state, interrupt 7 more chunks in.
    stats2 = TranslationMetrics.from_dict(state1.stats) if state1.stats else TranslationMetrics()
    stats2.total_chunks = total
    await run(state1.current_chunk_index, list(state1.translated_chunks), stats2, calls["n"] + 7)
    state2 = cm.load_xhtml_partial_state(tid, href)
    assert state2 is not None, "resumed partial state failed validate() (index double-counted)"
    # Must keep advancing (14), never reset toward 0.
    assert state2.current_chunk_index == 14


# ---------------------------------------------------------------------------
# translate_paragraphs_plain parallelism
# ---------------------------------------------------------------------------
def _make_paragraphs(n):
    # One paragraph per chunk: keep them large enough that the token chunker
    # emits a separate chunk for each (max_tokens_per_chunk=1 forces that).
    return [f"Paragraph number {i} with enough words to be its own chunk." for i in range(n)]


@pytest.mark.asyncio
async def test_plain_pipeline_parallel_matches_sequential_order(monkeypatch):
    """Sequential and parallel runs must produce the same ordered output."""

    async def fake_request(*, main_content, **kwargs):
        await asyncio.sleep(0)
        return f"T::{main_content}"

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake_request)
    # clean_translated_text is identity-ish; keep output deterministic.
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s, **kw: s)

    paragraphs = _make_paragraphs(7)

    seq, seq_stats, seq_interrupted = await plain_pipeline.translate_paragraphs_plain(
        paragraphs=paragraphs, source_language="English", target_language="French",
        model_name="m", llm_client=object(), max_tokens_per_chunk=20,
        parallel_workers=1,
    )
    par, par_stats, par_interrupted = await plain_pipeline.translate_paragraphs_plain(
        paragraphs=paragraphs, source_language="English", target_language="French",
        model_name="m", llm_client=object(), max_tokens_per_chunk=20,
        parallel_workers=4,
    )

    assert not seq_interrupted and not par_interrupted
    assert seq == par
    assert len(seq) == len(paragraphs)
    assert all(p.startswith("T::") for p in par)


@pytest.mark.asyncio
async def test_plain_pipeline_runs_concurrently(monkeypatch):
    """With workers>1, multiple requests are in flight at the same time."""
    in_flight = 0
    max_in_flight = 0
    gate = asyncio.Event()
    started = 0

    async def fake_request(*, main_content, **kwargs):
        nonlocal in_flight, max_in_flight, started
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        started += 1
        # Release all coroutines once the whole first window has started.
        if started >= 4:
            gate.set()
        try:
            await asyncio.wait_for(gate.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        in_flight -= 1
        return f"T::{main_content}"

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake_request)
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s, **kw: s)

    paragraphs = _make_paragraphs(4)
    await plain_pipeline.translate_paragraphs_plain(
        paragraphs=paragraphs, source_language="English", target_language="French",
        model_name="m", llm_client=object(), max_tokens_per_chunk=20,
        parallel_workers=4,
    )
    assert max_in_flight >= 2  # genuinely concurrent


@pytest.mark.asyncio
async def test_plain_pipeline_failed_chunk_keeps_original(monkeypatch):
    """A chunk returning None keeps the source text and counts as failed."""

    async def fake_request(*, main_content, **kwargs):
        if "number 1 " in main_content:
            return None
        return f"T::{main_content}"

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake_request)
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s, **kw: s)

    paragraphs = _make_paragraphs(3)
    out, stats, interrupted = await plain_pipeline.translate_paragraphs_plain(
        paragraphs=paragraphs, source_language="English", target_language="French",
        model_name="m", llm_client=object(), max_tokens_per_chunk=20,
        parallel_workers=3,
    )
    assert not interrupted
    assert stats.failed_chunks == 1
    # The failed paragraph keeps its original text (no "T::" prefix).
    assert any("number 1 " in p and not p.startswith("T::") for p in out)
