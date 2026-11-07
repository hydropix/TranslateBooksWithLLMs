"""The adaptive context window must not run away on a chunk that keeps failing.

Before this fix a repetition loop grew the window on every attempt with no
per-chunk cap, so a chunk the model kept looping on drove the
window up to the 131k maximum, and the inflated window then stayed in place
for every following chunk.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.context_optimizer import AdaptiveContextManager, MAX_REPETITION_RETRIES
from src.core.llm import RepetitionLoopError
from src.core import translator
from src.core.epub import xhtml_translator


def _manager(initial=4096, step=4096, max_context=131072):
    return AdaptiveContextManager(
        initial_context=initial, context_step=step, max_context=max_context
    )


class TestManagerBudget:
    def test_repetition_retries_are_capped_per_request(self):
        cm = _manager()
        cm.begin_request()
        allowed = 0
        while cm.should_retry_repetition_loop():
            cm.increase_context()
            cm.increase_context()
            allowed += 1
            assert allowed <= MAX_REPETITION_RETRIES
        assert allowed == MAX_REPETITION_RETRIES
        assert cm.get_context_size() == 4096 + 2 * MAX_REPETITION_RETRIES * 4096

    def test_begin_request_gives_each_chunk_a_fresh_budget(self):
        cm = _manager()
        cm.begin_request()
        while cm.should_retry_repetition_loop():
            pass
        cm.begin_request()
        assert cm.should_retry_repetition_loop() is True

    def test_abandon_restores_the_window_the_request_started_with(self):
        cm = _manager()
        cm.begin_request()
        cm.increase_context()
        cm.increase_context()
        assert cm.get_context_size() == 4096 + 2 * 4096
        cm.abandon_request()
        assert cm.get_context_size() == 4096

    def test_successful_growth_is_kept(self):
        cm = _manager()
        cm.begin_request()
        cm.increase_context()
        cm.begin_request()  # next chunk: growth from a successful chunk stays
        assert cm.get_context_size() == 8192


def _looping_client():
    client = Mock()
    client.context_window = 4096
    client.generate = AsyncMock(side_effect=RepetitionLoopError("loop"))
    client.make_request = AsyncMock(side_effect=RepetitionLoopError("loop"))
    return client


@pytest.mark.asyncio
async def test_translation_chunk_that_always_loops_is_bounded_and_restored():
    cm = _manager()
    client = _looping_client()

    result = await translator.generate_translation_request(
        "Some text to translate.", "", "", "",
        source_language="English", target_language="French", model="m",
        llm_client=client, context_manager=cm,
    )

    assert result is None
    # First attempt plus the capped repetition retries, not a climb to 131k.
    assert client.generate.await_count == 1 + MAX_REPETITION_RETRIES
    # The failed chunk does not leave an inflated window for the next one.
    assert cm.get_context_size() == 4096


@pytest.mark.asyncio
async def test_epub_refinement_request_that_always_loops_is_bounded_and_restored():
    cm = _manager()
    client = _looping_client()
    prompt_pair = SimpleNamespace(system="s", user="u")

    with pytest.raises(RepetitionLoopError):
        await xhtml_translator._request_refinement_with_adaptive_context(
            client, cm, prompt_pair, "m", None
        )

    assert client.make_request.await_count == 1 + MAX_REPETITION_RETRIES
    assert cm.get_context_size() == 4096


@pytest.mark.asyncio
async def test_retrying_the_same_chunk_starts_from_the_original_window():
    """The user-triggered retry of untranslated chunks must not keep climbing."""
    cm = _manager()
    for _ in range(3):
        client = _looping_client()
        await translator.generate_translation_request(
            "Some text to translate.", "", "", "",
            source_language="English", target_language="French", model="m",
            llm_client=client, context_manager=cm,
        )
        assert client.generate.await_count == 1 + MAX_REPETITION_RETRIES
    assert cm.get_context_size() == 4096
