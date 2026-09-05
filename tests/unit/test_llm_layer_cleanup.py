"""
Unit tests for the remaining LLM-layer correctness items of issue #231.

Covers item 4 (OllamaProvider referenced a context detector it never created),
item 5 (the repetition-loop threshold for very long phrases was unreachable)
and item 7 (the thinking cache persisted a monotonic clock value as if it were
a wall-clock timestamp).

https://github.com/hydropix/TranslateBooksWithLLMs/issues/231
"""
import json
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.llm.providers.ollama import OllamaProvider
from src.core.llm.thinking.behavior import ThinkingBehavior
from src.core.llm.thinking.cache import ThinkingCache
from src.core.llm.thinking.detection import detect_repetition_loop
from src.core.llm.utils.context_detection import ContextDetector


# ---------------------------------------------------------------------------
# Item 4: OllamaProvider.get_model_context_size()
# ---------------------------------------------------------------------------

class TestOllamaContextDetector:
    def test_detector_is_created_at_init(self):
        """__init__ used to skip it, so get_model_context_size() raised."""
        provider = OllamaProvider(model="llama3")
        assert isinstance(provider._context_detector, ContextDetector)

    @pytest.mark.asyncio
    async def test_get_model_context_size_reads_num_ctx_from_api_show(self):
        def handler(request):
            assert request.url.path == "/api/show"
            return httpx.Response(200, json={"parameters": 'num_ctx    16384'})

        provider = OllamaProvider(
            api_endpoint="http://localhost:11434/api/chat",
            model="llama3",
            context_window=2048,
        )
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            assert await provider.get_model_context_size() == 16384
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_get_model_context_size_falls_back_on_error(self):
        def handler(request):
            return httpx.Response(500, json={"error": "boom"})

        provider = OllamaProvider(
            api_endpoint="http://localhost:11434/api/chat",
            model="some-unknown-model",
            context_window=4096,
        )
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            assert await provider.get_model_context_size() == 4096
        finally:
            await provider.close()


# ---------------------------------------------------------------------------
# Item 5: repetition-loop thresholds
# ---------------------------------------------------------------------------

# 49 characters, no internal periodicity, so only a phrase_len of 49 matches it.
_LOOP_PHRASE = "I cannot determine the correct translation here. "

# Long enough that the scan window (len // min_repetitions) reaches phrase_len 49.
_FILLER = (
    "The narrator walks through a quiet town at dusk, noting shop signs, "
    "a bakery window, three bicycles leaning on a wall, and the smell of "
    "rain on warm stone before turning left toward the river where boats "
    "wait under low bridges for the tide to change again tonight. "
    "Children argue about a kite while an old man reads a newspaper on a "
    "bench, folding it twice, then once more, without ever looking up at "
    "the gulls circling above the market roof. "
)


class TestRepetitionThresholds:
    def test_very_long_phrase_needs_only_three_repetitions(self):
        """The `>= 40` branch used to sit after `>= 20` and never ran, so a
        45+ char phrase still required the 5 repetitions of the `>= 20` tier."""
        assert detect_repetition_loop(_FILLER + _LOOP_PHRASE * 3) is True

    def test_long_loop_still_detected(self):
        """Unambiguous loops keep being detected (guard against over-narrowing)."""
        assert detect_repetition_loop(_FILLER + _LOOP_PHRASE * 6) is True

    def test_normal_prose_is_not_flagged(self):
        """No false positive on text without a loop."""
        assert detect_repetition_loop(_FILLER) is False

    def test_short_phrase_still_needs_many_repetitions(self):
        """The lenient tiers must not leak down to short phrases."""
        assert detect_repetition_loop("ok! " * 3) is False


# ---------------------------------------------------------------------------
# Item 7: persisted timestamp
# ---------------------------------------------------------------------------

class TestThinkingCacheTimestamp:
    def test_tested_at_is_wall_clock(self, tmp_path):
        """loop.time() is monotonic and resets per process; the cache outlives
        the process, so the stored value must be comparable to time.time()."""
        cache_file = tmp_path / "thinking_cache.json"
        cache = ThinkingCache(cache_file)

        before = time.time()
        cache.set("llama3", ThinkingBehavior.STANDARD, endpoint="http://localhost:11434")
        after = time.time()

        stored = json.loads(cache_file.read_text(encoding="utf-8"))
        entry = stored["llama3@http://localhost:11434"]
        assert before <= entry["tested_at"] <= after
        assert entry["behavior"] == ThinkingBehavior.STANDARD.value

    def test_round_trip_still_works(self, tmp_path):
        cache_file = tmp_path / "thinking_cache.json"
        ThinkingCache(cache_file).set("qwq", ThinkingBehavior.UNCONTROLLABLE)

        reloaded = ThinkingCache(cache_file)
        assert reloaded.get("qwq") == ThinkingBehavior.UNCONTROLLABLE
