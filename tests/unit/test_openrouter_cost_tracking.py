"""Per-job cost tracking and usage.cost parsing (issue #218)."""
import asyncio
import threading

import pytest

from src.core.llm.cost_tracker import (
    begin_job_cost_tracking,
    end_job_cost_tracking,
    get_current_cost_tracker,
)
from src.core.llm.providers.openrouter import OpenRouterProvider
from src.core.llm.providers.poe import PoeProvider


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class _FakeClient:
    def __init__(self, body):
        self.body = body
        self.payloads = []

    async def post(self, url, headers=None, json=None, timeout=None):
        self.payloads.append(json)
        return _FakeResponse(self.body)


def _body(usage):
    return {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": usage,
    }


def _provider(body, cls=OpenRouterProvider, **kwargs):
    provider = cls(api_key="YOUR_API_KEY_HERE", model="test/model", **kwargs)
    client = _FakeClient(body)

    async def _get_client():
        return client

    provider._get_client = _get_client
    return provider, client


@pytest.fixture(autouse=True)
def _no_reasoning_lookup(monkeypatch):
    async def _none(self):
        return {}

    monkeypatch.setattr(OpenRouterProvider, "_get_reasoning_override", _none)


@pytest.mark.asyncio
async def test_usage_cost_is_used_and_not_flagged_estimated():
    provider, client = _provider(_body({"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.0123}))
    await provider.generate("hi")

    cost, tokens = provider.get_session_cost()
    assert cost == pytest.approx(0.0123)
    assert tokens == {"prompt": 100, "completion": 50}
    assert provider._cost_tracker.is_estimated is False


@pytest.mark.asyncio
async def test_missing_usage_cost_falls_back_to_flagged_estimate():
    provider, _ = _provider(_body({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}))
    await provider.generate("hi")

    cost, _ = provider.get_session_cost()
    assert cost == pytest.approx(0.50 + 1.50)
    assert provider._cost_tracker.is_estimated is True


@pytest.mark.asyncio
async def test_top_level_cost_is_ignored():
    # OpenRouter does not return a top-level "cost"; only usage.cost counts.
    body = _body({"prompt_tokens": 10, "completion_tokens": 10})
    body["cost"] = 99.0
    provider, _ = _provider(body)
    await provider.generate("hi")
    assert provider.get_session_cost()[0] < 1.0


@pytest.mark.asyncio
async def test_zero_usage_cost_is_a_real_value():
    provider, _ = _provider(_body({"prompt_tokens": 10, "completion_tokens": 10, "cost": 0}))
    await provider.generate("hi")
    assert provider.get_session_cost()[0] == 0.0
    assert provider._cost_tracker.is_estimated is False


def _run_job(cost, results, name, barrier):
    """Simulate one job: own thread, own loop, own tracker and callback."""
    seen = []
    tracker, token = begin_job_cost_tracking(seen.append)

    async def job():
        provider, _ = _provider(_body({"prompt_tokens": 10, "completion_tokens": 5, "cost": cost}))
        for _ in range(3):
            await provider.generate("hi")
            barrier.wait(timeout=10)  # force interleaving with the other job
            await asyncio.sleep(0)

    try:
        asyncio.run(job())
    finally:
        results[name] = (tracker.snapshot(), seen, get_current_cost_tracker() is tracker)
        end_job_cost_tracking(token)


def test_concurrent_jobs_keep_separate_totals_and_callbacks():
    results = {}
    barrier = threading.Barrier(2)
    threads = [
        threading.Thread(target=_run_job, args=(0.01, results, "a", barrier)),
        threading.Thread(target=_run_job, args=(1.00, results, "b", barrier)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    (cost_a, tokens_a), seen_a, ctx_a = results["a"]
    (cost_b, tokens_b), seen_b, ctx_b = results["b"]
    assert ctx_a and ctx_b
    assert cost_a == pytest.approx(0.03)
    assert cost_b == pytest.approx(3.00)
    assert tokens_a == tokens_b == {"prompt": 30, "completion": 15}
    # Each callback only saw its own job's requests, with running totals.
    assert [round(d["session_cost"], 2) for d in seen_a] == [0.01, 0.02, 0.03]
    assert [round(d["session_cost"], 2) for d in seen_b] == [1.00, 2.00, 3.00]


@pytest.mark.asyncio
async def test_job_tracker_spans_several_provider_instances():
    seen = []
    _, token = begin_job_cost_tracking(seen.append)
    try:
        p1, _ = _provider(_body({"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.1}))
        p2, _ = _provider(_body({"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.2}))
        await p1.generate("a")
        await p2.generate("b")
    finally:
        end_job_cost_tracking(token)
    assert seen[-1]["session_cost"] == pytest.approx(0.3)
    # Instances keep their own totals too.
    assert p1.get_session_cost()[0] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_no_job_tracker_after_job_ends():
    _, token = begin_job_cost_tracking()
    end_job_cost_tracking(token)
    assert get_current_cost_tracker() is None


@pytest.mark.asyncio
async def test_poe_records_tokens_per_job_without_cost():
    seen = []
    _, token = begin_job_cost_tracking(seen.append)
    try:
        provider = PoeProvider(api_key="YOUR_API_KEY_HERE", model="Test-Bot")
        client = _FakeClient(_body({"prompt_tokens": 7, "completion_tokens": 3}))

        async def _get_client():
            return client

        async def _no_overrides():
            return {}

        provider._get_client = _get_client
        provider._get_bot_overrides = _no_overrides
        await provider.generate("hi")
    finally:
        end_job_cost_tracking(token)
    assert seen[-1]["total_prompt_tokens"] == 7
    assert seen[-1]["session_cost"] == 0.0
