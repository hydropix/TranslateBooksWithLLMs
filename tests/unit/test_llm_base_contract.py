"""
Contract of the shared provider request loop (``LLMProvider.generate``).

A fake transport (``_send``) scripts each attempt; the loop alone decides
what to do with it: retry with backoff, rotate keys on 429, raise the
pipeline errors, fail fast on 4xx, and report usage to the job tracker.
"""

import asyncio
import threading
import time

import httpx
import pytest

from src.core.llm import base as base_mod
from src.core.llm import rate_limit_handler
from src.core.llm.base import LLMProvider, LLMResponse
from src.core.llm.cost_tracker import begin_job_cost_tracking, end_job_cost_tracking
from src.core.llm.exceptions import (
    ContextOverflowError, ProviderError, RateLimitError, RepetitionLoopError,
)
from src.core.llm.policy import (
    RetryPolicy, begin_job_retry_policy, current_retry_policy, end_job_retry_policy,
)

FAKE_KEYS = "YOUR_API_KEY_HERE_1,YOUR_API_KEY_HERE_2,YOUR_API_KEY_HERE_3"


class ScriptedProvider(LLMProvider):
    """Each attempt pops the next item: an LLMResponse or an exception."""

    def __init__(self, script, keys=None, policy=None):
        super().__init__("scripted-model", api_keys=keys, provider_name="scripted",
                         retry_policy=policy or RetryPolicy(max_attempts=3, jitter=0))
        self.script = list(script)
        self.requests = []

    async def _send(self, request):
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []

    async def fake_sleep(delay):
        recorded.append(delay)

    monkeypatch.setattr(base_mod.asyncio, "sleep", fake_sleep)
    return recorded


def _ok(text="ok", prompt=10, completion=5):
    return LLMResponse(content=text, prompt_tokens=prompt, completion_tokens=completion)


def _http(status, body="", headers=None):
    return ProviderError(body or f"HTTP {status}", status_code=status,
                         headers=headers or {}, body=body)


# --------------------------------------------------------------- retries
@pytest.mark.asyncio
async def test_transient_failure_is_retried_then_succeeds(sleeps):
    provider = ScriptedProvider([_http(503), httpx.ReadTimeout("slow"), _ok("done")])
    response = await provider.generate("hi")
    assert response.content == "done"
    assert [r.attempt for r in provider.requests] == [0, 1, 2]
    # Exponential backoff: base 2s, then 4s (jitter disabled).
    assert sleeps == [2.0, 4.0]


@pytest.mark.asyncio
async def test_exhausted_attempts_return_none(sleeps):
    provider = ScriptedProvider([_http(500)] * 3)
    assert await provider.generate("hi") is None
    assert len(provider.requests) == 3


@pytest.mark.asyncio
async def test_timeout_is_retried_with_the_policy_timeout(sleeps):
    policy = RetryPolicy(max_attempts=2, timeout=12, jitter=0)
    provider = ScriptedProvider([httpx.ConnectTimeout("t"), _ok()], policy=policy)
    assert (await provider.generate("hi")).content == "ok"
    assert {r.timeout for r in provider.requests} == {12}


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
@pytest.mark.asyncio
async def test_client_errors_fail_fast(sleeps, status):
    provider = ScriptedProvider([_http(status, "nope")])
    assert await provider.generate("hi") is None
    assert len(provider.requests) == 1
    assert sleeps == []


def test_jitter_stays_within_bounds():
    policy = RetryPolicy(base_delay=2, backoff_factor=2, max_delay=5, jitter=0.5)
    for attempt in range(1, 6):
        delay = policy.delay_before(attempt)
        base = min(2 * 2 ** (attempt - 1), 5)
        assert base <= delay <= base * 1.5


# ------------------------------------------------------------------ 429
@pytest.mark.asyncio
async def test_429_rotates_keys_without_consuming_attempts(sleeps):
    provider = ScriptedProvider(
        [_http(429), _http(429), _ok("third key")], keys=FAKE_KEYS,
        policy=RetryPolicy(max_attempts=1, jitter=0),
    )
    response = await provider.generate("hi")
    assert response.content == "third key"
    assert len({r.api_key for r in provider.requests}) == 3


@pytest.mark.asyncio
async def test_429_budget_exhaustion_raises_rate_limit_error(sleeps):
    provider = ScriptedProvider([_http(429, headers={"Retry-After": "7"})] * 10,
                                keys="YOUR_API_KEY_HERE",
                                policy=RetryPolicy(max_attempts=2, jitter=0))
    with pytest.raises(RateLimitError) as info:
        await provider.generate("hi")
    assert info.value.provider == "scripted"
    assert info.value.retry_after == 7


@pytest.mark.asyncio
async def test_keyless_429_pauses_after_the_attempts(sleeps):
    provider = ScriptedProvider([_http(429)] * 3)
    with pytest.raises(RateLimitError) as info:
        await provider.generate("hi")
    assert len(provider.requests) == 3
    assert info.value.retry_after is None  # no server hint


# ------------------------------------------------------- classification
@pytest.mark.asyncio
async def test_overflow_by_error_code(sleeps):
    error = ProviderError("bad request", status_code=400, error_code="context_length_exceeded")
    with pytest.raises(ContextOverflowError):
        await ScriptedProvider([error]).generate("hi")


@pytest.mark.asyncio
async def test_overflow_by_keyword_as_last_resort(sleeps):
    error = _http(400, "This model's maximum context length is 8192 tokens")
    with pytest.raises(ContextOverflowError):
        await ScriptedProvider([error]).generate("hi")


@pytest.mark.parametrize("status", [401, 402, 403, 429])
@pytest.mark.asyncio
async def test_auth_billing_and_rate_limit_are_never_overflow(sleeps, status):
    error = _http(status, "context length exceeded for your plan")
    provider = ScriptedProvider([error] * 3)
    try:
        await provider.generate("hi")
    except RateLimitError:
        assert status == 429
    except ContextOverflowError:  # pragma: no cover - the failure being tested
        pytest.fail(f"HTTP {status} classified as a context overflow")


@pytest.mark.asyncio
async def test_transport_errors_are_never_overflow(sleeps):
    provider = ScriptedProvider([httpx.ReadTimeout("context length")] * 3)
    assert await provider.generate("hi") is None


@pytest.mark.parametrize("error", [
    ContextOverflowError("too big"), RepetitionLoopError("loop"), RateLimitError("429"),
])
@pytest.mark.asyncio
async def test_pipeline_errors_propagate_untouched(sleeps, error):
    provider = ScriptedProvider([error])
    with pytest.raises(type(error)):
        await provider.generate("hi")
    assert len(provider.requests) == 1


# --------------------------------------------------------------- usage
@pytest.mark.asyncio
async def test_usage_is_reported_to_the_job_tracker(sleeps):
    tracker, token = begin_job_cost_tracking()
    try:
        provider = ScriptedProvider([_ok(prompt=100, completion=40)])
        await provider.generate("hi")
    finally:
        end_job_cost_tracking(token)
    cost, tokens = tracker.snapshot()
    assert tokens == {"prompt": 100, "completion": 40}
    assert cost == 0.0
    assert provider.get_session_cost()[1] == {"prompt": 100, "completion": 40}


@pytest.mark.asyncio
async def test_reported_cost_is_recorded(sleeps):
    response = _ok()
    response.cost = 0.25
    provider = ScriptedProvider([response])
    await provider.generate("hi")
    assert provider.get_session_cost()[0] == pytest.approx(0.25)


# ------------------------------------------------------- per request
@pytest.mark.asyncio
async def test_model_and_context_are_per_request(sleeps):
    provider = ScriptedProvider([_ok()])
    await provider.generate("hi", model="other-model", context_window=8192)
    assert provider.requests[0].model == "other-model"
    assert provider.requests[0].context_window == 8192
    assert provider.model == "scripted-model"  # never mutated


# ------------------------------------------------------------- policy
def test_policy_is_resolved_per_job_not_at_import():
    job_policy = RetryPolicy(max_attempts=7, timeout=99)
    token = begin_job_retry_policy(job_policy)
    try:
        assert current_retry_policy() is job_policy
        provider = ScriptedProvider([], policy=None)
        provider.retry_policy = None
        assert provider.effective_retry_policy() is job_policy
    finally:
        end_job_retry_policy(token)
    assert current_retry_policy() is not job_policy


def test_policy_from_web_job_config():
    policy = RetryPolicy.from_job_config(
        {"request_timeout": 600, "max_attempts": 4, "retry_delay": 5})
    assert (policy.timeout, policy.max_attempts, policy.base_delay) == (600, 4, 5)
    # Malformed values keep the defaults.
    default = RetryPolicy.from_config()
    bad = RetryPolicy.from_job_config({"max_attempts": "x", "retry_delay": -1})
    assert (bad.max_attempts, bad.base_delay) == (default.max_attempts, default.base_delay)


# --------------------------------------------------------------- pacing
# The cross-thread test uses real sleeps and measures the monotonic clock.
INTERVAL = 0.1
# Slack for a thread stalled between slot admission and its _send() timestamp;
# a broken limiter gives gaps near 0, so this stays discriminating.
SLACK = INTERVAL / 2


@pytest.fixture
def paced(monkeypatch):
    monkeypatch.setattr("src.config.REQUEST_INTERVAL", INTERVAL)
    monkeypatch.setattr(rate_limit_handler, "_last_request_start", {})


class StampedProvider(ScriptedProvider):
    """Records the monotonic start time of every attempt."""

    def __init__(self, script, name="scripted"):
        super().__init__(script, policy=RetryPolicy(max_attempts=3, base_delay=0, jitter=0))
        self.provider_name = name
        self.starts = []

    async def _send(self, request):
        self.starts.append(time.monotonic())
        return await super()._send(request)


def test_request_starts_are_spaced_across_jobs_and_retries(paced):
    # Web jobs run on their own thread and event loop; one of them retries.
    retrying, other = StampedProvider([_http(503), _ok()]), StampedProvider([_ok()])
    threads = [threading.Thread(target=asyncio.run, args=(p.generate("hi"),))
               for p in (retrying, other)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    starts = sorted(retrying.starts + other.starts)
    assert len(starts) == 3
    assert all(b - a >= INTERVAL - SLACK for a, b in zip(starts, starts[1:]))


@pytest.mark.asyncio
async def test_pacing_is_per_provider(paced, sleeps):
    await StampedProvider([_ok()], "gemini").generate("hi")
    await StampedProvider([_ok()], "mistral").generate("hi")
    assert sleeps == []  # mistral never waited for gemini's slot


@pytest.mark.asyncio
async def test_cancelled_waiter_leaves_no_reservation(paced, monkeypatch):
    monkeypatch.setattr("src.config.REQUEST_INTERVAL", 60.0)
    claimed = {"scripted": time.monotonic()}  # a request just started
    rate_limit_handler._last_request_start.update(claimed)
    asleep = asyncio.Event()

    async def sleep_until_cancelled(delay):
        asleep.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(base_mod.asyncio, "sleep", sleep_until_cancelled)
    waiter = asyncio.create_task(StampedProvider([_ok()]).generate("hi"))
    await asyncio.wait_for(asleep.wait(), 5)  # fail, don't hang, if it never sleeps
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    # A batch closed by a rate-limit pause must not push back the next slot.
    assert rate_limit_handler._last_request_start == claimed
