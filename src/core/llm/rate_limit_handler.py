"""
Centralized HTTP 429 handling with API key rotation support.

Used by cloud LLM providers to deduplicate the rate-limit retry/backoff logic
that was previously copy-pasted across all providers (gemini, openai-compatible,
openrouter, mistral, deepseek, poe).

Behavior on 429:
    1. Compute wait time from response headers (Retry-After or X-RateLimit-Reset)
    2. Mark the failed key as throttled in the pool
    3. If another key is available: rotate without sleeping (caller's next
       acquire() returns the new key)
    4. Else if rate-limit budget remains: sleep until the next key becomes
       available
    5. Else: raise RateLimitError to trigger upstream auto-pause

Rate-limit handling has its own budget, separate from the caller's transient
retry counter (MAX_TRANSLATION_ATTEMPTS): rotating to a spare key must never
consume a retry attempt, otherwise large pools exhaust the retry loop before
trying every key (issue #217).
"""

import asyncio
import time
from typing import Callable, Mapping, Optional

from .exceptions import RateLimitError
from .key_pool import KeyPool


def is_retryable_http_status(status_code: int) -> bool:
    """Whether an HTTP status is worth retrying with the same request.

    Client errors (4xx) are caused by the request itself and won't succeed on a
    retry: 404 (model retired/unknown), 400 (bad request), 401/403 (auth), 402
    (billing). Retrying them only wastes time and floods the log. Two exceptions
    are transient: 408 (request timeout, the server gave up waiting) and 429
    (rate limit, handled separately via handle_rate_limit()).

    Server errors (5xx) and anything else are treated as transient and retryable.
    """
    if status_code in (408, 429):
        return True
    return not (400 <= status_code < 500)


def compute_wait_time(headers: Mapping[str, str], attempt: int) -> int:
    """Derive a wait time in seconds from a 429 response.

    Priority:
        1. Retry-After header (seconds) — standard
        2. X-RateLimit-Reset header (UTC ms timestamp) — used by OpenRouter
        3. Exponential backoff fallback: min(2 ** (attempt + 2), 60)

    httpx returns case-insensitive headers, but we check both casings to stay
    robust against plain dict mocks in tests.

    Returns:
        Wait time in seconds, always >= 1.
    """
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            return max(int(retry_after), 1)
        except (ValueError, TypeError):
            pass

    reset_ms = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
    if reset_ms:
        try:
            wait = int((int(reset_ms) - time.time() * 1000) / 1000) + 1
            return max(min(wait, 65), 1)
        except (ValueError, TypeError):
            pass

    return min(2 ** (attempt + 2), 60)


def rate_limit_budget(pool_size: int, max_attempts: int) -> int:
    """Max number of 429 responses tolerated for a single request.

    One rotation per key in the pool, plus the same number of sleep-waits a
    single-key setup historically got (max_attempts - 1). Past this budget we
    raise RateLimitError even if a key has recovered in the meantime — it is
    the guard against an unbounded rotate loop when keys recover quickly but
    the provider keeps answering 429.
    """
    return pool_size + max_attempts - 1


async def handle_rate_limit(
    pool: KeyPool,
    failed_key: str,
    response_headers: Mapping[str, str],
    rate_limit_events: int,
    max_attempts: int,
    log_callback: Optional[Callable] = None,
) -> None:
    """Mark a key throttled, rotate if possible, sleep if needed, raise if exhausted.

    Caller pattern — `rate_limit_events` is a counter the caller owns, separate
    from its transient-retry counter, so rotations never consume retry attempts:

        if response.status_code == 429:
            rate_limit_events += 1
            await handle_rate_limit(
                self._key_pool, current_key, response.headers,
                rate_limit_events, MAX_TRANSLATION_ATTEMPTS, self.log_callback,
            )
            continue  # without consuming a transient-retry attempt

    Args:
        pool: KeyPool to update.
        failed_key: The key that just received a 429.
        response_headers: Headers from the 429 response.
        rate_limit_events: 1-based count of 429s handled for the current
            request, across all keys (caller increments before each call).
        max_attempts: MAX_TRANSLATION_ATTEMPTS; only used to size the
            rate-limit budget (see rate_limit_budget()).
        log_callback: Optional structured logging callback (key, message).

    Returns:
        None — caller should `continue` the retry loop after this returns.

    Raises:
        RateLimitError: Once `rate_limit_events` exceeds the budget, i.e. every
            key got its rotation and the sleep-waits are spent.
    """
    wait_time = compute_wait_time(response_headers, rate_limit_events - 1)
    await pool.mark_throttled(failed_key, time.monotonic() + wait_time)

    provider = pool.provider_name
    failed_idx = pool.index_of(failed_key)
    pool_size = pool.size
    budget = rate_limit_budget(pool_size, max_attempts)

    if rate_limit_events >= budget:
        raise RateLimitError(
            f"{provider} rate limit persisted after {rate_limit_events} "
            f"429 response(s) across {pool_size} key(s)",
            retry_after=wait_time,
            provider=provider,
        )

    # Fast path: another key is ready, rotate without sleeping.
    if pool_size > 1 and await pool.has_available():
        _log(
            log_callback,
            "llm_key_rotated",
            f"🔄 [{provider}] key #{failed_idx}/{pool_size} rate-limited "
            f"(retry-after {wait_time}s), rotating to next key",
        )
        return

    # All keys throttled (or single-key pool) — wait for the earliest recovery.
    remaining = await pool.time_until_next_available()
    sleep_for = max(int(remaining) if remaining > 0 else wait_time, 1)
    _log(
        log_callback,
        "llm_rate_limit",
        f"⚠️ [{provider}] all {pool_size} key(s) rate-limited "
        f"(429 #{rate_limit_events}/{budget}), waiting {sleep_for}s...",
    )
    await asyncio.sleep(sleep_for)


def _log(log_callback: Optional[Callable], event: str, message: str) -> None:
    """Best-effort logging: callback if provided, else stdout via print."""
    if log_callback:
        try:
            log_callback(event, message)
            return
        except Exception:
            pass
    print(message)
