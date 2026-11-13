"""
API key pool with throttle tracking and round-robin rotation.

Used by cloud LLM providers to support multiple API keys for the same provider.
On HTTP 429, the failing key is marked throttled and the next available key is
selected on the next request — failover happens without sleeping when possible.

Compatibility:
    - A pool with a single key behaves identically to today's single-key string.
    - All mutating operations are async-safe via asyncio.Lock so the same pool
      can be shared across coroutines (forward compatibility with parallel
      dispatch).
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Iterable, List, Union


@dataclass
class _KeyState:
    """Internal: throttle state for a single key.

    `throttled_until` is a `time.monotonic()` timestamp; 0 means available.
    Monotonic time avoids wall-clock-drift bugs when comparing expiry against
    the current moment.
    """
    throttled_until: float = 0.0


class KeyPool:
    """Round-robin pool of API keys with per-key throttle tracking.

    Typical usage from a provider (note the two separate counters — rotating
    on 429 must not consume a transient-retry attempt, see issue #217):

        attempt = 0
        rate_limit_events = 0
        while attempt < MAX_TRANSLATION_ATTEMPTS:
            current_key = await self._key_pool.acquire()
            headers = {"Authorization": f"Bearer {current_key}"}
            try:
                response = await client.post(url, headers=headers, ...)
                ...
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    rate_limit_events += 1
                    await handle_rate_limit(
                        self._key_pool, current_key, e.response.headers,
                        rate_limit_events, MAX_TRANSLATION_ATTEMPTS,
                    )
                    continue
                attempt += 1
                ...
    """

    def __init__(
        self,
        keys: Union[str, Iterable[str]],
        provider_name: str = "unknown",
    ):
        """
        Args:
            keys: A single key string or an iterable of keys. Empty strings and
                duplicates are silently dropped (preserving original order).
            provider_name: Logical name used in log messages and RateLimitError.

        Raises:
            ValueError: If no non-empty key is provided.
        """
        if isinstance(keys, str):
            keys = [keys]

        seen = set()
        cleaned: List[str] = []
        for k in keys:
            if k and k not in seen:
                cleaned.append(k)
                seen.add(k)

        if not cleaned:
            raise ValueError(
                f"KeyPool for '{provider_name}' requires at least one non-empty key"
            )

        self._keys: List[str] = cleaned
        self._states: dict = {k: _KeyState() for k in cleaned}
        self._cursor: int = 0
        self._lock = asyncio.Lock()
        self._provider_name = provider_name

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def peek(self) -> str:
        """Return the next key WITHOUT advancing the cursor or locking.

        Cheap, sync. Used by code paths outside translation (e.g. listing
        available models, context detection) that just need *a* valid key.
        """
        return self._keys[self._cursor % len(self._keys)]

    def index_of(self, key: str) -> int:
        """1-based index of `key` for human-readable logging. 0 if unknown."""
        try:
            return self._keys.index(key) + 1
        except ValueError:
            return 0

    async def acquire(self) -> str:
        """Acquire the next key, preferring non-throttled ones.

        Round-robin among non-throttled keys. If every key is throttled,
        returns the one with the earliest expiry — the caller is responsible
        for sleeping or raising via the rate-limit handler.

        Always returns a key; never blocks.
        """
        async with self._lock:
            now = time.monotonic()
            n = len(self._keys)
            for offset in range(n):
                idx = (self._cursor + offset) % n
                key = self._keys[idx]
                if self._states[key].throttled_until <= now:
                    self._cursor = (idx + 1) % n
                    return key
            # All throttled — return the one that recovers soonest.
            return min(self._keys, key=lambda k: self._states[k].throttled_until)

    async def mark_throttled(self, key: str, until_monotonic: float) -> None:
        """Mark `key` as throttled until `until_monotonic` (`time.monotonic()`).

        No-op if `key` is not in the pool. We take the max of existing and new
        expiry so a longer existing throttle isn't shortened by a fresh 429.
        """
        async with self._lock:
            state = self._states.get(key)
            if state is not None:
                state.throttled_until = max(state.throttled_until, until_monotonic)

    async def has_available(self) -> bool:
        """True if at least one key is currently non-throttled."""
        async with self._lock:
            now = time.monotonic()
            return any(s.throttled_until <= now for s in self._states.values())

    async def time_until_next_available(self) -> float:
        """Seconds until the earliest key becomes available.

        Returns 0.0 if any key is currently available.
        """
        async with self._lock:
            now = time.monotonic()
            soonest = min(
                (s.throttled_until for s in self._states.values()),
                default=now,
            )
            return max(0.0, soonest - now)
