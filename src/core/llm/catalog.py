"""
Process-wide cache for a provider's model catalog.

OpenRouter and Poe derive their reasoning / web-search overrides from the
provider's model listing, fetched once per process. Jobs run on their own
event loops (one per worker thread in ``src/api/handlers.py``), so the cache
must not share one ``asyncio.Lock`` between loops: a lock is bound to the loop
that first waits on it. This cache keeps one lock per running loop (to avoid a
burst of identical fetches inside a job) behind a thread lock for the shared
state. A failed fetch is retried once ``retry_seconds`` have passed, so a
transient outage does not disable the overrides for the life of the process.
"""

import asyncio
import logging
import threading
import time
import weakref
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger("llm.catalog")


class CatalogCache:
    """Fetch-once cache with a failure TTL, usable from any event loop."""

    def __init__(self, name: str, retry_seconds: float = 300,
                 clock: Callable[[], float] = time.monotonic):
        self.name = name
        self.retry_seconds = retry_seconds
        self._clock = clock
        self._data: Optional[Dict[str, Any]] = None
        self._failed_at: Optional[float] = None
        self._state_lock = threading.Lock()
        self._loop_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
            weakref.WeakKeyDictionary()
        )

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        return self._data

    @property
    def failed_at(self) -> Optional[float]:
        return self._failed_at

    def reset(self) -> None:
        with self._state_lock:
            self._data = None
            self._failed_at = None

    def _failed_recently(self) -> bool:
        failed_at = self._failed_at
        return failed_at is not None and self._clock() - failed_at < self.retry_seconds

    def _loop_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        with self._state_lock:
            lock = self._loop_locks.get(loop)
            if lock is None:
                lock = asyncio.Lock()
                self._loop_locks[loop] = lock
            return lock

    async def get(self, fetch: Callable[[], Awaitable[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Return the cached catalog, fetching it if needed.

        None means "unknown" (fetch failed recently), not "empty catalog".
        """
        if self._data is not None:
            return self._data
        if self._failed_recently():
            return None
        async with self._loop_lock():
            if self._data is not None:
                return self._data
            if self._failed_recently():
                return None
            try:
                data = await fetch()
            except Exception as e:
                logger.warning(f"[{self.name}] could not read the model catalog ({e}); "
                               f"defaults stay in place, retry in {self.retry_seconds:.0f}s")
                with self._state_lock:
                    self._failed_at = self._clock()
                return None
            with self._state_lock:
                self._data = data
                self._failed_at = None
            return data
