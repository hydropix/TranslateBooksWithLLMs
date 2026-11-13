"""
A transient failure fetching the OpenRouter / Poe model catalog must not
disable the reasoning overrides for the life of the server process: the
fetch is skipped for CATALOG_RETRY_SECONDS, then retried.
"""

import asyncio
import threading

import httpx
import pytest

from src.core.llm.catalog import CatalogCache

FAKE_KEY = "YOUR_API_KEY_HERE"


def _openrouter():
    from src.core.llm.providers.openrouter import OpenRouterProvider
    return (OpenRouterProvider(api_key=FAKE_KEY, model="openai/gpt-4o"),
            OpenRouterProvider, "_load_model_catalog")


def _poe():
    from src.core.llm.providers.poe import PoeProvider
    return (PoeProvider(api_key=FAKE_KEY, model="GPT-4o"),
            PoeProvider, "_load_model_parameters")


@pytest.mark.asyncio
@pytest.mark.parametrize("build", [_openrouter, _poe], ids=["openrouter", "poe"])
async def test_failed_catalog_fetch_is_retried_after_the_ttl(monkeypatch, build):
    provider, cls, loader_name = build()
    clock = [1000.0]
    # Fresh process-wide cache for this test only, on a fake clock (the event
    # loop keeps the real one).
    monkeypatch.setattr(cls, "_catalog", CatalogCache(
        cls.__name__, retry_seconds=cls.CATALOG_RETRY_SECONDS, clock=lambda: clock[0]))

    responses = [httpx.Response(503, text="temporarily unavailable"),
                 httpx.Response(200, json={"data": [{"id": "GPT-4o"}]})]
    requests = []

    def handler(request):
        requests.append(request)
        return responses.pop(0)

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    load = getattr(provider, loader_name)
    try:
        assert await load() is None
        assert len(requests) == 1

        # Within the TTL: no new request, still unknown.
        clock[0] += cls.CATALOG_RETRY_SECONDS - 1
        assert await load() is None
        assert len(requests) == 1

        # Past the TTL: fetched again, and the success is cached.
        clock[0] += 2
        catalog = await load()
        assert catalog is not None and "GPT-4o" in catalog
        assert len(requests) == 2
        assert cls._catalog.failed_at is None
    finally:
        await provider.close()


def test_catalog_cache_is_usable_from_several_event_loops():
    """Jobs run on their own event loops (one per worker thread): the cache
    must not hand one loop a lock bound to another."""
    cache = CatalogCache("test", retry_seconds=300)
    fetches = []

    async def fetch():
        fetches.append(1)
        await asyncio.sleep(0)
        return {"model": {}}

    results, errors = [], []

    def run():
        try:
            results.append(asyncio.run(cache.get(fetch)))
        except Exception as e:  # pragma: no cover - the failure being tested
            errors.append(e)

    # First loop holds the per-loop lock while a second loop asks too.
    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert all(r == {"model": {}} for r in results)
    # Later callers, whatever their loop, are served from the cache.
    assert asyncio.run(cache.get(fetch)) == {"model": {}}
    assert len(fetches) <= 4
