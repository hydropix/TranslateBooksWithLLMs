"""
Unit tests for listing OpenRouter presets alongside the model catalog.

GET /api/v1/presets returns the presets visible to the API key; each one is
selectable as the model string "@preset/<slug>". The HTTP client is faked so
no network access is needed.
"""

import pytest

from src.core.llm.providers.openrouter import OpenRouterProvider


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class _FakeClient:
    """Routes GET calls by URL; records preset request params."""

    def __init__(self, models_body, presets_pages=None, presets_error=None):
        self.models_body = models_body
        self.presets_pages = presets_pages or []
        self.presets_error = presets_error
        self.preset_calls = []

    async def get(self, url, headers=None, params=None, timeout=None):
        if url == OpenRouterProvider.MODELS_URL:
            return _FakeResponse(self.models_body)
        if url == OpenRouterProvider.PRESETS_URL:
            if self.presets_error:
                raise self.presets_error
            self.preset_calls.append(params)
            return _FakeResponse(self.presets_pages[len(self.preset_calls) - 1])
        raise AssertionError(f"unexpected URL {url}")


def _models_body(n=6):
    return {"data": [
        {"id": f"vendor/model-{i}", "name": f"Model {i}",
         "pricing": {"prompt": str(i * 1e-6), "completion": str(i * 2e-6)},
         "architecture": {"modality": "text->text"}}
        for i in range(1, n + 1)
    ]}


def _preset(slug, status="active", description=None):
    return {"slug": slug, "name": slug, "status": status, "description": description}


def _provider(client, model="anthropic/claude-sonnet-4"):
    provider = OpenRouterProvider(api_key="sk-or-xxxxxxxx", model=model)

    async def _get_client():
        return client

    provider._get_client = _get_client
    return provider


@pytest.mark.asyncio
async def test_presets_listed_first_as_model_ids():
    client = _FakeClient(_models_body(), [{
        "data": [_preset("luna-translation", description="Literary FR"), _preset("alpha")],
        "total_count": 2,
    }])

    models = await _provider(client).get_available_models()

    assert [m["id"] for m in models[:2]] == ["@preset/alpha", "@preset/luna-translation"]
    assert models[1]["is_preset"] is True
    assert models[1]["description"] == "Literary FR"
    assert "pricing" not in models[0]
    assert models[2]["id"] == "vendor/model-1"


@pytest.mark.asyncio
async def test_inactive_presets_are_skipped():
    client = _FakeClient(_models_body(), [{
        "data": [_preset("old", status="archived"), _preset("off", status="disabled"),
                 _preset("live")],
        "total_count": 3,
    }])

    presets = await _provider(client).get_presets()

    assert [p["id"] for p in presets] == ["@preset/live"]


@pytest.mark.asyncio
async def test_presets_paginate_until_total_count():
    size = OpenRouterProvider.PRESETS_PAGE_SIZE
    first = [_preset(f"p{i:03d}") for i in range(size)]
    client = _FakeClient(_models_body(), [
        {"data": first, "total_count": size + 1},
        {"data": [_preset("last")], "total_count": size + 1},
    ])

    presets = await _provider(client).get_presets()

    assert len(presets) == size + 1
    assert [c["offset"] for c in client.preset_calls] == [0, size]


@pytest.mark.asyncio
async def test_preset_fetch_failure_keeps_model_list():
    client = _FakeClient(_models_body(), presets_error=RuntimeError("403"))

    models = await _provider(client).get_available_models()

    assert len(models) == 6
    assert not any(m.get("is_preset") for m in models)


@pytest.mark.asyncio
async def test_preset_model_gets_no_reasoning_override():
    client = _FakeClient(_models_body())
    provider = _provider(client, model="@preset/luna-translation")

    assert await provider._get_reasoning_override() == {}
    # The model catalog is never consulted for a preset
    assert client.preset_calls == []
