"""Unit tests for the Opencode provider.

Opencode is a plain OpenAI-compatible provider whose only difference is the
mandatory ``x-opencode-session`` header. These tests pin that header (present
on every request, stable per provider/conversation, not leaked onto the shared
HTTP client) and the factory/endpoint wiring.
"""

import re
import uuid

import httpx
import pytest

from src.core.llm.providers.opencode import OpencodeProvider


def _chat_response(content="translated"):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


def _provider_with_transport(handler, **kwargs):
    provider = OpencodeProvider(
        model="opencode-go/deepseek-v4.1-flash",
        api_key="test-key",
        **kwargs,
    )
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


@pytest.mark.asyncio
async def test_generate_sends_session_and_auth_headers():
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json=_chat_response())

    provider = _provider_with_transport(handler)
    try:
        result = await provider.generate("Translate: hello")
    finally:
        await provider.close()

    assert result is not None
    assert result.content == "translated"
    assert seen["headers"]["x-opencode-session"] == provider.session_id
    assert seen["headers"]["authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_session_id_is_stable_across_calls():
    session_ids = []

    def handler(request):
        session_ids.append(request.headers["x-opencode-session"])
        return httpx.Response(200, json=_chat_response())

    provider = _provider_with_transport(handler)
    try:
        await provider.generate("first")
        await provider.generate("second")
    finally:
        await provider.close()

    assert len(session_ids) == 2
    assert session_ids[0] == session_ids[1] == provider.session_id


def test_distinct_instances_get_distinct_session_ids():
    a = OpencodeProvider(model="m", api_key="k")
    b = OpencodeProvider(model="m", api_key="k")
    assert a.session_id != b.session_id


def test_explicit_session_id_is_reused():
    provider = OpencodeProvider(model="m", api_key="k", session_id="conv-123")
    assert provider.session_id == "conv-123"
    assert provider.extra_headers["x-opencode-session"] == "conv-123"


def test_session_id_is_not_machine_derived():
    """Guard the intent of the removed install fingerprint: random per job.

    The id must be a fresh UUIDv4 (not a machine-derived or monotonic token),
    so a reintroduction of a host-derived prefix is caught.
    """
    ids = {OpencodeProvider(model="m", api_key="k").session_id for _ in range(5)}
    assert len(ids) == 5
    for session_id in ids:
        assert session_id.startswith("opencode-session-")
        raw = session_id.removeprefix("opencode-session-")
        assert uuid.UUID(raw).version == 4
        assert not re.fullmatch(r"[0-9a-fA-F]{12,}", session_id)


@pytest.mark.asyncio
async def test_extra_headers_cannot_override_auth_or_content_type():
    """A caller-supplied extra_headers dict must not clobber the base headers."""
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json=_chat_response())

    provider = OpencodeProvider(
        model="m",
        api_key="real-key",
        extra_headers={"Authorization": "Bearer evil", "Content-Type": "text/evil"},
    )
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await provider.generate("hi")
    finally:
        await provider.close()

    assert seen["headers"]["authorization"] == "Bearer real-key"
    assert seen["headers"]["content-type"] == "application/json"
    assert seen["headers"]["x-opencode-session"] == provider.session_id


@pytest.mark.asyncio
async def test_context_detection_probe_carries_session_header():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["headers"] = dict(request.headers)
        return httpx.Response(
            200, json={"default_generation_settings": {"n_ctx": 4096}}
        )

    provider = _provider_with_transport(handler)
    try:
        ctx = await provider.get_model_context_size()
    finally:
        await provider.close()

    assert ctx == 4096
    assert seen["path"].endswith("/props")
    assert seen["headers"]["x-opencode-session"] == provider.session_id


@pytest.mark.asyncio
async def test_session_header_is_not_on_shared_client():
    """The header travels per-request, never on the shared client defaults."""
    provider = OpencodeProvider(model="m", api_key="k")
    try:
        client = await provider._get_client()
        assert "x-opencode-session" not in {name.lower() for name in client.headers}
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_get_available_models_sends_header_and_parses_list():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"data": [
            {"id": "b-model", "name": "B model", "context_length": 32000},
            {"id": "a-model"},
        ]})

    provider = _provider_with_transport(handler)
    try:
        models = await provider.get_available_models()
    finally:
        await provider.close()

    assert seen["url"].endswith("/models")
    assert seen["headers"]["x-opencode-session"] == provider.session_id
    assert seen["headers"]["authorization"] == "Bearer test-key"
    assert [m["id"] for m in models] == ["a-model", "b-model"]
    assert models[1]["context_length"] == 32000


def test_default_endpoint_comes_from_config():
    from src.config import OPENCODE_API_ENDPOINT

    provider = OpencodeProvider(model="m", api_key="k")
    assert provider.api_endpoint == OPENCODE_API_ENDPOINT


def test_factory_builds_opencode_provider():
    from src.core.llm import create_llm_provider

    provider = create_llm_provider(
        "opencode", api_key="factory-key", model="opencode-go/deepseek-v4.1-flash"
    )
    assert isinstance(provider, OpencodeProvider)
    assert provider.api_key == "factory-key"
    assert provider.extra_headers["x-opencode-session"] == provider.session_id


def test_factory_ignores_request_endpoint():
    """Unlike ollama/openai, the fixed cloud endpoint must not be hijackable."""
    from src.config import OPENCODE_API_ENDPOINT
    from src.core.llm import create_llm_provider

    provider = create_llm_provider(
        "opencode",
        api_key="k",
        model="m",
        endpoint="http://evil.example",
        api_endpoint="http://evil2.example",
    )
    assert provider.api_endpoint == OPENCODE_API_ENDPOINT


def test_factory_requires_model(monkeypatch):
    import src.core.llm.factory as factory

    monkeypatch.setattr(factory, "OPENCODE_MODEL", "")
    with pytest.raises(ValueError, match="requires a model"):
        factory.create_llm_provider("opencode", api_key="k", model="")


def test_factory_requires_key(monkeypatch):
    import src.core.llm.factory as factory

    monkeypatch.setattr(factory, "OPENCODE_API_KEY", "")
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires an API key"):
        factory.create_llm_provider("opencode", model="m")


def test_factory_forwards_explicit_session_id():
    from src.core.llm import create_llm_provider

    p = create_llm_provider("opencode", api_key="k", model="m", session_id="conv-9")
    assert p.session_id == "conv-9"
    p2 = create_llm_provider("opencode", api_key="k", model="m", conversation_id="conv-10")
    assert p2.session_id == "conv-10"


@pytest.mark.asyncio
async def test_get_available_models_without_key_returns_empty():
    provider = OpencodeProvider(model="m")
    try:
        assert await provider.get_available_models() == []
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_get_available_models_http_error_returns_empty():
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    provider = _provider_with_transport(handler)
    try:
        assert await provider.get_available_models() == []
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_get_available_models_skips_empty_id_and_falls_back_to_name():
    def handler(request):
        return httpx.Response(200, json={"data": [
            {"id": "", "name": "no id"},
            {"id": "x-model"},
        ]})

    provider = _provider_with_transport(handler)
    try:
        models = await provider.get_available_models()
    finally:
        await provider.close()

    assert models == [{"id": "x-model", "name": "x-model"}]
