"""Unit tests for the Opencode provider.

Opencode is a plain OpenAI-compatible provider whose only difference is the
mandatory ``x-opencode-session`` header. These tests pin that header (present
on every request, stable per provider/conversation, not leaked onto the shared
HTTP client) and the factory/endpoint wiring.
"""

import re

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

    Two ids must not share a machine-derived prefix, and the value must not
    look like a bare hex token (see tests/unit/test_no_install_fingerprint.py).
    """
    ids = {OpencodeProvider(model="m", api_key="k").session_id for _ in range(5)}
    assert len(ids) == 5
    for session_id in ids:
        assert not re.fullmatch(r"[0-9a-fA-F]{12,}", session_id)


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
