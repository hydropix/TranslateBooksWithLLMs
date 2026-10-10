"""Unit tests for the Opencode provider.

These tests pin the mandatory ``x-opencode-session`` header (present on every
request, stable per provider/conversation, not leaked onto the shared HTTP
client), catalog and API-family routing, and the factory/endpoint wiring.
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
        model="deepseek-v4.1-flash",
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
    seen = {"urls": []}

    def handler(request):
        seen["urls"].append(str(request.url))
        seen["headers"] = dict(request.headers)
        if request.url.path == "/zen/go/v1/models":
            return httpx.Response(200, json={"data": [
                {"id": "b-model", "context_length": 32000},
                {"id": "a-model"},
            ]})
        return httpx.Response(200, json={"data": [{"id": "z-model"}]})

    provider = _provider_with_transport(handler)
    try:
        models = await provider.get_available_models()
    finally:
        await provider.close()

    # Both catalogs are fetched concurrently, so request order is not fixed.
    assert sorted(seen["urls"]) == [
        "https://opencode.ai/zen/go/v1/models",
        "https://opencode.ai/zen/v1/models",
    ]
    assert seen["headers"]["x-opencode-session"] == provider.session_id
    assert seen["headers"]["authorization"] == "Bearer test-key"
    # Go first, then Zen, each sorted; ids carry the catalog prefix.
    assert [m["id"] for m in models] == [
        "opencode-go/a-model", "opencode-go/b-model", "opencode/z-model",
    ]
    assert [m["group"] for m in models] == ["OpenCode Go", "OpenCode Go", "OpenCode Zen"]
    assert models[1]["context_length"] == 32000


@pytest.mark.asyncio
async def test_get_available_models_raises_when_both_catalogs_fail():
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    provider = _provider_with_transport(handler)
    try:
        with pytest.raises(RuntimeError, match="401"):
            await provider.get_available_models()
    finally:
        await provider.close()


def test_assigning_prefixed_model_strips_prefix_and_switches_catalog():
    """LLMClient.make_request assigns the settings id to provider.model."""
    provider = OpencodeProvider(model="opencode-go/glm-5.3", api_key="k")
    provider.model = "opencode/kimi-k2.6"
    assert provider.model == "kimi-k2.6"
    assert provider.catalog == "opencode"
    assert provider.api_endpoint.endswith("/zen/v1/chat/completions")
    provider.model = "opencode-go/gpt-6-luna"
    assert provider.api_family_name == "responses"
    with pytest.raises(ValueError, match="System One"):
        provider.model = "opencode/jev-1.13"


def test_plain_text_model_access_disabled_gets_hint():
    provider = OpencodeProvider(model="glm-5.3", api_key="k")
    response = httpx.Response(403, text="Model access is disabled")
    message = provider._describe_http_error(response, "403 - Model access is disabled")
    assert "not enabled for your OpenCode workspace" in message


def test_untyped_insufficient_funds_gets_balance_hint():
    provider = OpencodeProvider(model="opencode/gemini-3.5-flash", api_key="k")
    response = httpx.Response(402, json={"error": {
        "code": 402, "message": "Upstream request failed: Insufficient account funds",
    }})
    message = provider._describe_http_error(
        response, "Upstream request failed: Insufficient account funds")
    assert "workspace balance" in message


def test_default_endpoint_comes_from_config():
    from src.config import OPENCODE_API_BASE

    provider = OpencodeProvider(model="m", api_key="k")
    assert provider.api_endpoint == OPENCODE_API_BASE + "/go/v1/chat/completions"


def test_factory_builds_opencode_provider():
    from src.core.llm import create_llm_provider

    provider = create_llm_provider(
        "opencode", api_key="factory-key", model="deepseek-v4.1-flash"
    )
    assert isinstance(provider, OpencodeProvider)
    assert provider.api_key == "factory-key"
    assert provider.extra_headers["x-opencode-session"] == provider.session_id


def test_factory_ignores_request_endpoint():
    """Unlike ollama/openai, the fixed cloud endpoint must not be hijackable."""
    from src.config import OPENCODE_API_BASE
    from src.core.llm import create_llm_provider

    provider = create_llm_provider(
        "opencode",
        api_key="k",
        model="m",
        endpoint="http://evil.example",
        api_endpoint="http://evil2.example",
    )
    assert provider.api_endpoint == OPENCODE_API_BASE + "/go/v1/chat/completions"


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
async def test_get_available_models_skips_a_failing_catalog():
    def handler(request):
        if request.url.path == "/zen/v1/models":
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"data": [{"id": "glm-5.3"}]})

    provider = _provider_with_transport(handler)
    try:
        models = await provider.get_available_models()
    finally:
        await provider.close()
    assert [m["id"] for m in models] == ["opencode-go/glm-5.3"]


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

    assert models == [
        {"id": "opencode-go/x-model", "name": "opencode-go/x-model", "group": "OpenCode Go"},
        {"id": "opencode/x-model", "name": "opencode/x-model", "group": "OpenCode Zen"},
    ]


# --- Model routing: OpenCode Go serves each model through one API family ---


@pytest.mark.asyncio
async def test_config_prefix_is_stripped_from_model_id():
    """The API rejects "opencode-go/<id>"; only the bare id may be sent."""
    seen = {}

    def handler(request):
        import json
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json=_chat_response())

    provider = OpencodeProvider(model="opencode-go/deepseek-v4.1-flash", api_key="test-key")
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await provider.generate("hello")
    finally:
        await provider.close()

    assert provider.model == "deepseek-v4.1-flash"
    assert seen["model"] == "deepseek-v4.1-flash"


@pytest.mark.parametrize("model,family", [
    ("gpt-5.6-luna", "responses"), ("grok-4.7", "responses"),
    ("muse-spark-1.3-contributor", "responses"), ("claude-haiku-5-5", "messages"),
    ("opencode-go/gpt-6-luna", "responses"), ("opencode/gpt-5.5", "responses"),
    ("opencode/claude-sonnet-5", "messages"), ("opencode/qwen3.8-max", "messages"),
    ("opencode/gemini-3.5-flash", "google"), ("opencode/jev-1.13", "unsupported"),
    # Go documents MiniMax and Qwen under Messages; Zen MiniMax is chat.
    ("minimax-m3", "messages"), ("qwen3.8-max", "messages"), ("glm-5.3", "chat"),
    ("opencode/minimax-m3", "chat"),
    # Google is a Zen-only family.
    ("gemini-3.5-flash", "chat"),
])
def test_api_family_routing(model, family):
    assert OpencodeProvider.api_family(model) == family


def test_system_one_models_are_rejected():
    with pytest.raises(ValueError, match="System One"):
        OpencodeProvider(model="opencode/jev-1.13", api_key="k")


async def _generate_with(model, response_json):
    import json
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=response_json)

    provider = OpencodeProvider(model=model, api_key="test-key")
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await provider.generate("hello", system_prompt="be a translator")
    finally:
        await provider.close()
    return result, seen


@pytest.mark.asyncio
async def test_responses_family_round_trip():
    result, seen = await _generate_with("opencode-go/gpt-6-luna", {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": "bonjour"}]},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 2},
    })
    assert seen["url"] == "https://opencode.ai/zen/go/v1/responses"
    assert seen["headers"]["authorization"] == "Bearer test-key"
    assert seen["headers"]["x-opencode-session"]
    assert seen["body"]["model"] == "gpt-6-luna"
    assert seen["body"]["input"] == "hello"
    assert seen["body"]["instructions"] == "be a translator"
    assert result.content == "bonjour"
    assert (result.prompt_tokens, result.completion_tokens) == (5, 2)
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_responses_family_flags_output_truncation():
    result, _ = await _generate_with("gpt-6-luna", {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "bon"}]}],
    })
    assert result.content == "bon"
    assert result.output_truncated


@pytest.mark.asyncio
async def test_messages_family_round_trip():
    result, seen = await _generate_with("opencode/claude-sonnet-5", {
        "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "bonjour"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    })
    assert seen["url"] == "https://opencode.ai/zen/v1/messages"
    assert seen["headers"]["x-api-key"] == "test-key"
    assert "authorization" not in seen["headers"]
    assert seen["headers"]["anthropic-version"]
    assert seen["headers"]["x-opencode-session"]
    assert seen["body"]["model"] == "claude-sonnet-5"
    assert seen["body"]["system"] == "be a translator"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert seen["body"]["max_tokens"] > 0
    assert result.content == "bonjour"
    assert (result.prompt_tokens, result.completion_tokens) == (5, 2)


@pytest.mark.asyncio
async def test_google_family_round_trip():
    result, seen = await _generate_with("opencode/gemini-3.5-flash", {
        "candidates": [{
            "content": {"parts": [{"text": "...", "thought": True}, {"text": "bonjour"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
    })
    assert seen["url"] == "https://opencode.ai/zen/v1/models/gemini-3.5-flash:generateContent"
    assert seen["headers"]["x-goog-api-key"] == "test-key"
    assert "authorization" not in seen["headers"]
    assert seen["headers"]["x-opencode-session"]
    assert seen["body"]["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert seen["body"]["systemInstruction"] == {"parts": [{"text": "be a translator"}]}
    assert result.content == "bonjour"
    assert (result.prompt_tokens, result.completion_tokens) == (5, 2)


@pytest.mark.parametrize("model", [
    "deepseek-v4.1-flash", "glm-5.3", "kimi-k3", "minimax-m3", "qwen3.8-max",
    "some-new-free-model", "opencode/kimi-k2.6", "opencode/big-pickle",
])
def test_chat_and_unknown_models_are_accepted(model):
    """Unknown ids fall back to chat/completions so rotating free models work."""
    assert OpencodeProvider(model=model, api_key="k").model == model.split("/")[-1]


@pytest.mark.asyncio
async def test_get_available_models_lists_every_routed_family():
    def handler(request):
        if request.url.path == "/zen/go/v1/models":
            return httpx.Response(200, json={"data": [
                {"id": "glm-5.3"}, {"id": "gpt-5.6-luna"}, {"id": "grok-4.7"},
                {"id": "muse-spark-1.3-contributor"}, {"id": "claude-haiku-5-5"},
                {"id": "minimax-m3"},
            ]})
        return httpx.Response(200, json={"data": [
            {"id": "kimi-k2.6"}, {"id": "gpt-5.5"}, {"id": "claude-sonnet-5"},
            {"id": "gemini-3.5-flash"}, {"id": "jev-1.13"}, {"id": "big-pickle"},
        ]})

    provider = _provider_with_transport(handler)
    try:
        models = await provider.get_available_models()
    finally:
        await provider.close()

    # Only System One (jev-) is hidden. Free Zen models (big-pickle) stay
    # listed; the gateway explains refusals.
    assert [m["id"] for m in models] == [
        "opencode-go/claude-haiku-5-5", "opencode-go/glm-5.3", "opencode-go/gpt-5.6-luna",
        "opencode-go/grok-4.7", "opencode-go/minimax-m3", "opencode-go/muse-spark-1.3-contributor",
        "opencode/big-pickle", "opencode/claude-sonnet-5", "opencode/gemini-3.5-flash",
        "opencode/gpt-5.5", "opencode/kimi-k2.6",
    ]


@pytest.mark.asyncio
async def test_protocol_unsupported_error_is_explained():
    """A model outside the known prefixes that the gateway refuses on chat."""
    logs = []

    def handler(request):
        return httpx.Response(400, json={"type": "error", "error": {
            "type": "ModelProtocolUnsupported",
            "message": "Model does not support this protocol.",
        }})

    provider = _provider_with_transport(handler, log_callback=lambda k, m: logs.append(m))
    try:
        result = await provider.generate("hello")
    finally:
        await provider.close()

    assert result is None
    assert any("serves 'opencode-go/deepseek-v4.1-flash' through a different API" in m for m in logs)


@pytest.mark.asyncio
async def test_zen_model_routes_to_zen_catalog_with_bare_id():
    import json
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json=_chat_response())

    provider = OpencodeProvider(model="opencode/kimi-k2.6", api_key="test-key")
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await provider.generate("hello")
    finally:
        await provider.close()

    assert seen["url"] == "https://opencode.ai/zen/v1/chat/completions"
    assert seen["model"] == "kimi-k2.6"


@pytest.mark.parametrize("model, expected", [
    ("deepseek-v4.1-flash", "opencode-go/deepseek-v4.1-flash"),
    ("opencode-go/glm-5.3", "opencode-go/glm-5.3"),
    ("opencode/kimi-k2.6", "opencode/kimi-k2.6"),
    ("", ""),
    (None, ""),
])
def test_canonical_model_id(model, expected):
    assert OpencodeProvider.canonical_model_id(model) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("status, body, expected", [
    (403, {"type": "error", "error": {"type": "FreeTierError",
        "message": "OpenCode's free tier can only be used from within OpenCode"}},
     "restricts its free Zen models to its own app"),
    (401, {"type": "error", "error": {"type": "CreditsError",
        "message": "Insufficient balance."}},
     "paid from your OpenCode workspace balance"),
    (403, {"error": {"type": "server_error",
        "message": "Upstream request failed: Model access is disabled"}},
     "is not enabled for your OpenCode workspace"),
])
async def test_gateway_refusals_are_explained(status, body, expected):
    logs = []

    def handler(request):
        return httpx.Response(status, json=body)

    provider = OpencodeProvider(
        model="opencode/kimi-k2.6", api_key="test-key",
        log_callback=lambda k, m: logs.append(m),
    )
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert await provider.generate("hello") is None
    finally:
        await provider.close()

    assert any(expected in m for m in logs), logs
