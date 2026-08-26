"""ChatGPT OAuth helpers: PKCE, JWT account id, token store, Codex import."""
import json
import time
from pathlib import Path

import pytest

from src.core.llm import chatgpt_oauth as oauth


def _jwt(payload: dict) -> str:
    import base64
    body = base64.urlsafe_b64encode(json.dumps(payload).encode("ascii")).decode("ascii").rstrip("=")
    return f"aaa.{body}.ccc"


def test_account_id_from_nested_jwt():
    token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}})
    assert oauth.account_id_from_jwt(token) == "acct-123"


def test_save_and_load_tokens(tmp_path: Path):
    store = tmp_path / "chatgpt_oauth.json"
    record = {
        "access_token": "access-one",
        "refresh_token": "refresh-one",
        "id_token": "",
        "account_id": "acct-123",
        "expires_at": time.time() + 3600,
    }
    oauth.save_tokens(record, store)
    loaded = oauth.load_tokens(store)
    assert loaded["access_token"] == "access-one"
    assert loaded["account_id"] == "acct-123"


def test_import_codex_auth(tmp_path: Path):
    source = tmp_path / "auth.json"
    dest = tmp_path / "chatgpt_oauth.json"
    token = _jwt({"chatgpt_account_id": "acct-from-codex"})
    source.write_text(json.dumps({
        "tokens": {
            "access_token": token,
            "refresh_token": "refresh-codex",
            "id_token": "",
        }
    }), encoding="utf-8")
    imported = oauth.import_codex_auth(source, dest)
    assert imported["refresh_token"] == "refresh-codex"
    assert imported["account_id"] == "acct-from-codex"
    assert dest.is_file()


def test_build_authorize_url_contains_pkce():
    pkce = oauth.generate_pkce()
    url = oauth.build_authorize_url(pkce)
    assert "code_challenge=" in url
    assert pkce["state"] in url
    assert oauth.CLIENT_ID in url
    assert "originator=codex_cli_rs" in url


def test_request_headers_use_codex_originator():
    headers = oauth.request_headers({"access_token": "tok", "account_id": "acct-1"})
    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"].startswith("codex_cli_rs/")
    assert headers["version"]
    assert headers["ChatGPT-Account-Id"] == "acct-1"


def test_parse_sse_response_collects_deltas():
    from src.core.llm.providers.chatgpt import parse_sse_response

    lines = [
        "event: response.output_text.delta",
        'data: {"type":"response.output_text.delta","delta":"Bon"}',
        'data: {"type":"response.output_text.delta","delta":"jour"}',
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":3,"output_tokens":2},"status":"completed"}}',
    ]
    text, usage, truncated = parse_sse_response(lines)
    assert text == "Bonjour"
    assert usage["input_tokens"] == 3
    assert truncated is False


@pytest.mark.asyncio
async def test_chatgpt_extracts_output_text():
    from src.core.llm.providers.chatgpt import ChatGPTProvider

    provider = ChatGPTProvider(model="gpt-5.4", tokens={"access_token": "t", "account_id": "a"})
    text = provider._extract_output_text({
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "Bonjour"}]}
        ]
    })
    assert text == "Bonjour"


@pytest.mark.asyncio
async def test_ollama_cloud_lists_models(monkeypatch):
    from src.core.llm.providers.ollama_cloud import OllamaCloudProvider

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "kimi-k2.6"}, {"id": "glm-5.1"}]}

    class _Client:
        async def get(self, url, **kwargs):
            assert url.endswith("/models")
            assert "Bearer" in kwargs["headers"]["Authorization"]
            return _Resp()

    provider = OllamaCloudProvider(api_key="ollama-cloud-key")

    async def _get_client():
        return _Client()

    monkeypatch.setattr(provider, "_get_client", _get_client)
    models = await provider.get_available_models()
    assert [m["id"] for m in models] == ["kimi-k2.6", "glm-5.1"]


def test_oauth_store_path_uses_data_dir_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert oauth.oauth_store_path() == data_dir / "chatgpt_oauth.json"


def test_oauth_store_path_uses_cwd_without_data_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert oauth.oauth_store_path() == tmp_path / "chatgpt_oauth.json"
