"""
Unit tests for the invalid-API-key fail-fast path of the Mistral and DeepSeek providers.

An HTTP 401 means the key is wrong and no retry can fix it, so `generate()` must stop
after the first request and report it clearly (as the Poe provider does). The check
used to raise a ValueError inside the same `try` that ends in a generic
`except Exception`, which logged it as an "Unknown Error", slept and retried it.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm.providers.deepseek import DeepSeekProvider
from src.core.llm.providers.mistral import MistralProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_class, label",
    [
        (MistralProvider, "Mistral"),
        (DeepSeekProvider, "DeepSeek"),
    ],
)
async def test_invalid_api_key_is_not_retried(monkeypatch, capsys, provider_class, label):
    provider = provider_class(api_key="sk-xxxxxxxx")
    unauthorized = MagicMock(status_code=401)
    client = MagicMock()
    client.post = AsyncMock(return_value=unauthorized)
    provider._get_client = AsyncMock(return_value=client)
    sleep = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep)

    result = await provider.generate("Hello")

    output = capsys.readouterr().out
    assert result is None
    assert client.post.await_count == 1
    sleep.assert_not_awaited()
    assert f"{label}: Invalid API key" in output
    assert "Unknown Error" not in output
