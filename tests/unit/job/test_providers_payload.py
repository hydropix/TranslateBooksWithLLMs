"""OpenAI payload parameters: temperature, reasoning_effort.

Goes through ``OpenAICompatibleProvider._build_payload()``. Fully offline.
The config defaults are pinned so a local ``.env`` cannot change the outcome.
"""
from typing import Any, Dict, Optional

import pytest

from src import config
from src.core.llm.base import LLMRequest
from src.core.llm.providers.openai import OpenAICompatibleProvider


@pytest.fixture(autouse=True)
def _pinned_config(monkeypatch):
    monkeypatch.setattr(config, "TEMPERATURE", 0.3)
    monkeypatch.setattr(config, "REASONING_EFFORT", None)


async def _openai_payload(model: str, reasoning_effort: Optional[str] = None) -> Dict[str, Any]:
    llm = OpenAICompatibleProvider(api_endpoint="", model=model, reasoning_effort=reasoning_effort)
    req = LLMRequest(prompt="", system_prompt=None, model=model, api_key="", timeout=0.0)
    return await llm._build_payload(req)


@pytest.mark.asyncio
async def test_openai_payload_reasoning_effort() -> None:
    payload = await _openai_payload("gpt-5.6-sol", reasoning_effort="high")
    assert "temperature" not in payload and payload["reasoning_effort"] == "high"

    payload = await _openai_payload("gpt-6-sol", reasoning_effort="medium")
    assert "temperature" not in payload and payload["reasoning_effort"] == "medium"


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [None, ""])
async def test_openai_payload_sends_nothing_when_effort_unset(monkeypatch, configured) -> None:
    # Unset and empty (``REASONING_EFFORT=`` copied from .env.example) both
    # leave the choice to the model's own default.
    monkeypatch.setattr(config, "REASONING_EFFORT", configured)
    payload = await _openai_payload("gpt-6.2-astra")
    assert "reasoning_effort" not in payload and "temperature" not in payload


@pytest.mark.asyncio
async def test_openai_payload_reads_effort_from_config(monkeypatch) -> None:
    monkeypatch.setattr(config, "REASONING_EFFORT", "low")
    payload = await _openai_payload("gpt-5.5")
    assert "temperature" not in payload and payload["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_openai_payload_temperature() -> None:
    payload = await _openai_payload("gpt-5.3-codex")
    assert "reasoning_effort" not in payload and payload["temperature"] == 0.3

    payload = await _openai_payload("gpt-5.4", reasoning_effort="max")
    assert "reasoning_effort" not in payload and payload["temperature"] == 0.3
