"""OpenAI payload parameters: temperature, reasoning_effort.

Goes through ``OpenAICompatibleProvider._build_payload()``. Fully offline.
"""
import pytest

from typing import Any, Dict

from src.core.llm.base import LLMRequest
from src.core.llm.providers.openai import OpenAICompatibleProvider


@pytest.mark.asyncio
async def _openai_payload(model: str, reasoning_effort: str | None = None) -> Dict[str, Any]:
    llm: OpenAICompatibleProvider = OpenAICompatibleProvider(api_endpoint="", model=model, reasoning_effort=reasoning_effort)
    req: LLMRequest = LLMRequest(prompt="", system_prompt=None, model=model, api_key="", timeout=0.0)
    return await llm._build_payload(req)


@pytest.mark.asyncio
async def test_openai_payload_reasoning_effort() -> None:
    payload: Dict[str, Any] = await _openai_payload("gpt-5.6-sol", reasoning_effort="high")
    assert "temperature" not in payload and payload["reasoning_effort"] == "high"

    payload = await _openai_payload("gpt-6-sol", reasoning_effort="medium")
    assert "temperature" not in payload and payload["reasoning_effort"] == "medium"

    payload = await _openai_payload("gpt-6.2-astra")
    assert "reasoning_effort" not in payload and "temperature" not in payload


@pytest.mark.asyncio
async def test_openai_payload_temperature() -> None:
    payload: Dict[str, Any] = await _openai_payload("gpt-5.3-codex")
    assert "reasoning_effort" not in payload and payload["temperature"] == 0.3

    payload = await _openai_payload("gpt-5.4", reasoning_effort="max")
    assert "reasoning_effort" not in payload and payload["temperature"] == 0.3
