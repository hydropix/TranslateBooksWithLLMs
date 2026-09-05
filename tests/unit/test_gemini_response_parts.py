"""
Unit tests for Gemini response parsing.

Regression tests for item 3 of issue #231: only parts[0] was read, so a reply
split across several parts was silently truncated, and a leading "thought" part
could be returned instead of the answer.

https://github.com/hydropix/TranslateBooksWithLLMs/issues/231
"""
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.llm.providers.gemini import GeminiProvider


def _provider_returning(payload, status_code=200):
    def handler(request):
        return httpx.Response(status_code, json=payload)

    provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


def _candidate(parts, finish_reason="STOP"):
    return {
        "candidates": [{"content": {"parts": parts}, "finishReason": finish_reason}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }


@pytest.mark.asyncio
async def test_multiple_text_parts_are_joined():
    """A reply split across parts must be reassembled, not truncated."""
    provider = _provider_returning(_candidate([
        {"text": "First half. "},
        {"text": "Second half."},
    ]))
    try:
        result = await provider.generate("prompt")
        assert result.content == "First half. Second half."
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_thought_part_is_skipped():
    """A part flagged as a thought is internal reasoning, not the answer."""
    provider = _provider_returning(_candidate([
        {"text": "Let me think about this.", "thought": True},
        {"text": "The answer."},
    ]))
    try:
        result = await provider.generate("prompt")
        assert result.content == "The answer."
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_single_part_is_unchanged():
    """The common case keeps returning exactly the single part's text."""
    provider = _provider_returning(_candidate([{"text": "Bonjour"}]))
    try:
        result = await provider.generate("prompt")
        assert result.content == "Bonjour"
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_empty_parts_list_yields_empty_content():
    """No parts at all must not raise (SAFETY blocks send an empty content)."""
    provider = _provider_returning(_candidate([], finish_reason="SAFETY"))
    try:
        result = await provider.generate("prompt")
        assert result is None or result.content == ""
    finally:
        await provider.close()
