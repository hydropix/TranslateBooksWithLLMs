"""The refine request path must keep the draft on a bad response.

The translation path rejects truncated, empty and unclosed-tag answers; the
refine path had none of those guards: an empty refinement was accepted and
erased the chunk, a truncated one replaced it with its first half, and a None
content raised TypeError.
"""

from unittest.mock import AsyncMock, Mock

import pytest

import src.core.translator as translator
from src.core.llm.base import LLMResponse

DRAFT = (
    "Le vieux gardien du phare montait l'escalier chaque soir, "
    "en comptant les marches comme son père le lui avait appris."
)


def _client(content, extracted=None, **response_fields):
    client = Mock()
    client.generate = AsyncMock(return_value=LLMResponse(
        content=content, prompt_tokens=10, completion_tokens=5,
        context_used=15, context_limit=4096, was_truncated=False,
        **response_fields,
    ))
    client.extract_translation = Mock(return_value=extracted)
    return client


async def _refine(client):
    logs = []
    refined, _ = await translator._make_refinement_request(
        draft_translation=DRAFT, context_before="", context_after="",
        previous_refined_context="", target_language="French", model="m",
        llm_client=client, log_callback=lambda kind, msg, data=None: logs.append(kind),
        has_placeholders=False,
    )
    return refined, logs


@pytest.mark.asyncio
@pytest.mark.parametrize("client", [
    _client(None),
    _client("   "),
    _client("<TRANSLATION></TRANSLATION>", extracted=""),
    _client("<TRANSLATION>Le vieux gardien", extracted=None),
    _client("<TRANSLATION>Le vieux gardien</TRANSLATION>", extracted="Le vieux gardien",
            output_truncated=True, finish_reason="length"),
    _client("I'm sorry, I can't help with this request."),
], ids=["none", "blank", "empty-tags", "unclosed", "truncated", "refusal"])
async def test_bad_refinement_keeps_the_draft(client):
    refined, logs = await _refine(client)
    assert refined is None


@pytest.mark.asyncio
async def test_valid_refinement_is_returned():
    better = DRAFT.replace("vieux", "vieil homme,")
    refined, _ = await _refine(_client(f"<TRANSLATION>{better}</TRANSLATION>", extracted=better))
    assert refined == better


@pytest.mark.asyncio
async def test_refine_chunks_keeps_draft_when_refinement_is_empty(monkeypatch):
    class _Client:
        async def detect_thinking_model(self):
            return False

        async def close(self):
            return None

    async def empty_refinement(**kwargs):
        return "\n\n", Mock(prompt_tokens=1, completion_tokens=1, context_limit=4096)

    monkeypatch.setattr(translator, "create_provider_from_args", lambda *a, **k: _Client())
    monkeypatch.setattr(translator, "_make_refinement_request", empty_refinement)

    result = await translator.refine_chunks(
        [DRAFT], [{}], target_language="French", model_name="m",
        api_endpoint="http://localhost", llm_provider="openai",
        log_callback=lambda *a, **k: None,
    )
    assert result == [DRAFT]
