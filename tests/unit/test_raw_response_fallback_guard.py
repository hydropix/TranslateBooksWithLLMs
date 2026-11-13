"""The plain-text raw-response fallback must not save non-translations.

When the model drops the translation tags, plain-text mode used to save the
whole raw reply (a preamble, the model's reasoning or a refusal) as the
translation, and `was_fallback` was then discarded so nobody could tell.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from src.core.llm.base import LLMResponse
from src.core.translator import (
    _make_llm_request_with_adaptive_context,
    generate_translation_request,
    raw_fallback_rejection_reason,
)

SOURCE = (
    "The old lighthouse keeper climbed the stairs every evening, "
    "counting the steps as his father had taught him."
)


def _client(raw):
    client = Mock()
    client.extract_translation = Mock(return_value=None)
    client.generate = AsyncMock(return_value=LLMResponse(
        content=raw, prompt_tokens=10, completion_tokens=5,
        context_used=15, context_limit=2048, was_truncated=False,
    ))
    return client


async def _request(raw, source=SOURCE, logs=None):
    def log(kind, msg, data=None):
        if logs is not None:
            logs.append(kind)
    return await _make_llm_request_with_adaptive_context(
        main_content=source, context_before="", context_after="",
        previous_translation_context="", source_language="English",
        target_language="French", model="m", llm_client=_client(raw),
        log_callback=log, has_placeholders=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    "I'm sorry, but I can't help with translating this content.",
    "I cannot provide a translation of this request as it violates my guidelines.",
    # A reasoning dump far longer than the passage.
    "Let me think about this. " * 60,
    # A one-word answer for a full sentence.
    "D'accord.",
])
async def test_non_translations_are_rejected(raw):
    logs = []
    translated, _, response = await _request(raw, logs=logs)
    assert translated is None
    assert response.was_fallback is False
    assert "raw_response_fallback_rejected" in logs


@pytest.mark.asyncio
async def test_plausible_untagged_translation_is_kept_and_flagged():
    raw = ("Le vieux gardien du phare montait l'escalier chaque soir, "
           "en comptant les marches comme son père le lui avait appris.")
    translated, _, response = await _request(raw)
    assert translated == raw
    assert response.was_fallback is True


def test_cjk_latin_pairs_are_within_bounds():
    zh = "老灯塔看守人每天晚上都爬楼梯，像他父亲教他的那样数着台阶。"
    assert raw_fallback_rejection_reason(zh, SOURCE) is None
    # And the reverse direction.
    assert raw_fallback_rejection_reason(SOURCE, zh) is None


def test_translated_apology_dialogue_is_not_a_refusal():
    source = "I'm sorry, I can't help you with the luggage, said the porter."
    assert raw_fallback_rejection_reason(
        "I'm sorry, I can't help you with the luggage, said the porter.", source
    ) is None


@pytest.mark.asyncio
async def test_generate_translation_request_reports_the_fallback():
    raw = ("Le vieux gardien du phare montait l'escalier chaque soir, "
           "en comptant les marches comme son père le lui avait appris.")
    hits = []
    result = await generate_translation_request(
        SOURCE, "", "", "", source_language="English", target_language="French",
        model="m", llm_client=_client(raw), on_raw_fallback=lambda: hits.append(1),
    )
    assert result == raw
    assert hits == [1]
