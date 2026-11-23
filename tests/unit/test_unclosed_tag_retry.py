"""
An answer that opens the translation tag but never closes it (issue #354).

On a cloud provider there is no adaptive context manager, so the
"retry with a larger context" path never ran: the chunk was marked failed at
once and the file kept a stretch of source text until a manual Resume, where
the same prompt usually went through. The request core now repeats the request
once before giving up, and the warning names the provider's finish_reason.

These tests drive the real Plain Text Mode pipeline and the real request core
with a scripted client (no network), the way an OpenRouter job runs them.
"""

from types import SimpleNamespace

import pytest

from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.common.plain_text_pipeline import translate_paragraphs_plain
from src.core.llm.base import LLMResponse
from src.core.llm.utils.extraction import TranslationExtractor
from src.core.request_core import ChunkStatus, REFINEMENT_LABELS, request_chunk

SOURCE = "The rain kept falling on the empty avenue, and nobody came looking."
TRANSLATION = "La pluie tombait sur l'avenue vide, et personne ne vint chercher."


def _resp(content, finish_reason="stop"):
    return LLMResponse(content=content, prompt_tokens=500, completion_tokens=40,
                       context_used=540, finish_reason=finish_reason)


def _unclosed():
    return _resp(f"{TRANSLATE_TAG_IN}\nLa pluie tombait sur l'avenue")


def _tagged(text=TRANSLATION):
    return _resp(f"{TRANSLATE_TAG_IN}\n{text}\n{TRANSLATE_TAG_OUT}")


class _CloudClient:
    """A provider without adaptive context, answering from a script."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0
        self._extractor = TranslationExtractor(TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT)

    async def generate(self, prompt, system_prompt=None, **kwargs):
        self.calls += 1
        return self.answers.pop(0)

    def extract_translation(self, text):
        return self._extractor.extract(text)


class _Log:
    def __init__(self):
        self.events = []

    def __call__(self, event, message="", **kwargs):
        self.events.append((event, message))

    def messages(self, event):
        return [m for e, m in self.events if e == event]


async def _translate(client, log):
    translated, _metrics, _ = await translate_paragraphs_plain(
        [SOURCE], "English", "French", "test-model", client,
        max_tokens_per_chunk=1800, log_callback=log, context_manager=None,
    )
    return translated


@pytest.mark.asyncio
async def test_plain_text_chunk_recovers_from_one_unclosed_tag():
    client = _CloudClient(_unclosed(), _tagged())
    log = _Log()

    translated = await _translate(client, log)

    assert translated == [TRANSLATION]
    assert client.calls == 2
    assert not log.messages("plain_text_chunk_failed")
    retry = log.messages("unclosed_tag_retry")
    assert len(retry) == 1 and "finish_reason: stop" in retry[0]


@pytest.mark.asyncio
async def test_plain_text_chunk_fails_after_a_second_unclosed_tag():
    client = _CloudClient(_unclosed(), _unclosed())
    log = _Log()

    translated = await _translate(client, log)

    # Today's behaviour once the single retry is spent: source text kept,
    # chunk recorded as failed, no loop.
    assert translated == [SOURCE]
    assert client.calls == 2
    assert log.messages("plain_text_chunk_failed")
    final = log.messages("unclosed_translation_tag")
    assert len(final) == 1 and "finish_reason: stop" in final[0]


@pytest.mark.asyncio
async def test_output_limit_is_not_retried():
    """finish_reason=length is a real limit: repeating the request would only
    cut the answer at the same place again."""
    response = _resp(f"{TRANSLATE_TAG_IN}\nLa pluie", finish_reason="length")
    response.output_truncated = True
    client = _CloudClient(response)

    outcome = await request_chunk(client, SimpleNamespace(system="s", user="u"),
                                  source_text=SOURCE, allow_raw_fallback=True)

    assert outcome.status is ChunkStatus.TRUNCATED
    assert outcome.detail == "output token limit"
    assert client.calls == 1


@pytest.mark.asyncio
async def test_refinement_pass_gets_the_same_single_retry():
    client = _CloudClient(_unclosed(), _tagged())
    log = _Log()

    outcome = await request_chunk(client, SimpleNamespace(system="s", user="u"),
                                  source_text=SOURCE, allow_raw_fallback=False,
                                  log_callback=log, labels=REFINEMENT_LABELS)

    assert outcome.ok and outcome.text == TRANSLATION
    assert client.calls == 2
    assert len(log.messages("refinement_unclosed_tag_retry")) == 1
