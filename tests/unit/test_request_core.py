"""
Outcome statuses of the shared request/validate core (``request_chunk``).

One fake client answers from a script; each test checks the typed outcome
the core returns and, where relevant, how it drove the context manager.
"""

from types import SimpleNamespace

import pytest

from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.context_optimizer import AdaptiveContextManager
from src.core.llm.base import LLMResponse
from src.core.llm.exceptions import ContextOverflowError, RepetitionLoopError
from src.core.llm.utils.extraction import TranslationExtractor
from src.core.request_core import (
    ChunkStatus, ContextBudget, REFINEMENT_LABELS, request_chunk,
)

PROMPT = SimpleNamespace(system="sys", user="user prompt")
SOURCE = "The rain kept falling on the empty avenue, and nobody came looking."


class _Client:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []
        self._extractor = TranslationExtractor(TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT)

    async def generate(self, prompt, system_prompt=None, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def extract_translation(self, text):
        return self._extractor.extract(text)


def _resp(content, **kwargs):
    return LLMResponse(content=content, prompt_tokens=10, completion_tokens=10, **kwargs)


def _tagged(text):
    return _resp(f"{TRANSLATE_TAG_IN}{text}{TRANSLATE_TAG_OUT}")


async def _run(client, allow_raw_fallback=True, **kwargs):
    return await request_chunk(client, PROMPT, source_text=SOURCE,
                               allow_raw_fallback=allow_raw_fallback, **kwargs)


def _manager():
    return AdaptiveContextManager(initial_context=2048, context_step=2048, max_context=8192)


@pytest.mark.asyncio
async def test_ok():
    outcome = await _run(_Client(_tagged("La pluie tombait sur l'avenue vide.")))
    assert outcome.status is ChunkStatus.OK and outcome.ok
    assert outcome.text == "La pluie tombait sur l'avenue vide."


@pytest.mark.asyncio
async def test_failed_when_the_provider_gives_nothing():
    outcome = await _run(_Client(None))
    assert outcome.status is ChunkStatus.FAILED and not outcome.ok
    assert outcome.response is None


@pytest.mark.parametrize("content", ["", "   ", f"{TRANSLATE_TAG_IN}{TRANSLATE_TAG_OUT}"])
@pytest.mark.asyncio
async def test_empty(content):
    outcome = await _run(_Client(_resp(content)))
    assert outcome.status is ChunkStatus.EMPTY


@pytest.mark.asyncio
async def test_output_truncated():
    response = _resp(f"{TRANSLATE_TAG_IN}La pluie{TRANSLATE_TAG_OUT}",
                     finish_reason="length", output_truncated=True)
    outcome = await _run(_Client(response))
    assert outcome.status is ChunkStatus.TRUNCATED and outcome.text is None


@pytest.mark.asyncio
async def test_unclosed_tag_is_truncated_without_context_manager():
    outcome = await _run(_Client(_resp(f"{TRANSLATE_TAG_IN}La pluie tombait")))
    assert outcome.status is ChunkStatus.TRUNCATED


@pytest.mark.asyncio
async def test_unclosed_tag_grows_the_context_and_retries():
    cm = _manager()
    client = _Client(_resp(f"{TRANSLATE_TAG_IN}La pluie"), _tagged("La pluie tombait."))
    outcome = await _run(client, context_manager=cm)
    assert outcome.status is ChunkStatus.OK
    assert [c["context_window"] for c in client.calls] == [2048, 4096]


@pytest.mark.asyncio
async def test_untagged_answer_is_a_fallback_in_plain_text():
    outcome = await _run(_Client(_resp(
        "Here is the translation:\nLa pluie tombait sur l'avenue vide, et personne ne venait.")))
    assert outcome.status is ChunkStatus.FALLBACK and outcome.ok
    assert outcome.text.startswith("La pluie")


@pytest.mark.asyncio
async def test_untagged_answer_is_unusable_for_structured_content():
    outcome = await _run(_Client(_resp("La pluie tombait [id0] sur l'avenue.")),
                         allow_raw_fallback=False)
    assert outcome.status is ChunkStatus.UNEXTRACTABLE


@pytest.mark.parametrize("content, detail", [
    ("I'm sorry, but I can't help translate this content.", "refusal"),
    (f"Sure. {SOURCE}", "input echoed"),
    ("Non.", "length"),
])
@pytest.mark.asyncio
async def test_refused(content, detail):
    outcome = await _run(_Client(_resp(content)))
    assert outcome.status is ChunkStatus.REFUSED
    assert detail in outcome.detail


@pytest.mark.asyncio
async def test_overflow_grows_then_reports_overflow():
    cm = AdaptiveContextManager(initial_context=2048, context_step=2048, max_context=4096)
    client = _Client(ContextOverflowError("big"), ContextOverflowError("big"))
    outcome = await _run(client, context_manager=cm)
    assert outcome.status is ChunkStatus.OVERFLOW
    assert isinstance(outcome.error, ContextOverflowError)
    assert [c["context_window"] for c in client.calls] == [2048, 4096]


@pytest.mark.asyncio
async def test_overflow_without_context_manager_is_reported_for_splitting():
    outcome = await _run(_Client(ContextOverflowError("big")))
    assert outcome.status is ChunkStatus.OVERFLOW


@pytest.mark.asyncio
async def test_repetition_loop_grows_two_steps_then_gives_up():
    cm = _manager()
    client = _Client(RepetitionLoopError("loop"), _tagged("La pluie."))
    outcome = await _run(client, context_manager=cm)
    assert outcome.status is ChunkStatus.OK
    assert [c["context_window"] for c in client.calls] == [2048, 6144]

    outcome = await _run(_Client(RepetitionLoopError("loop")))
    assert outcome.status is ChunkStatus.REPETITION


@pytest.mark.asyncio
async def test_raw_mode_returns_the_answer_untouched():
    outcome = await _run(_Client(_resp('<snippet id="1">fixed</snippet>')),
                         allow_raw_fallback=False, extract=False)
    assert outcome.status is ChunkStatus.OK
    assert outcome.text == '<snippet id="1">fixed</snippet>'


@pytest.mark.asyncio
async def test_events_follow_the_pass_labels():
    events = []
    await _run(_Client(_resp("")), labels=REFINEMENT_LABELS,
               log_callback=lambda event, message, **_: events.append(event))
    assert events == ["refinement_empty_response"]


@pytest.mark.asyncio
async def test_raw_response_callback_sees_every_answer_and_cannot_break_it():
    seen = []

    def record(response):
        seen.append(response.content)
        raise RuntimeError("disk full")

    outcome = await _run(_Client(_tagged("La pluie.")), raw_response_callback=record)
    assert outcome.ok and len(seen) == 1


def test_context_budget_gives_back_growth_of_a_failed_request():
    cm = _manager()
    with ContextBudget(cm):
        cm.increase_context()
    assert cm.get_context_size() == 2048

    with ContextBudget(cm) as budget:
        cm.increase_context()
        budget.succeeded()
    assert cm.get_context_size() == 4096
