"""The EPUB/DOCX refine pass must grow the context window on overflow.

Issue #282: `_refine_epub_chunks` caught the overflow raised by the Ollama
stream guard in a generic `except Exception` and silently kept the unrefined
chunk, so the adaptive context never grew past 2048 during refinement. It now
retries with a larger window, like the translation pass does.
"""

import sys
from pathlib import Path

import pytest

from tests.job_helpers import make_job

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.context_optimizer import AdaptiveContextManager, REFINEMENT_MIN_CONTEXT
from src.core.epub.chunk_refinement import refine_chunks
from src.core.llm import ContextOverflowError, RepetitionLoopError
from src.core.refine import client_setup


class _FakeResponse:
    def __init__(self, content):
        self.content = content
        self.prompt_tokens = 100
        self.completion_tokens = 50
        self.context_used = 150
        self.context_limit = 4096
        self.was_truncated = False


class _OverflowOnceClient:
    """Raises `error` on the first request, then answers normally."""

    def __init__(self, error, context_window=2048):
        self.error = error
        self.context_window = context_window
        self.calls = []  # context_window seen by each request

    async def generate(self, prompt, system_prompt=None, **kwargs):
        # The context size travels with each request (the provider is never mutated).
        self.calls.append(kwargs.get("context_window", self.context_window))
        if len(self.calls) == 1:
            raise self.error
        return _FakeResponse("<TRANSLATION>Refined text.</TRANSLATION>")

    def extract_translation(self, content):
        return content.replace("<TRANSLATION>", "").replace("</TRANSLATION>", "").strip()


def _chunks():
    return ["Draft text."], [{"local_tag_map": {}, "global_indices": []}]


@pytest.mark.asyncio
@pytest.mark.parametrize("error, grown_context", [
    (ContextOverflowError("overflow"), 4096),
    # A repetition loop grows the window two steps, as in every pass (shared
    # request core).
    (RepetitionLoopError("Context overflow detected during streaming."), 6144),
])
async def test_refine_retries_with_larger_context_on_overflow(error, grown_context):
    translated, chunks = _chunks()
    client = _OverflowOnceClient(error)
    manager = AdaptiveContextManager(initial_context=2048, context_step=2048, max_context=8192)

    refined = await refine_chunks(
        translated_chunks=translated,
        chunks=chunks,
        target_language="French",
        model_name="dummy",
        llm_client=client,
        context_manager=manager,
        placeholder_format=("[id", "]"),
        log_callback=None,
        prompt_options={},
    )

    assert refined == ["Refined text."]
    assert client.calls == [2048, grown_context]


@pytest.mark.asyncio
async def test_refine_falls_back_to_draft_without_context_manager():
    translated, chunks = _chunks()
    client = _OverflowOnceClient(ContextOverflowError("overflow"))

    refined = await refine_chunks(
        translated_chunks=translated,
        chunks=chunks,
        target_language="French",
        model_name="dummy",
        llm_client=client,
        context_manager=None,
        placeholder_format=("[id", "]"),
        log_callback=None,
        prompt_options={},
    )

    assert refined == translated
    assert len(client.calls) == 1


@pytest.mark.parametrize("auto_adjust,context_window", [(True, 2048), (False, 2048)])
def test_refine_client_starts_at_refinement_floor(monkeypatch, auto_adjust, context_window):
    seen = {}

    def fake_create_llm_client(job, initial_context, log_callback=None, *, refine=False):
        seen["initial_context"] = initial_context
        seen["refine"] = refine
        return object()

    monkeypatch.setattr(client_setup, "_create_llm_client", fake_create_llm_client)

    client_setup.build_refine_client(make_job(
        provider="ollama", model="dummy",
        endpoint="http://localhost:11434/api/generate",
        auto_adjust_context=auto_adjust,
        context_window=context_window,
    ))

    assert seen["initial_context"] >= REFINEMENT_MIN_CONTEXT
    assert seen["refine"] is True
