"""
Wiring tests for the target-language inflection line (issue #255, phase 3).

Phase 2 proved `build_glossary_block` can emit the line. These tests prove the
real translation path actually feeds it a target language, so the block the
model receives matches the one the preview endpoints show.

The seam is deliberately as close to the model as possible: each test captures
the `glossary_block` keyword argument that `src.core.translator` passes to the
prompt builder, driving the genuine async request functions with a stubbed LLM
client. No test calls `build_glossary_block` directly.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core import translator
from src.core.llm.base import LLMResponse
from src.core.translator import (
    _build_chunk_glossary_block,
    _make_llm_request_with_adaptive_context,
    _make_refinement_request,
)

# Fragment of TARGET_INFLECTION_INSTRUCTION, per the plan's contract. The test
# asserts on the "choice of rendering" clause rather than the whole sentence so
# a wording tweak in the injector does not turn into a wiring failure here.
INFLECTION_MARKER = "What must never change is the choice of rendering"

RUSSIAN_TERMS = {"Muggle": "магл"}
CHUNK = "The Muggle spoke to nobody."


def _fake_client(content="<TRANSLATION>Магл</TRANSLATION>", extracted="Магл"):
    """Minimal LLM stub, shaped like the one in test_translator_fallback_context.

    The translation path calls `generate`, the refinement path calls
    `make_request`; both then call `extract_translation` on the content.
    """
    client = Mock()
    response = LLMResponse(
        content=content,
        prompt_tokens=10,
        completion_tokens=5,
        context_used=15,
        context_limit=2048,
        was_truncated=False,
    )
    client.generate = AsyncMock(return_value=response)
    client.make_request = AsyncMock(return_value=response)
    client.extract_translation = Mock(return_value=extracted)
    return client


def _capture(monkeypatch, prompt_func_name):
    """Wrap a prompt builder in src.core.translator and record its kwargs."""
    captured = {}
    original = getattr(translator, prompt_func_name)

    def spy(*args, **kwargs):
        captured["glossary_block"] = kwargs.get("glossary_block")
        return original(*args, **kwargs)

    monkeypatch.setattr(translator, prompt_func_name, spy)
    return captured


class TestTranslationPathCarriesTargetLanguage:
    """Seam: `generate_translation_prompt` as imported into src.core.translator,
    driven through the real `_make_llm_request_with_adaptive_context`."""

    @pytest.mark.asyncio
    async def test_russian_target_emits_inflection_line(self, monkeypatch):
        captured = _capture(monkeypatch, "generate_translation_prompt")

        await _make_llm_request_with_adaptive_context(
            main_content=CHUNK,
            previous_translation_context="",
            source_language="English",
            target_language="Russian",
            model="test-model",
            llm_client=_fake_client(),
            log_callback=None,
            has_placeholders=False,
            prompt_options={"glossary_terms": RUSSIAN_TERMS},
        )

        block = captured["glossary_block"]
        assert "Muggle -> магл" in block
        assert INFLECTION_MARKER in block
        assert "Russian" in block

    @pytest.mark.asyncio
    async def test_chinese_target_omits_inflection_line(self, monkeypatch):
        captured = _capture(monkeypatch, "generate_translation_prompt")

        await _make_llm_request_with_adaptive_context(
            main_content=CHUNK,
            previous_translation_context="",
            source_language="English",
            target_language="Chinese",
            model="test-model",
            llm_client=_fake_client(),
            log_callback=None,
            has_placeholders=False,
            prompt_options={"glossary_terms": {"Muggle": "麻瓜"}},
        )

        block = captured["glossary_block"]
        assert "Muggle -> 麻瓜" in block
        assert INFLECTION_MARKER not in block


class TestRefinementPathCarriesTargetLanguage:
    """Seam: `generate_refinement_prompt` as imported into src.core.translator,
    driven through the real `_make_refinement_request`. Glossary drift during
    refinement is the same failure mode as during the first pass."""

    @pytest.mark.asyncio
    async def test_russian_target_emits_inflection_line(self, monkeypatch):
        captured = _capture(monkeypatch, "generate_refinement_prompt")

        await _make_refinement_request(
            # The refinement input is the Russian draft: the glossary is
            # matched on its target form, here inflected (магла).
            draft_translation="Он увидел магла у двери.",
            context_before="",
            context_after="",
            previous_refined_context="",
            target_language="Russian",
            model="test-model",
            llm_client=_fake_client(),
            log_callback=None,
            has_placeholders=False,
            prompt_options={"glossary_terms": RUSSIAN_TERMS},
        )

        block = captured["glossary_block"]
        assert "Muggle -> магл" in block
        assert INFLECTION_MARKER in block
        assert "Russian" in block


class TestBackwardCompatibleDefault:
    """The new parameter is keyword-only and defaulted, so a caller that does
    not know about it keeps the previous output."""

    def test_call_without_target_language_has_no_inflection_line(self):
        block = _build_chunk_glossary_block(
            CHUNK, {"glossary_terms": RUSSIAN_TERMS}
        )
        assert "Muggle -> магл" in block
        assert INFLECTION_MARKER not in block

    def test_default_matches_explicit_empty_target_language(self):
        opts = {"glossary_terms": RUSSIAN_TERMS}
        assert _build_chunk_glossary_block(CHUNK, opts) == (
            _build_chunk_glossary_block(CHUNK, opts, target_language="")
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
