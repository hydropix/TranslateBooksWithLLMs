"""
TXT translations must not silently lose text.

A user translated a 141-chapter Chinese TXT novel and ~20 consecutive chapters
were missing from an output that looked complete. Three gaps combined:

1. The chunker only split on blank lines and on ASCII sentence punctuation.
   A one-paragraph-per-line Chinese file (the usual web-novel layout) became a
   single chunk holding the whole book or several chapters at once.
2. Nothing checked that a response covered its whole chunk: a model that
   stopped early (output token limit) or skipped passages was saved as a
   successful translation.
3. Raw model responses were never kept, so there was no way to tell whether an
   odd word came from the model or from the pipeline.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import src.config as config_module
import src.core.llm_client as llm_client_module
from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.adapters.generic_translator import GenericTranslator
from src.core.adapters.txt_adapter import TxtAdapter
from src.core.chunking.token_chunker import TokenChunker, count_tokens
from src.core.llm.base import LLMResponse, is_output_limit_finish
from src.core.llm.providers.gemini import GeminiProvider
from src.core.llm.providers.openrouter import OpenRouterProvider
from src.core.llm.utils.extraction import TranslationExtractor
from src.core.translator import _make_llm_request_with_adaptive_context
from src.persistence.checkpoint_manager import CheckpointManager
from src.persistence.raw_response_log import raw_response_log_path


SENTENCE = "他看着远方的山，心里想着很多事情。“你为什么要走？”她问道。阿言没有回答，只是默默地转过身去！"


def _chinese_book(chapters=12, paragraphs=15, blank_lines=False):
    """A Chinese novel laid out one paragraph per line, like most web novels."""
    sep = "\n\n" if blank_lines else "\n"
    body = []
    for c in range(1, chapters + 1):
        paras = ["　　" + SENTENCE * (1 + (p % 3)) for p in range(paragraphs)]
        body.append(f"第{c}章" + sep + sep.join(paras))
    return sep.join(body)


# ---------------------------------------------------------------------------
# 1. Chunking
# ---------------------------------------------------------------------------

class TestChunkerHandlesCjkLayouts:

    def test_one_paragraph_per_line_is_split_under_the_limit(self):
        text = _chinese_book()
        chunker = TokenChunker(max_tokens=450)
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 10
        assert max(chunker.count_tokens(c["main_content"]) for c in chunks) <= 450

    def test_line_split_round_trips_to_the_source(self):
        """Joining the untouched chunks gives back every line, in order."""
        text = _chinese_book()
        chunks = TokenChunker(max_tokens=450).chunk_text(text)

        rebuilt = chunks[0]["main_content"] + "".join(
            c["join_with"] + c["main_content"] for c in chunks[1:]
        )
        assert rebuilt == text

    def test_line_split_chunks_rejoin_on_a_single_newline(self):
        chunks = TokenChunker(max_tokens=450).chunk_text(_chinese_book())
        assert {c["join_with"] for c in chunks[1:]} == {"\n"}

    def test_context_is_one_line_not_the_whole_neighbour_chunk(self):
        chunks = TokenChunker(max_tokens=450).chunk_text(_chinese_book())
        middle = chunks[len(chunks) // 2]

        assert "\n" not in middle["context_before"]
        assert "\n" not in middle["context_after"]
        assert middle["context_before"]
        assert middle["context_after"]

    def test_long_cjk_line_splits_on_fullwidth_punctuation(self):
        """A single oversized line with only 。！？ still splits into sentences."""
        text = SENTENCE * 60
        chunker = TokenChunker(max_tokens=200)
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        assert max(chunker.count_tokens(c["main_content"]) for c in chunks) <= 200
        assert "".join(c["main_content"] for c in chunks).replace(" ", "") == text

    def test_closing_quote_stays_with_its_sentence(self):
        sentences = TokenChunker().split_paragraph_into_sentences("“你好。”她说。")
        assert sentences == ["“你好。”", "她说。"]

    def test_blank_line_layout_is_unchanged(self):
        chunks = TokenChunker(max_tokens=450).chunk_text(_chinese_book(blank_lines=True))
        assert {c["join_with"] for c in chunks[1:]} == {"\n\n"}


# ---------------------------------------------------------------------------
# 2a. Completeness check (TxtAdapter.validate_unit_translation)
# ---------------------------------------------------------------------------

ENGLISH_PARAGRAPH = (
    "The river ran quietly past the old mill, carrying leaves and the last light "
    "of the afternoon toward the sea. Nobody in the village remembered when the "
    "wheel had stopped turning, only that it had, and that the miller's son had "
    "left soon after without saying where he was going or when he would return. "
    "In spring the water rose over the lower field, and the children who had once "
    "played on the wheel watched it from the bridge, arguing about whether it "
    "would ever move again, and what the miller would have said if he had seen it."
)


async def _prepared_adapter(tmp_path, text, max_tokens=450):
    source = tmp_path / "in.txt"
    source.write_text(text, encoding="utf-8")
    adapter = TxtAdapter(str(source), str(tmp_path / "out.txt"),
                         {"max_tokens_per_chunk": max_tokens})
    assert await adapter.prepare_for_translation()
    return adapter


class TestLengthValidation:

    @pytest.mark.asyncio
    async def test_complete_translation_passes(self, tmp_path):
        adapter = await _prepared_adapter(tmp_path, ENGLISH_PARAGRAPH)
        assert adapter.validate_unit_translation("chunk_0", ENGLISH_PARAGRAPH.upper()) is None

    @pytest.mark.asyncio
    async def test_translation_missing_most_of_the_chunk_is_flagged(self, tmp_path):
        adapter = await _prepared_adapter(tmp_path, ENGLISH_PARAGRAPH)
        feedback = adapter.validate_unit_translation("chunk_0", "The river ran quietly.")
        assert feedback is not None
        assert "skipped" in feedback

    @pytest.mark.asyncio
    async def test_short_source_is_not_checked(self, tmp_path):
        adapter = await _prepared_adapter(tmp_path, "第一章 山中\n\n" + "短。")
        assert count_tokens(adapter.chunks[0]["main_content"]) < config_module.MIN_TOKENS_FOR_LENGTH_CHECK
        assert adapter.validate_unit_translation("chunk_0", "I") is None

    @pytest.mark.asyncio
    async def test_ratio_zero_disables_the_check(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config_module, "MIN_TRANSLATION_LENGTH_RATIO", 0.0)
        adapter = await _prepared_adapter(tmp_path, ENGLISH_PARAGRAPH)
        assert adapter.validate_unit_translation("chunk_0", "x") is None

    @pytest.mark.asyncio
    async def test_cjk_to_spanish_normal_length_passes(self, tmp_path):
        """A faithful Chinese -> Spanish translation must not trip the check."""
        adapter = await _prepared_adapter(tmp_path, SENTENCE * 4)
        spanish = (
            "Miraba las montañas a lo lejos, pensando en muchas cosas. "
            "\"¿Por qué te vas?\", preguntó ella. A Yan no respondió; "
            "simplemente se dio la vuelta en silencio. "
        ) * 4
        assert adapter.validate_unit_translation("chunk_0", spanish) is None


# ---------------------------------------------------------------------------
# 2b. Truncated output is rejected by the request layer
# ---------------------------------------------------------------------------

class _StubClient:
    def __init__(self, response):
        self._response = response
        self._extractor = TranslationExtractor(TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT)

    async def generate(self, prompt, system_prompt=None):
        return self._response

    def extract_translation(self, text):
        return self._extractor.extract(text)


async def _request(response, has_placeholders=False, raw_response_callback=None):
    return await _make_llm_request_with_adaptive_context(
        main_content="Hello world, this is a test.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="French",
        model="test-model",
        llm_client=_StubClient(response),
        log_callback=None,
        has_placeholders=has_placeholders,
        raw_response_callback=raw_response_callback,
    )


class TestTruncatedOutputRejected:

    @pytest.mark.asyncio
    async def test_output_limit_finish_reason_fails_the_chunk(self):
        response = LLMResponse(
            content=f"{TRANSLATE_TAG_IN}Bonjour le monde{TRANSLATE_TAG_OUT}",
            finish_reason="length",
            output_truncated=True,
        )
        translated, _, _ = await _request(response)
        assert translated is None

    @pytest.mark.asyncio
    async def test_unclosed_tag_is_not_saved_as_raw_fallback(self):
        response = LLMResponse(content=f"{TRANSLATE_TAG_IN}\nBonjour le mo")
        translated, _, _ = await _request(response)
        assert translated is None

    @pytest.mark.asyncio
    async def test_normal_response_still_passes(self):
        response = LLMResponse(
            content=f"{TRANSLATE_TAG_IN}Bonjour le monde{TRANSLATE_TAG_OUT}",
            finish_reason="stop",
        )
        translated, _, _ = await _request(response)
        assert translated == "Bonjour le monde"

    @pytest.mark.asyncio
    async def test_raw_callback_sees_the_response_before_rejection(self):
        seen = []
        response = LLMResponse(content=f"{TRANSLATE_TAG_IN}\nBonjour le mo")
        await _request(response, raw_response_callback=seen.append)
        assert seen == [response]

    @pytest.mark.asyncio
    async def test_failing_raw_callback_does_not_break_the_request(self):
        def boom(_response):
            raise OSError("disk full")

        response = LLMResponse(content=f"{TRANSLATE_TAG_IN}Bonjour{TRANSLATE_TAG_OUT}")
        translated, _, _ = await _request(response, raw_response_callback=boom)
        assert translated == "Bonjour"


class TestProvidersReportOutputLimit:

    @pytest.mark.parametrize("reason, expected", [
        ("length", True), ("MAX_TOKENS", True), ("max_output_tokens", True),
        ("stop", False), ("STOP", False), (None, False), ("", False),
    ])
    def test_is_output_limit_finish(self, reason, expected):
        assert is_output_limit_finish(reason) is expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("finish_reason, truncated", [("length", True), ("stop", False)])
    async def test_openrouter_reads_finish_reason(self, finish_reason, truncated):
        payload = {
            "choices": [{"message": {"content": "partial"}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "cost": 0,
        }
        provider = OpenRouterProvider(api_key="sk-or-xxxxxxxx", model="qwen/qwen3-235b",
                                      disable_thinking=False)
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        )
        try:
            result = await provider.generate("prompt")
        finally:
            await provider.close()
        assert result.finish_reason == finish_reason
        assert result.output_truncated is truncated

    @pytest.mark.asyncio
    async def test_gemini_max_tokens_sets_output_truncated(self):
        payload = {
            "candidates": [{"content": {"parts": [{"text": "partial"}]},
                            "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        )
        try:
            result = await provider.generate("prompt")
        finally:
            await provider.close()
        assert result.output_truncated is True


# ---------------------------------------------------------------------------
# End to end through GenericTranslator (real request layer, stubbed LLM)
# ---------------------------------------------------------------------------

class _ScriptedLLMClient:
    """Stands in for LLMClient: answers from a script, extracts like the real one."""

    script = []
    calls = []

    def __init__(self, **kwargs):
        self._extractor = TranslationExtractor(TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT)

    async def generate(self, prompt, system_prompt=None):
        type(self).calls.append({"prompt": prompt, "system": system_prompt})
        responder = type(self).script[min(len(type(self).calls) - 1, len(type(self).script) - 1)]
        return responder(prompt)

    def extract_translation(self, text):
        return self._extractor.extract(text)


def _tagged(text, finish_reason="stop"):
    return LLMResponse(
        content=f"{TRANSLATE_TAG_IN}{text}{TRANSLATE_TAG_OUT}",
        prompt_tokens=100,
        completion_tokens=50,
        finish_reason=finish_reason,
        output_truncated=is_output_limit_finish(finish_reason),
    )


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    # start_job copies the input under uploads_dir; keep it out of the real data/.
    manager.uploads_dir = tmp_path / "uploads"
    yield manager
    manager.close()


@pytest.fixture
def scripted_llm(monkeypatch):
    _ScriptedLLMClient.script = []
    _ScriptedLLMClient.calls = []
    monkeypatch.setattr(llm_client_module, "LLMClient", _ScriptedLLMClient)
    # The scripted "translations" are the English source upper-cased, which the
    # source-residue check (issue #270) rightly reads as an echo. These tests
    # count calls for other reasons, so keep that check out of them.
    monkeypatch.setattr(config_module, "SOURCE_RESIDUE_CHECK_ENABLED", False)
    return _ScriptedLLMClient


async def _translate(tmp_path, cm, text, logs=None):
    source = tmp_path / "in.txt"
    output = tmp_path / "out.txt"
    source.write_text(text, encoding="utf-8")
    adapter = TxtAdapter(str(source), str(output), {"max_tokens_per_chunk": 450})
    translator = GenericTranslator(adapter=adapter, checkpoint_manager=cm, translation_id="job-1")

    def log_callback(msg_type, message, data=None):
        if logs is not None:
            logs.append((msg_type, message))

    success = await translator.translate(
        source_language="English",
        target_language="French",
        model_name="fake-model",
        llm_provider="ollama",
        log_callback=log_callback,
    )
    return success, output


class TestGenericTranslatorIntegrity:

    @pytest.mark.asyncio
    async def test_skipped_passage_is_retried_then_succeeds(self, tmp_path, cm, scripted_llm):
        scripted_llm.script = [
            lambda prompt: _tagged("La rivière."),
            lambda prompt: _tagged(ENGLISH_PARAGRAPH.upper()),
        ]
        success, output = await _translate(tmp_path, cm, ENGLISH_PARAGRAPH)

        assert success is True
        assert len(scripted_llm.calls) == 2
        retry_prompt = scripted_llm.calls[1]["prompt"] + (scripted_llm.calls[1]["system"] or "")
        assert "Translate the ENTIRE text" in retry_prompt
        assert "[N] index marker" not in retry_prompt
        assert output.read_text(encoding="utf-8") == ENGLISH_PARAGRAPH.upper()

    @pytest.mark.asyncio
    async def test_persistently_short_chunk_ends_the_job_partial(self, tmp_path, cm, scripted_llm):
        scripted_llm.script = [lambda prompt: _tagged("La rivière.")]
        logs = []
        success, _ = await _translate(tmp_path, cm, ENGLISH_PARAGRAPH, logs)

        assert success is False
        assert 1 + config_module.UNIT_VALIDATION_RETRIES == len(scripted_llm.calls)
        assert any(t == "translation_partial" for t, _ in logs)
        assert cm.db.get_job("job-1")["status"] == "partial"

    @pytest.mark.asyncio
    async def test_output_limit_hit_fails_the_unit(self, tmp_path, cm, scripted_llm):
        scripted_llm.script = [lambda prompt: _tagged(ENGLISH_PARAGRAPH, finish_reason="length")]
        logs = []
        success, output = await _translate(tmp_path, cm, ENGLISH_PARAGRAPH, logs)

        assert success is False
        assert any(t == "output_truncated" for t, _ in logs)
        # The source stays in place of the failed chunk: visible, never silently dropped.
        assert output.read_text(encoding="utf-8") == ENGLISH_PARAGRAPH


class TestRawResponseLog:

    @pytest.mark.asyncio
    async def test_disabled_by_default_writes_nothing(self, tmp_path, cm, scripted_llm, monkeypatch):
        monkeypatch.setattr(config_module, "SAVE_RAW_LLM_RESPONSES", False)
        scripted_llm.script = [lambda prompt: _tagged(ENGLISH_PARAGRAPH.upper())]
        await _translate(tmp_path, cm, ENGLISH_PARAGRAPH)

        assert not raw_response_log_path(tmp_path, "job-1").exists()

    @pytest.mark.asyncio
    async def test_every_call_is_recorded_with_its_attempt(self, tmp_path, cm, scripted_llm, monkeypatch):
        monkeypatch.setattr(config_module, "SAVE_RAW_LLM_RESPONSES", True)
        scripted_llm.script = [
            lambda prompt: _tagged("signifi- cada"),
            lambda prompt: _tagged(ENGLISH_PARAGRAPH.upper()),
        ]
        success, _ = await _translate(tmp_path, cm, ENGLISH_PARAGRAPH)
        assert success is True

        path = raw_response_log_path(tmp_path, "job-1")
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        assert [(e["unit_index"], e["attempt"]) for e in entries] == [(0, 1), (0, 2)]
        assert entries[0]["raw_response"] == f"{TRANSLATE_TAG_IN}signifi- cada{TRANSLATE_TAG_OUT}"
        assert entries[0]["model"] == "fake-model"
        assert entries[1]["finish_reason"] == "stop"

    def test_translation_id_cannot_escape_the_directory(self, tmp_path):
        path = raw_response_log_path(tmp_path, "../../etc/passwd")
        assert path.parent == tmp_path / "raw_responses"
