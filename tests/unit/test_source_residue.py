"""Text left in the source language after translation (issue #270).

A chunk used to count as translated as soon as its placeholders survived, so a
model that skipped a paragraph, or echoed the whole chunk back, produced a
"successful" chunk that no retry ever looked at again. These tests pin:

1. the detector itself (what counts as residue, what is tolerated);
2. the one-shot corrective retry inside generate_translation_request;
3. the placeholder pipeline recording a residue chunk as untranslated;
4. the Plain Text Mode pipeline doing the same;
5. the translation of <head><title>, which the body pipeline never sees.

No network: every LLM call is stubbed.
"""
import asyncio

import pytest
from lxml import etree

import src.config as config_module
import src.core.common.plain_text_pipeline as plain_pipeline
import src.core.epub.xhtml_translator as xt
import src.core.translator as core_translator
from src.core.common.source_residue import detect_blocking_residue, detect_source_residue
from src.core.epub.head_title import snapshot_source_title, translate_head_title
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.xhtml_translation_state import (
    CHUNK_TRANSLATED,
    CHUNK_UNTRANSLATED,
)


ENGLISH = "He opened the door and walked slowly into the cold night air."


# ---------------------------------------------------------------------------
# 1. Detector
# ---------------------------------------------------------------------------

class TestDetector:

    def test_leftover_korean_sentence_is_flagged(self):
        report = detect_source_residue(
            "[id0]그는 문을 열었다.[id1][id2]바람이 차가웠다.[id3]",
            "[id0]He opened the door.[id1][id2]바람이 차가웠다.[id3]",
            "Korean", "English")
        assert report is not None
        assert report.kind == "script"
        assert "바람이" in report.sample

    def test_kept_korean_name_is_tolerated(self):
        assert detect_source_residue(
            "그는 김민준을 만났다.", "He met 김민준.", "Korean", "English") is None

    def test_latin_brand_names_in_japanese_are_tolerated(self):
        assert detect_source_residue(
            "He met Kim at the Starbucks on Main Street yesterday.",
            "彼は昨日Main Streetのスターバックスでキムに会った。",
            "English", "Japanese") is None

    def test_leftover_russian_sentence_is_flagged(self):
        report = detect_source_residue(
            "Он открыл дверь и медленно вышел в холодную ночь.",
            "Il ouvrit la porte. Он медленно вышел в холодную ночь.",
            "Russian", "French")
        assert report is not None and report.kind == "script"

    def test_echo_is_flagged_for_same_script_pairs(self):
        report = detect_source_residue(ENGLISH, ENGLISH, "English", "French")
        assert report is not None and report.kind == "echo"

    def test_echo_and_latin_residue_are_retry_only(self):
        echo = detect_source_residue(ENGLISH, ENGLISH, "English", "French")
        latin = detect_source_residue(
            ENGLISH, "彼は言った：He opened the door and walked slowly into the cold night air.",
            "English", "Japanese")
        korean = detect_source_residue("바람이 차가웠다.", "바람이 차가웠다.", "Korean", "English")
        assert echo.retry_only and latin.retry_only
        assert korean.blocking
        assert detect_blocking_residue(ENGLISH, ENGLISH, "English", "French") is None

    @pytest.mark.parametrize("text", [
        "Rowling, J. K. Harry Potter and the Philosopher's Stone. Bloomsbury, 1997.",
        "print('hello world')\nfor item in items: process(item)",
        "Characters: Kim Minjun, Park Seoyeon, Lee Jiho",
    ])
    def test_legitimately_identical_text_never_blocks(self, text):
        assert detect_blocking_residue(text, text, "English", "French") is None

    def test_real_translation_is_accepted(self):
        assert detect_source_residue(
            ENGLISH, "Il ouvrit la porte et s'avança lentement dans l'air froid.",
            "English", "French") is None

    def test_short_identical_text_is_not_an_echo(self):
        assert detect_source_residue("Paris", "Paris", "English", "French") is None

    def test_shared_han_script_is_not_residue(self):
        # Kanji are valid Chinese: only kana would be Japanese residue.
        assert detect_source_residue(
            "彼は扉を開けた。", "他打开了门。", "Japanese", "Chinese (Simplified)") is None

    def test_unknown_language_disables_the_check(self):
        assert detect_source_residue("안녕하세요 여러분 반갑습니다", "안녕하세요 여러분 반갑습니다",
                                     "Korean", "Auto") is None

    def test_same_language_disables_the_check(self):
        assert detect_source_residue(ENGLISH, ENGLISH, "English", "English") is None

    def test_config_switch_disables_the_check(self, monkeypatch):
        monkeypatch.setattr(config_module, "SOURCE_RESIDUE_CHECK_ENABLED", False)
        assert detect_source_residue(ENGLISH, ENGLISH, "English", "French") is None


# ---------------------------------------------------------------------------
# 2. Corrective retry in generate_translation_request
# ---------------------------------------------------------------------------

def _stub_adaptive_request(monkeypatch, answers):
    """Answer successive LLM requests from `answers`, recording prompt_options."""
    seen_options = []

    async def fake(**kwargs):
        seen_options.append(kwargs.get("prompt_options") or {})
        return answers[min(len(seen_options) - 1, len(answers) - 1)], "", None

    monkeypatch.setattr(core_translator, "_make_llm_request_with_adaptive_context", fake)
    return seen_options


async def _request(text, **kwargs):
    return await core_translator.generate_translation_request(
        text, "", "", "", source_language="Korean", target_language="English",
        model="m", llm_client=object(), **kwargs)


class TestCorrectiveRetry:

    @pytest.mark.asyncio
    async def test_residue_triggers_one_retry_with_the_hint(self, monkeypatch):
        seen = _stub_adaptive_request(monkeypatch, [
            "He opened the door. 바람이 차가웠다.",
            "He opened the door. The wind was cold.",
        ])
        result = await _request("그는 문을 열었다. 바람이 차가웠다.")

        assert result == "He opened the door. The wind was cold."
        assert len(seen) == 2
        assert not seen[0].get("source_residue_retry")
        assert seen[1]["source_residue_retry"] is True

    @pytest.mark.asyncio
    async def test_clean_answer_costs_no_extra_request(self, monkeypatch):
        seen = _stub_adaptive_request(monkeypatch, ["He opened the door. The wind was cold."])
        await _request("그는 문을 열었다. 바람이 차가웠다.")
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_worse_retry_keeps_the_first_answer(self, monkeypatch):
        first = "He opened the door. 바람이 차가웠다."
        _stub_adaptive_request(monkeypatch, [first, "그는 문을 열었다. 바람이 차가웠다."])
        assert await _request("그는 문을 열었다. 바람이 차가웠다.") == first

    @pytest.mark.asyncio
    async def test_retry_that_breaks_placeholders_is_discarded(self, monkeypatch):
        first = "[id0]He opened the door. 바람이 차가웠다.[id1]"
        _stub_adaptive_request(monkeypatch, [first, "He opened the door. The wind was cold."])
        result = await _request("[id0]그는 문을 열었다. 바람이 차가웠다.[id1]",
                                has_placeholders=True, placeholder_format=("[id", "]"))
        assert result == first

    def test_retry_hint_reaches_the_user_prompt(self):
        from src.prompts.prompts import generate_translation_prompt
        plain = generate_translation_prompt("x", "", "", "", "Korean", "English")
        hinted = generate_translation_prompt("x", "", "", "", "Korean", "English",
                                             prompt_options={"source_residue_retry": True})
        assert "PREVIOUS ATTEMPT INCOMPLETE" in hinted.user
        assert "PREVIOUS ATTEMPT INCOMPLETE" not in plain.user
        # The cacheable system prompt never changes.
        assert hinted.system == plain.system


# ---------------------------------------------------------------------------
# 3. Placeholder pipeline
# ---------------------------------------------------------------------------

def _stub_xhtml_generate(monkeypatch, answer):
    async def fake(main_content, *args, **kwargs):
        return answer(main_content)
    monkeypatch.setattr(xt, "generate_translation_request", fake)


async def _translate_chunk(text, tag_map, stats):
    return await xt.translate_chunk_with_fallback(
        chunk_text=text, local_tag_map=tag_map, global_indices=list(range(len(tag_map))),
        source_language="Korean", target_language="English", model_name="m",
        llm_client=object(), stats=stats, max_retries=1,
        placeholder_format=("[id", "]"), chunk_index=0)


class TestPlaceholderPipeline:

    @pytest.mark.asyncio
    async def test_residue_chunk_is_recorded_untranslated_but_keeps_its_translation(self, monkeypatch):
        _stub_xhtml_generate(monkeypatch, lambda t: t.replace("그는 문을 열었다.", "He opened the door."))
        stats = TranslationMetrics()
        result = await _translate_chunk(
            "[id0]그는 문을 열었다.[id1][id2]바람이 차가웠다.[id3]",
            {"[id0]": "<p>", "[id1]": "</p>", "[id2]": "<p>", "[id3]": "</p>"}, stats)

        assert stats.chunk_outcomes == {0: CHUNK_UNTRANSLATED}
        assert stats.fallback_used == 1
        assert stats.successful_first_try == 0
        assert "He opened the door." in result

    @pytest.mark.asyncio
    async def test_echoed_korean_chunk_is_recorded_untranslated(self, monkeypatch):
        # The echo itself is retry-only, but the Hangul left in it blocks.
        _stub_xhtml_generate(monkeypatch, lambda t: t)
        stats = TranslationMetrics()
        await _translate_chunk("[id0]그는 문을 열었다. 바람이 차가웠다.[id1]",
                               {"[id0]": "<p>", "[id1]": "</p>"}, stats)
        assert stats.chunk_outcomes == {0: CHUNK_UNTRANSLATED}

    @pytest.mark.asyncio
    async def test_echoed_bibliography_stays_translated(self, monkeypatch):
        _stub_xhtml_generate(monkeypatch, lambda t: t)
        stats = TranslationMetrics()
        await xt.translate_chunk_with_fallback(
            chunk_text="[id0]Rowling, J. K. Harry Potter and the Philosopher's Stone. Bloomsbury, 1997.[id1]",
            local_tag_map={"[id0]": "<p>", "[id1]": "</p>"}, global_indices=[0, 1],
            source_language="English", target_language="French", model_name="m",
            llm_client=object(), stats=stats, max_retries=1,
            placeholder_format=("[id", "]"), chunk_index=0)
        assert stats.chunk_outcomes == {0: CHUNK_TRANSLATED}
        assert stats.fallback_used == 0

    @pytest.mark.asyncio
    async def test_clean_chunk_is_still_translated(self, monkeypatch):
        _stub_xhtml_generate(monkeypatch, lambda t: "[id0]He opened the door.[id1]")
        stats = TranslationMetrics()
        await _translate_chunk("[id0]그는 문을 열었다.[id1]", {"[id0]": "<p>", "[id1]": "</p>"}, stats)
        assert stats.chunk_outcomes == {0: CHUNK_TRANSLATED}
        assert stats.fallback_used == 0


# ---------------------------------------------------------------------------
# 4. Plain Text Mode pipeline
# ---------------------------------------------------------------------------

class TestPlainTextPipeline:

    @pytest.mark.asyncio
    async def test_residue_segment_is_recorded_untranslated(self, monkeypatch):
        async def fake(*, main_content, **kwargs):
            await asyncio.sleep(0)
            return main_content.replace("그는 문을 열었다.", "He opened the door.")

        monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake)
        persisted = []

        def hook(segments, prefix, next_index, stats_dict, chunk_statuses=None):
            persisted.append(list(chunk_statuses))

        translated, stats, interrupted = await plain_pipeline.translate_paragraphs_plain(
            ["그는 문을 열었다. 바람이 차가웠다."],
            "Korean", "English", "m", object(), max_tokens_per_chunk=2000,
            checkpoint_hook=hook, count_failures_as_fallback=True)

        assert not interrupted
        assert persisted[-1] == [CHUNK_UNTRANSLATED]
        assert stats.fallback_used == 1
        assert stats.successful_first_try == 0
        assert translated[0].startswith("He opened the door.")


    @pytest.mark.asyncio
    async def test_docx_echo_is_not_a_failed_chunk(self, monkeypatch):
        # DOCX and PDF have no per-segment retry: a blocking finding there is
        # a failed chunk, so a legitimate echo must never become one.
        async def fake(*, main_content, **kwargs):
            await asyncio.sleep(0)
            return main_content

        monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake)
        text = "Rowling, J. K. Harry Potter and the Philosopher's Stone. Bloomsbury, 1997."
        translated, stats, _ = await plain_pipeline.translate_paragraphs_plain(
            [text], "English", "French", "m", object(), max_tokens_per_chunk=2000)
        assert stats.failed_chunks == 0
        assert stats.fallback_used == 0
        assert translated == [text]


# ---------------------------------------------------------------------------
# 5. <head><title>
# ---------------------------------------------------------------------------

XHTML = """<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>{title}</title></head>
<body><h1>{heading}</h1><p>본문</p></body></html>"""


def _doc(title, heading):
    return etree.fromstring(XHTML.format(title=title, heading=heading).encode("utf-8"))


def _title(doc):
    return doc.find(".//{http://www.w3.org/1999/xhtml}title").text


class _FakeTranslate:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def __call__(self, text, *args, **kwargs):
        self.calls.append(text)
        return self.answer


async def _translate_title(doc, snapshot, translate, cache=None):
    return await translate_head_title(
        doc, snapshot, source_language="Korean", target_language="English",
        model_name="m", llm_client=object(),
        cache={} if cache is None else cache, translate=translate)


class TestHeadTitle:

    @pytest.mark.asyncio
    async def test_title_equal_to_heading_reuses_its_translation(self):
        doc = _doc("제1장 시작", "제1장 시작")
        snapshot = snapshot_source_title(doc)
        doc.find(".//{http://www.w3.org/1999/xhtml}h1").text = "Chapter 1: The Beginning"
        translate = _FakeTranslate("unused")

        assert await _translate_title(doc, snapshot, translate) is True
        assert _title(doc) == "Chapter 1: The Beginning"
        assert translate.calls == []

    @pytest.mark.asyncio
    async def test_other_title_is_translated_once_per_job(self):
        cache = {}
        translate = _FakeTranslate("The Seventh Lamp")
        for _ in range(3):
            doc = _doc("일곱 번째 등불", "제1장")
            await _translate_title(doc, snapshot_source_title(doc), translate, cache)
            assert _title(doc) == "The Seventh Lamp"
        assert translate.calls == ["일곱 번째 등불"]

    @pytest.mark.asyncio
    async def test_untranslated_heading_is_not_reused(self):
        # Bilingual output (or a heading chunk left untranslated) keeps the
        # source heading first: it must not be mistaken for a translation.
        doc = _doc("제1장 시작", "제1장 시작")
        translate = _FakeTranslate("Chapter 1: The Beginning")
        await _translate_title(doc, snapshot_source_title(doc), translate)
        assert _title(doc) == "Chapter 1: The Beginning"
        assert translate.calls == ["제1장 시작"]

    @pytest.mark.asyncio
    async def test_rejected_answer_keeps_the_source_title(self):
        doc = _doc("일곱 번째 등불", "제1장")
        translate = _FakeTranslate("일곱 번째 등불 그대로입니다")
        assert await _translate_title(doc, snapshot_source_title(doc), translate) is False
        assert _title(doc) == "일곱 번째 등불"

    @pytest.mark.asyncio
    async def test_rate_limit_keeps_the_source_title_without_pausing(self):
        from src.core.llm.exceptions import RateLimitError

        async def limited(*args, **kwargs):
            raise RateLimitError("429", retry_after=1, provider="test")

        doc = _doc("일곱 번째 등불", "제1장")
        assert await _translate_title(doc, snapshot_source_title(doc), limited) is False
        assert _title(doc) == "일곱 번째 등불"

    @pytest.mark.asyncio
    async def test_title_without_letters_is_left_alone(self):
        doc = _doc("12", "12")
        translate = _FakeTranslate("x")
        assert await _translate_title(doc, snapshot_source_title(doc), translate) is False
        assert translate.calls == []
