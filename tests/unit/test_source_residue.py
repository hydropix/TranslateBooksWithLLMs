"""Text left in the source language after translation (issue #270).

A chunk used to count as translated as soon as its placeholders survived, so a
model that skipped a paragraph, or echoed the whole chunk back, produced a
"successful" chunk that no retry ever looked at again. These tests pin:

1. the detector itself (what counts as residue, what is tolerated);
2. the one-shot corrective retry inside generate_translation_request;
3. the placeholder pipeline recording a residue chunk as untranslated;
4. the Plain Text Mode pipeline doing the same;

No network: every LLM call is stubbed.
"""
import asyncio

import pytest

import src.config as config_module
import src.core.common.plain_text_pipeline as plain_pipeline
import src.core.epub.chunk_fallback as chunk_fallback
import src.core.translator as core_translator
from src.core.common.source_residue import detect_blocking_residue, detect_source_residue
from src.core.epub.chunk_settings import ChunkLoopSettings
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
        text, "", source_language="Korean", target_language="English",
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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("first_fallback, retry_fallback, retry_wins, expected", [
        (True, False, True, 0),   # the raw first answer was replaced
        (False, True, True, 1),   # the kept retry came from a raw reply
        (True, True, False, 1),   # the raw first answer was kept
        (False, True, False, 0),  # the raw retry was thrown away
    ])
    async def test_raw_fallback_is_counted_for_the_kept_answer_only(
            self, monkeypatch, first_fallback, retry_fallback, retry_wins, expected):
        class _Response:
            def __init__(self, was_fallback):
                self.was_fallback = was_fallback

        first = "He opened the door. 바람이 차가웠다."
        retry = "He opened the door. The wind was cold." if retry_wins else "그는 문을 열었다. 바람이 차가웠다."
        replies = [(first, first_fallback), (retry, retry_fallback)]
        calls = []

        async def fake(**kwargs):
            text, was_fallback = replies[len(calls)]
            calls.append(text)
            return text, "", _Response(was_fallback)

        monkeypatch.setattr(core_translator, "_make_llm_request_with_adaptive_context", fake)
        counted = []
        result = await _request("그는 문을 열었다. 바람이 차가웠다.",
                                on_raw_fallback=lambda: counted.append(1))
        assert result == (retry if retry_wins else first)
        assert len(calls) == 2
        assert len(counted) == expected

    def test_retry_hint_reaches_the_user_prompt(self):
        from src.prompts.prompts import generate_translation_prompt
        plain = generate_translation_prompt("x", "", "Korean", "English")
        hinted = generate_translation_prompt("x", "", "Korean", "English",
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
    monkeypatch.setattr(chunk_fallback, "generate_translation_request", fake)


async def _translate_chunk(text, tag_map, stats, source_language="Korean",
                           target_language="English"):
    settings = ChunkLoopSettings(
        source_language=source_language, target_language=target_language,
        model_name="m", llm_client=object(), max_retries=1)
    return await chunk_fallback.translate_chunk_with_fallback(
        chunk_text=text, local_tag_map=tag_map, global_indices=list(range(len(tag_map))),
        settings=settings, stats=stats, placeholder_format=("[id", "]"), chunk_index=0)


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
        await _translate_chunk(
            "[id0]Rowling, J. K. Harry Potter and the Philosopher's Stone. Bloomsbury, 1997.[id1]",
            {"[id0]": "<p>", "[id1]": "</p>"}, stats,
            source_language="English", target_language="French")
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
# 5. Single lowercase source words inside a non-Latin target (issue #353)
# ---------------------------------------------------------------------------

class TestLowercaseWordResidue:

    @pytest.mark.parametrize("source, output, word", [
        ("I tried not to fidget too much under his very intense gaze.",
         "Я постарался не слишком ёрзать под его очень intense взглядом.", "intense"),
        ("Frustration in the class steadily grew.",
         "Разочарование в классе steadily росло.", "steadily"),
        ("It would be a gross understatement.",
         "Это было бы gross understatement.", "gross, understatement"),
        ("Now I had to pack my things.",
         "Сейчас мне нужно было pack вещи.", "pack"),
    ])
    def test_issue_examples_are_retry_only_findings(self, source, output, word):
        report = detect_source_residue(source, output, "English", "Russian")
        assert report is not None
        assert report.kind == "word"
        assert report.retry_only and not report.blocking
        assert report.sample == word
        assert detect_blocking_residue(source, output, "English", "Russian") is None

    def test_other_non_latin_targets(self):
        assert detect_source_residue(
            "His gaze was intense.", "Το βλέμμα του ήταν intense.",
            "English", "Greek").kind == "word"
        assert detect_source_residue(
            "His gaze was intense.", "彼の視線はintenseだった。",
            "English", "Japanese").kind == "word"

    @pytest.mark.parametrize("source, output", [
        # Names and acronyms are capitalised.
        ("Harry met Hermione at NASA.", "Harry встретил Hermione в NASA."),
        # URLs, emails, paths, file names, code.
        ("See https://example.com/docs for details.",
         "Подробности на https://example.com/docs."),
        ("Write to support@example.com today.", "Напишите на support@example.com сегодня."),
        ("Open the config/settings.yaml file.", "Откройте файл config/settings.yaml."),
        ("Call the my_function helper.", "Вызовите помощник my_function."),
        ("Run `pip install` first.", "Сначала выполните `pip install`."),
        # camelCase brands.
        ("She bought an iPhone.", "Она купила iPhone."),
        # A lowercase Latin word that is not in the source is not residue.
        ("He shrugged.", "Он пожал плечами, как бы говоря «ну и что», et cetera."),
        # Single letters are variables or initials.
        ("Let x be the answer.", "Пусть x будет ответом."),
        # A Latin word in a line with no target-script text (a quotation or
        # a code line on its own) is left to the long-run check.
        ("intense\nHe looked at her.", "intense\nОн посмотрел на неё."),
    ])
    def test_false_positives_are_not_flagged(self, source, output):
        assert detect_source_residue(source, output, "English", "Russian") is None

    def test_glossary_terms_are_whitelisted(self):
        source = "He whispered lumos and the wand lit up."
        output = "Он прошептал lumos, и палочка засветилась."
        assert detect_source_residue(source, output, "English", "Russian").kind == "word"
        assert detect_source_residue(source, output, "English", "Russian",
                                     allowed_terms=["Lumos"]) is None
        assert detect_source_residue(source, output, "English", "Russian",
                                     allowed_terms={"lumos": "люмос"}.values()) is not None

    def test_latin_target_is_unchanged(self):
        assert detect_source_residue(
            "His gaze was intense and steady.",
            "Son regard était intense et steady.", "English", "French") is None

    def test_serbian_target_allows_latin(self):
        assert detect_source_residue(
            "His gaze was intense.", "Његов поглед је био intense.",
            "English", "Serbian") is None

    @pytest.mark.asyncio
    async def test_word_residue_earns_one_retry_using_the_glossary(self, monkeypatch):
        seen = _stub_adaptive_request(monkeypatch, [
            "Он прошептал lumos очень intense.",
            "Он прошептал lumos очень напряжённо.",
        ])
        result = await core_translator.generate_translation_request(
            "He whispered lumos very intense.", "",
            source_language="English", target_language="Russian",
            model="m", llm_client=object(),
            prompt_options={"glossary_terms": {"lumos": "lumos"}})
        assert result == "Он прошептал lumos очень напряжённо."
        assert len(seen) == 2 and seen[1]["source_residue_retry"] is True

    @pytest.mark.asyncio
    async def test_glossary_only_latin_word_triggers_no_retry(self, monkeypatch):
        seen = _stub_adaptive_request(monkeypatch, ["Он прошептал lumos."])
        result = await core_translator.generate_translation_request(
            "He whispered lumos.", "",
            source_language="English", target_language="Russian",
            model="m", llm_client=object(),
            prompt_options={"glossary_terms": {"lumos": "lumos"}})
        assert result == "Он прошептал lumos."
        assert len(seen) == 1
