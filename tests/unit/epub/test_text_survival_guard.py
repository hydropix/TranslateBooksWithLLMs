"""Regression tests: a chunk whose text did not survive translation is retried.

Placeholder-only validation accepted "[id0][id1]" as the translation of a
whole paragraph, so the paragraph vanished from the book.
"""
import pytest

import src.core.epub.xhtml_translator as xt
from src.core.epub.text_survival import text_survival_problem
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.xhtml_translation_state import CHUNK_TRANSLATED, CHUNK_UNTRANSLATED

EN = "[id0]It was a bright cold day in April, and the clocks were striking thirteen.[id1]"


@pytest.mark.parametrize("source,translated", [
    (EN, "[id0]Il faisait beau et froid en avril, et les horloges sonnaient treize heures.[id1]"),
    (EN, "[id0]四月里一个晴朗寒冷的日子，时钟敲了十三下。[id1]"),          # EN -> ZH shrinks
    ("[id0]四月里一个晴朗寒冷的日子，时钟敲了十三下。[id1]",
     "[id0]It was a bright cold day in April, and the clocks were striking thirteen.[id1]"),  # ZH -> EN grows
    ("[id0]Oui.[id1]", "[id0]Yes.[id1]"),
    ("[id0]OK[id1]", "[id0]D'accord, entendu[id1]"),
])
def test_plausible_translations_pass(source, translated):
    assert text_survival_problem(source, translated) is None


@pytest.mark.parametrize("source,translated", [
    (EN, "[id0][id1]"),
    (EN, "[id0]   [id1]"),
    (EN, "[id0]Il[id1]"),
    (EN, "[id0]" + "Voici une explication détaillée de la traduction. " * 12 + "[id1]"),
])
def test_lost_or_runaway_text_is_rejected(source, translated):
    assert text_survival_problem(source, translated) is not None


async def _translate(monkeypatch, replies, max_retries=2):
    calls = []

    async def fake_request(main_content, *args, **kwargs):
        calls.append(main_content)
        return replies[min(len(calls), len(replies)) - 1]

    monkeypatch.setattr(xt, "generate_translation_request", fake_request)
    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", False)
    stats = TranslationMetrics()
    result = await xt.translate_chunk_with_fallback(
        chunk_text=EN,
        local_tag_map={"[id0]": "<p>", "[id1]": "</p>"},
        global_indices=[7, 8],
        source_language="English",
        target_language="French",
        model_name="fake",
        llm_client=object(),
        stats=stats,
        max_retries=max_retries,
        chunk_index=0,
    )
    return result, stats, calls


@pytest.mark.asyncio
async def test_placeholder_only_reply_is_retried(monkeypatch):
    good = "[id0]Il faisait beau et froid en avril, et les horloges sonnaient treize heures.[id1]"
    result, stats, calls = await _translate(monkeypatch, ["[id0][id1]", good])
    assert len(calls) == 2
    assert result == good.replace("[id0]", "[id7]").replace("[id1]", "[id8]")
    assert stats.chunk_outcomes[0] == CHUNK_TRANSLATED


@pytest.mark.asyncio
async def test_persistent_text_loss_falls_back_to_source(monkeypatch):
    result, stats, calls = await _translate(monkeypatch, ["[id0][id1]"])
    assert len(calls) == 2
    assert "bright cold day" in result
    assert stats.chunk_outcomes[0] == CHUNK_UNTRANSLATED
    assert stats.fallback_used == 1
