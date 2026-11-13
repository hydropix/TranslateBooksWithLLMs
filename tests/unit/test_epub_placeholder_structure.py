"""Regression tests: placeholder validation must reject mis-nested restorations.

validate_strict used to sort the found indices before comparing them, so a
translation such as "[id1]Bonjour[id0]" for {"[id0]": "<p>", "[id1]": "</p>"}
passed and produced "</p>Bonjour<p>" in the final XHTML.
"""
import asyncio

from src.core.epub import xhtml_translator
from src.core.epub.placeholder_validator import PlaceholderValidator, nesting_signature
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.xhtml_translation_state import CHUNK_TRANSLATED


def test_swapped_block_placeholders_are_rejected():
    tag_map = {"[id0]": "<p>", "[id1]": "</p>"}
    ok, error = PlaceholderValidator.validate_strict("[id1]Bonjour[id0]", tag_map)
    assert not ok
    assert "nested" in error or "unbalanced" in error


def test_correct_order_is_accepted():
    tag_map = {"[id0]": "<p>", "[id1]": "</p>"}
    assert PlaceholderValidator.validate_strict("[id0]Bonjour[id1]", tag_map) == (True, "")


def test_reordered_inline_spans_are_accepted():
    # The target language may legitimately move an inline span.
    tag_map = {
        "[id0]": "<p>", "[id1]": "<em>", "[id2]": "</em>",
        "[id3]": "<b>", "[id4]": "</b>", "[id5]": "</p>",
    }
    ok, _ = PlaceholderValidator.validate_strict(
        "[id0][id3]rouge[id4] et [id1]bleu[id2][id5]", tag_map
    )
    assert ok


def test_crossed_inline_spans_are_rejected():
    tag_map = {
        "[id0]": "<p>", "[id1]": "<em>", "[id2]": "</em>", "[id3]": "</p>",
    }
    ok, _ = PlaceholderValidator.validate_strict("[id0][id2]a[id1][id3]", tag_map)
    assert not ok


def test_chunk_slices_that_close_outer_tags_are_accepted():
    # A chunk may close tags opened by an earlier chunk and open tags for a later one.
    tag_map = {"[id0]": "</p><p>", "[id1]": "</p></div><div><p>"}
    assert PlaceholderValidator.validate_strict("[id0]Salut[id1]", tag_map)[0]


def test_moving_a_block_boundary_outside_its_slice_is_rejected():
    tag_map = {"[id0]": "<div><p>", "[id1]": "</p>", "[id2]": "<p>", "[id3]": "</p></div>"}
    ok, _ = PlaceholderValidator.validate_strict("[id0]a[id1][id3]b[id2]", tag_map)
    assert not ok


def test_nesting_signature_ignores_void_and_self_closing_tags():
    assert nesting_signature("<p>a<br/>b<br>c<img src='x'/></p>") == ((), ())
    assert nesting_signature("</p><p>") == (("p",), ("p",))
    assert nesting_signature("<p><em></p></em>") is None


def test_mis_nested_translation_feeds_the_retry_phase(monkeypatch):
    replies = iter(["[id1]Bonjour[id0]", "[id0]Bonjour[id1]"])
    calls = []

    async def fake_request(chunk_text, **kwargs):
        calls.append(chunk_text)
        return next(replies)

    monkeypatch.setattr(xhtml_translator, "generate_translation_request", fake_request)
    stats = TranslationMetrics()
    result = asyncio.run(xhtml_translator.translate_chunk_with_fallback(
        chunk_text="[id0]Hello[id1]",
        local_tag_map={"[id0]": "<p>", "[id1]": "</p>"},
        global_indices=[4, 5],
        source_language="English",
        target_language="French",
        model_name="fake",
        llm_client=None,
        stats=stats,
        max_retries=2,
        chunk_index=0,
    ))
    assert len(calls) == 2
    assert result == "[id4]Bonjour[id5]"
    assert stats.placeholder_errors == 1
    assert stats.chunk_outcomes[0] == CHUNK_TRANSLATED
