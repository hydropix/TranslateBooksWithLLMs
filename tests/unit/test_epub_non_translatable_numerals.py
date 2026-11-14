"""Regression tests: real words made of Roman-numeral letters must be translated.

The EPUB tag preserver used to freeze any segment matching
``[IVXLCDM]+`` case-insensitively, so words such as "did", "Civil" or "vivid"
were hidden inside placeholders and never reached the LLM.
"""
import pytest

from src.core.epub.tag_preservation import TagPreserver, is_non_translatable


@pytest.mark.parametrize("word", [
    "did", "Civil", "vivid", "mild", "Mix", "dim", "lid", "Mimi.", "civic",
    "DID", "Liv", "mid", "I am", "XVX", "IIII",
])
def test_words_are_translatable(word):
    assert not is_non_translatable(word)
    assert not is_non_translatable(word, standalone=True)


@pytest.mark.parametrize("text", [
    "1", "1.", "1.2", "42", "1-", "1)", "(1)", "  ", " ", "",
    "IV.", "XII)", "(III)", "I.", "MCMXCIV.",
])
def test_numbering_is_non_translatable(text):
    assert is_non_translatable(text)


def test_bare_roman_numeral_needs_standalone_context():
    assert not is_non_translatable("IV")
    assert is_non_translatable("IV", standalone=True)
    assert not is_non_translatable("I")
    assert is_non_translatable("I", standalone=True)


@pytest.mark.parametrize("html,word", [
    ("<p>He <em>did</em> it.</p>", "did"),
    ("<p>The <span>Civil</span> War</p>", "Civil"),
    ("<p><i>vivid</i></p>", "vivid"),
    ("<p><em>I</em> went home.</p>", "I"),
    ("<p>Yes, <b>Mimi.</b></p>", "Mimi."),
])
def test_words_inside_inline_tags_reach_the_llm(html, word):
    preserver = TagPreserver()
    text, tag_map = preserver.preserve_tags(html)
    assert word in text
    assert not any(word in v for v in tag_map.values())


@pytest.mark.parametrize("html", [
    "<h2>IV</h2><p>Hello world</p>",
    "<h2><span>XII</span></h2><p>Hello world</p>",
    "<p>I</p><p>Hello world</p>",
    "<h2>IV.</h2><p>Hello world</p>",
])
def test_standalone_chapter_numerals_stay_frozen(html):
    preserver = TagPreserver()
    text, tag_map = preserver.preserve_tags(html)
    assert text.count("Hello world") == 1
    # Only the translatable sentence remains outside placeholders
    stripped = text
    for placeholder in tag_map:
        stripped = stripped.replace(placeholder, "")
    assert stripped == "Hello world"
