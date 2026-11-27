"""Unit tests for the literal emphasis markers of Plain Text Mode (issue #352)."""
import pytest
from lxml import etree

from src.core.common.emphasis_markers import (
    EMPHASIS_TAGS,
    append_marked_text,
    count_spans,
    has_literal_markers,
    is_well_formed,
    recover_markdown_emphasis,
    sanitize,
    strip_markers,
)


def test_emphasis_tags():
    assert EMPHASIS_TAGS == ("i", "em")


@pytest.mark.parametrize("text,expected", [
    ("plain text", False),
    ("a <i>b</i>", True),
    ("stray </em> closer", True),
    ("upper <EM>case</EM>", True),
    ("<I>", True),
    ("attributes <i class='x'>are not markers</i>", True),  # the closer is
    ("<b>bold</b> and <img/>", False),
    ("<iframe>", False),
    ("", False),
])
def test_has_literal_markers(text, expected):
    assert has_literal_markers(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("", 0),
    ("no spans", 0),
    ("<i>a</i>", 1),
    ("<i>a</i> and <em>b</em>", 2),
    ("<EM>a</EM> <I>b</I>", 2),
    ("</i> only closers </em>", 0),
])
def test_count_spans(text, expected):
    assert count_spans(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("a <i>b</i> c", "a b c"),
    ("<em>whole</em>", "whole"),
    ("a  <I>b</I>   c", "a b c"),
    ("broken <i>span", "broken span"),
    ("glued<i>word</i>s", "gluedwords"),
    ("", ""),
])
def test_strip_markers(text, expected):
    assert strip_markers(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("plain", True),
    ("", True),
    ("a <i>b</i> c", True),
    ("<i>a</i> <em>b</em>", True),
    ("<EM>a</em>", True),                 # case-insensitive match
    ("a <i>b", False),                    # unclosed
    ("a </i>b", False),                   # stray closer
    ("<i>a</em>", False),                 # mismatched closer
    ("<i>a <em>b</em></i>", False),       # nested
    ("<i>a <i>b</i></i>", False),         # nested, same tag
    ("<i></i>", False),                   # empty span
    ("a <em>  </em> b", False),           # whitespace-only span
    ("<i>a</i></i>", False),              # extra closer
])
def test_is_well_formed(text, expected):
    assert is_well_formed(text) is expected


def test_sanitize_keeps_valid_markers():
    assert sanitize("un <i>mot</i>", "a <i>word</i>") == ("un <i>mot</i>", 1)


def test_sanitize_allows_fewer_spans_than_source():
    assert sanitize("un <em>mot</em> et un autre",
                    "a <em>word</em> and <em>another</em>") == ("un <em>mot</em> et un autre", 1)


def test_sanitize_allows_mixed_tags():
    out, kept = sanitize("<em>a</em> <i>b</i>", "<i>x</i> <em>y</em>")
    assert (out, kept) == ("<em>a</em> <i>b</i>", 2)


def test_sanitize_rejects_more_spans_than_source():
    assert sanitize("<i>a</i> <i>b</i>", "<i>x</i> y") == ("a b", 0)


def test_sanitize_rejects_when_source_has_no_span():
    assert sanitize("un <i>mot</i>", "a word") == ("un mot", 0)


@pytest.mark.parametrize("bad", [
    "un <i>mot", "un </i>mot", "<i>un <em>mot</em></i>", "<i></i>mot", "<i>a</em>",
])
def test_sanitize_rejects_malformed(bad):
    out, kept = sanitize(bad, "<i>a</i> <em>b</em>")
    assert kept == 0
    assert not has_literal_markers(out)


def test_sanitize_never_raises_on_none():
    assert sanitize(None, None) == ("", 0)


def _serialize(elem):
    return etree.tostring(elem, encoding="unicode")


def test_append_marked_text_builds_children_with_tails():
    p = etree.Element("p")
    append_marked_text(p, "He <em>did</em> say <i>Ciao</i>.")
    assert _serialize(p) == "<p>He <em>did</em> say <i>Ciao</i>.</p>"
    assert p.text == "He "
    assert [c.tag for c in p] == ["em", "i"]
    assert p[0].tail == " say "
    assert p[1].tail == "."


def test_append_marked_text_lowercases_tags():
    p = etree.Element("p")
    append_marked_text(p, "<EM>a</Em> b")
    assert _serialize(p) == "<p><em>a</em> b</p>"


def test_append_marked_text_plain():
    p = etree.Element("p")
    append_marked_text(p, "just text")
    assert _serialize(p) == "<p>just text</p>"
    assert len(p) == 0


def test_append_marked_text_malformed_falls_back_to_plain():
    p = etree.Element("p")
    append_marked_text(p, "broken <i>span  here")
    assert _serialize(p) == "<p>broken span here</p>"
    assert len(p) == 0


def test_append_marked_text_escapes_text():
    p = etree.Element("p")
    append_marked_text(p, "a &amp; <i>b < c</i>")
    assert _serialize(p) == "<p>a &amp;amp; <i>b &lt; c</i></p>"


# ---------------------------------------------------------------------------
# Markdown emphasis the LLM wrote instead of a marker
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("translated,source,expected", [
    # The missing span comes back as a marker of the source's tag.
    ("Я *могу* быть объективным.", "I <i>can</i> be objective.",
     ("Я <i>могу</i> быть объективным.", 1)),
    ("— *Алохомора*.", "<em>Alohomora</em>.", ("— <em>Алохомора</em>.", 1)),
    ("Это _важно_.", "That is <i>important</i>.", ("Это <i>важно</i>.", 1)),
    # Only as many as are missing, in reading order; the rest lose their stars.
    ("<i>a</i> *b* *c*", "<i>x</i> <i>y</i>", ("<i>a</i> <i>b</i> c", 1)),
    ("*a* *b*", "<i>x</i> y", ("<i>a</i> b", 1)),
    # Nothing missing: the invented pair is stripped, never left in the book.
    ("<i>a</i> and *b*", "<i>x</i> and y", ("<i>a</i> and b", 0)),
    # A pair inside an existing span is stripped, not nested.
    ("<i>a *b* c</i>", "<i>x</i> <i>y</i>", ("<i>a b c</i>", 0)),
    # Broken markers: strip the pairs, leave the markers to sanitize.
    ("<i>a *b*", "<i>x</i> y", ("<i>a b", 0)),
    # The source uses the delimiter itself: leave it alone.
    ("*a* <i>b</i>", "*x* <i>y</i> <i>z</i>", ("*a* <i>b</i>", 0)),
    # Not a pair: lone stars, bold, word-internal underscores, spaced stars.
    ("2 * 3 = 6", "<i>x</i>", ("2 * 3 = 6", 0)),
    ("**bold**", "<i>x</i>", ("**bold**", 0)),
    ("snake_case_name", "<i>x</i>", ("snake_case_name", 0)),
    ("a * b * c", "<i>x</i>", ("a * b * c", 0)),
    ("", "<i>x</i>", ("", 0)),
])
def test_recover_markdown_emphasis(translated, source, expected):
    assert recover_markdown_emphasis(translated, source) == expected


def test_recovered_text_passes_sanitize():
    text, recovered = recover_markdown_emphasis(
        "*Один*, <em>два</em> и *три*.", "<em>One</em>, <em>two</em> and <em>three</em>.")
    assert recovered == 2
    assert sanitize(text, "<em>One</em>, <em>two</em> and <em>three</em>.") == (
        "<em>Один</em>, <em>два</em> и <em>три</em>.", 3)
