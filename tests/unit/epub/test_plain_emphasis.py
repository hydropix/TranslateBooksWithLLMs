"""
Opt-in emphasis in Plain Text Mode (issue #352).

With keep_emphasis, the extractor keeps <i>/<em> spans as literal markers in
the paragraph text sent to the LLM, and the rebuild turns validated markers
back into real elements. With it off (the default), nothing changes.
"""
import pytest
from lxml import etree

from src.core.common.emphasis_markers import strip_markers
from src.core.common.plain_text_pipeline import strip_hallucinated_markup
from src.core.epub.plain_extractor import (
    PLAIN_EMPHASIS_MARKED_KEY,
    PLAIN_EMPHASIS_WRAP_KEY,
    extract_plain_paragraphs,
    replace_body_with_paragraphs,
)

XHTML = "http://www.w3.org/1999/xhtml"
NS = f'xmlns="{XHTML}" xmlns:epub="http://www.idpf.org/2007/ops"'


def _body(inner: str) -> etree._Element:
    root = etree.fromstring(f"<html {NS}><body>{inner}</body></html>".encode())
    return root[0]


def _extract(inner: str, keep_emphasis: bool = True):
    return extract_plain_paragraphs(_body(inner), keep_emphasis=keep_emphasis)


def _texts(inner: str, keep_emphasis: bool = True):
    return _extract(inner, keep_emphasis)[0]


def _local(tag) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


# ---------------------------------------------------------------- extraction

@pytest.mark.parametrize("inner,expected", [
    ("<p>A <em>big</em> deal.</p>", "A <em>big</em> deal."),
    ("<p>A <i>big</i> deal.</p>", "A <i>big</i> deal."),
    ("<p>A <strong>very <em>big</em></strong> deal.</p>", "A very <em>big</em> deal."),
    ("<p>A <em>very <strong>big</strong></em> deal.</p>", "A <em>very big</em> deal."),
    ("<p>A <i>very <em>big</em> one</i> here.</p>", "A <i>very big one</i> here."),
    ("<p>A <i>very <i>big</i></i> deal.</p>", "A <i>very big</i> deal."),
    ("<p>A <em>line<br/>break</em> here.</p>", "A <em>line break</em> here."),
    ("<p>An <em>empty</em><em></em> span.</p>", "An <em>empty</em> span."),
    ("<p>An <em>  </em> span.</p>", "An span."),
    ("<p>Word<em> spaced </em>out.</p>", "Word <em>spaced</em> out."),
    ("<p>Glued<i>word</i>s</p>", "Glued<i>word</i>s"),
    ("<p>A <EM>caps</EM> tag.</p>", "A <em>caps</em> tag."),
])
def test_extraction_markers(inner, expected):
    assert _texts(inner) == [expected]


def test_markers_strip_back_to_default_extraction():
    inner = ("<p>Word<em> spaced </em>out, <i><b>bold</b> it</i>, "
             "<em><img src='x.png'/></em> and <em>end </em></p>")
    marked = _texts(inner)[0]
    assert strip_markers(marked) == _texts(inner, keep_emphasis=False)[0]


def test_image_inside_emphasis_is_anchored_and_span_dropped_if_empty():
    texts, _, images, _ = _extract("<p>See <em><img src='a.png'/></em> <em>here<img src='b.png'/></em>.</p>")
    assert texts == ["See <em>here</em>."]
    assert [img.get("src") for img in images[0]] == ["a.png", "b.png"]


def test_pre_block_has_no_markers():
    texts = _texts("<pre>code <em>x</em></pre><p>a <i>b</i></p>")
    assert texts == ["code x", "a <i>b</i>"]


def test_pre_nested_in_table_cell_has_no_markers():
    texts = _texts("<table><tr><td>a <pre>b <em>c</em></pre> <em>d</em></td></tr></table>")
    assert texts == ["a b c <em>d</em>"]


def test_literal_markers_in_source_disable_emphasis_for_that_paragraph():
    inner = "<p>The tag &lt;i&gt; means <em>italic</em>.</p><p>Other <em>one</em>.</p>"
    texts, _, _, attrib = _extract(inner)
    assert texts == ["The tag <i> means italic.", "Other <em>one</em>."]
    assert PLAIN_EMPHASIS_WRAP_KEY not in attrib[0]


def test_literal_whole_span_in_source_is_not_a_wrap():
    texts, _, _, attrib = _extract("<p>&lt;i&gt;not a tag&lt;/i&gt;</p>")
    assert texts == ["<i>not a tag</i>"]
    assert PLAIN_EMPHASIS_WRAP_KEY not in attrib[0]


@pytest.mark.parametrize("inner,tag", [
    ("<p><i>All of it.</i></p>", "i"),
    ("<p> <em>All of it.</em> </p>", "em"),
    ("<div><em>Loose div.</em></div>", "em"),
    ("<table><tr><td><i>Cell</i></td></tr></table>", "i"),
])
def test_whole_paragraph_emphasis_is_flagged_not_marked(inner, tag):
    texts, _, _, attrib = _extract(inner)
    assert len(texts) == 1
    assert "<" not in texts[0]
    assert attrib[0][PLAIN_EMPHASIS_WRAP_KEY] == tag


def test_two_spans_covering_everything_are_not_a_wrap():
    texts, _, _, attrib = _extract("<p><i>a</i> <i>b</i></p>")
    assert texts == ["<i>a</i> <i>b</i>"]
    assert PLAIN_EMPHASIS_WRAP_KEY not in attrib[0]


def test_source_attribute_named_like_private_key_is_dropped():
    body = _body("<p>x</p>")
    body[0].set(PLAIN_EMPHASIS_WRAP_KEY, "em")
    _, _, _, attrib = extract_plain_paragraphs(body, keep_emphasis=True)
    assert PLAIN_EMPHASIS_WRAP_KEY not in attrib[0]


def test_keep_emphasis_false_is_unchanged():
    inner = ("<p>A <em>big</em> deal.</p><p><i>Whole</i></p>"
             "<div>Loose <i>text</i></div><table><tr><td><em>c</em></td></tr></table>")
    default = extract_plain_paragraphs(_body(inner))
    explicit = _extract(inner, keep_emphasis=False)
    assert default[0] == explicit[0] == ["A big deal.", "Whole", "Loose text", "c"]
    assert default[3] == explicit[3]
    assert all(PLAIN_EMPHASIS_WRAP_KEY not in a for a in default[3])


# ------------------------------------------------------------------- rebuild

def _rebuild(inner, translate, keep_emphasis=True, **kwargs):
    body = _body(inner)
    texts, tags, images, attrib = extract_plain_paragraphs(body, keep_emphasis=keep_emphasis)
    translations = translate(texts) if callable(translate) else translate
    stats = replace_body_with_paragraphs(
        body, translations, tags, images, source_paragraphs=texts,
        paragraphs_attrib=attrib, keep_emphasis=keep_emphasis, **kwargs)
    return body, stats


def _children_xml(body) -> str:
    """Serialize the rebuilt body and reparse it, proving it is valid XML."""
    xml = etree.tostring(body, encoding="unicode")
    reparsed = etree.fromstring(xml.encode())
    return reparsed


def test_rebuild_valid_markers_become_real_elements():
    body, stats = _rebuild("<p class='c' id='p1'>A <em>big</em> deal, <i>ciao</i>.</p>",
                           ["Une <em>grosse</em> affaire, <i>ciao</i>."])
    reparsed = _children_xml(body)
    p = reparsed[0]
    assert p.get("class") == "c" and p.get("id") == "p1"
    assert p.text == "Une "
    assert [_local(c.tag) for c in p] == ["em", "i"]
    assert p[0].text == "grosse" and p[0].tail == " affaire, "
    assert p[1].text == "ciao" and p[1].tail == "."
    assert stats == {"source_spans": 2, "kept_spans": 2, "recovered_spans": 0, "rejected_paragraphs": 0}


def test_rebuild_invalid_markers_fall_back_to_plain_text():
    body, stats = _rebuild("<p>A <em>big</em> deal.</p>", ["Une <em>grosse affaire."])
    p = body[0]
    assert len(p) == 0
    assert p.text == "Une grosse affaire."
    assert stats == {"source_spans": 1, "kept_spans": 0, "recovered_spans": 0, "rejected_paragraphs": 1}


def test_rebuild_dropped_markers_are_not_counted_as_rejected():
    _, stats = _rebuild("<p>A <em>big</em> deal.</p>", ["Une grosse affaire."])
    assert stats == {"source_spans": 1, "kept_spans": 0, "recovered_spans": 0, "rejected_paragraphs": 0}


def test_rebuild_bilingual_source_twin_keeps_emphasis():
    body, _ = _rebuild("<p id='x'>A <em>big</em> deal.</p>", ["Une grosse affaire."], bilingual=True)
    src, tgt = body[0], body[1]
    assert "plain-text-source" in src.get("class")
    assert src.get("id") == "x" and tgt.get("id") is None
    assert src.text == "A " and src[0].text == "big" and src[0].tail == " deal."
    assert len(tgt) == 0 and tgt.text == "Une grosse affaire."


def test_rebuild_empty_translation_falls_back_to_marked_source():
    body, stats = _rebuild("<p>A <em>big</em> deal.</p>", [""])
    p = body[0]
    assert "plain-text-untranslated" in p.get("class")
    assert p.text == "A " and _local(p[0].tag) == "em" and p[0].text == "big"
    assert stats == {"source_spans": 1, "kept_spans": 1, "recovered_spans": 0, "rejected_paragraphs": 0}


def test_rebuild_whole_paragraph_wrap():
    body, stats = _rebuild("<h2><i>Part One</i></h2>", ["Première partie"])
    h2 = body[0]
    assert h2.text is None
    assert len(h2) == 1 and _local(h2[0].tag) == "i" and h2[0].text == "Première partie"
    assert stats == {"source_spans": 1, "kept_spans": 1, "recovered_spans": 0, "rejected_paragraphs": 0}


def test_rebuild_whole_paragraph_wrap_strips_llm_markers():
    body, _ = _rebuild("<p><em>All</em></p>", ["<i>Tout</i>"])
    p = body[0]
    assert len(p) == 1 and _local(p[0].tag) == "em"
    assert p[0].text == "Tout" and len(p[0]) == 0


def test_rebuild_whole_paragraph_wrap_bilingual_and_fallback():
    body, _ = _rebuild("<p><em>All</em></p>", [""], bilingual=True)
    # Bilingual with an empty translation: only the source twin is emitted.
    assert len(body) == 1
    assert _local(body[0][0].tag) == "em" and body[0][0].text == "All"

    body, _ = _rebuild("<p><em>All</em></p>", [""])
    assert "plain-text-untranslated" in body[0].get("class")
    assert _local(body[0][0].tag) == "em" and body[0][0].text == "All"


def test_private_key_never_serialized():
    body, _ = _rebuild("<p class='k'><i>Whole</i></p><p>x <em>y</em></p>",
                       ["Tout", "x <em>y</em>"], bilingual=True)
    xml = etree.tostring(body, encoding="unicode")
    assert PLAIN_EMPHASIS_WRAP_KEY not in xml
    assert PLAIN_EMPHASIS_MARKED_KEY not in xml
    assert "plain_emphasis" not in xml


def test_marked_flag_only_on_paragraphs_that_carry_markers():
    _, _, _, attrib = _extract(
        "<p>a <i>b</i></p><p>plain</p><pre>x <em>y</em></pre><p>&lt;i&gt; <em>z</em></p>")
    assert [PLAIN_EMPHASIS_MARKED_KEY in a for a in attrib] == [True, False, False, False]


# ------------------------------------------- review regressions (option on)

PRE_TEXT = "line1\n  line2\n\tline3"


@pytest.mark.parametrize("bilingual", [False, True])
def test_pre_layout_kept_with_option_on(bilingual):
    body, stats = _rebuild(f"<pre>{PRE_TEXT}</pre>", lambda texts: list(texts), bilingual=bilingual)
    assert [b.text for b in body] == [PRE_TEXT] * (2 if bilingual else 1)
    assert all(len(b) == 0 for b in body)
    assert stats == {"source_spans": 0, "kept_spans": 0, "recovered_spans": 0, "rejected_paragraphs": 0}


def test_pre_layout_kept_with_option_on_when_translated():
    body, _ = _rebuild(f"<pre>{PRE_TEXT}</pre>", ["ligne1\n  ligne2\n\tligne3"])
    assert body[0].text == "ligne1\n  ligne2\n\tligne3"


@pytest.mark.parametrize("inner,text", [
    ("<p>Use the &lt;i&gt; tag here.</p>", "Use the <i> tag here."),
    ("<p>Write &lt;i&gt;x&lt;/i&gt; in HTML.</p>", "Write <i>x</i> in HTML."),
    ("<pre>&lt;em&gt;code&lt;/em&gt;\nnext</pre>", "<em>code</em>\nnext"),
    # A literal marker next to real italics: the paragraph is sent plain.
    ("<p>The &lt;i&gt; tag makes <em>italics</em>.</p>", "The <i> tag makes italics."),
])
@pytest.mark.parametrize("bilingual", [False, True])
@pytest.mark.parametrize("translation", ["identity", "empty"])
def test_literal_markers_in_book_text_survive(inner, text, bilingual, translation):
    translate = (lambda texts: list(texts)) if translation == "identity" else [""]
    body, stats = _rebuild(inner, translate, bilingual=bilingual)
    # Identity in bilingual mode: source twin + target; empty: twin only.
    expected = 2 if bilingual and translation == "identity" else 1
    assert len(body) == expected
    for block in body:
        assert len(block) == 0, etree.tostring(block, encoding="unicode")
        assert block.text == text
    assert stats["source_spans"] == 0 and stats["rejected_paragraphs"] == 0


def test_literal_markers_survive_hallucinated_markup_filter():
    source = "Use the <i> tag here."
    translated = "Utilisez la balise <i> ici.<sup>1</sup>"
    assert strip_hallucinated_markup(translated, source, keep_emphasis=True) == (
        "Utilisez la balise <i> ici.1")


def test_markers_only_answer_falls_back_to_source():
    body, stats = _rebuild("<p>Some <i>source</i> words.</p>", ["<i></i>"])
    p = body[0]
    assert "plain-text-untranslated" in p.get("class")
    assert p.text == "Some " and _local(p[0].tag) == "i" and p[0].text == "source"
    assert stats == {"source_spans": 1, "kept_spans": 1, "recovered_spans": 0, "rejected_paragraphs": 0}


def test_markers_only_answer_skips_target_in_bilingual():
    body, _ = _rebuild("<p>Some <i>source</i> words.</p>", ["<i> </i>"], bilingual=True)
    assert len(body) == 1
    assert "plain-text-source" in body[0].get("class")


def test_markers_only_answer_on_wrapped_paragraph_falls_back():
    body, _ = _rebuild("<p><em>All</em></p>", ["<em></em>"])
    assert "plain-text-untranslated" in body[0].get("class")
    assert _local(body[0][0].tag) == "em" and body[0][0].text == "All"


def test_rejected_markers_keep_the_rest_of_the_text_verbatim():
    body, stats = _rebuild("<p>A <em>big</em> deal.</p>", ["Une  <em>grosse affaire,\nvraiment."])
    assert len(body[0]) == 0
    assert body[0].text == "Une  grosse affaire,\nvraiment."
    assert stats["rejected_paragraphs"] == 1


def test_rebuild_without_keep_emphasis_returns_none_and_writes_plain_text():
    body = _body("<p>A <em>big</em> deal.</p>")
    texts, tags, images, attrib = extract_plain_paragraphs(body)
    result = replace_body_with_paragraphs(
        body, ["Une <em>grosse</em> affaire."], tags, images,
        source_paragraphs=texts, paragraphs_attrib=attrib)
    assert result is None
    # Default path: the text is written verbatim, markers included, as today.
    assert len(body[0]) == 0
    assert body[0].text == "Une <em>grosse</em> affaire."


def test_rebuild_wrap_key_ignored_without_keep_emphasis():
    body = _body("<p>x</p>")
    replace_body_with_paragraphs(
        body, ["y"], ["p"], {}, source_paragraphs=["x"],
        paragraphs_attrib=[{PLAIN_EMPHASIS_WRAP_KEY: "i"}])
    assert len(body[0]) == 0 and body[0].text == "y"


def test_rebuild_empty_page_backstop_returns_stats():
    body = _body("<svg xmlns='http://www.w3.org/2000/svg'/>")
    texts, tags, images, attrib = extract_plain_paragraphs(body, keep_emphasis=True)
    stats = replace_body_with_paragraphs(body, [], tags, images, source_paragraphs=texts,
                                         paragraphs_attrib=attrib, keep_emphasis=True)
    assert stats == {"source_spans": 0, "kept_spans": 0, "recovered_spans": 0, "rejected_paragraphs": 0}
    assert len(body) == 1


def test_round_trip_identity_translation_preserves_markup():
    inner = "<p>He <em>did</em> it.</p><p><i>Title</i></p><p>No emphasis.</p>"
    body, stats = _rebuild(inner, lambda texts: list(texts))
    xml = etree.tostring(body, encoding="unicode")
    assert "<em>did</em>" in xml
    assert "<i>Title</i>" in xml
    assert stats == {"source_spans": 2, "kept_spans": 2, "recovered_spans": 0, "rejected_paragraphs": 0}


def test_rebuild_recovers_markdown_italics():
    body, stats = _rebuild("<p>I <i>can</i> be <i>objective</i>.</p>",
                           ["Я *могу* быть <i>объективным</i>."])
    p = _children_xml(body)[0]
    assert p.text == "Я " and [_local(c.tag) for c in p] == ["i", "i"]
    assert p[0].text == "могу" and p[1].text == "объективным"
    assert "*" not in etree.tostring(p, encoding="unicode")
    assert stats == {"source_spans": 2, "kept_spans": 2, "recovered_spans": 1,
                     "rejected_paragraphs": 0}


def test_rebuild_strips_markdown_italics_from_rejected_paragraph():
    body, stats = _rebuild("<p>A <em>big</em> deal.</p>", ["Une <em>*grosse* affaire."])
    p = body[0]
    assert len(p) == 0 and p.text == "Une grosse affaire."
    assert stats == {"source_spans": 1, "kept_spans": 0, "recovered_spans": 0,
                     "rejected_paragraphs": 1}
