"""
Unit tests for the stray angle-bracket escape helper.

Source EPUBs (especially Korean webnovels) often use literal `<Word>` markers
in text content as stylistic quotation, e.g. `<Skill>` or `<ItemName>` for
status windows. Calibre encodes these as `&lt;Word&gt;` in the XHTML, but
once parsed by lxml the text node contains real `<` and `>` characters. The
LLM passes them through, and without escaping they corrupt the reinjected
document because the XML parser treats them as phantom HTML tags.

A bare '&' is just as dangerous (issue #202): lxml's recover-mode parser
silently deletes malformed or undefined entity references along with the
adjacent text ('AT&T' -> 'AT'), so the helper also escapes every '&' that is
not part of a reference the XML parser understands.
"""
from lxml import etree

from src.core.epub.body_serializer import replace_body_content
from src.core.epub.body_reassembly import escape_stray_markup


def test_escapes_simple_angle_brackets():
    assert escape_stray_markup("<Cloud>") == "&lt;Cloud&gt;"


def test_escapes_brackets_around_korean_text():
    assert escape_stray_markup("<엔젤릭>") == "&lt;엔젤릭&gt;"


def test_escapes_brackets_around_spanish_translation():
    assert escape_stray_markup("<Ángelico>") == "&lt;Ángelico&gt;"


def test_leaves_placeholder_brackets_untouched():
    text = "[id0]Hola [id1]<Skill>[id2]"
    assert escape_stray_markup(text) == "[id0]Hola [id1]&lt;Skill&gt;[id2]"


def test_preserves_existing_entities():
    # Predefined XML entities already produced by the LLM stay intact instead
    # of being double-escaped to &amp;lt;.
    assert escape_stray_markup("&lt;Cloud&gt;") == "&lt;Cloud&gt;"


def test_handles_mixed_text_with_multiple_brackets():
    raw = "Las clientes entraron en <Cloud> con paso firme hacia <Angelico>"
    expected = "Las clientes entraron en &lt;Cloud&gt; con paso firme hacia &lt;Angelico&gt;"
    assert escape_stray_markup(raw) == expected


def test_empty_string_returns_empty():
    assert escape_stray_markup("") == ""


def test_text_without_brackets_unchanged():
    assert escape_stray_markup("Hola mundo") == "Hola mundo"


# --- Ampersand handling (issue #202) ---


def test_escapes_bare_ampersand():
    assert escape_stray_markup("AT&T") == "AT&amp;T"
    assert escape_stray_markup("Tom & Jerry") == "Tom &amp; Jerry"


def test_preserves_predefined_xml_entities():
    assert escape_stray_markup("a &amp; b") == "a &amp; b"
    assert escape_stray_markup("&quot;hi&quot; &apos;y&apos;") == \
        "&quot;hi&quot; &apos;y&apos;"


def test_preserves_valid_numeric_references():
    assert escape_stray_markup("&#233;&#xE9;") == "&#233;&#xE9;"


def test_escapes_invalid_numeric_references():
    # NUL and out-of-range codepoints are not valid XML characters; the
    # recover parser would drop them, so the reference is kept as plain text.
    assert escape_stray_markup("&#0;") == "&amp;#0;"
    assert escape_stray_markup("&#x110000;") == "&amp;#x110000;"


def test_replaces_html_only_entities_with_literal_characters():
    # &nbsp; / &hellip; are undefined in DTD-less XHTML and would be deleted
    # by the recover parser, so they become their literal characters.
    assert escape_stray_markup("a&nbsp;b") == "a b"
    assert escape_stray_markup("wait&hellip;") == "wait…"


def test_html_entity_decoding_to_markup_chars_is_reescaped():
    # &LT; is a valid HTML5 name for '<' but not an XML predefined entity;
    # the decoded character must come back escaped, not as a raw bracket.
    assert escape_stray_markup("&LT;tag&GT;") == "&lt;tag&gt;"


def test_escapes_unknown_entity_names():
    assert escape_stray_markup("&foobar123;") == "&amp;foobar123;"


def test_double_escaped_source_ampersand_survives():
    # Source extraction emits '&amp;amp;' for a literal '&amp;' in the text;
    # the leading '&amp;' is a valid reference and 'amp;' is plain text.
    assert escape_stray_markup("&amp;amp;") == "&amp;amp;"


def _roundtrip_through_body(payload: str) -> str:
    """Escape `payload`, reinject it via replace_body_content, return the
    text the EPUB reader would actually display."""
    doc = etree.fromstring(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>old</p></body></html>'
    )
    body = doc.find('.//{http://www.w3.org/1999/xhtml}body')
    replace_body_content(body, escape_stray_markup(payload))
    return ''.join(body.itertext())


def test_roundtrip_issue_202_no_text_loss():
    # Before the fix this round-tripped to 'AT and Tom  Jerry  ok' — the
    # recover parser ate '&T', both bare '&', and even the valid '&amp;'.
    assert _roundtrip_through_body("AT&T and Tom & Jerry &amp; ok") == \
        "AT&T and Tom & Jerry & ok"


def test_roundtrip_html_entities_no_text_loss():
    assert _roundtrip_through_body("a&nbsp;b&hellip;c") == "a b…c"
