"""
Unit tests for <ruby> annotation folding (issue #242).

Covers the four ruby usages the reporter documented (contextual reading,
loanword, double meaning, invented terminology), the <rp> and group-ruby
markup variants, and the guarantee that documents without ruby are untouched.
Both extraction paths are exercised end to end, since the bug showed up
differently in each.
"""
import pytest
from lxml import etree

from src.core.epub.body_serializer import extract_body_html
from src.core.epub.plain_extractor import extract_plain_paragraphs
from src.core.epub.ruby_annotations import fold_ruby_annotations
from src.core.epub.tag_preservation import TagPreserver

XHTML_NS = "http://www.w3.org/1999/xhtml"


def _body(inner_html: str) -> etree._Element:
    """Parse an XHTML body fragment and return the <body> element."""
    doc = f'<html xmlns="{XHTML_NS}"><body>{inner_html}</body></html>'
    root = etree.fromstring(doc.encode("utf-8"))
    return root.find(f".//{{{XHTML_NS}}}body")


def _doc(inner_html: str) -> etree._Element:
    """Parse an XHTML document and return its root."""
    doc = f'<html xmlns="{XHTML_NS}"><body>{inner_html}</body></html>'
    return etree.fromstring(doc.encode("utf-8"))


def _text(body: etree._Element) -> str:
    return "".join(body.itertext())


class TestFoldRubyAnnotations:
    def test_contextual_reading(self):
        """Case 1: a meaning-based reading (宇宙 written, そら intended)."""
        body = _body("<p>彼は<ruby>宇宙<rt>そら</rt></ruby>を見上げた。</p>")
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "彼は宇宙（そら）を見上げた。"

    def test_foreign_loanword(self):
        """Case 2: katakana reading of a native kanji."""
        body = _body("<p><ruby>地球<rt>アース</rt></ruby></p>")
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "地球（アース）"

    def test_double_meaning(self):
        """Case 3: the base is the inner thought, the reading is spoken aloud."""
        body = _body("<p>お前は俺の<ruby>友<rt>ライバル</rt></ruby>だ。</p>")
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "お前は俺の友（ライバル）だ。"

    def test_invented_terminology(self):
        """Case 4: a stylized reading for a fictional technique."""
        body = _body("<p>これが<ruby>滅竜魔法<rt>ドラゴンスレイヤー</rt></ruby>だ。</p>")
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "これが滅竜魔法（ドラゴンスレイヤー）だ。"

    def test_rp_delimiters_are_kept_verbatim(self):
        """When the source ships its own <rp> fallback, we do not impose ours."""
        body = _body("<p><ruby>友<rp>(</rp><rt>ライバル</rt><rp>)</rp></ruby></p>")
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "友(ライバル)"

    def test_group_ruby_merges_all_segments(self):
        """Several <rb>/<rt> pairs fold into one base and one reading."""
        body = _body("<p><ruby><rb>地</rb><rb>球</rb><rt>アー</rt><rt>ス</rt></ruby></p>")
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "地球（アース）"

    def test_empty_reading_yields_base_only(self):
        """An <rt> with no content adds no parentheses."""
        body = _body("<p><ruby>宇宙<rt></rt></ruby></p>")
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "宇宙"

    def test_ruby_without_rt_yields_base_only(self):
        body = _body("<p><ruby>宇宙</ruby></p>")
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "宇宙"

    def test_stray_rp_without_rt_emits_no_empty_parentheses(self):
        """Malformed markup must not leave a bare "友()" behind."""
        body = _body("<p><ruby>友<rp>(</rp><rp>)</rp></ruby></p>")
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "友"

    def test_comment_inside_ruby_is_skipped(self):
        """lxml's itertext() rejects comment nodes; the base text must survive."""
        body = _body("<p><ruby>友<!--c-->と<rt>ライバル</rt></ruby></p>")
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "友と（ライバル）"

    def test_inline_markup_in_base_is_flattened(self):
        body = _body("<p><ruby><em>宇宙</em><rt>そら</rt></ruby></p>")
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "宇宙（そら）"

    def test_rtc_double_sided_ruby_degrades_gracefully(self):
        """Both annotation layers merge into one reading rather than being lost."""
        body = _body(
            "<p><ruby><rb>東</rb><rt>とう</rt><rtc>east</rtc></ruby></p>"
        )
        assert fold_ruby_annotations(body) == 1
        assert _text(body) == "東（とうeast）"

    def test_nested_ruby_does_not_crash(self):
        body = _body("<p><ruby><ruby>友<rt>とも</rt></ruby><rt>ライバル</rt></ruby></p>")
        # Only the outer annotation survives to be folded; the inner is detached
        # with it and skipped rather than double-counted.
        assert fold_ruby_annotations(body) == 1
        assert "ruby" not in etree.tostring(body, encoding="unicode")

    def test_tail_text_after_ruby_is_preserved(self):
        body = _body("<p>a<ruby>宇宙<rt>そら</rt></ruby>b<em>c</em>d</p>")
        fold_ruby_annotations(body)
        assert _text(body) == "a宇宙（そら）bcd"

    def test_multiple_annotations_in_one_paragraph(self):
        body = _body(
            "<p><ruby>友<rt>ライバル</rt></ruby>と<ruby>宇宙<rt>そら</rt></ruby></p>"
        )
        assert fold_ruby_annotations(body) == 2
        assert _text(body) == "友（ライバル）と宇宙（そら）"

    def test_annotation_inside_heading_and_list(self):
        body = _body(
            "<h1><ruby>序<rt>プロローグ</rt></ruby></h1>"
            "<ul><li><ruby>友<rt>ライバル</rt></ruby></li></ul>"
        )
        assert fold_ruby_annotations(body) == 2
        assert _text(body) == "序（プロローグ）友（ライバル）"


class TestNoOpGuarantees:
    def test_document_without_ruby_is_untouched(self):
        """The overwhelmingly common case must produce a byte-identical body."""
        body = _body("<p>Plain prose.</p><div><p>More <em>prose</em>.</p></div>")
        before = etree.tostring(body, encoding="unicode")
        assert fold_ruby_annotations(body) == 0
        assert etree.tostring(body, encoding="unicode") == before

    def test_none_body_is_tolerated(self):
        assert fold_ruby_annotations(None) == 0

    def test_comments_and_pis_are_ignored(self):
        body = _body("<p>text</p><!-- a comment --><?pi data?>")
        assert fold_ruby_annotations(body) == 0

    def test_idempotent(self):
        """No <ruby> survives the pass, so a second call changes nothing."""
        body = _body("<p><ruby>宇宙<rt>そら</rt></ruby></p>")
        assert fold_ruby_annotations(body) == 1
        after_first = etree.tostring(body, encoding="unicode")
        assert fold_ruby_annotations(body) == 0
        assert etree.tostring(body, encoding="unicode") == after_first

    def test_log_callback_fires_only_when_something_was_folded(self):
        events = []

        def log(event, message):
            events.append(event)

        fold_ruby_annotations(_body("<p>no ruby here</p>"), log_callback=log)
        assert events == []

        fold_ruby_annotations(_body("<p><ruby>友<rt>ライバル</rt></ruby></p>"), log_callback=log)
        assert events == ["ruby_annotations_folded"]


class TestPlainTextPath:
    """Plain Text Mode used to glue base and reading into a nonexistent word."""

    def test_reading_no_longer_glued_to_base(self):
        body = _body("<p>彼は<ruby>宇宙<rt>そら</rt></ruby>を見上げた。</p>")
        paragraphs, _tags, _images, _attrib = extract_plain_paragraphs(body)
        assert paragraphs == ["彼は宇宙（そら）を見上げた。"]
        assert "宇宙そら" not in paragraphs[0]

    def test_paragraph_count_is_unchanged_by_folding(self):
        """Folding stays inside inline markup, so checkpoint resume still matches.

        `resume_plain_segments` validates a stored `paragraph_count`; a fold that
        changed it would invalidate every in-flight checkpoint.
        """
        with_ruby = _body(
            "<p><ruby>友<rt>ライバル</rt></ruby></p><p>b</p><h2><ruby>序<rt>序</rt></ruby></h2>"
        )
        without_ruby = _body("<p>友</p><p>b</p><h2>序</h2>")
        assert len(extract_plain_paragraphs(with_ruby)[0]) == len(
            extract_plain_paragraphs(without_ruby)[0]
        )

    def test_images_still_anchored_around_an_annotation(self):
        body = _body(
            '<p><ruby>宇宙<rt>そら</rt></ruby><img src="a.png"/></p>'
        )
        paragraphs, tags, images, _attrib = extract_plain_paragraphs(body)
        assert paragraphs == ["宇宙（そら）"]
        assert tags == ["p"]
        assert len(images[0]) == 1


class TestStructuredHtmlPath:
    """Structured mode used to split each annotation into two fragments."""

    def test_annotation_is_one_translatable_unit(self):
        inner, _body_el = extract_body_html(
            _doc("<p>彼は<ruby>宇宙<rt>そら</rt></ruby>を見上げた。</p>")
        )
        assert "<ruby" not in inner
        assert "宇宙（そら）" in inner

        preserved, tag_map = TagPreserver().preserve_tags(inner)
        # One contiguous sentence between the <p> tags, instead of the former
        # "[id0]彼は[id1]宇宙[id2]そら[id3]を見上げた。[id4]".
        assert preserved == "[id0]彼は宇宙（そら）を見上げた。[id1]"
        assert tag_map["[id0]"] == "<p>"

    def test_surrounding_markup_is_preserved(self):
        inner, _body_el = extract_body_html(
            _doc('<p class="x">a<em>b</em><ruby>友<rt>ライバル</rt></ruby></p>')
        )
        assert 'class="x"' in inner
        assert "<em>b</em>" in inner
        assert "友（ライバル）" in inner
