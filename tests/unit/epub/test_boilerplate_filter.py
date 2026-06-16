"""
Unit tests for web-scraping boilerplate removal (issue #239).

Covers the artifact families seen in scraped EPUBs (social share bars, related
posts, prev/next navigation, footers, hidden elements) and verifies that the
EPUB3 TOC navigation document is preserved.
"""
import pytest
from lxml import etree

from src.core.epub.boilerplate_filter import strip_web_boilerplate
from src.core.epub.body_serializer import extract_body_html

XHTML_NS = "http://www.w3.org/1999/xhtml"


def _body(inner_html: str) -> etree._Element:
    """Parse an XHTML body fragment and return the <body> element."""
    doc = (
        f'<html xmlns="{XHTML_NS}" xmlns:epub="http://www.idpf.org/2007/ops">'
        f"<body>{inner_html}</body></html>"
    )
    root = etree.fromstring(doc.encode("utf-8"))
    return root.find(f".//{{{XHTML_NS}}}body")


def _text(body: etree._Element) -> str:
    return " ".join(t.strip() for t in body.itertext() if t.strip())


class TestStripWebBoilerplate:
    def test_removes_jetpack_share_bar(self):
        body = _body(
            "<p>Real prose.</p>"
            '<div class="sharedaddy sd-sharing-enabled">'
            "<h3>Share this:</h3>"
            '<a class="share-twitter">Share on X (Opens in new window)</a>'
            "</div>"
        )
        removed = strip_web_boilerplate(body)
        assert removed == 1
        assert "Real prose." in _text(body)
        assert "Share on X" not in _text(body)

    def test_removes_related_posts(self):
        body = _body('<p>Keep me.</p><div id="jp-relatedposts">Related</div>')
        assert strip_web_boilerplate(body) == 1
        assert "Related" not in _text(body)
        assert "Keep me." in _text(body)

    def test_removes_prev_next_navigation(self):
        body = _body(
            '<div class="post-navigation"><a>PREV</a><a>NEXT</a></div>'
            "<p>Chapter body.</p>"
        )
        assert strip_web_boilerplate(body) == 1
        assert "PREV" not in _text(body)
        assert "Chapter body." in _text(body)

    def test_removes_footer_tag(self):
        body = _body("<p>Story.</p><footer>SukaMemuat... PREV TOC NEXT</footer>")
        assert strip_web_boilerplate(body) == 1
        assert "Story." in _text(body)
        assert "SukaMemuat" not in _text(body)

    def test_removes_hidden_elements(self):
        body = _body(
            '<p>Visible.</p>'
            '<span class="screen-reader-text">Posted in Uncategorized</span>'
            '<div style="display:none">tracking pixel alt</div>'
            '<div aria-hidden="true">decorative</div>'
        )
        assert strip_web_boilerplate(body) == 3
        assert _text(body) == "Visible."

    def test_preserves_epub3_toc_nav(self):
        body = _body(
            '<nav epub:type="toc"><ol><li><a href="c1.xhtml">Chapter 1</a></li></ol></nav>'
        )
        assert strip_web_boilerplate(body) == 0
        assert "Chapter 1" in _text(body)

    def test_preserves_landmarks_nav(self):
        body = _body('<nav epub:type="landmarks"><a href="c1.xhtml">Start</a></nav>')
        assert strip_web_boilerplate(body) == 0
        assert "Start" in _text(body)

    def test_removes_non_toc_nav(self):
        body = _body('<p>Text.</p><nav><a>PREV</a><a>NEXT</a></nav>')
        assert strip_web_boilerplate(body) == 1
        assert "PREV" not in _text(body)

    def test_nested_widgets_counted_once(self):
        body = _body(
            '<div class="sharedaddy">'
            '<div class="sd-block sd-social"><a>X</a></div>'
            "</div>"
        )
        # Outer match short-circuits the nested one.
        assert strip_web_boilerplate(body) == 1

    def test_preserves_ordinary_content(self):
        body = _body(
            "<h1>Chapter 1</h1>"
            "<p>A paragraph that merely mentions the word share casually.</p>"
            '<div class="chapter-text">More prose.</div>'
        )
        assert strip_web_boilerplate(body) == 0
        assert "Chapter 1" in _text(body)
        assert "More prose." in _text(body)

    def test_preserves_tail_text(self):
        # Text trailing a stripped element must not be lost.
        body = _body('<p>Before.</p><footer>junk</footer> After.')
        strip_web_boilerplate(body)
        assert "After." in _text(body)
        assert "junk" not in _text(body)

    def test_none_body_is_safe(self):
        assert strip_web_boilerplate(None) == 0

    def test_logs_when_removed(self):
        events = []
        body = _body('<footer>x</footer>')
        strip_web_boilerplate(body, log_callback=lambda e, m, *a, **k: events.append(e))
        assert "boilerplate_stripped" in events


class TestExtractBodyHtmlIntegration:
    def _root(self, inner_html: str) -> etree._Element:
        doc = (
            f'<html xmlns="{XHTML_NS}" xmlns:epub="http://www.idpf.org/2007/ops">'
            f"<body>{inner_html}</body></html>"
        )
        return etree.fromstring(doc.encode("utf-8"))

    def test_extract_strips_by_default(self):
        root = self._root('<p>Prose.</p><div class="sharedaddy"><a>Share on X</a></div>')
        html, body = extract_body_html(root)
        assert "Prose." in html
        assert "Share on X" not in html

    def test_extract_can_opt_out(self):
        root = self._root('<p>Prose.</p><div class="sharedaddy"><a>Share on X</a></div>')
        html, body = extract_body_html(root, strip_boilerplate=False)
        assert "Share on X" in html
