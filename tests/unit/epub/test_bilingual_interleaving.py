"""Tests for paragraph-level bilingual interleaving.

Regression for https://github.com/hydropix/TranslateBooksWithLLMs/discussions/199

Two bugs were reported on bilingual EPUB output:

1. Layout: bilingual output grouped an entire chunk's source in one block and
   the whole translation in another. A short chapter is often a single chunk,
   so readers saw a long run of source paragraphs followed by a long run of
   translations instead of source-then-translation per paragraph.

2. Data loss: chapters whose content is wrapped in a container element
   (e.g. ``<div class="Section13">``) spanning many chunks lost almost all
   text. The original per-chunk reconstruction reparsed unbalanced fragments
   in isolation, which corrupted the document and truncated it.

The fix reconstructs the full source and full translation as two complete,
well-formed documents (identical tag skeleton) and merges them structurally,
recursing into containers and interleaving at the leaf-paragraph level.
"""
import re

from src.core.epub.body_reassembly import interleave_bilingual


def _strip(html):
    """Drop tags and all whitespace for a spacing/markup-agnostic text view."""
    return re.sub(r'\s+', '', re.sub(r'<[^>]+>', '', html))


class TestInterleaveTopLevelParagraphs:
    def test_source_then_translation_per_paragraph(self):
        orig = '<p>S1</p><p>S2</p>'
        trans = '<p>T1</p><p>T2</p>'
        out = interleave_bilingual(orig, trans)

        # Each source paragraph is immediately followed by its translation,
        # and translation #1 precedes source #2 (the core layout fix).
        assert out.index('S1') < out.index('T1') < out.index('S2') < out.index('T2')
        assert out.count('bilingual-original') == 2
        assert out.count('bilingual-translation') == 2

    def test_blank_separators_emitted_once(self):
        orig = '<p> </p><p>S1</p><p class="blank"><br/></p><p>S2</p>'
        trans = '<p> </p><p>T1</p><p class="blank"><br/></p><p>T2</p>'
        out = interleave_bilingual(orig, trans)

        assert out.count('bilingual-original') == 2
        assert out.count('bilingual-translation') == 2
        # The <br/> blank separator is kept once, not duplicated/wrapped.
        assert out.count('class="blank"') == 1
        assert out.index('S1') < out.index('T1') < out.index('S2') < out.index('T2')


class TestInterleaveNestedContainer:
    """The data-loss case: content wrapped in a container spanning the body."""

    def test_container_preserved_and_text_complete(self):
        orig = (
            '<div class="Section13">'
            '<p>Para one source.</p>'
            '<p>Para two source.</p>'
            '<p>Para three source.</p>'
            '</div>'
        )
        trans = (
            '<div class="Section13">'
            '<p>Para one translated.</p>'
            '<p>Para two translated.</p>'
            '<p>Para three translated.</p>'
            '</div>'
        )
        out = interleave_bilingual(orig, trans)

        assert out is not None
        # The container is rebuilt ONCE (not duplicated per chunk).
        assert out.count('class="Section13"') == 1
        # One source + one translation wrapper per paragraph.
        assert out.count('bilingual-original') == 3
        assert out.count('bilingual-translation') == 3
        # No data loss: every source and translation paragraph is present.
        for frag in ('Paraonesource', 'Paratwosource', 'Parathreesource',
                     'Paraonetranslated', 'Paratwotranslated', 'Parathreetranslated'):
            assert frag in _strip(out)
        # Interleaved order inside the container.
        assert out.index('Para one source') < out.index('Para one translated') \
            < out.index('Para two source')

    def test_container_with_undeclared_namespace_prefix_attribute(self):
        # Body fragments don't carry the xmlns:epub declaration (it lives on
        # the <html> root), so the recovering parser keeps epub:type under its
        # literal colon name. Rebuilding the container must not choke on it
        # (regression: ValueError "Invalid attribute name 'epub:type'" failed
        # the whole file).
        orig = '<section epub:type="chapter"><p>S1</p><p>S2</p></section>'
        trans = '<section epub:type="chapter"><p>T1</p><p>T2</p></section>'
        out = interleave_bilingual(orig, trans)

        assert out is not None
        assert out.count('epub:type="chapter"') == 1
        assert out.count('bilingual-original') == 2
        assert out.index('S1') < out.index('T1') < out.index('S2') < out.index('T2')

    def test_deeply_nested_text_complete(self):
        orig = '<div><section><p>Deep source.</p><p>More source.</p></section></div>'
        trans = '<div><section><p>Deep translated.</p><p>More translated.</p></section></div>'
        out = interleave_bilingual(orig, trans)

        assert out is not None
        assert out.count('<section') == 1
        assert out.count('bilingual-original') == 2
        for frag in ('Deepsource', 'Moresource', 'Deeptranslated', 'Moretranslated'):
            assert frag in _strip(out)


class TestInterleaveFallback:
    def test_returns_none_on_structural_mismatch(self):
        # Different container child counts -> caller uses lossless fallback.
        orig = '<div><p>S1</p><p>S2</p></div>'
        trans = '<div><p>T1 merged</p></div>'
        assert interleave_bilingual(orig, trans) is None

    def test_matching_skeleton_with_inline_differences_ok(self):
        # Leaves may differ inside (LLM may restructure inline tags) — that is
        # fine, we don't recurse into leaf paragraphs.
        orig = '<p>Hello <b>world</b></p>'
        trans = '<p>Bonjour le monde</p>'
        out = interleave_bilingual(orig, trans)
        assert out is not None
        assert 'Hello' in out and 'Bonjour le monde' in out
        assert out.count('bilingual-original') == 1
        assert out.count('bilingual-translation') == 1
