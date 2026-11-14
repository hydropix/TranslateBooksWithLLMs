"""Regression tests: TOC fragments resolve to the heading they point at.

A fragment on an empty anchor (`<a id="ch2"/><h2>...`) used to fall back to
the file's FIRST heading, so every section of a multi-section file got the
same label, and a fragment on a heading-less wrapper used the whole section
text as the label.
"""
import os
from pathlib import Path

import pytest
from lxml import etree

from src.core.epub.toc_labels import (
    _get_translated_title_for_src,
    update_ncx_toc_labels_from_translated_docs,
)

XHTML_NS = "http://www.w3.org/1999/xhtml"
NCX_NS = "http://www.daisy.org/z3986/2005/ncx/"

BODY = """
<h1>Livre premier</h1>
<p>Introduction.</p>
<a id="ch2"/>
<h2>Chapitre deux</h2>
<p>Texte du chapitre deux.</p>
<section id="ch3"><div><h2>Chapitre trois</h2><p>Texte trois.</p></div></section>
<div id="wrapper"><p>Un long paragraphe sans titre qui ne doit jamais devenir une etiquette.</p></div>
<a id="before-text"/>
<p>Du texte avant le titre.</p>
<h2>Chapitre quatre</h2>
<h3 id="ch5">Chapitre <em>cinq</em></h3>
"""


def _docs(tmp_path: Path):
    path = tmp_path / "book.xhtml"
    doc = etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><body>{BODY}</body></html>'.encode("utf-8")
    )
    return {str(path): doc}, path


def _title(tmp_path, src):
    docs, _path = _docs(tmp_path)
    docs_by_path = {os.path.normcase(os.path.abspath(p)): d for p, d in docs.items()}
    return _get_translated_title_for_src(src=src, base_dir=str(tmp_path), docs_by_path=docs_by_path)


@pytest.mark.parametrize("src,expected", [
    ("book.xhtml", "Livre premier"),
    ("book.xhtml#ch2", "Chapitre deux"),          # empty anchor -> next heading
    ("book.xhtml#ch3", "Chapitre trois"),         # wrapper -> contained heading
    ("book.xhtml#ch5", "Chapitre cinq"),          # the heading itself
    ("book.xhtml#wrapper", None),                 # heading-less wrapper -> keep label
    ("book.xhtml#before-text", None),             # text before the next heading -> unsure
    ("book.xhtml#missing", None),                 # unknown fragment -> keep label
])
def test_fragment_resolution(tmp_path, src, expected):
    assert _title(tmp_path, src) == expected


def test_ncx_keeps_original_label_when_unsure(tmp_path):
    docs, _path = _docs(tmp_path)
    (tmp_path / "toc.ncx").write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="{NCX_NS}" version="2005-1"><navMap>
  <navPoint id="a"><navLabel><text>Chapter Two</text></navLabel><content src="book.xhtml#ch2"/></navPoint>
  <navPoint id="b"><navLabel><text>Wrapper</text></navLabel><content src="book.xhtml#wrapper"/></navPoint>
</navMap></ncx>""",
        encoding="utf-8",
    )
    result = update_ncx_toc_labels_from_translated_docs(str(tmp_path), docs)
    labels = [el.text for el in etree.parse(str(tmp_path / "toc.ncx")).iter(f"{{{NCX_NS}}}text")]
    assert labels == ["Chapitre deux", "Wrapper"]
    assert result["updated"] == 1
