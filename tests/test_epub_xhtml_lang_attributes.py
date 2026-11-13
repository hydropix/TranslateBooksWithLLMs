"""
Tests for issue #159: XHTML lang/xml:lang attributes must reflect target language.

When an EPUB is translated, the <html> root element of every XHTML file should
have its `lang` and `xml:lang` attributes updated to the target language ISO
code. Without this, e-readers still apply hyphenation, dictionary lookup and
TTS using the source language.

These tests cover:
  1. The language-name -> ISO code mapping (extended beyond RTL).
  2. The helper that mutates a single lxml document.
  3. The directory walker that processes every XHTML/HTML file in an extracted
     EPUB tree.

Run:  pytest tests/test_epub_xhtml_lang_attributes.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from lxml import etree

from src.core.epub.lang_support import (
    get_language_code,
    set_xhtml_lang_attributes,
    apply_target_language_to_xhtml_directory,
)
from src.core.epub.rtl_support import is_rtl_language


XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


# -------------------- get_language_code --------------------

@pytest.mark.parametrize("name,expected", [
    ("English", "en"),
    ("english", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Portuguese (Brazil)", "pt-BR"),
    ("Portuguese (Portugal)", "pt-PT"),
    ("portuguese (brazil)", "pt-BR"),
    ("portuguese (portugal)", "pt-PT"),
    ("Russian", "ru"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Turkish", "tr"),
    ("Greek", "el"),
    ("Czech", "cs"),
    ("Swedish", "sv"),
    ("Norwegian", "no"),
    ("Danish", "da"),
    ("Finnish", "fi"),
    ("Vietnamese", "vi"),
    ("Indonesian", "id"),
    ("Thai", "th"),
    ("Hindi", "hi"),
    ("Ukrainian", "uk"),
    # RTL languages must still resolve (regression check vs rtl_support map)
    ("Arabic", "ar"),
    ("Hebrew", "he"),
    ("Persian", "fa"),
    ("Farsi", "fa"),
    ("Urdu", "ur"),
])
def test_get_language_code_by_name(name, expected):
    assert get_language_code(name) == expected


@pytest.mark.parametrize("code", ["en", "es", "fr", "de", "ar", "he", "zh", "ja"])
def test_get_language_code_passthrough_for_known_codes(code):
    assert get_language_code(code) == code


@pytest.mark.parametrize("locale,expected", [
    ("en-US", "en"),
    ("es-ES", "es"),
    ("zh-CN", "zh"),
    ("pt-BR", "pt"),
    ("fr-CA", "fr"),
    ("ar-SA", "ar"),
])
def test_get_language_code_strips_locale_suffix(locale, expected):
    assert get_language_code(locale) == expected


def test_get_language_code_unknown_returns_none():
    assert get_language_code("Klingon") is None
    assert get_language_code("") is None
    assert get_language_code(None) is None


def test_rtl_detection_still_works_after_map_extension():
    """The map now includes LTR languages — is_rtl_language must not regress."""
    assert is_rtl_language("Arabic") is True
    assert is_rtl_language("Hebrew") is True
    assert is_rtl_language("Persian") is True
    assert is_rtl_language("Spanish") is False
    assert is_rtl_language("English") is False
    assert is_rtl_language("French") is False
    assert is_rtl_language("Chinese") is False


# -------------------- set_xhtml_lang_attributes --------------------

def _xhtml(lang_attr: str = "", xml_lang_attr: str = "") -> etree._Element:
    extra = ""
    if lang_attr:
        extra += f' lang="{lang_attr}"'
    if xml_lang_attr:
        extra += f' xml:lang="{xml_lang_attr}"'
    src = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"'
        f'{extra}>'
        '<head><title>t</title></head>'
        '<body><p>hello</p></body>'
        '</html>'
    )
    return etree.fromstring(src.encode("utf-8"))


def test_set_lang_overrides_source_lang():
    doc = _xhtml(lang_attr="en-US", xml_lang_attr="en-US")
    changed = set_xhtml_lang_attributes(doc, "es")
    assert changed is True
    assert doc.get("lang") == "es"
    assert doc.get(XML_LANG) == "es"


def test_set_lang_when_attributes_missing():
    doc = _xhtml()
    changed = set_xhtml_lang_attributes(doc, "fr")
    assert changed is True
    assert doc.get("lang") == "fr"
    assert doc.get(XML_LANG) == "fr"


def test_set_lang_idempotent_when_already_correct():
    doc = _xhtml(lang_attr="es", xml_lang_attr="es")
    changed = set_xhtml_lang_attributes(doc, "es")
    assert changed is False
    assert doc.get("lang") == "es"
    assert doc.get(XML_LANG) == "es"


def test_set_lang_partial_state():
    """One attribute set, the other missing — both must end up correct."""
    doc = _xhtml(lang_attr="en", xml_lang_attr="")
    set_xhtml_lang_attributes(doc, "de")
    assert doc.get("lang") == "de"
    assert doc.get(XML_LANG) == "de"


def test_set_lang_noop_when_root_not_html():
    doc = etree.fromstring(b'<root><child/></root>')
    changed = set_xhtml_lang_attributes(doc, "es")
    assert changed is False
    assert doc.get("lang") is None


# -------------------- apply_target_language_to_xhtml_directory --------------------

XHTML_EN_US = '''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en-US" xml:lang="en-US">
<head><title>Chapter 1</title></head>
<body><p>Hola mundo.</p></body>
</html>
'''

XHTML_NO_LANG = '''<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 2</title></head>
<body><p>Adios.</p></body>
</html>
'''

XHTML_JA = '''<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="ja" xml:lang="ja">
<head><title>Chapter 3</title></head>
<body><p>Adios.</p></body>
</html>
'''


def _read_lang_attrs(path: Path) -> tuple[str | None, str | None]:
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(str(path), parser)
    root = tree.getroot()
    return root.get("lang"), root.get(XML_LANG)


def test_apply_to_directory_updates_all_xhtml_files(tmp_path: Path):
    (tmp_path / "OEBPS").mkdir()
    f1 = tmp_path / "OEBPS" / "ch1.xhtml"
    f2 = tmp_path / "OEBPS" / "ch2.xhtml"
    f3 = tmp_path / "OEBPS" / "ch3.xhtml"
    f1.write_text(XHTML_EN_US, encoding="utf-8")
    f2.write_text(XHTML_NO_LANG, encoding="utf-8")
    f3.write_text(XHTML_JA, encoding="utf-8")

    # Non-XHTML files must be left alone.
    other = tmp_path / "OEBPS" / "stylesheet.css"
    other.write_text("body { color: red; }", encoding="utf-8")

    result = apply_target_language_to_xhtml_directory(str(tmp_path), "Spanish")

    assert result["updated"] == 3
    for f in (f1, f2, f3):
        lang, xml_lang = _read_lang_attrs(f)
        assert lang == "es", f"{f.name} lang should be 'es', got {lang!r}"
        assert xml_lang == "es", f"{f.name} xml:lang should be 'es', got {xml_lang!r}"

    # CSS file untouched
    assert other.read_text(encoding="utf-8") == "body { color: red; }"


def test_apply_to_directory_handles_html_extension(tmp_path: Path):
    f = tmp_path / "page.html"
    f.write_text(XHTML_EN_US, encoding="utf-8")

    apply_target_language_to_xhtml_directory(str(tmp_path), "French")
    lang, xml_lang = _read_lang_attrs(f)
    assert lang == "fr"
    assert xml_lang == "fr"


def test_apply_to_directory_with_unknown_target_language(tmp_path: Path):
    """If we cannot resolve a code, leave files alone (don't write garbage)."""
    f = tmp_path / "ch.xhtml"
    f.write_text(XHTML_EN_US, encoding="utf-8")
    before = f.read_bytes()

    result = apply_target_language_to_xhtml_directory(str(tmp_path), "Klingon")

    assert result["updated"] == 0
    assert result["skipped_no_code"] is True
    assert f.read_bytes() == before


def test_apply_to_directory_skips_unchanged_files(tmp_path: Path):
    """If lang already correct, file should not be rewritten unnecessarily."""
    f = tmp_path / "ch.xhtml"
    src = XHTML_EN_US.replace('lang="en-US" xml:lang="en-US"', 'lang="es" xml:lang="es"')
    f.write_text(src, encoding="utf-8")

    result = apply_target_language_to_xhtml_directory(str(tmp_path), "Spanish")
    assert result["updated"] == 0
    lang, xml_lang = _read_lang_attrs(f)
    assert lang == "es"
    assert xml_lang == "es"


def test_apply_to_directory_recursive(tmp_path: Path):
    nested = tmp_path / "OEBPS" / "Text" / "deep"
    nested.mkdir(parents=True)
    f = nested / "deep.xhtml"
    f.write_text(XHTML_EN_US, encoding="utf-8")

    apply_target_language_to_xhtml_directory(str(tmp_path), "German")
    lang, xml_lang = _read_lang_attrs(f)
    assert lang == "de"
    assert xml_lang == "de"


# -------------------- end-to-end against a real EPUB-like tree --------------------

def test_end_to_end_repackaged_epub_has_correct_lang(tmp_path: Path):
    """Build a fake EPUB tree, run the lang-update step, repackage, re-extract,
    and verify the XHTML files inside the resulting EPUB have lang='es'."""
    import zipfile

    work = tmp_path / "work"
    work.mkdir()
    (work / "META-INF").mkdir()
    (work / "META-INF" / "container.xml").write_text(
        '<?xml version="1.0"?><container/>', encoding="utf-8"
    )
    (work / "OEBPS").mkdir()
    (work / "OEBPS" / "chapter1.xhtml").write_text(XHTML_EN_US, encoding="utf-8")
    (work / "OEBPS" / "chapter2.xhtml").write_text(XHTML_JA, encoding="utf-8")

    apply_target_language_to_xhtml_directory(str(work), "Spanish")

    out_epub = tmp_path / "out.epub"
    with zipfile.ZipFile(out_epub, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(work):
            for name in files:
                p = Path(root) / name
                z.write(p, p.relative_to(work))

    extract = tmp_path / "extract"
    extract.mkdir()
    with zipfile.ZipFile(out_epub, "r") as z:
        z.extractall(extract)

    for fname in ("chapter1.xhtml", "chapter2.xhtml"):
        lang, xml_lang = _read_lang_attrs(extract / "OEBPS" / fname)
        assert lang == "es", f"{fname}: lang={lang!r}"
        assert xml_lang == "es", f"{fname}: xml:lang={xml_lang!r}"


if __name__ == "__main__":
    # Allow `python tests/test_epub_xhtml_lang_attributes.py` for quick local runs.
    raise SystemExit(pytest.main([__file__, "-v"]))
