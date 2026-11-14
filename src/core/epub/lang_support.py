"""
Language attribute support for translated EPUB XHTML files.

When an EPUB is translated, every XHTML file's <html> root must carry `lang`
and `xml:lang` attributes that match the *target* language. Without this,
e-readers continue to apply hyphenation, dictionary lookup, font selection
and TTS using the source language (issue #159).

This module owns:
  - A comprehensive language-name -> ISO 639-1 code mapping (LTR + RTL).
  - `get_language_code(name)` — normalize a user-facing language name or locale.
  - `set_xhtml_lang_attributes(doc_root, code)` — mutate a parsed XHTML document.
  - `apply_target_language_to_xhtml_directory(dir, target_language)` — walk an
    extracted EPUB tree and rewrite every .xhtml/.html file in place.

RTL-specific concerns (CSS, page-progression-direction) remain in rtl_support.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Iterator, Optional

from lxml import etree

from .rtl_support import LANGUAGE_NAME_TO_CODE as _RTL_NAME_TO_CODE


# Comprehensive map: keys are lower-case language names, values are IANA
# primary language subtags (BCP 47 / RFC 5646). Validated against
# https://www.iana.org/assignments/language-subtag-registry — see
# tests/test_lang_codes_bcp47.py for the conformance checks.
#
# All entries are ISO 639-1 (2-letter) codes except where no 639-1 exists
# (e.g. Filipino -> "fil", which is ISO 639-2/3 but registered in IANA).
#
# Macrolanguage notes: a few entries are macrolanguage subtags
# ('zh', 'no', 'ms', 'ar', 'fa', 'sq', 'et', 'ku', 'lv', 'sw', 'yi', 'ps').
# These render correctly in e-readers but are coarser than e.g. 'zh-Hans' or
# 'nb' (Bokmål). Refining them is a product decision, not a correctness issue.
_LTR_NAME_TO_CODE: Dict[str, str] = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "dutch": "nl",
    "polish": "pl",
    "turkish": "tr",
    "swedish": "sv",
    "norwegian": "no",
    "danish": "da",
    "finnish": "fi",
    "czech": "cs",
    "greek": "el",
    "thai": "th",
    "vietnamese": "vi",
    "indonesian": "id",
    "malay": "ms",
    "ukrainian": "uk",
    "romanian": "ro",
    "bulgarian": "bg",
    "croatian": "hr",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "estonian": "et",
    "latvian": "lv",
    "lithuanian": "lt",
    "hindi": "hi",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "marathi": "mr",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "punjabi": "pa",
    "hungarian": "hu",
    "catalan": "ca",
    "basque": "eu",
    "galician": "gl",
    "irish": "ga",
    "welsh": "cy",
    "icelandic": "is",
    "albanian": "sq",
    "macedonian": "mk",
    "afrikaans": "af",
    "swahili": "sw",
    "filipino": "fil",
    "tagalog": "tl",
}

# Regional language variants that map to BCP47 tags with region subtags.
# Kept separate from LANGUAGE_NAME_TO_CODE so the BCP47 conformance test
# (which validates primary subtags only) is not affected.
_REGIONAL_NAME_TO_CODE: Dict[str, str] = {
    "portuguese (brazil)": "pt-BR",
    "portuguese (portugal)": "pt-PT",
}

# Merge RTL entries first so any conflict is resolved by LTR keys (none expected).
LANGUAGE_NAME_TO_CODE: Dict[str, str] = {**_RTL_NAME_TO_CODE, **_LTR_NAME_TO_CODE}

# Pre-computed set of valid output codes for membership tests.
_KNOWN_CODES = set(LANGUAGE_NAME_TO_CODE.values())

XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"
XML_LANG_ATTR = f"{{{XML_NAMESPACE}}}lang"


def get_language_code(language: Optional[str]) -> Optional[str]:
    """Resolve a language name or locale to an ISO 639-1 code.

    For regional variants like "Portuguese (Brazil)" / "Portuguese (Portugal)",
    returns the full BCP47 tag with region subtag ("pt-BR" / "pt-PT") so
    e-readers apply the correct hyphenation and TTS.

    Examples:
        get_language_code("English")  -> "en"
        get_language_code("Spanish")  -> "es"
        get_language_code("Portuguese (Brazil)")  -> "pt-BR"
        get_language_code("Portuguese (Portugal)") -> "pt-PT"
        get_language_code("en-US")    -> "en"
        get_language_code("fr")       -> "fr"
        get_language_code("Klingon")  -> None
    """
    if not language:
        return None

    lang_lower = language.lower().strip()
    if not lang_lower:
        return None

    # Check regional variants first (e.g. "portuguese (brazil)" -> "pt-BR")
    if lang_lower in _REGIONAL_NAME_TO_CODE:
        return _REGIONAL_NAME_TO_CODE[lang_lower]

    if lang_lower in LANGUAGE_NAME_TO_CODE:
        return LANGUAGE_NAME_TO_CODE[lang_lower]

    base = lang_lower.split("-")[0]
    if base in _KNOWN_CODES:
        return base

    if base in LANGUAGE_NAME_TO_CODE:
        return LANGUAGE_NAME_TO_CODE[base]

    return None


def _is_html_root(elem: etree._Element) -> bool:
    tag = elem.tag
    if not isinstance(tag, str):
        return False
    # Strip XHTML namespace if present: "{http://www.w3.org/1999/xhtml}html"
    local = tag.rsplit("}", 1)[-1].lower()
    return local == "html"


def set_xhtml_lang_attributes(doc_root: etree._Element, lang_code: str) -> bool:
    """Set `lang` and `xml:lang` on the <html> root.

    Returns True if either attribute was changed, False otherwise (already
    correct, or root is not an <html> element).
    """
    if doc_root is None or not _is_html_root(doc_root):
        return False

    changed = False
    if doc_root.get("lang") != lang_code:
        doc_root.set("lang", lang_code)
        changed = True
    if doc_root.get(XML_LANG_ATTR) != lang_code:
        doc_root.set(XML_LANG_ATTR, lang_code)
        changed = True
    return changed


def _iter_markup_paths(directory: str) -> Iterator[str]:
    """Yield every .xhtml/.html/.htm file under `directory`, in os.walk order."""
    for root_dir, _, files in os.walk(directory):
        for name in files:
            lower = name.lower()
            if not (lower.endswith(".xhtml") or lower.endswith(".html") or lower.endswith(".htm")):
                continue
            yield os.path.join(root_dir, name)


def _apply_lang_to_file(
    path: str,
    code: str,
    parser: etree.XMLParser,
    result: Dict,
    log_callback: Optional[Callable],
) -> None:
    """Set lang/xml:lang in one file, counting the outcome into `result`."""
    try:
        with open(path, "rb") as f:
            content = f.read()

        doc_root = etree.fromstring(content, parser)
        if doc_root is None:
            result["skipped"] += 1
            return

        if not set_xhtml_lang_attributes(doc_root, code):
            result["skipped"] += 1
            return

        with open(path, "wb") as f:
            f.write(
                etree.tostring(
                    doc_root,
                    encoding="utf-8",
                    xml_declaration=True,
                    pretty_print=True,
                    method="xml",
                )
            )
        result["updated"] += 1
    except Exception as e:
        result["errors"] += 1
        if log_callback:
            log_callback(
                "epub_lang_update_error",
                f"Could not update lang attributes in {path}: {e}",
            )


def apply_target_language_to_xhtml_directory(
    directory: str,
    target_language: str,
    log_callback: Optional[Callable] = None,
) -> Dict:
    """Walk `directory`, set lang/xml:lang on every XHTML/HTML file's root.

    Designed to run on the extracted EPUB tree just before repackaging.
    Files whose root isn't <html>, or that fail to parse, are skipped. Files
    whose attributes are already correct are not rewritten.

    Returns:
        {
          "updated": int,       # files actually rewritten
          "skipped": int,       # files visited but not rewritten
          "errors": int,        # parse/write failures
          "lang_code": str | None,
          "skipped_no_code": bool,  # True if target_language couldn't be resolved
        }
    """
    result = {
        "updated": 0,
        "skipped": 0,
        "errors": 0,
        "lang_code": None,
        "skipped_no_code": False,
    }

    code = get_language_code(target_language)
    if not code:
        result["skipped_no_code"] = True
        if log_callback:
            log_callback(
                "epub_lang_unknown",
                f"WARNING: Could not resolve ISO code for target language "
                f"'{target_language}'. XHTML lang attributes left unchanged.",
            )
        return result

    result["lang_code"] = code

    parser = etree.XMLParser(encoding="utf-8", recover=True, remove_blank_text=False)

    for path in _iter_markup_paths(directory):
        _apply_lang_to_file(path, code, parser, result, log_callback)

    if log_callback:
        log_callback(
            "epub_lang_attrs_applied",
            f"🌐 XHTML lang attributes set to '{code}': "
            f"{result['updated']} updated, {result['skipped']} unchanged, "
            f"{result['errors']} errors",
        )

    return result
