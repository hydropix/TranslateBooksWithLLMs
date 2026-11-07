"""
Translation of the XHTML `<head><title>` (issue #270).

Only `<body>` goes through the chunk pipeline, so every content document kept
its source-language `<title>`. Readers show it in the window or tab title, in
some library views and in TTS announcements.

Strategy, cheapest first:
  1. The title repeats the document's first heading (the usual chapter file):
     reuse the heading's translation. Free, and identical to the TOC label the
     pipeline derives from the same heading.
  2. The same title was already translated earlier in this job (a book title
     repeated on every page): reuse it from the job cache.
  3. Otherwise one short LLM request. A rejected answer keeps the source title,
     which is always a valid document.

Never fails the file: every error, a rate limit included, is logged and
swallowed, and the source title is kept.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from lxml import etree

_XHTML_NS = "http://www.w3.org/1999/xhtml"
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# A title is one line. Anything much longer than its source is a model talking
# instead of translating.
_MAX_LENGTH_RATIO = 4
_MIN_LENGTH_ALLOWANCE = 120


def _local_name(element: etree._Element) -> str:
    try:
        return etree.QName(element).localname.lower()
    except ValueError:
        return str(element.tag).split("}", 1)[-1].lower()


def _normalized(text: Optional[str]) -> str:
    return " ".join((text or "").split())


def find_head_title(doc_root: etree._Element) -> Optional[etree._Element]:
    """The `<head><title>` element, with or without the XHTML namespace."""
    title = doc_root.find(f".//{{{_XHTML_NS}}}head/{{{_XHTML_NS}}}title")
    if title is None:
        title = doc_root.find(".//head/title")
    return title


def first_heading_text(doc_root: etree._Element) -> str:
    """Normalized text of the first non-empty h1-h6 of the body, or ''."""
    for element in doc_root.iter():
        if not isinstance(element.tag, str):
            continue
        if _local_name(element) in _HEADINGS:
            text = _normalized("".join(element.itertext()))
            if text:
                return text
    return ""


def snapshot_source_title(doc_root: etree._Element) -> Dict[str, str]:
    """Record, before translation, what is needed to translate the title after."""
    title = find_head_title(doc_root)
    return {
        "title": _normalized(title.text if title is not None else ""),
        "heading": first_heading_text(doc_root),
    }


def _acceptable(source: str, candidate: Optional[str]) -> bool:
    if not candidate or "\n" in candidate.strip():
        return False
    cap = max(_MAX_LENGTH_RATIO * len(source), _MIN_LENGTH_ALLOWANCE)
    return len(candidate) <= cap


async def translate_head_title(
    doc_root: etree._Element,
    snapshot: Dict[str, str],
    *,
    source_language: str,
    target_language: str,
    model_name: str,
    llm_client: Any,
    cache: Dict[str, str],
    translate: Callable,
    log_callback: Optional[Callable] = None,
) -> bool:
    """Replace the source `<title>` of a translated document. Returns True if changed.

    Args:
        doc_root: The document, with its body already translated.
        snapshot: Output of `snapshot_source_title` taken before translation.
        cache: Job-level source title -> translated title map, shared across
            files so a title repeated on every page costs one request.
        translate: `generate_translation_request`, injected so this module
            does not import the translator (and tests need no network).
    """
    source_title = snapshot.get("title", "")
    if not source_title or not any(ch.isalpha() for ch in source_title):
        return False
    title_el = find_head_title(doc_root)
    if title_el is None:
        return False

    try:
        translated = cache.get(source_title)

        if translated is None and snapshot.get("heading") == source_title:
            # In bilingual mode the first heading is the source twin, and a
            # heading whose chunk stayed untranslated is still the source:
            # neither is a translation to reuse.
            heading = first_heading_text(doc_root)
            if heading and heading != source_title:
                translated = heading

        if translated is None:
            from src.core.common.source_residue import detect_blocking_residue

            answer = await translate(
                source_title, "", "", "",
                source_language=source_language,
                target_language=target_language,
                model=model_name,
                llm_client=llm_client,
                log_callback=None,
                has_placeholders=False,
            )
            answer = _normalized(answer)
            if (_acceptable(source_title, answer)
                    and detect_blocking_residue(source_title, answer,
                                              source_language, target_language) is None):
                translated = answer
            elif log_callback:
                log_callback("epub_head_title_rejected",
                             f"⚠️ Could not translate document title \"{source_title}\" - kept as is")

        if not translated:
            return False
        cache[source_title] = translated
        if title_el.text == translated:
            return False
        title_el.text = translated
        return True
    except Exception as exc:  # noqa: BLE001 - the title must never fail a file
        # A rate limit included: the body of this file is already translated,
        # and pausing the whole job over a cosmetic title would throw that
        # file's result away. The next request will meet the limit anyway.
        if log_callback:
            log_callback("epub_head_title_failed",
                         f"⚠️ Document title translation failed and was skipped: {exc}")
        return False
