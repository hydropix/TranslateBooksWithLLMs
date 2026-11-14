"""
Packaging-metadata localization for translated EPUBs.

The pipeline translates every content document but leaves the *packaging* in the
source language: `content.opf` keeps its original `dc:title` and
`dc:description`, and `toc.ncx` keeps its original `docTitle/text`. A reader's
library shelf, its "book information" panel and most file managers read exactly
those fields, so a fully translated book still shows up under its Chinese title
(finding F5 of `plan/PLAN_CjkSourceRendering.md`).

This module owns that fix, and nothing else:
  - `translate_opf_metadata(...)` — one LLM call for both fields, in-place edit
    of the OPF tree, its own OPF write, then propagation of the accepted title
    to every `*.ncx` `docTitle/text`.

Deliberate design constraints, all from the plan:
  - **One call per book.** Both fields travel in a single request; there is no
    retry, no correction pass and no fallback provider. A rejected result keeps
    the original value, which is always a valid EPUB.
  - **`dc:creator` is never touched.** Author names are neither translated nor
    transliterated.
  - **Never fails the job.** Every failure mode (no response, unparseable
    response, a raising client, a broken NCX) is logged and swallowed; the
    caller's EPUB is written either way.
  - **The client is the caller's.** `llm_client` / `model_name` are the ones
    `translate_epub_file` already built; this module creates no client, reads no
    provider setting and needs no API key of its own.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from lxml import etree

from src.config import (
    GENERATOR_NAME,
    GENERATOR_SOURCE,
    INPUT_TAG_IN,
    INPUT_TAG_OUT,
    NAMESPACES,
    TRANSLATE_TAG_IN,
    TRANSLATE_TAG_OUT,
)

# The CJK character class is imported rather than redefined. `cjk_typography`
# already documents why it holds a copy of the canonical class from
# `src/core/glossary/filter.py` (avoiding an `src.core.epub` -> `src.core.glossary`
# dependency); adding a third literal here is exactly what the plan forbids.
# The name is private, but this is a sibling module of the same package, and a
# same-package private reference is cheaper than promoting a public alias that
# only this call site would use.
from .cjk_typography import _CJK_CSS_RE as _CJK_CHAR_RE, is_cjk_language

# Field delimiters for the single request. Deliberately not `<TITLE>`: a book
# description can legitimately contain HTML-ish text, and `<TITLE>` is a real
# HTML element name.
_TITLE_TAG_IN = "<BOOK_TITLE>"
_TITLE_TAG_OUT = "</BOOK_TITLE>"
_DESC_TAG_IN = "<BOOK_DESCRIPTION>"
_DESC_TAG_OUT = "</BOOK_DESCRIPTION>"

DESCRIPTION_MAX_CHARS = 4000
"""Descriptions longer than this are not sent at all.

In most CJK web-novel EPUBs the description is a verbatim copy of the book's
intro page, which the pipeline has already translated as a content document, so
an unbounded extra call buys nothing. The title is still translated when the
description is skipped.
"""

MAX_EXPANSION_RATIO = 4
"""Reject a field whose translation is more than this many times longer.

Guards against a model that answers with an explanation, a commentary or the
whole prompt instead of a translation.
"""

MIN_LENGTH_ALLOWANCE = 120
"""Absolute character allowance applied on top of `MAX_EXPANSION_RATIO`.

The ratio alone is unusable on short CJK sources: the reported book's title is
11 characters, so a 4x cap is 44 characters, while any faithful French
rendering of it runs past 50. A ratio is meaningless at that scale, so the
effective cap is `max(4 * len(source), 120)` — still an order of magnitude
below the runaway output the ratio exists to catch, and a title is additionally
required to be single-line.
"""

# Tokens accepted as a "context window" bump target. A 4000-character CJK
# description is roughly 4000 tokens, so the pipeline's default context (2048)
# would truncate the request and guarantee a rejection.
_CONTEXT_HEADROOM_TOKENS = 1024
_MAX_CONTEXT_TOKENS = 32768


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _attribution_signature() -> str:
    """The exact signature `_update_epub_metadata` appends to dc:description.

    Kept byte-identical with `src/core/epub/translator.py`'s
    `_update_epub_metadata` (the source of truth) by building it from the same
    two config constants. Never hardcode the sentence.
    """
    return f"\n\nTranslated using {GENERATOR_NAME}\n{GENERATOR_SOURCE}"


def _split_signature(raw: str) -> Tuple[str, str]:
    """Split a dc:description into (body, signature) with `raw == body + sig`.

    `_update_epub_metadata` appends the signature in two shapes: as-is after an
    existing description, and `.strip()`ed when the description was empty or
    absent (in which case the whole element text *is* the signature and the body
    is empty). Both are recognized; anything else yields an empty signature,
    which is the ATTRIBUTION_ENABLED=false case.
    """
    if not raw:
        return "", ""
    signature = _attribution_signature()
    if raw.endswith(signature):
        return raw[:-len(signature)], signature
    stripped = signature.strip()
    if raw.endswith(stripped):
        body = raw[:-len(stripped)]
        if not body.strip():
            # The whole element text is the signature: the book had no
            # description of its own, so there is nothing to translate.
            return "", raw
        # A body followed by the stripped form is not a shape
        # `_update_epub_metadata` produces; normalize it to the canonical one
        # so the re-appended signature keeps its blank-line separator.
        return body.rstrip(), signature
    return raw, ""


def _extract_field(payload: str, tag_in: str, tag_out: str) -> Optional[str]:
    """Return the text between the first `tag_in` and the next `tag_out`.

    None when either delimiter is missing — treated as a rejection by the
    caller, never as an empty translation.
    """
    start = payload.find(tag_in)
    if start == -1:
        return None
    start += len(tag_in)
    end = payload.find(tag_out, start)
    if end == -1:
        return None
    return payload[start:end].strip()


def _rejection_reason(candidate: Optional[str], source: str,
                      target_language: str, single_line: bool) -> Optional[str]:
    """Why `candidate` must not replace `source`, or None when it is acceptable.

    The returned string is a short English label used in the log line.
    """
    if candidate is None:
        return "the model's answer could not be parsed"
    if not candidate.strip():
        return "the model returned an empty value"
    if single_line and ("\n" in candidate or "\r" in candidate):
        return "a title must be a single line"
    if not is_cjk_language(target_language) and _CJK_CHAR_RE.search(candidate):
        return "the answer still contains CJK characters"
    if len(candidate) > max(MAX_EXPANSION_RATIO * len(source), MIN_LENGTH_ALLOWANCE):
        return (f"the answer is {len(candidate)} characters for a "
                f"{len(source)}-character source (runaway output)")
    return None


def build_metadata_prompt(title: Optional[str], description: Optional[str],
                          source_language: str,
                          target_language: str) -> Tuple[str, str]:
    """Build the (system, user) prompt pair for the single metadata request.

    Only the fields actually present are asked for, and the answer must mirror
    the same delimiters, which makes parsing deterministic: a field whose tag
    pair is missing from the answer is rejected rather than guessed at.
    """
    fields: List[str] = []
    if title is not None:
        fields.append(f"{_TITLE_TAG_IN}{title}{_TITLE_TAG_OUT}")
    if description is not None:
        fields.append(f"{_DESC_TAG_IN}{description}{_DESC_TAG_OUT}")

    expected = "\n".join(
        line for line, wanted in (
            (f"{_TITLE_TAG_IN}the translated title{_TITLE_TAG_OUT}", title is not None),
            (f"{_DESC_TAG_IN}the translated description{_DESC_TAG_OUT}",
             description is not None),
        ) if wanted
    )

    system_prompt = f"""You are translating the packaging metadata of an EPUB book from {source_language} into {target_language}.

RULES
1. Translate the meaning into natural {target_language}. Do not transliterate, do not romanize, do not explain, do not comment.
2. A title stays a title: keep it on a single line, invent no subtitle, add no quotation marks and add no trailing punctuation.
3. Keep the description close to the source in length and in structure. Never add a note about the translation itself.
4. Never translate or transliterate a person's name.
5. The answer must be written entirely in {target_language}.

OUTPUT FORMAT
{TRANSLATE_TAG_IN}
{expected}
{TRANSLATE_TAG_OUT}

Emit exactly the field tags listed above, in that order, and nothing else: no preamble, no explanation, no markdown code fences."""

    joined_fields = "\n".join(fields)
    user_prompt = f"""# SOURCE METADATA ({source_language})

{INPUT_TAG_IN}
{joined_fields}
{INPUT_TAG_OUT}

Translate every field above into {target_language}. Output the result between {TRANSLATE_TAG_IN} and {TRANSLATE_TAG_OUT}, nothing else."""

    return system_prompt.strip(), user_prompt.strip()


# ---------------------------------------------------------------------------
# Private plumbing
# ---------------------------------------------------------------------------

def _log(log_callback: Optional[Callable], event: str, message: str) -> None:
    if log_callback:
        log_callback(event, message)


def _set_reason(result: dict, reason: str) -> None:
    """Record the first reason something was skipped; later ones only log."""
    if result['skipped_reason'] is None:
        result['skipped_reason'] = reason


def _local_name(tag) -> str:
    """Local name of an lxml tag, namespace-agnostic (comments included)."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit('}', 1)[-1]


def _iter_ncx_paths(opf_dir: str) -> List[str]:
    """Every `*.ncx` under `opf_dir`, sorted for deterministic logging."""
    found: List[str] = []
    for root, _dirs, files in os.walk(opf_dir):
        for name in files:
            if name.lower().endswith('.ncx'):
                found.append(os.path.join(root, name))
    return sorted(found)


def _update_ncx_doctitle(ncx_path: str, title: str,
                         log_callback: Optional[Callable]) -> bool:
    """Set `docTitle/text` in one NCX. Returns True when the file was rewritten.

    An NCX with no `docTitle/text` element is left alone: the pass propagates a
    translated title, it never adds a title where the book had none.
    """
    try:
        tree = etree.parse(
            ncx_path,
            etree.XMLParser(recover=True, remove_blank_text=False, huge_tree=True),
        )
        root = tree.getroot()
        if root is None:
            return False
        changed = False
        for element in root.iter():
            if _local_name(element.tag) != 'docTitle':
                continue
            for child in element:
                if _local_name(child.tag) != 'text':
                    continue
                if child.text == title:
                    continue
                child.text = title
                changed = True
        if changed:
            tree.write(ncx_path, encoding='utf-8', xml_declaration=True,
                       pretty_print=True)
        return changed
    except Exception as exc:
        _log(log_callback, "epub_metadata_ncx_failed",
             f"⚠️ Could not update the NCX title in "
             f"{os.path.basename(ncx_path)}: {exc}")
        return False


async def _request_metadata_translation(
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
) -> Optional[str]:
    """Send the single request and return the payload to parse, or None.

    Uses the pipeline's own client interface (`generate` + `extract_translation`
    + `context_window`), so no new request path, timeout policy or provider
    branch is introduced here. The model is not passed per call: exactly like
    the pipeline's main translation path (`src/core/translator.py`), the client
    was built for `model_name` already, which is why that argument is only used
    for logging.

    When the prompt needs a larger window than the client's own, the larger
    window is passed for this request only (``generate(context_window=...)``,
    the same per-request path the chunk requests use); the shared client is
    never mutated.
    """
    kwargs = {'system_prompt': system_prompt}
    current = getattr(llm_client, 'context_window', None)
    if isinstance(current, int):
        # One CJK character is worth roughly one token, so character count
        # is a safe upper bound for what this prompt needs.
        needed = min(
            len(system_prompt) + len(user_prompt) + _CONTEXT_HEADROOM_TOKENS,
            _MAX_CONTEXT_TOKENS,
        )
        if needed > current:
            kwargs['context_window'] = needed

    response = await llm_client.generate(user_prompt, **kwargs)

    if response is None or not getattr(response, 'content', None):
        return None
    extracted = llm_client.extract_translation(response.content)
    return extracted if extracted else response.content


def _requested_fields(
    metadata: etree._Element,
    result: dict,
    log_callback: Optional[Callable],
) -> Tuple[Any, Any, Optional[str], Optional[str], str]:
    """Locate the fields to translate and decide which ones are requested.

    Returns ``(title_el, desc_el, requested_title, requested_description,
    signature)``. A description over the size limit is dropped from the request
    (and the skip recorded) while the title stays requested.
    """
    title_el = metadata.find('.//dc:title', namespaces=NAMESPACES)
    desc_el = metadata.find('.//dc:description', namespaces=NAMESPACES)
    # dc:creator is deliberately absent from this function: author names are
    # never translated nor transliterated.

    source_title = (title_el.text or "").strip() if title_el is not None else ""
    raw_description = desc_el.text or "" if desc_el is not None else ""
    description_body, signature = _split_signature(raw_description)

    requested_title: Optional[str] = source_title or None
    requested_description: Optional[str] = description_body.strip() or None

    if requested_description and len(requested_description) > DESCRIPTION_MAX_CHARS:
        _set_reason(result, 'description_too_long')
        _log(log_callback, "epub_metadata_description_skipped",
             f"ℹ️ Book description left untranslated: "
             f"{len(requested_description)} characters exceeds the "
             f"{DESCRIPTION_MAX_CHARS}-character limit")
        requested_description = None

    return title_el, desc_el, requested_title, requested_description, signature


async def _fetch_payload(
    llm_client: Any,
    requested_title: Optional[str],
    requested_description: Optional[str],
    source_language: str,
    target_language: str,
    result: dict,
    log_callback: Optional[Callable],
) -> Optional[str]:
    """Run the single LLM request; on failure record the reason and return None."""
    system_prompt, user_prompt = build_metadata_prompt(
        requested_title, requested_description, source_language, target_language)

    try:
        payload = await _request_metadata_translation(
            llm_client, system_prompt, user_prompt)
    except Exception as exc:
        _set_reason(result, 'llm_error')
        _log(log_callback, "epub_metadata_translation_failed",
             f"⚠️ Packaging metadata left in {source_language}, the request "
             f"failed: {exc}")
        return None

    if payload is None:
        _set_reason(result, 'llm_no_response')
        _log(log_callback, "epub_metadata_translation_failed",
             f"⚠️ Packaging metadata left in {source_language}: the model "
             f"returned no usable answer")
        return None

    return payload


def _apply_title(payload: str, title_el: Any, requested_title: str,
                 source_language: str, target_language: str, result: dict,
                 log_callback: Optional[Callable]) -> bool:
    """Write the accepted title into `title_el`. Returns True when it changed."""
    candidate = _extract_field(payload, _TITLE_TAG_IN, _TITLE_TAG_OUT)
    reason = _rejection_reason(candidate, requested_title, target_language,
                               single_line=True)
    if reason is None:
        title_el.text = candidate.strip()
        result['title_translated'] = True
        return True
    _set_reason(result, 'title_rejected')
    _log(log_callback, "epub_metadata_title_rejected",
         f"⚠️ Book title kept in {source_language}: {reason}")
    return False


def _apply_description(payload: str, desc_el: Any, requested_description: str,
                       signature: str, source_language: str,
                       target_language: str, result: dict,
                       log_callback: Optional[Callable]) -> bool:
    """Write the accepted description into `desc_el`. Returns True when it changed."""
    candidate = _extract_field(payload, _DESC_TAG_IN, _DESC_TAG_OUT)
    reason = _rejection_reason(candidate, requested_description,
                               target_language, single_line=False)
    if reason is None:
        # The attribution signature was stripped before the request and is
        # re-appended verbatim here, so it survives exactly once.
        desc_el.text = candidate.strip() + signature
        result['description_translated'] = True
        return True
    _set_reason(result, 'description_rejected')
    _log(log_callback, "epub_metadata_description_rejected",
         f"⚠️ Book description kept in {source_language}: {reason}")
    return False


def _write_opf(opf_tree: etree._ElementTree, opf_path: str,
               source_language: str, result: dict,
               log_callback: Optional[Callable]) -> bool:
    """Persist the edited OPF. Returns False (and resets the result) on failure."""
    # This pass owns its OPF write: it runs after `_update_epub_metadata`,
    # which performs the pipeline's single OPF write, so these edits would
    # otherwise be dropped. See the ordering note at the call site.
    try:
        opf_tree.write(opf_path, encoding='utf-8', xml_declaration=True,
                       pretty_print=True)
    except Exception as exc:
        # Nothing reached the disk, so nothing is reported as translated and
        # nothing is propagated to the NCX: the book keeps its original
        # packaging metadata, consistently across all its files.
        _set_reason(result, 'opf_write_failed')
        _log(log_callback, "epub_metadata_opf_write_failed",
             f"⚠️ Packaging metadata left in {source_language}, the OPF "
             f"could not be written: {exc}")
        result['title_translated'] = False
        result['description_translated'] = False
        return False
    return True


def _propagate_title_to_ncx(opf_dir: str, translated_title: str, result: dict,
                            log_callback: Optional[Callable]) -> None:
    """Copy the accepted title into every NCX `docTitle/text`."""
    for ncx_path in _iter_ncx_paths(opf_dir):
        if _update_ncx_doctitle(ncx_path, translated_title, log_callback):
            result['ncx_doctitle_updated'] += 1


def _log_localized_summary(result: dict, log_callback: Optional[Callable]) -> None:
    """Log which packaging fields ended up localized."""
    localized = []
    if result['title_translated']:
        localized.append("title")
    if result['description_translated']:
        localized.append("description")
    message = f"🏷️ Packaging metadata localized: {', '.join(localized)}"
    if result['ncx_doctitle_updated']:
        message += f", {result['ncx_doctitle_updated']} NCX title(s)"
    _log(log_callback, "epub_metadata_translated", message)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def translate_opf_metadata(
    opf_tree: etree._ElementTree,
    opf_path: str,
    opf_dir: str,
    source_language: str,
    target_language: str,
    llm_client: Any,
    model_name: str,
    log_callback: Optional[Callable] = None,
) -> dict:
    """Translate dc:title and dc:description in place; propagate the title to
    every *.ncx docTitle/text.

    Returns:
        {'title_translated': bool,
         'description_translated': bool,
         'ncx_doctitle_updated': int,
         'skipped_reason': Optional[str]}

    `skipped_reason` is the first reason nothing (or only part) of the work was
    done, out of: 'no_translatable_metadata', 'description_too_long',
    'llm_no_response', 'llm_error', 'title_rejected', 'description_rejected',
    'opf_write_failed'.

    Behaviour is fully specified in this module's docstring; the short version
    is one LLM call for both fields, `dc:creator` never touched, a rejected
    answer keeps the original, and no failure ever propagates to the caller.
    """
    result: Dict[str, Any] = {
        'title_translated': False,
        'description_translated': False,
        'ncx_doctitle_updated': 0,
        'skipped_reason': None,
    }

    root = opf_tree.getroot() if opf_tree is not None else None
    metadata = root.find('.//opf:metadata', namespaces=NAMESPACES) if root is not None else None
    if metadata is None:
        _set_reason(result, 'no_translatable_metadata')
        return result

    (title_el, desc_el, requested_title, requested_description,
     signature) = _requested_fields(metadata, result, log_callback)

    if requested_title is None and requested_description is None:
        _set_reason(result, 'no_translatable_metadata')
        _log(log_callback, "epub_metadata_nothing_to_translate",
             "ℹ️ No packaging metadata to localize")
        return result

    _log(log_callback, "epub_metadata_translation_start",
         f"🏷️ Localizing packaging metadata with {model_name}...")

    payload = await _fetch_payload(
        llm_client, requested_title, requested_description,
        source_language, target_language, result, log_callback)
    if payload is None:
        return result

    changed = False
    if requested_title is not None:
        changed |= _apply_title(payload, title_el, requested_title,
                                source_language, target_language, result,
                                log_callback)
    if requested_description is not None:
        changed |= _apply_description(payload, desc_el, requested_description,
                                      signature, source_language,
                                      target_language, result, log_callback)

    if changed and not _write_opf(opf_tree, opf_path, source_language, result,
                                  log_callback):
        return result

    if result['title_translated']:
        # Never invent an NCX title: propagation happens only with an accepted
        # translation of a title the book already had.
        _propagate_title_to_ncx(opf_dir, title_el.text or "", result,
                                log_callback)

    if changed:
        _log_localized_summary(result, log_callback)

    return result
