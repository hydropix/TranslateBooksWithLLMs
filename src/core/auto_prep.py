"""
Auto mode: derive a throwaway glossary and style block from the document that
is about to be translated.

The manual paths (glossary NER review, style-extraction presets) require a
human workflow; a user who skips it translates with nothing at all. This
module runs the very same extraction primitives — `suggest_terms`,
`extract_style`, `assemble_instructions`, `document_sampler` — over excerpts of
the job's own source text, and returns a `prompt_options` fragment that the
existing translate/refine machinery already knows how to consume.

Everything here is best-effort: no function raises, whatever the input or the
provider does. A failure degrades to "nothing injected", never to a failed
translation job. The module knows nothing about Flask, `config` dicts or the
CLI — callers own the LLM client's lifecycle and the logging channel.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from src.core.glossary.models import normalize_gender
from src.core.glossary.name_scan import take_glossary_samples
from src.core.glossary.ner import suggest_terms
from src.core.style.assembler import assemble_instructions
from src.core.style.extractor import extract_style
from src.utils import document_sampler

logger = logging.getLogger("core.auto_prep")

#: Value both dropdowns send when the user picks "Auto".
AUTO_SENTINEL = "__auto__"
#: Name reported for the derived glossary (it is never persisted).
AUTO_GLOSSARY_NAME = "auto"

# Budgets mirror the manual endpoints so auto and manual quality match: a
# divergence here would mean the auto path silently produces worse glossaries
# and style rules than the reviewed one, for no visible reason.
GLOSSARY_MAX_CHARS = 6000        # == glossary_routes._NER_MAX_CHARS_HARD_CAP
GLOSSARY_SAMPLE_COUNT = 10       # == glossary_routes default sample_count
GLOSSARY_MIN_SAMPLE_SIZE = 500   # == glossary_routes._NER_MIN_SAMPLE_SIZE
GLOSSARY_MAX_TERMS = 60          # cap on injected terms (prompt-budget guard)
STYLE_MAX_CHARS = 10000          # == custom_instruction_routes._EXTRACT_DEFAULT_MAX_CHARS
STYLE_SAMPLE_COUNT = 6           # == custom_instruction_routes._EXTRACT_DEFAULT_SAMPLE_COUNT
STYLE_MIN_SAMPLE_SIZE = 1200     # == custom_instruction_routes._EXTRACT_MIN_SAMPLE_SIZE
AUTO_PREP_CONTEXT_WINDOW = 16384 # == custom_instruction_routes._EXTRACT_CONTEXT_WINDOW
AUTO_PREP_TIMEOUT_S = 420        # hard wall-clock ceiling for both passes together

#: Value written to `glossary_source`; suppresses the legacy "Loaded glossary" log line.
_AUTO_GLOSSARY_SOURCE = "auto"


def normalize_auto_flags(prompt_options: dict) -> tuple[bool, bool]:
    """Read (and consume) the auto selectors from `prompt_options` IN PLACE.

    Returns (glossary_auto_requested, style_auto_requested).

    Recognises, for glossary:
      - prompt_options['glossary_auto'] is truthy            → True
      - prompt_options['glossary_id'] == AUTO_SENTINEL       → True, and the
        'glossary_id' key is DELETED (it is not an int id).
    and for style:
      - prompt_options['style_auto'] is truthy               → True
      - prompt_options['custom_instruction_file'] == AUTO_SENTINEL → True, and
        that key is set to '' (so resolve_custom_instructions stays a no-op and
        logs no "invalid filename" warning).

    Mutates only those keys. Never raises. A None/empty mapping yields
    (False, False).
    """
    if not isinstance(prompt_options, dict) or not prompt_options:
        return False, False

    try:
        glossary_auto = bool(prompt_options.get("glossary_auto"))
        if prompt_options.get("glossary_id") == AUTO_SENTINEL:
            glossary_auto = True
            prompt_options.pop("glossary_id", None)

        style_auto = bool(prompt_options.get("style_auto"))
        if prompt_options.get("custom_instruction_file") == AUTO_SENTINEL:
            style_auto = True
            prompt_options["custom_instruction_file"] = ""

        return glossary_auto, style_auto
    except Exception as exc:  # pragma: no cover - defensive: dict-like inputs
        logger.debug("normalize_auto_flags failed: %s", exc)
        return False, False


def candidates_to_glossary(
    candidates: list[dict],
    max_terms: int = GLOSSARY_MAX_TERMS,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Convert `suggest_terms` output into the two mappings the prompt path wants.

    Returns (terms, metadata) where terms is {source: target} and metadata is
    {source: {"category": ..., "gender": ...}} carrying only the fields that are
    actually present — byte-compatible with `Glossary.terms_metadata` and with
    `cli_loader.load_glossary_from_file`.

    Rules, in order, applied to each candidate:
      1. `source` and `target` are stripped; a falsy `source` or `target` drops
         the entry (an entry with no proposed translation cannot be injected).
      2. `source.casefold() == target.casefold()` drops the entry (an identity
         mapping teaches the model nothing and costs prompt budget).
      3. First occurrence of a `source` wins; later duplicates are dropped.
      4. `gender` is passed through `models.normalize_gender`; an unrecognised
         value becomes no gender at all (key omitted).
      5. `category` is kept verbatim when non-empty (it is already validated by
         the NER parser), omitted otherwise.
      6. After filtering, the list is truncated to the first `max_terms` entries.
    Order is preserved (dicts are insertion-ordered). Pure; never raises.
    """
    terms: dict[str, str] = {}
    metadata: dict[str, dict[str, str]] = {}

    if not candidates:
        return terms, metadata

    try:
        limit = int(max_terms)
    except (TypeError, ValueError):
        limit = GLOSSARY_MAX_TERMS

    for candidate in candidates:
        if limit > 0 and len(terms) >= limit:
            break
        if not isinstance(candidate, dict):
            continue

        source = _clean(candidate.get("source"))
        target = _clean(candidate.get("target"))
        if not source or not target:
            continue
        if source.casefold() == target.casefold():
            continue
        if source in terms:
            continue

        terms[source] = target

        entry: dict[str, str] = {}
        category = _clean(candidate.get("category"))
        if category:
            entry["category"] = category
        gender = normalize_gender(candidate.get("gender"))
        if gender:
            entry["gender"] = gender
        # An all-empty metadata entry would only add noise to the checkpoint
        # and to the per-chunk glossary block, so it is omitted entirely.
        if entry:
            metadata[source] = entry

    return terms, metadata


def extract_source_text(
    *,
    file_path: str | None = None,
    text: str | None = None,
    hard_cap: int = document_sampler.FULL_TEXT_CAP,
) -> str:
    """Return the document's readable text, or '' when it cannot be obtained.

    Exactly one of `file_path` / `text` is meaningful; `text` wins when both are
    given (it is the inline-TXT case, where no file exists yet). For a path, the
    bytes are read and handed to `document_sampler.extract_full_text`, which
    dispatches on the extension (.txt/.srt/.epub/.docx/.pdf) and returns None for
    anything else. Any OSError / unsupported extension / empty result yields ''.
    Never raises.
    """
    try:
        if text and text.strip():
            # Same memory guard as the file path: the caller may hand us a
            # multi-megabyte paste.
            return text[:hard_cap] if hard_cap and hard_cap > 0 else text

        if not file_path:
            return ""

        path = Path(file_path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.debug("auto prep could not read %s: %s", file_path, exc)
            return ""

        extracted = document_sampler.extract_full_text(data, path.name, hard_cap)
        return extracted or ""
    except Exception as exc:  # pragma: no cover - defensive: never raise
        logger.debug("extract_source_text failed: %s", exc)
        return ""


async def auto_glossary(
    text: str, source_language: str, target_language: str, llm_client
) -> tuple[dict[str, str], dict[str, dict[str, str]], int, list[str]]:
    """One NER pass over excerpts of `text` that target its recurring names.

    Samples with take_glossary_samples(text, GLOSSARY_MAX_CHARS,
    GLOSSARY_SAMPLE_COUNT, min_sample_size=GLOSSARY_MIN_SAMPLE_SIZE) — a few
    evenly spaced excerpts plus a short window around each recurring name the
    whole-text scan found (plain distributed excerpts when it finds none) — then calls
    `suggest_terms(sample, source_language, target_language, llm_client,
    max_chars=GLOSSARY_MAX_CHARS)` and converts via candidates_to_glossary.

    Returns (terms, metadata, effective_excerpt_count, warnings). On an empty
    document, an LLM error, or zero usable candidates: ({}, {}, n, warnings).
    Never raises: an exception from the provider is caught and appended to
    `warnings` as f"auto glossary failed: {exc}".
    """
    warnings: list[str] = []
    excerpt_count = 0

    if not text or not text.strip():
        return {}, {}, 0, warnings

    try:
        sample, excerpt_count, names_targeted = take_glossary_samples(
            text,
            GLOSSARY_MAX_CHARS,
            GLOSSARY_SAMPLE_COUNT,
            min_sample_size=GLOSSARY_MIN_SAMPLE_SIZE,
        )
        if names_targeted:
            logger.info(
                "auto glossary sample targets %d recurring names found across %d chars",
                names_targeted, len(text),
            )
        candidates, ner_warnings = await suggest_terms(
            sample,
            source_language,
            target_language,
            llm_client,
            max_chars=GLOSSARY_MAX_CHARS,
        )
        warnings.extend(ner_warnings or [])
        terms, metadata = candidates_to_glossary(candidates)
        return terms, metadata, excerpt_count, warnings
    except Exception as exc:
        logger.debug("auto glossary failed", exc_info=True)
        warnings.append(f"auto glossary failed: {exc}")
        return {}, {}, excerpt_count, warnings


async def auto_style(
    text: str, source_language: str, target_language: str, llm_client
) -> tuple[str | None, str | None, int, list[str]]:
    """One style-extraction pass over distributed excerpts of `text`.

    Samples with take_distributed_samples(text, STYLE_MAX_CHARS,
    STYLE_SAMPLE_COUNT, min_sample_size=STYLE_MIN_SAMPLE_SIZE), calls
    `extract_style(sample, mode="source", source_language, target_language,
    llm_client, max_chars=STYLE_MAX_CHARS)`, keeps ONLY rules whose `flags` list
    is empty (decision D8), and assembles with
    `assemble_instructions("source", unflagged_rules, style["context"])`.

    Returns (translation_block, refinement_block, kept_rule_count, warnings);
    the blocks are None when nothing was assembled. `mode` is always "source":
    "model" means "imitate a different reference work", which auto mode has no
    way to choose. Never raises (same catch-and-warn contract as auto_glossary).
    """
    translation, refinement, kept, _excerpts, warnings = await _auto_style_pass(
        text, source_language, target_language, llm_client
    )
    return translation, refinement, kept, warnings


async def _auto_style_pass(
    text: str, source_language: str, target_language: str, llm_client
) -> tuple[str | None, str | None, int, int, list[str]]:
    """`auto_style` plus the effective excerpt count.

    The public signature is frozen at four values, but the log line reports how
    many excerpts were read — so the count is carried by this internal variant
    rather than re-sampling the document a second time just to count.
    """
    warnings: list[str] = []
    excerpt_count = 0

    if not text or not text.strip():
        return None, None, 0, 0, warnings

    try:
        sample, excerpt_count = document_sampler.take_distributed_samples(
            text,
            STYLE_MAX_CHARS,
            STYLE_SAMPLE_COUNT,
            min_sample_size=STYLE_MIN_SAMPLE_SIZE,
        )
        style, style_warnings = await extract_style(
            sample,
            "source",
            source_language,
            target_language,
            llm_client,
            max_chars=STYLE_MAX_CHARS,
        )
        warnings.extend(style_warnings or [])

        rules = style.get("rules") or []
        # D8: the manual path hides lint-flagged rules behind human review.
        # With no review at all, a flagged rule must be dropped, not applied.
        unflagged = [rule for rule in rules if not rule.get("flags")]

        blocks = assemble_instructions("source", unflagged, style.get("context", "") or "")
        return (
            blocks.get("translation"),
            blocks.get("refinement"),
            len(unflagged),
            excerpt_count,
            warnings,
        )
    except Exception as exc:
        logger.debug("auto style failed", exc_info=True)
        warnings.append(f"auto style failed: {exc}")
        return None, None, 0, excerpt_count, warnings


async def build_auto_prompt_options(
    *,
    source_text: str,
    source_language: str,
    target_language: str,
    want_glossary: bool,
    want_style: bool,
    llm_client,
    log: Callable[[str, str], None] | None = None,
) -> dict:
    """Run the requested auto passes and return a `prompt_options` fragment.

    Behaviour:
      - Returns {} immediately when both wants are False, when `source_text` is
        blank, or when `llm_client` is None (no LLM call is made).
      - Runs the enabled passes CONCURRENTLY via asyncio.gather(...,
        return_exceptions=True), wrapped in asyncio.wait_for(AUTO_PREP_TIMEOUT_S).
        On timeout: returns whatever is already available — in practice {} —
        after logging the timeout. Never propagates the TimeoutError.
      - Fragment keys are exactly those listed in the plan §3.3, and a key is
        present only when its value is non-empty. `glossary_name`/
        `glossary_source` are written together with `glossary_terms` and never
        alone.
      - Does NOT own the client: the caller creates and closes `llm_client`.
      - Does NOT mutate any input.
      - `log(key, message)` is called at most twice, once per enabled pass, with
        keys "auto_glossary" / "auto_style". A None `log` disables logging, and
        a raising `log` is swallowed. Warnings from the underlying parsers are
        NOT surfaced to `log` (they are review-oriented) but ARE logged at DEBUG
        level via the module logger.
      - Never raises, for any input.
    """
    try:
        if not want_glossary and not want_style:
            return {}
        if not source_text or not source_text.strip():
            return {}
        if llm_client is None:
            return {}

        # Announce the work BEFORE it starts. Both passes run before the first
        # chunk exists, so without this line the job sits silent — and looks
        # hung — for as long as the provider takes to answer.
        _safe_log(log, "auto_prep_start", _start_message(want_glossary, want_style))

        passes: list[str] = []
        coroutines: list[Any] = []
        if want_glossary:
            passes.append("glossary")
            coroutines.append(
                auto_glossary(source_text, source_language, target_language, llm_client)
            )
        if want_style:
            passes.append("style")
            coroutines.append(
                _auto_style_pass(source_text, source_language, target_language, llm_client)
            )

        try:
            # Read the ceiling from the module at call time so it stays tunable
            # (and testable) without rebinding a default argument.
            results = await asyncio.wait_for(
                asyncio.gather(*coroutines, return_exceptions=True),
                timeout=AUTO_PREP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "auto prep timed out after %ss — translating without auto glossary/style",
                AUTO_PREP_TIMEOUT_S,
            )
            # A timeout emits no per-pass line (no pass completed), so without
            # this the user would see an unexplained multi-minute pause before
            # chunk 1. Mutually exclusive with the two per-pass lines.
            _safe_log(
                log,
                "auto_prep_timeout",
                f"⚠️ Auto mode timed out after {AUTO_PREP_TIMEOUT_S}s — "
                f"translating without an auto glossary or style.",
            )
            return {}

        fragment: dict[str, Any] = {}
        for name, result in zip(passes, results):
            if name == "glossary":
                _merge_glossary_result(fragment, result, log)
            else:
                _merge_style_result(fragment, result, log)
        return fragment
    except Exception as exc:  # pragma: no cover - defensive: never raise
        logger.warning("auto prep failed unexpectedly: %s", exc, exc_info=True)
        return {}


def _merge_glossary_result(fragment: dict, result: Any, log) -> None:
    """Fold one `auto_glossary` outcome into `fragment` and emit its log line."""
    if isinstance(result, BaseException):
        # auto_glossary never raises, so this is a bug guard, not a path.
        logger.debug("auto glossary raised: %s", result)
        terms, metadata, excerpts, warnings = {}, {}, 0, []
    else:
        terms, metadata, excerpts, warnings = result

    _log_warnings("auto glossary", warnings)

    if terms:
        fragment["glossary_terms"] = terms
        fragment["glossary_name"] = AUTO_GLOSSARY_NAME
        fragment["glossary_source"] = _AUTO_GLOSSARY_SOURCE
        if metadata:
            fragment["glossary_term_metadata"] = metadata
        _safe_log(
            log,
            "auto_glossary",
            f"🧠 Auto glossary: {len(terms)} terms extracted from {excerpts} "
            f"excerpts of this document.",
        )
    else:
        _safe_log(
            log,
            "auto_glossary",
            "⚠️ Auto glossary: no usable terms found — translating without a glossary.",
        )


def _merge_style_result(fragment: dict, result: Any, log) -> None:
    """Fold one `_auto_style_pass` outcome into `fragment` and emit its log line."""
    if isinstance(result, BaseException):
        # _auto_style_pass never raises, so this is a bug guard, not a path.
        logger.debug("auto style raised: %s", result)
        translation, refinement, kept, excerpts, warnings = None, None, 0, 0, []
    else:
        translation, refinement, kept, excerpts, warnings = result

    _log_warnings("auto style", warnings)

    if translation:
        fragment["custom_instructions"] = translation
    if refinement:
        fragment["refinement_instructions"] = refinement

    if kept and (translation or refinement):
        _safe_log(
            log,
            "auto_style",
            f"🧠 Auto style: {kept} rules extracted from {excerpts} excerpts of this document.",
        )
    else:
        _safe_log(
            log,
            "auto_style",
            "⚠️ Auto style: no usable style rules found — translating without style instructions.",
        )


def _start_message(want_glossary: bool, want_style: bool) -> str:
    """The "work has begun" line, naming only the passes actually enabled."""
    if want_glossary and want_style:
        what = "a glossary and style instructions"
    elif want_glossary:
        what = "a glossary"
    else:
        what = "style instructions"
    return (
        f"🧠 Auto mode: reading this document to derive {what}. "
        f"This runs before the first chunk, so the progress bar stays at 0% until it completes."
    )


def _log_warnings(label: str, warnings: list[str] | None) -> None:
    """Parser warnings are review-oriented: they belong in the log file, not in
    the user-facing job log."""
    for warning in warnings or []:
        logger.debug("%s warning: %s", label, warning)


def _safe_log(log, key: str, message: str) -> None:
    if log is None:
        return
    try:
        log(key, message)
    except Exception as exc:
        logger.debug("auto prep log callback failed: %s", exc)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()
