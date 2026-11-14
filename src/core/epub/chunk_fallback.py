"""The per-chunk fallback ladder of the placeholder pipeline.

A chunk goes down three phases until one produces text:

1. Phase 1 - normal translation, up to ``max_retries`` attempts; a reply must
   keep every placeholder and actually carry the text.
2. Phase 2 - token alignment: translate the text without placeholders, then
   reinsert them proportionally (``EPUB_TOKEN_ALIGNMENT_ENABLED``).
3. Phase 3 - keep the source text.

Whatever the phase, the chunk's terminal outcome is recorded in
``stats.chunk_outcomes`` (CHUNK_TRANSLATED / CHUNK_TOKEN_ALIGNED /
CHUNK_UNTRANSLATED) and the result comes back with global placeholders.
"""

from typing import Dict, List, Optional, Tuple

from .chunk_settings import ChunkLoopSettings
from .engine_log import report_error
from .placeholder_renumberer import globalize_placeholders
from .placeholder_validator import PlaceholderValidator
from .text_survival import text_survival_problem
from .translation_metrics import TranslationMetrics
from .xhtml_translation_state import (
    CHUNK_TOKEN_ALIGNED,
    CHUNK_TRANSLATED,
    CHUNK_UNTRANSLATED,
)
from ..translator import generate_translation_request

# TokenAlignmentFallback is expensive to build; built on first Phase 2 use.
_aligner = None


def keep_edge_whitespace(source: str, translated: str) -> str:
    """Give ``translated`` the leading/trailing whitespace of ``source``.

    Chunks are joined back with ''.join(), so the whitespace at the edge of a
    chunk cut inside a paragraph is the only separator between two sentences.
    LLM replies are stripped on the way back, which glued them together.
    """
    leading = source[:len(source) - len(source.lstrip())]
    trailing = source[len(source.rstrip()):]
    return leading + translated.strip() + trailing


def validate_placeholders(translated_text: str, local_tag_map: Dict[str, str]) -> bool:
    """True when ``translated_text`` holds every placeholder of ``local_tag_map``
    exactly as expected (strict validation)."""
    is_valid, _error = PlaceholderValidator.validate_strict(translated_text, local_tag_map)
    return is_valid


async def _request(text: str, settings: ChunkLoopSettings, *, has_placeholders: bool,
                   placeholder_format: Optional[Tuple[str, str]]) -> Optional[str]:
    """One translation request for a chunk (no cross-chunk context)."""
    return await generate_translation_request(
        text,
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language=settings.source_language,
        target_language=settings.target_language,
        model=settings.model_name,
        llm_client=settings.llm_client,
        log_callback=settings.log_callback,
        has_placeholders=has_placeholders,
        context_manager=settings.context_manager,
        placeholder_format=placeholder_format,
        prompt_options=settings.prompt_options,
    )


async def _phase1_translate(chunk_text: str, local_tag_map: Dict[str, str],
                            settings: ChunkLoopSettings, stats: TranslationMetrics,
                            placeholder_format: Optional[Tuple[str, str]]) -> Optional[str]:
    """Normal translation with retries; the validated reply, or None."""
    log_callback = settings.log_callback
    max_retries = settings.max_retries
    has_placeholders = len(local_tag_map) > 0

    for attempt in range(max_retries):
        # Only retry attempts are logged (not the first attempt).
        if log_callback and attempt > 0:
            log_callback("translation_attempt", f"🔄 Translation retry attempt {attempt + 1}/{max_retries}")

        # The chunk goes as-is to the LLM (it already has local indices 0, 1, 2...).
        translated = await _request(chunk_text, settings, has_placeholders=has_placeholders,
                                    placeholder_format=placeholder_format)
        if translated is None:
            report_error(log_callback, "chunk_translation_failed",
                         f"Attempt {attempt + 1}/{max_retries}: Translation returned None")
            stats.retry_attempts += 1
            continue

        # The reply comes back stripped; a chunk cut inside a paragraph must
        # keep the space that separates it from the next chunk.
        translated = keep_edge_whitespace(chunk_text, translated)
        valid = validate_placeholders(translated, local_tag_map)

        # Placeholders alone do not prove the text came back: "[id0][id1]"
        # passes them for a whole paragraph. Route such replies to a retry.
        survival_problem = text_survival_problem(chunk_text, translated) if valid else None
        if survival_problem:
            report_error(log_callback, "chunk_text_lost",
                         f"Attempt {attempt + 1}/{max_retries}: rejected translation, {survival_problem}")
            stats.retry_attempts += 1
            continue

        if not valid:
            stats.placeholder_errors += 1
            stats.retry_attempts += 1
            continue

        if attempt == 0:
            stats.successful_first_try += 1
        else:
            stats.successful_after_retry += 1
            if log_callback:
                log_callback("retry_success", f"✓ Translation succeeded after {attempt + 1} attempt(s)")
        return translated
    return None


def _get_aligner():
    global _aligner
    if _aligner is None:
        from .token_alignment_fallback import TokenAlignmentFallback
        _aligner = TokenAlignmentFallback()
    return _aligner


async def _phase2_token_alignment(chunk_text: str, local_tag_map: Dict[str, str],
                                  settings: ChunkLoopSettings,
                                  stats: TranslationMetrics) -> Optional[str]:
    """Translate without placeholders, then reinsert them; the result, or None."""
    log_callback = settings.log_callback
    try:
        stats.token_alignment_used += 1
        if log_callback:
            log_callback("phase2_warning",
                f"⚠️ Placeholder validation failed after {settings.max_retries} attempts - using fallback")
            log_callback("phase2_hint",
                "💡 Tip: A more capable LLM model may better preserve placeholders and avoid layout issues")

        from src.common.placeholder_format import PlaceholderFormat
        clean_text = PlaceholderFormat.from_config().remove_all(chunk_text)

        # No placeholder instructions in this prompt.
        translated_clean = await _request(clean_text, settings, has_placeholders=False,
                                          placeholder_format=None)
        if translated_clean is None:
            raise Exception("LLM returned None for clean translation")
        survival_problem = text_survival_problem(clean_text, translated_clean)
        if survival_problem:
            raise Exception(f"rejected clean translation, {survival_problem}")

        placeholders_list = list(local_tag_map.keys())
        realigned = _get_aligner().align_and_insert_placeholders(
            original_with_placeholders=chunk_text,
            translated_without_placeholders=translated_clean,
            placeholders=placeholders_list,
        )
        realigned = keep_edge_whitespace(chunk_text, realigned)

        # Should always pass, checked anyway.
        if validate_placeholders(realigned, local_tag_map):
            stats.token_alignment_success += 1
            if log_callback:
                log_callback("phase2_success",
                    f"✓ Phase 2 successful: Token alignment repositioned {len(placeholders_list)} tags")
                log_callback("phase2_warning",
                    "⚠️ Note: Proportional repositioning may cause minor layout imperfections")
            return realigned
        report_error(log_callback, "phase2_validation_failed", "✗ Phase 2 validation failed")
    except Exception as e:
        report_error(log_callback, "phase2_error", f"✗ Phase 2 error: {str(e)}")
    return None


def _phase3_keep_source(settings: ChunkLoopSettings, stats: TranslationMetrics) -> None:
    """Count and log the untranslated fallback."""
    stats.fallback_used += 1
    report_error(settings.log_callback, "fallback_untranslated",
                 "✗ Phase 3: All translation attempts failed - returning original untranslated text")
    if settings.log_callback:
        settings.log_callback("phase3_warning", "⚠️ This chunk will remain in the source language")


async def translate_chunk_with_fallback(
    chunk_text: str,
    local_tag_map: Dict[str, str],
    global_indices: List[int],
    settings: ChunkLoopSettings,
    stats: TranslationMetrics,
    placeholder_format: Optional[Tuple[str, str]] = None,
    *,
    chunk_index: Optional[int] = None,
) -> str:
    """Translate one chunk down the fallback ladder (see the module docstring).

    Args:
        chunk_text: Text with local placeholders (0, 1, 2...).
        local_tag_map: Local placeholder -> markup map.
        global_indices: Global index of each local placeholder.
        settings: Languages, model, client, retries, prompt options, log.
        stats: Metrics updated in place.
        placeholder_format: ``(prefix, suffix)`` for the prompt.
        chunk_index: Index of the chunk in its file. When given, the terminal
            outcome is recorded in ``stats.chunk_outcomes`` so the caller can
            persist which chunks are still untranslated (issue #261).

    Returns:
        The translated (or source) text with global placeholders.
    """
    from src.config import EPUB_TOKEN_ALIGNMENT_ENABLED

    result = await _phase1_translate(chunk_text, local_tag_map, settings, stats, placeholder_format)
    outcome = CHUNK_TRANSLATED
    if result is None and EPUB_TOKEN_ALIGNMENT_ENABLED:
        result = await _phase2_token_alignment(chunk_text, local_tag_map, settings, stats)
        outcome = CHUNK_TOKEN_ALIGNED
    if result is None:
        outcome = CHUNK_UNTRANSLATED
        result = chunk_text
        _phase3_keep_source(settings, stats)

    stats.record_chunk_outcome(chunk_index, outcome)
    restored = globalize_placeholders(result, global_indices)
    stats.record_processed()  # the chunk is fully processed, even on failure
    return restored
