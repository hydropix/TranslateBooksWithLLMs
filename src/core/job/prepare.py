"""
The prepare phase of a job: everything that is decided once, at job start,
and persisted with the job so a resume reuses it.

* the glossary snapshot (an edited or deleted glossary must not change a book
  half way through),
* the auto glossary / auto style passes (one extra LLM call each),
* the custom-instructions snapshot.

Every helper here works on a mutable ``prompt_options`` dict; the runner
copies the job's options, applies the helpers and stores the result back in
a new ``JobConfig``. Moved verbatim (apart from that signature change) from
``src/api/handlers.py`` so the CLI gets the same behaviour.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from src.core import auto_prep
from src.core.llm.exceptions import LLMClientConfigError
from src.utils.custom_instructions import is_safe_filename, load_custom_instructions

_logger = logging.getLogger(__name__)

# Set in prompt_options once the preset text has been resolved into the config.
CUSTOM_INSTRUCTIONS_SNAPSHOT_FLAG = 'custom_instructions_snapshot'

# How often the auto passes tell the UI they are still alive. They run before
# the first chunk exists, so nothing else emits for as long as they take.
AUTO_PREP_HEARTBEAT_S = 2.0


def resolve_custom_instructions(prompt_options, custom_instructions_dir, log_callback):
    """Resolve the instruction prose of the selected custom-instructions preset.

    Args:
        prompt_options (dict): the job's prompt options.
        custom_instructions_dir (Path): directory presets are loaded from.
        log_callback: callable(message_key, message_content="", data=None).

    Returns:
        tuple[str | None, str | None]: (translation_instructions, refinement_instructions).
    """
    custom_instruction_file = prompt_options.get('custom_instruction_file', '')

    translation_instructions = None
    refinement_instructions = None

    if custom_instruction_file:
        if not is_safe_filename(custom_instruction_file):
            log_callback(
                "custom_instructions_invalid",
                f"⚠️ Custom instructions file name '{custom_instruction_file}' is invalid "
                f"(no path separators or any of <>:\"|?*; must end in .txt, .yaml, or .yml). "
                f"Translation will proceed without it."
            )
        else:
            try:
                loaded = load_custom_instructions(custom_instruction_file, custom_instructions_dir)
                translation_instructions = loaded.get('translation')
                refinement_instructions = loaded.get('refinement')

                if translation_instructions or refinement_instructions:
                    phases = []
                    if translation_instructions:
                        phases.append('translation')
                    if refinement_instructions:
                        phases.append('refinement')
                    log_callback(
                        "custom_instructions",
                        f"📝 Loaded custom instructions: {custom_instruction_file} "
                        f"(phases: {', '.join(phases)})"
                    )
                else:
                    log_callback(
                        "custom_instructions_empty",
                        f"⚠️ Custom instructions file '{custom_instruction_file}' is empty. "
                        f"Translation will proceed without it."
                    )
            except FileNotFoundError:
                log_callback(
                    "custom_instructions_missing",
                    f"⚠️ Custom instructions file '{custom_instruction_file}' was selected "
                    f"but not found in {custom_instructions_dir}. Translation will proceed "
                    f"without it."
                )
            except Exception as e:
                log_callback(
                    "custom_instructions_error",
                    f"⚠️ Failed to load custom instructions '{custom_instruction_file}': {e}. "
                    f"Translation will proceed without it."
                )

    return translation_instructions, refinement_instructions


def snapshot_custom_instructions(prompt_options, custom_instructions_dir, log_callback):
    """Resolve the selected preset into ``prompt_options``, exactly once.

    The resolved text is stored in ``prompt_options['custom_instructions']`` /
    ``['refinement_instructions']`` and flagged, so a resume reuses it instead
    of re-reading the file: editing or deleting the preset mid-book must not
    change the instructions for the rest of that book. A config without the
    flag (a job created before the snapshot existed) is resolved from disk.
    """
    if prompt_options.get(CUSTOM_INSTRUCTIONS_SNAPSHOT_FLAG):
        return
    translation_instructions, refinement_instructions = resolve_custom_instructions(
        prompt_options, custom_instructions_dir, log_callback
    )
    if translation_instructions:
        prompt_options['custom_instructions'] = translation_instructions
    if refinement_instructions:
        prompt_options['refinement_instructions'] = refinement_instructions
    if (prompt_options.get('custom_instruction_file') or '').strip():
        prompt_options[CUSTOM_INSTRUCTIONS_SNAPSHOT_FLAG] = True


def snapshot_glossary(prompt_options, glossary_store_getter: Optional[Callable[[], Any]]) -> None:
    """Copy the selected glossary into ``prompt_options`` (never raises).

    A load failure is kept in ``prompt_options['glossary_load_error']`` and
    reported to the job log once the job has started.
    """
    glossary_id = prompt_options.get('glossary_id')
    if not glossary_id or prompt_options.get('glossary_terms') or glossary_store_getter is None:
        return
    try:
        store = glossary_store_getter()
        glossary = store.get_glossary(int(glossary_id))
        if glossary and glossary.terms:
            prompt_options['glossary_terms'] = glossary.terms_dict
            prompt_options['glossary_name'] = glossary.name
            metadata = glossary.terms_metadata
            if metadata:
                prompt_options['glossary_term_metadata'] = metadata
    except Exception as e:
        prompt_options['glossary_load_error'] = str(e)


def auto_prep_wants(job, prompt_options) -> "tuple[bool, bool]":
    """Decide whether each auto pass should run for this job.

    Calls ``auto_prep.normalize_auto_flags(prompt_options)`` (which consumes
    the ``__auto__`` sentinels of stale clients), then applies the precedence
    rules: an explicitly selected glossary wins (its snapshot is already in
    ``glossary_terms``, or ``glossary_id`` is set when the snapshot failed);
    auto-glossary is skipped in refine-only mode; an explicitly selected
    custom-instructions preset wins over auto-style; a term fix wants neither.
    Never raises.
    """
    try:
        glossary_requested, style_requested = auto_prep.normalize_auto_flags(prompt_options)
        if job.is_term_fix:
            return False, False
        want_glossary = bool(
            glossary_requested
            and not prompt_options.get('glossary_terms')
            and not prompt_options.get('glossary_id')
            and not job.is_refine_only
        )
        want_style = bool(
            style_requested
            and not (prompt_options.get('custom_instruction_file') or '').strip()
        )
        return want_glossary, want_style
    except Exception as exc:  # pragma: no cover - defensive: never raise
        _logger.debug("auto prep want-detection failed: %s", exc, exc_info=True)
        return False, False


async def _auto_prep_heartbeat(progress_callback, interval):
    """Re-emit the "preparing" state every ``interval`` seconds until cancelled,
    so the progress panel does not look frozen while the passes run."""
    while True:
        await asyncio.sleep(interval)
        try:
            progress_callback(True)
        except Exception as exc:
            _logger.debug("auto prep heartbeat emit failed: %s", exc, exc_info=True)


async def apply_auto_prep(job, prompt_options: Dict[str, Any], log_callback,
                          progress_callback=None, *, create_provider=None) -> None:
    """Compute the auto glossary/style for this job and merge them in place.

    Invariants:
      - any unexpected error is logged and swallowed: this can never make a
        translation job fail;
      - only ``prompt_options`` is mutated;
      - no LLM client is created and ``progress_callback`` is not called when
        neither pass is wanted.

    Args:
        job: the ``JobConfig`` (input, languages, provider).
        prompt_options: the job's prompt options, mutated in place.
        log_callback: callable(message_key, message_content="", data=None).
        progress_callback: optional callable(active: bool) advertising the
            "preparing" state: True when the passes start (then periodically),
            exactly once False when they end.
        create_provider: provider factory (tests); ``create_llm_provider`` by default.
    """
    def _signal(active):
        if progress_callback is None:
            return
        try:
            progress_callback(active)
        except Exception as exc:
            _logger.debug("auto prep progress emit failed: %s", exc, exc_info=True)

    try:
        want_glossary, want_style = auto_prep_wants(job, prompt_options)
        if not want_glossary and not want_style:
            return

        _signal(True)
        try:
            source_text = auto_prep.extract_source_text(file_path=job.input_path, text=job.text)
            if not source_text or not source_text.strip():
                log_callback(
                    "auto_prep_no_text",
                    "⚠️ Auto mode: could not read the source document — "
                    "translating without an auto glossary or style."
                )
                return

            source_language = job.source_language
            target_language = job.target_language
            if job.is_refine_only:
                # The input is already in the target language, so the style
                # pass must read it as such ("keep this text's voice").
                source_language = target_language

            if create_provider is None:
                from src.core.llm.factory import create_llm_provider as create_provider
            try:
                client = create_provider(
                    job.provider_for(context_window=auto_prep.AUTO_PREP_CONTEXT_WINDOW))
            except LLMClientConfigError as exc:
                log_callback(
                    "auto_prep_client_warning",
                    f"⚠️ Auto mode: no LLM client could be built for provider "
                    f"'{job.llm_provider}' ({exc}) — translating without an auto glossary or style."
                )
                return

            heartbeat = None
            if progress_callback is not None:
                heartbeat = asyncio.create_task(_auto_prep_heartbeat(_signal, AUTO_PREP_HEARTBEAT_S))
            try:
                fragment = await auto_prep.build_auto_prompt_options(
                    source_text=source_text,
                    source_language=source_language,
                    target_language=target_language,
                    want_glossary=want_glossary,
                    want_style=want_style,
                    llm_client=client,
                    log=log_callback,
                )
            finally:
                if heartbeat is not None:
                    heartbeat.cancel()
                    try:
                        await heartbeat
                    except asyncio.CancelledError:
                        pass
                try:
                    await client.close()
                except Exception as close_exc:
                    _logger.debug("auto prep client close failed: %s", close_exc, exc_info=True)

            if fragment:
                prompt_options.update(fragment)
        finally:
            _signal(False)
    except Exception as exc:
        log_callback("auto_prep_error", f"⚠️ Auto mode failed: {exc} — translating without it.")
