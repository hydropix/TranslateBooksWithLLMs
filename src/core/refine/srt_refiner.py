"""SRT refine-only mode.

Runs the existing refine_subtitle_translations helper on each subtitle of
an already-translated SRT and writes a polished file. Timestamps and
subtitle indices are preserved verbatim.
"""

import os
import aiofiles
from typing import Optional, Callable, Dict, Any

from src.config import (
    DEFAULT_MODEL,
    API_ENDPOINT,
    SRT_LINES_PER_BLOCK,
)

# Disable the char cap when grouping: block sizing is purely fixed-count
# (every block holds exactly SRT_LINES_PER_BLOCK subtitles).
_NO_CHAR_CAP = 10 ** 12
from src.core.llm.exceptions import RateLimitError
from src.core.llm_client import create_llm_client
from src.core.srt_processor import SRTProcessor
from src.core.subtitle_translator import refine_subtitle_translations


async def _write_refined_srt(
    srt_processor: SRTProcessor,
    subtitles,
    refined: Dict[int, str],
    output_filepath: str,
    log_callback: Optional[Callable] = None,
) -> bool:
    """Apply the refined texts to the cues and write the SRT file.

    Shared by the nominal completion path and the rate-limit partial-save path
    so both produce the same layout. Cue numbers and timings come from the
    parsed input, so they are preserved verbatim. Returns True when the file
    was written; logs and returns False otherwise.
    """
    refined_subs = srt_processor.update_translated_subtitles(subtitles, refined)
    refined_srt = srt_processor.reconstruct_srt(refined_subs)
    try:
        async with aiofiles.open(output_filepath, 'w', encoding='utf-8') as f:
            await f.write(refined_srt)
        return True
    except Exception as e:
        if log_callback:
            log_callback("srt_save_error",
                         f"ERROR: Saving SRT file '{output_filepath}': {e}")
        return False


async def _load_resumed_cues(
    srt_processor: SRTProcessor,
    subtitles,
    output_filepath: str,
    resume_from_index: int,
    log_callback: Optional[Callable] = None,
) -> Dict[int, str]:
    """Read back the cues a previous, rate-limited pass already refined.

    The partial output written before the pause holds every cue, refined up
    to `resume_from_index`. It is trusted only when it still lines up with
    the input cue for cue (same count, same timings); otherwise an empty map
    is returned and the pass starts over from the first cue.
    """
    def _restart(reason: str) -> Dict[int, str]:
        if log_callback:
            log_callback("srt_refine_resume_restart",
                         f"⚠️ Cannot reuse the partial refined SRT ({reason}). "
                         f"Refining from the first subtitle.")
        return {}

    if resume_from_index > len(subtitles):
        return _restart("resume point is past the end of the file")
    try:
        async with aiofiles.open(output_filepath, 'r', encoding='utf-8') as f:
            previous = srt_processor.parse_srt(await f.read())
    except Exception as e:
        return _restart(f"cannot read '{output_filepath}': {e}")

    if len(previous) != len(subtitles) or any(
        prev['start_time'] != sub['start_time'] or prev['end_time'] != sub['end_time']
        for prev, sub in zip(previous, subtitles)
    ):
        return _restart("it no longer matches the input file")

    if log_callback:
        log_callback("srt_refine_resume",
                     f"↻ Reusing {resume_from_index} already refined subtitles, "
                     f"resuming at subtitle {resume_from_index + 1}.")
    return {idx: previous[idx]['text'] for idx in range(resume_from_index)}


async def refine_srt_file(
    input_filepath: str,
    output_filepath: str,
    target_language: str,
    model_name: str = DEFAULT_MODEL,
    cli_api_endpoint: str = API_ENDPOINT,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    check_interruption_callback: Optional[Callable] = None,
    llm_provider: str = "ollama",
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    poe_api_key: Optional[str] = None,
    nim_api_key: Optional[str] = None,
    prompt_options: Optional[Dict[str, Any]] = None,
    checkpoint_manager: Any = None,
    translation_id: Optional[str] = None,
    resume_from_index: int = 0,
) -> bool:
    """Run a refinement-only pass on an already-translated SRT file.

    On a rate limit the cues refined so far are written to `output_filepath`
    and, when a checkpoint manager is supplied, the last finished cue is
    recorded so a resume (`resume_from_index`) skips the finished ones.
    """
    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input SRT file '{input_filepath}' not found."
        if log_callback:
            log_callback("srt_file_not_found", err_msg)
        return False

    try:
        async with aiofiles.open(input_filepath, 'r', encoding='utf-8') as f:
            srt_content = await f.read()
    except Exception as e:
        if log_callback:
            log_callback("srt_read_error",
                         f"ERROR: Reading SRT file '{input_filepath}': {e}")
        return False

    srt_processor = SRTProcessor()
    if not srt_processor.validate_srt(srt_content):
        if log_callback:
            log_callback("srt_invalid_format", "Invalid SRT file format")
        return False

    subtitles = srt_processor.parse_srt(srt_content)
    if not subtitles:
        if log_callback:
            log_callback("srt_no_subtitles", "No subtitles found in file")
        return False

    # Cues already refined by a previous, rate-limited pass (resume only).
    done_texts: Dict[int, str] = {}
    if resume_from_index > 0:
        done_texts = await _load_resumed_cues(
            srt_processor, subtitles, output_filepath, resume_from_index,
            log_callback)
    start_index = len(done_texts)

    if log_callback:
        log_callback("srt_refine_start",
                     f"✨ Refining {len(subtitles) - start_index} subtitles in {target_language}...")

    if stats_callback:
        stats_callback({
            'total_chunks': len(subtitles),
            'completed_chunks': start_index,
            'failed_chunks': 0,
        })

    # On resume the refine helper only sees the remaining cues, so shift its
    # progress back onto the whole file.
    refine_stats_callback = stats_callback
    if stats_callback and start_index:
        def refine_stats_callback(stats: Dict[str, Any]) -> None:
            stats_callback({
                **stats,
                'total_chunks': len(subtitles),
                'completed_chunks': stats.get('completed_chunks', 0) + start_index,
            })

    # Key translations by list position, NOT by the cue number printed in
    # the file: update_translated_subtitles applies by position, and real
    # files have gaps, restarts or 0-based numbering (issue #205).
    translations: Dict[int, str] = {
        idx: sub.get('text', '') for idx, sub in enumerate(subtitles)
        if idx >= start_index
    }
    subtitle_positions = {id(sub): idx for idx, sub in enumerate(subtitles)}

    llm_client = create_llm_client(
        llm_provider, gemini_api_key, cli_api_endpoint, model_name,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        log_callback=log_callback,
    )

    # Fixed-count grouping for refine (no char cap): every block sent to
    # the LLM has the same shape, which keeps [N] marker accounting
    # predictable across the whole file.
    refine_blocks = srt_processor.group_subtitles_for_translation(
        subtitles[start_index:], SRT_LINES_PER_BLOCK, _NO_CHAR_CAP
    )

    try:
        refined = await refine_subtitle_translations(
            translations=translations,
            target_language=target_language,
            model_name=model_name,
            llm_client=llm_client,
            log_callback=log_callback,
            prompt_options=prompt_options,
            post_processing_instructions=(
                prompt_options.get('refinement_instructions', '')
                if prompt_options else ''
            ),
            stats_callback=refine_stats_callback,
            check_interruption_callback=check_interruption_callback,
            subtitle_blocks=refine_blocks,
            subtitle_positions=subtitle_positions,
        )
    except RateLimitError as e:
        # Persist whatever the pass managed to refine before pausing, then let
        # the error propagate: handlers.py drives the pause / auto-resume.
        payload = e.partial_result
        if not isinstance(payload, dict) or 'translations' not in payload:
            raise
        partial = {**done_texts, **payload['translations']}
        if len(partial) != len(subtitles):
            raise
        last_done = payload.get('next_index', start_index) - 1
        # Read before the write below, which applies the refined texts in place.
        original_last = subtitles[last_done]['text'] if last_done >= 0 else ''
        written = await _write_refined_srt(
            srt_processor, subtitles, partial, output_filepath, log_callback)
        if written:
            if log_callback:
                log_callback("srt_refine_partial_saved",
                             f"💾 Partial refined SRT saved before rate-limit pause: '{output_filepath}'")
            # Record the last finished cue so a resume restarts right after it.
            if checkpoint_manager and translation_id and last_done >= 0:
                try:
                    checkpoint_manager.save_checkpoint(
                        translation_id,
                        chunk_index=last_done,
                        original_text=original_last,
                        translated_text=partial[last_done],
                        total_chunks=len(subtitles),
                        completed_chunks=last_done + 1,
                        failed_chunks=0,
                    )
                except Exception as cp_err:
                    if log_callback:
                        log_callback("srt_refine_checkpoint_error",
                                     f"⚠️ Could not save the refine checkpoint: {cp_err}")
        raise
    finally:
        if llm_client:
            try:
                await llm_client.close()
            except Exception:
                pass

    if check_interruption_callback and check_interruption_callback():
        if log_callback:
            log_callback("srt_refine_interrupted",
                         "Refinement interrupted before save")
        return False

    if not await _write_refined_srt(
            srt_processor, subtitles, {**done_texts, **refined},
            output_filepath, log_callback):
        return False
    if log_callback:
        log_callback("srt_refine_done",
                     f"✅ Refined SRT saved: {output_filepath}")
    return True
