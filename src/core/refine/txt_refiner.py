"""
TXT refine-only mode.

Reads an already-translated plain-text file, chunks it the same way as
translation would, then runs refine_chunks() and writes the polished output.
"""

import os
import aiofiles
from typing import Optional, Callable

from src.core.chunking.reassembly import join_translated_chunks
from src.core.llm.exceptions import RateLimitError
from src.core.text_processor import split_text_into_chunks
from src.core.job import JobConfig, JobContext
from src.core.translator import refine_chunks


async def _write_refined_output(
    refined_parts,
    structured_chunks,
    output_filepath: str,
    log_callback: Optional[Callable] = None,
) -> bool:
    """Reassemble the refined parts and write the final file.

    Shared by the nominal completion path and the rate-limit partial-save path
    so both produce byte-identical output for the same parts. Returns True when
    the file was written; logs and returns False otherwise.
    """
    from src.config import ATTRIBUTION_ENABLED, GENERATOR_NAME, GENERATOR_SOURCE
    final_text = join_translated_chunks(refined_parts, structured_chunks)
    if ATTRIBUTION_ENABLED:
        footer = f"\n\n{'=' * 60}\n"
        footer += f"Refined with {GENERATOR_NAME}\n"
        footer += f"{GENERATOR_SOURCE}\n"
        footer += f"{'=' * 60}\n"
        final_text += footer

    try:
        async with aiofiles.open(output_filepath, 'w', encoding='utf-8') as f:
            await f.write(final_text)
        return True
    except Exception as e:
        err_msg = f"ERROR: Saving output file '{output_filepath}': {e}"
        if log_callback:
            log_callback("refine_save_error", err_msg)
        else:
            print(err_msg)
        return False


async def refine_txt_file(
    job: JobConfig,
    ctx: JobContext,
    *,
    soft_limit_ratio: Optional[float] = None,
) -> bool:
    """Run a refinement-only pass on an already-translated text file.

    Reads ``job.input_path`` and writes ``job.output_path``.
    ``job.target_language`` names the language the file is already in:
    refinement is monolingual and does not translate. Returns True when the
    refined file was written.
    """
    input_filepath = job.input_path
    output_filepath = job.output_path
    target_language = job.target_language
    log_callback = ctx.log_callback
    max_tokens_per_chunk = job.max_tokens_per_chunk

    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input file '{input_filepath}' not found."
        if log_callback:
            log_callback("file_not_found_error", err_msg)
        else:
            print(err_msg)
        return False

    try:
        async with aiofiles.open(input_filepath, 'r', encoding='utf-8') as f:
            translated_text = await f.read()
    except Exception as e:
        err_msg = f"ERROR: Reading input file '{input_filepath}': {e}"
        if log_callback:
            log_callback("file_read_error", err_msg)
        else:
            print(err_msg)
        return False

    if not translated_text.strip():
        if log_callback:
            log_callback("txt_empty_input", "Empty input file. Nothing to refine.")
        try:
            async with aiofiles.open(output_filepath, 'w', encoding='utf-8') as f:
                await f.write("")
        except Exception:
            pass
        return True

    if log_callback:
        log_callback("refine_split_start", "Splitting translated text for refinement...")

    structured_chunks = split_text_into_chunks(
        translated_text,
        max_tokens_per_chunk=max_tokens_per_chunk,
        soft_limit_ratio=soft_limit_ratio,
    )
    total_chunks = len(structured_chunks)

    if total_chunks == 0:
        if log_callback:
            log_callback("txt_no_chunks_warning",
                         "WARNING: No segments generated for non-empty text. Processing as a single block.")
        structured_chunks = [{
            "context_before": "",
            "main_content": translated_text,
            "context_after": "",
        }]
        total_chunks = 1

    if log_callback:
        log_callback("refine_info_chunks",
                     f"Refining {total_chunks} segment(s) in {target_language}.")

    # refine_chunks uses original_chunks only for context_before/after, so in
    # refine-only mode we pass main_content as both draft and original.
    draft_chunks = [c["main_content"] for c in structured_chunks]

    try:
        refined_parts = await refine_chunks(draft_chunks, structured_chunks, job, ctx)
    except RateLimitError as e:
        # Persist whatever the pass managed to refine before pausing, then let
        # the error propagate: handlers.py drives the pause / auto-resume.
        parts = e.partial_result
        if parts is None or len(parts) != len(structured_chunks):
            raise
        written = await _write_refined_output(
            parts, structured_chunks, output_filepath, log_callback)
        if written and log_callback:
            log_callback("refine_partial_saved",
                         f"💾 Partial refined output saved before rate-limit pause: '{output_filepath}'")
        raise

    if not await _write_refined_output(
            refined_parts, structured_chunks, output_filepath, log_callback):
        return False

    if log_callback:
        log_callback("refine_save_success", f"Refined output saved: '{output_filepath}'")
    return True
