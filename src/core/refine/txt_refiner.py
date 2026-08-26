"""
TXT refine-only mode.

Reads an already-translated plain-text file, chunks it the same way as
translation would, then runs refine_chunks() and writes the polished output.
"""

import os
import aiofiles
from typing import Optional, Callable, Dict, Any

from src.core.chunking.reassembly import join_translated_chunks
from src.core.llm.exceptions import RateLimitError
from src.core.text_processor import split_text_into_chunks
from src.core.translator import refine_chunks
from src.config import DEFAULT_MODEL, API_ENDPOINT


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
    anthropic_api_key: Optional[str] = None,
    xai_api_key: Optional[str] = None,
    opencode_api_key: Optional[str] = None,
    opencodego_api_key: Optional[str] = None,
    ollamacloud_api_key: Optional[str] = None,
    context_window: int = 2048,
    auto_adjust_context: bool = True,
    max_tokens_per_chunk: Optional[int] = None,
    soft_limit_ratio: Optional[float] = None,
    prompt_options: Optional[Dict[str, Any]] = None,
) -> bool:
    """Run a refinement-only pass on an already-translated text file.

    `target_language` names the language the file is already in: refinement
    is monolingual and does not translate.
    """
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

    if stats_callback:
        stats_callback({'total_chunks': total_chunks, 'completed_chunks': 0, 'failed_chunks': 0})

    if log_callback:
        log_callback("refine_info_chunks",
                     f"Refining {total_chunks} segment(s) in {target_language}.")

    # refine_chunks uses original_chunks only for context_before/after, so in
    # refine-only mode we pass main_content as both draft and original.
    draft_chunks = [c["main_content"] for c in structured_chunks]

    try:
        refined_parts = await refine_chunks(
            translated_chunks=draft_chunks,
            original_chunks=structured_chunks,
            target_language=target_language,
            model_name=model_name,
            api_endpoint=cli_api_endpoint,
            log_callback=log_callback,
            stats_callback=stats_callback,
            check_interruption_callback=check_interruption_callback,
            llm_provider=llm_provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            mistral_api_key=mistral_api_key,
            deepseek_api_key=deepseek_api_key,
            poe_api_key=poe_api_key,
            nim_api_key=nim_api_key,
            anthropic_api_key=anthropic_api_key,
            xai_api_key=xai_api_key,
            opencode_api_key=opencode_api_key,
            opencodego_api_key=opencodego_api_key,
            ollamacloud_api_key=ollamacloud_api_key,
            context_window=context_window,
            auto_adjust_context=auto_adjust_context,
            prompt_options=prompt_options,
        )
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
