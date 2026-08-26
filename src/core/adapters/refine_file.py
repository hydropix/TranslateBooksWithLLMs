"""Unified refine-only entry point.

Mirrors translate_file() but skips the translation phase: the input file
is assumed to already be in the target language, and only a refinement
pass is applied.
"""

import os
from typing import Optional, Callable, Dict, Any

from .exceptions import UnsupportedFormatError
from src.utils.file_detector import detect_file_type, detect_file_type_by_content


async def refine_file(
    input_filepath: str,
    output_filepath: str,
    target_language: str,
    model_name: str,
    llm_provider: str,
    checkpoint_manager: Any = None,
    translation_id: Optional[str] = None,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    check_interruption_callback: Optional[Callable] = None,
    resume_from_index: int = 0,
    llm_api_endpoint: Optional[str] = None,
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
    context_window: Optional[int] = None,
    auto_adjust_context: bool = True,
    max_tokens_per_chunk: Optional[int] = None,
    prompt_options: Optional[Dict[str, Any]] = None,
    **additional_config,
) -> bool:
    """Run a refinement-only pass on an already-translated file.

    `target_language` names the language the file is already in: refinement
    is monolingual and does not translate.

    Raises UnsupportedFormatError when the file format cannot be refined.
    """
    if prompt_options is None:
        prompt_options = {}

    # Resolve max_tokens_per_chunk lazily so a reload_config() between calls is
    # honoured for subsequent runs (the .env value can change at runtime via
    # the /api/settings endpoint).
    if max_tokens_per_chunk is None:
        from src.config import MAX_TOKENS_PER_CHUNK as _DEFAULT_MAX_TOKENS
        max_tokens_per_chunk = _DEFAULT_MAX_TOKENS

    _, ext = os.path.splitext(input_filepath.lower())
    try:
        detected_type = detect_file_type(input_filepath)
    except ValueError:
        detected_type = detect_file_type_by_content(input_filepath)
        if detected_type is None:
            raise UnsupportedFormatError(
                f"Cannot determine file type for: {ext}. "
                f"The file does not appear to be a supported format."
            )

    if log_callback and detected_type != ext.lstrip('.'):
        log_callback("file_type_detected",
                     f"📄 File with extension '{ext}' detected as '{detected_type.upper()}' format")

    if detected_type == 'txt':
        from src.core.refine.txt_refiner import refine_txt_file
        return await refine_txt_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            target_language=target_language,
            model_name=model_name,
            cli_api_endpoint=llm_api_endpoint,
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
            context_window=context_window or 2048,
            auto_adjust_context=auto_adjust_context,
            max_tokens_per_chunk=max_tokens_per_chunk,
            prompt_options=prompt_options,
        )

    if detected_type == 'epub':
        from src.core.refine.epub_refiner import refine_epub_file
        return await refine_epub_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            target_language=target_language,
            model_name=model_name,
            cli_api_endpoint=llm_api_endpoint,
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
            context_window=context_window or 2048,
            auto_adjust_context=auto_adjust_context,
            max_tokens_per_chunk=max_tokens_per_chunk,
            prompt_options=prompt_options,
        )

    if detected_type == 'docx':
        from src.core.refine.docx_refiner import refine_docx_file
        return await refine_docx_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            target_language=target_language,
            model_name=model_name,
            cli_api_endpoint=llm_api_endpoint,
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
            context_window=context_window or 2048,
            auto_adjust_context=auto_adjust_context,
            max_tokens_per_chunk=max_tokens_per_chunk,
            prompt_options=prompt_options,
        )

    if detected_type == 'srt':
        from src.core.refine.srt_refiner import refine_srt_file
        return await refine_srt_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            target_language=target_language,
            model_name=model_name,
            cli_api_endpoint=llm_api_endpoint,
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
            prompt_options=prompt_options,
        )

    supported = ', '.join(['txt', 'epub', 'srt', 'docx'])
    raise UnsupportedFormatError(
        f"Unsupported file format for refine-only: {detected_type}. "
        f"Supported formats: {supported}"
    )
