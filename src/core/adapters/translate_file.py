"""
Unified file translation function using the adapter pattern.

This module provides a single entry point for translating files of any supported
format (TXT, SRT, EPUB, DOCX) using format-specific adapters.

File type detection supports:
1. Known extensions (.txt, .epub, .srt, .docx)
2. Content-based detection for unknown extensions (e.g., .log, .md, .text -> txt)
3. Automatic routing to appropriate processors
"""

import os
from typing import Optional, Callable, Dict, Any
from pathlib import Path

from .generic_translator import GenericTranslator
from .txt_adapter import TxtAdapter
from .srt_adapter import SrtAdapter
from .epub_adapter import EpubAdapter
from .exceptions import UnsupportedFormatError
from src.utils.file_detector import detect_file_type, detect_file_type_by_content


async def translate_file(
    input_filepath: str,
    output_filepath: str,
    source_language: str,
    target_language: str,
    model_name: str,
    llm_provider: str,
    checkpoint_manager: Any,
    translation_id: str,
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
    context_window: Optional[int] = None,
    auto_adjust_context: bool = True,
    min_chunk_size: int = 5,
    prompt_options: Optional[Dict[str, Any]] = None,
    bilingual_output: bool = False,
    **additional_config
) -> bool:
    """
    Translate a file using the adapter pattern.

    This is the unified entry point for all file format translations, replacing:
    - translate_text_file_with_callbacks() for TXT
    - translate_srt_file_with_callbacks() for SRT
    - translate_epub_file() for EPUB

    Args:
        input_filepath: Path to the input file
        output_filepath: Path to the output file
        source_language: Source language name
        target_language: Target language name
        model_name: LLM model identifier
        llm_provider: LLM provider name (ollama, gemini, openai, openrouter)
        checkpoint_manager: CheckpointManager instance for resume capability
        translation_id: Unique identifier for this translation job
        log_callback: Optional callback for logging (receives type and message)
        stats_callback: Optional callback for statistics updates
        check_interruption_callback: Optional callback to check if translation should be interrupted
        resume_from_index: Index to resume from (if resuming from checkpoint)
        llm_api_endpoint: LLM API endpoint URL
        gemini_api_key: Google Gemini API key (required for gemini provider)
        openai_api_key: OpenAI API key (required for openai provider)
        openrouter_api_key: OpenRouter API key (required for openrouter provider)
        mistral_api_key: Mistral API key (required for mistral provider)
        deepseek_api_key: DeepSeek API key (required for deepseek provider)
        poe_api_key: Poe API key (required for poe provider)
        context_window: Maximum context window size in tokens
        auto_adjust_context: Whether to automatically adjust context size
        min_chunk_size: Minimum chunk size for text splitting
        prompt_options: Optional prompt customization options
        bilingual_output: If True, output will contain both original and translated text
        **additional_config: Additional configuration passed to the adapter

    Returns:
        True if translation completed successfully, False otherwise

    Raises:
        UnsupportedFormatError: If the file format is not supported

    Example:
        >>> from src.persistence.checkpoint_manager import CheckpointManager
        >>> from src.core.adapters import translate_file
        >>>
        >>> checkpoint_mgr = CheckpointManager()
        >>> success = await translate_file(
        ...     input_filepath="book.epub",
        ...     output_filepath="book_fr.epub",
        ...     source_language="English",
        ...     target_language="French",
        ...     model_name="llama3.2",
        ...     llm_provider="ollama",
        ...     checkpoint_manager=checkpoint_mgr,
        ...     translation_id="job_123",
        ...     llm_api_endpoint="http://localhost:11434/api/generate"
        ... )
    """
    # Initialize prompt_options if not provided
    if prompt_options is None:
        prompt_options = {}

    # Detect file format - first by extension, then by content for unknown extensions
    _, ext = os.path.splitext(input_filepath.lower())

    # Use content-based detection for file type determination
    # This handles files with non-standard extensions (e.g., .log, .text, .md)
    try:
        detected_type = detect_file_type(input_filepath)
    except ValueError:
        # If detection fails, try content-based detection as fallback
        detected_type = detect_file_type_by_content(input_filepath)
        if detected_type is None:
            raise UnsupportedFormatError(
                f"Cannot determine file type for: {ext}. "
                f"The file does not appear to be a supported format."
            )

    # Log detected type if different from extension
    if log_callback and detected_type != ext.lstrip('.'):
        log_callback("file_type_detected",
            f"📄 File with extension '{ext}' detected as '{detected_type.upper()}' format")

    # TEMPORARY WORKAROUND: For EPUB files, use the legacy translate_epub_file() directly
    # The generic adapter pattern doesn't work well with EPUB's complex XHTML processing
    # that requires HTML chunking, tag preservation, technical content protection, etc.
    # TODO: Refactor EPUB translation to properly work with the adapter pattern
    if detected_type == 'epub':
        from src.core.epub.translator import translate_epub_file
        await translate_epub_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            source_language=source_language,
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
            context_window=context_window or 2048,
            auto_adjust_context=auto_adjust_context,
            min_chunk_size=min_chunk_size,
            checkpoint_manager=checkpoint_manager,
            translation_id=translation_id,
            resume_from_index=resume_from_index,
            prompt_options=prompt_options,
            bilingual=bilingual_output,
            **additional_config
        )
        return True  # Legacy function doesn't return success status

    # DOCX translation using EPUB pipeline (Phase 1 implementation)
    # Similar to EPUB, DOCX requires HTML chunking, tag preservation, etc.
    # This reuses the EPUB pipeline for rapid deployment
    if detected_type == 'docx':
        from src.core.docx.translator import translate_docx_file
        from src.core.llm import create_llm_provider

        # Create LLM client
        llm_client = create_llm_provider(
            provider_type=llm_provider,
            endpoint=llm_api_endpoint,
            model=model_name,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            mistral_api_key=mistral_api_key,
            deepseek_api_key=deepseek_api_key,
            poe_api_key=poe_api_key
        )

        result = await translate_docx_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            source_language=source_language,
            target_language=target_language,
            model_name=model_name,
            llm_client=llm_client,
            max_tokens_per_chunk=context_window or 450,
            log_callback=log_callback,
            stats_callback=stats_callback,
            prompt_options=prompt_options,
            max_retries=1,
            context_manager=None,
            check_interruption_callback=check_interruption_callback,
            checkpoint_manager=checkpoint_manager,
            translation_id=translation_id
        )
        return result.get('success', False)

    # Map detected file types to adapters
    adapter_map = {
        'txt': TxtAdapter,
        'srt': SrtAdapter,
        # Note: 'epub' uses legacy path above
        # Note: 'docx' uses legacy path above
    }

    adapter_class = adapter_map.get(detected_type)
    if not adapter_class:
        supported = ', '.join(['txt', 'srt', 'epub', 'docx'])
        raise UnsupportedFormatError(
            f"Unsupported file format: {detected_type}. Supported formats: {supported}"
        )

    # Prepare adapter configuration (format-specific settings)
    adapter_config = {
        'context_window': context_window,
        'auto_adjust_context': auto_adjust_context,
        'min_chunk_size': min_chunk_size,
        'prompt_options': prompt_options,
        **additional_config
    }

    # Create adapter instance
    adapter = adapter_class(
        input_file_path=input_filepath,
        output_file_path=output_filepath,
        config=adapter_config
    )

    # Create generic translator
    translator = GenericTranslator(
        adapter=adapter,
        checkpoint_manager=checkpoint_manager,
        translation_id=translation_id
    )

    # Prepare LLM configuration (provider-specific settings)
    llm_config = {
        'endpoint': llm_api_endpoint,
        'gemini_api_key': gemini_api_key,
        'openai_api_key': openai_api_key,
        'openrouter_api_key': openrouter_api_key,
        'mistral_api_key': mistral_api_key,
        'deepseek_api_key': deepseek_api_key,
        'poe_api_key': poe_api_key,
        'prompt_options': prompt_options,
    }

    # Execute translation
    return await translator.translate(
        source_language=source_language,
        target_language=target_language,
        model_name=model_name,
        llm_provider=llm_provider,
        log_callback=log_callback,
        stats_callback=stats_callback,
        check_interruption_callback=check_interruption_callback,
        bilingual_output=bilingual_output,
        **llm_config
    )


def get_file_type_from_path(filepath: str) -> str:
    """
    Get the file type identifier from a file path.

    Args:
        filepath: Path to the file

    Returns:
        File type string: 'txt', 'srt', 'epub', or 'unknown'
    """
    _, ext = os.path.splitext(filepath.lower())

    type_map = {
        '.txt': 'txt',
        '.srt': 'srt',
        '.epub': 'epub',
        '.docx': 'docx',
    }

    return type_map.get(ext, 'unknown')


async def build_translated_output(
    translation_id: str,
    checkpoint_manager: Any,
    **adapter_config
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Rebuild the translated output file from a checkpoint.

    This function is used by the web API to reconstruct the output file
    from checkpoint data when resuming or downloading an interrupted translation.

    Args:
        translation_id: Translation job identifier
        checkpoint_manager: CheckpointManager instance
        **adapter_config: Additional configuration for the adapter

    Returns:
        Tuple of (output_bytes, error_message)
        - If successful: (bytes, None)
        - If failed: (None, error_message)

    Example:
        >>> from src.persistence.checkpoint_manager import CheckpointManager
        >>> from src.core.adapters import build_translated_output
        >>>
        >>> checkpoint_mgr = CheckpointManager()
        >>> output_bytes, error = await build_translated_output(
        ...     translation_id="job_123",
        ...     checkpoint_manager=checkpoint_mgr
        ... )
        >>> if output_bytes:
        ...     with open("output.epub", "wb") as f:
        ...         f.write(output_bytes)
    """
    # Load job from checkpoint
    job = checkpoint_manager.db.get_job(translation_id)
    if not job:
        return None, "Job not found"

    config = job['config']
    file_type = job['file_type']

    # Map file types to adapters
    adapter_map = {
        'txt': TxtAdapter,
        'srt': SrtAdapter,
        'epub': EpubAdapter,
        # Note: docx doesn't support checkpoint reconstruction yet
    }

    adapter_class = adapter_map.get(file_type)
    if not adapter_class:
        return None, f"Unsupported file type: {file_type}"

    # Get file paths from config
    input_file_path = config.get('preserved_input_path') or config.get('input_file_path')
    output_file_path = config.get('output_file_path')

    if not input_file_path or not output_file_path:
        return None, "Missing file paths in checkpoint configuration"

    # Create adapter instance
    try:
        adapter = adapter_class(
            input_file_path=input_file_path,
            output_file_path=output_file_path,
            config={**config, **adapter_config}
        )

        # Prepare adapter
        if not await adapter.prepare_for_translation():
            return None, "Failed to prepare adapter for reconstruction"

        # Load checkpoint data
        checkpoint_data = checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint_data:
            return None, "No checkpoint data found"

        # Resume from checkpoint (restores translated content)
        await adapter.resume_from_checkpoint(checkpoint_data)

        # Reconstruct output
        output_bytes = await adapter.reconstruct_output()

        # Cleanup
        await adapter.cleanup()

        return output_bytes, None

    except Exception as e:
        return None, f"Error reconstructing output: {str(e)}"
