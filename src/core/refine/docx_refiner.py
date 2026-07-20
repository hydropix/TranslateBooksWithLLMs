"""DOCX refine-only mode.

Converts an already-translated DOCX to HTML, refines, then writes a new
DOCX. Reuses the EPUB tag-preservation + chunking machinery. No resume
support in v1.
"""

import os
import tempfile
from typing import Optional, Callable, Dict

from src.config import DEFAULT_MODEL, API_ENDPOINT, MAX_TOKENS_PER_CHUNK
from src.core.epub.xhtml_translator import _refine_epub_chunks, _escape_stray_angle_brackets
from src.core.epub.container import TranslationContainer
from src.core.docx.converter import DocxHtmlConverter
from .client_setup import build_refine_client
from .epub_refiner import _globalize_chunk_text


async def refine_docx_file(
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
    atlascloud_api_key: Optional[str] = None,
    poe_api_key: Optional[str] = None,
    nim_api_key: Optional[str] = None,
    context_window: int = 2048,
    auto_adjust_context: bool = True,
    prompt_options: Optional[Dict] = None,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
) -> bool:
    """Run a refinement-only pass on an already-translated DOCX file."""
    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input DOCX file '{input_filepath}' not found."
        if log_callback:
            log_callback("docx_input_file_not_found", err_msg)
        return False

    llm_client, context_manager = build_refine_client(
        model_name=model_name,
        llm_provider=llm_provider,
        cli_api_endpoint=cli_api_endpoint,
        auto_adjust_context=auto_adjust_context,
        context_window=context_window,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        atlascloud_api_key=atlascloud_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        log_callback=log_callback,
    )
    if llm_client is None:
        return False

    try:
        if log_callback:
            log_callback("docx_refine_start", "✨ Starting DOCX refine pass...")

        converter = DocxHtmlConverter()
        container = TranslationContainer()
        tag_preserver = container.tag_preserver

        html_content, metadata = converter.to_html(input_filepath)
        if log_callback:
            log_callback("docx_html_extracted",
                         f"Extracted {len(html_content)} chars HTML from DOCX")

        text_with_placeholders, tag_map = tag_preserver.preserve_tags(html_content)
        placeholder_format = (
            tag_preserver.placeholder_format.prefix,
            tag_preserver.placeholder_format.suffix,
        )
        if log_callback:
            log_callback("docx_tags_preserved",
                         f"Preserved {len(tag_map)} tag groups")

        chunks = container.chunker.chunk_html_with_placeholders(
            text_with_placeholders, tag_map
        )
        if not chunks:
            if log_callback:
                log_callback("docx_no_chunks", "Nothing to refine in this DOCX")
            return True

        if log_callback:
            log_callback("docx_chunks_created", f"Created {len(chunks)} chunks")

        if stats_callback:
            stats_callback({'total_chunks': len(chunks), 'completed_chunks': 0, 'failed_chunks': 0})

        draft_globalized = [_globalize_chunk_text(c, placeholder_format) for c in chunks]

        refined_chunks = await _refine_epub_chunks(
            translated_chunks=draft_globalized,
            chunks=chunks,
            target_language=target_language,
            model_name=model_name,
            llm_client=llm_client,
            context_manager=context_manager,
            placeholder_format=placeholder_format,
            log_callback=log_callback,
            prompt_options=prompt_options,
            stats_callback=stats_callback,
        )

        if check_interruption_callback and check_interruption_callback():
            if log_callback:
                log_callback("docx_refine_interrupted",
                             "Refinement interrupted before reconstruction")
            return False

        full_text = ''.join(refined_chunks)
        full_text = _escape_stray_angle_brackets(full_text)
        final_html = tag_preserver.restore_tags(full_text, tag_map)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name
        try:
            converter.from_html(final_html, metadata, tmp_path)
            with open(tmp_path, 'rb') as src, open(output_filepath, 'wb') as dst:
                dst.write(src.read())
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        if log_callback:
            log_callback("docx_refine_done",
                         f"✅ DOCX refine complete, saved: {output_filepath}")
        return True
    finally:
        if llm_client and hasattr(llm_client, 'close'):
            try:
                await llm_client.close()
            except Exception:
                pass
