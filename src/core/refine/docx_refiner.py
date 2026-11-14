"""DOCX refine-only mode.

Converts an already-translated DOCX to HTML, refines, then writes a new
DOCX. Reuses the EPUB tag-preservation + chunking machinery. No resume
support in v1.
"""

import os
import tempfile

from src.config import DEFAULT_MODEL
from src.core.epub.body_reassembly import escape_stray_markup
from src.core.epub.chunk_refinement import refine_chunks
from src.core.epub.placeholder_renumberer import globalize_chunks
from src.core.epub.container import TranslationContainer
from src.core.epub.html_chunker import HtmlChunker
from src.core.docx.converter import DocxHtmlConverter
from src.core.job import JobConfig, JobContext
from src.core.llm.exceptions import LLMClientConfigError, RateLimitError
from src.core.progress import Phase, ProgressReporter
from .client_setup import build_refine_client


def _write_docx(converter, tag_preserver, refined_chunks, tag_map, metadata, output_filepath) -> None:
    """Rebuild the DOCX from refined chunks (GLOBAL placeholders) and write it."""
    full_text = escape_stray_markup(''.join(refined_chunks))
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


async def refine_docx_file(job: JobConfig, ctx: JobContext) -> bool:
    """Run a refinement-only pass on an already-translated DOCX file.

    Reads ``job.input_path`` and writes ``job.output_path``.
    ``job.max_tokens_per_chunk`` sizes the refine chunks; None reads the
    current MAX_TOKENS_PER_CHUNK setting at call time. Returns True when the
    refined file was written.
    """
    input_filepath = job.input_path
    output_filepath = job.output_path
    target_language = job.target_language
    model_name = job.refine_provider_config().model or DEFAULT_MODEL
    prompt_options = job.engine_prompt_options()
    max_tokens_per_chunk = job.max_tokens_per_chunk
    log_callback = ctx.log_callback
    check_interruption_callback = ctx.check_interruption

    if max_tokens_per_chunk is None:
        from src.config import MAX_TOKENS_PER_CHUNK
        max_tokens_per_chunk = MAX_TOKENS_PER_CHUNK
    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input DOCX file '{input_filepath}' not found."
        if log_callback:
            log_callback("docx_input_file_not_found", err_msg)
        return False

    try:
        llm_client, context_manager = build_refine_client(job, log_callback=log_callback)
    except LLMClientConfigError:
        # Already reported to the job log by build_refine_client.
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

        # The container's chunker is sized with the MAX_TOKENS_PER_CHUNK
        # frozen at import; build one with the caller's value instead.
        chunker = HtmlChunker(max_tokens=max_tokens_per_chunk)
        chunks = chunker.chunk_html_with_placeholders(
            text_with_placeholders, tag_map
        )
        if not chunks:
            if log_callback:
                log_callback("docx_no_chunks", "Nothing to refine in this DOCX")
            return True

        if log_callback:
            log_callback("docx_chunks_created", f"Created {len(chunks)} chunks")

        progress = ProgressReporter(ctx.stats_callback, phase=Phase.REFINING)
        progress.begin(len(chunks))
        progress.emit()

        draft_globalized = globalize_chunks(chunks, placeholder_format)

        try:
            refined_chunks = await refine_chunks(
                translated_chunks=draft_globalized,
                chunks=chunks,
                target_language=target_language,
                model_name=model_name,
                llm_client=llm_client,
                context_manager=context_manager,
                placeholder_format=placeholder_format,
                log_callback=log_callback,
                prompt_options=prompt_options,
                progress=progress,
            )
        except RateLimitError as e:
            # Save what was refined before pausing (the rest keeps its draft),
            # then let the error propagate: the job runner drives the pause.
            # Same contract as the TXT and SRT refiners.
            parts = e.partial_result
            if parts is not None and len(parts) == len(chunks):
                _write_docx(converter, tag_preserver, parts, tag_map, metadata, output_filepath)
                if log_callback:
                    log_callback("refine_partial_saved",
                                 f"💾 Partial refined output saved before rate-limit pause: '{output_filepath}'")
            raise

        if check_interruption_callback and check_interruption_callback():
            if log_callback:
                log_callback("docx_refine_interrupted",
                             "Refinement interrupted before reconstruction")
            return False

        _write_docx(converter, tag_preserver, refined_chunks, tag_map, metadata, output_filepath)

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
