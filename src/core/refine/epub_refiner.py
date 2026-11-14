"""EPUB refine-only mode.

Walks each XHTML content file of an already-translated EPUB, refines its
body in place, and repackages. Interruption stops cleanly between chunks
but does not persist partial state (no resume support in v1).
"""

import os
import tempfile
from typing import Optional, Callable, Dict, Any, List, Tuple
from lxml import etree

from src.config import (
    DEFAULT_MODEL, MAX_TOKENS_PER_CHUNK,
)
from src.core.epub.archive import (
    extract_epub, parse_epub_manifest, repackage_epub,
)
from src.core.epub.xhtml_translator import (
    _setup_translation, _preserve_tags, _create_chunks,
    _replace_body, _escape_stray_angle_brackets, _refine_epub_chunks,
)
from src.core.epub.container import TranslationContainer
from src.core.job import JobConfig, JobContext
from src.core.llm.exceptions import LLMClientConfigError
from .client_setup import build_refine_client


def _globalize_chunk_text(
    chunk: Dict,
    placeholder_format: Tuple[str, str],
) -> str:
    """Convert a chunk's placeholders from local to global indices.

    `_refine_epub_chunks` expects globally-numbered placeholders because it
    re-localizes internally before sending to the LLM. In translate-mode the
    chunks already carry global indices; in refine-only mode they come fresh
    from HtmlChunker with local indices, so we re-globalize here.
    """
    text = chunk['text']
    global_indices = chunk.get('global_indices', [])
    if not global_indices:
        return text

    prefix, suffix = placeholder_format
    for local_idx, global_idx in enumerate(global_indices):
        local_ph = f"{prefix}{local_idx}{suffix}"
        text = text.replace(local_ph, f"__TEMP_GLOBAL_{global_idx}__")
    for global_idx in global_indices:
        text = text.replace(f"__TEMP_GLOBAL_{global_idx}__",
                            f"{prefix}{global_idx}{suffix}")
    return text


def _count_file_chunks(doc_root, max_tokens_per_chunk: int) -> int:
    """Count the chunks a content file will be split into (no LLM call)."""
    try:
        body_html, body_element, tag_preserver = _setup_translation(doc_root, None, None)
        if not body_html or body_element is None:
            return 0
        text, tag_map, _ = _preserve_tags(
            body_html, tag_preserver, None, protect_technical=True
        )
        return len(_create_chunks(text, tag_map, max_tokens_per_chunk, None, None))
    except Exception:
        return 0


async def _refine_one_xhtml(
    doc_root: etree._Element,
    target_language: str,
    model_name: str,
    llm_client: Any,
    max_tokens_per_chunk: int,
    log_callback: Optional[Callable],
    context_manager: Optional[Any],
    prompt_options: Optional[Dict],
    check_interruption_callback: Optional[Callable],
    container: Optional[TranslationContainer] = None,
    stats_callback: Optional[Callable] = None,
) -> bool:
    """Refine a single parsed XHTML document in place."""
    body_html, body_element, tag_preserver = _setup_translation(
        doc_root, log_callback, container
    )
    if not body_html or body_element is None:
        if log_callback:
            log_callback("no_body", "No <body> element found")
        return False

    text_with_placeholders, global_tag_map, placeholder_format = _preserve_tags(
        body_html, tag_preserver, log_callback, protect_technical=True
    )

    chunks = _create_chunks(
        text_with_placeholders, global_tag_map, max_tokens_per_chunk,
        log_callback, container,
    )

    if not chunks:
        if log_callback:
            log_callback("no_chunks", "No translatable chunks in this XHTML, skipping")
        return True

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
            log_callback("refine_interrupted",
                         "Refinement interrupted before reconstruction")
        return False

    full_text = ''.join(refined_chunks)
    full_text = _escape_stray_angle_brackets(full_text)
    final_html = tag_preserver.restore_tags(full_text, global_tag_map)

    return _replace_body(body_element, final_html, log_callback)


async def refine_epub_file(job: JobConfig, ctx: JobContext) -> bool:
    """Run a refinement-only pass on an already-translated EPUB.

    Reads ``job.input_path`` and writes ``job.output_path`` (the
    ``[partial ...]`` sibling when interrupted). Returns True only when every
    content file was refined and the run was not interrupted.
    """
    input_filepath = job.input_path
    output_filepath = job.output_path
    target_language = job.target_language
    model_name = job.refine_provider_config().model or DEFAULT_MODEL
    prompt_options = job.engine_prompt_options()
    max_tokens_per_chunk = (job.max_tokens_per_chunk if job.max_tokens_per_chunk is not None
                            else MAX_TOKENS_PER_CHUNK)
    log_callback = ctx.log_callback
    stats_callback = ctx.stats_callback
    check_interruption_callback = ctx.check_interruption

    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input EPUB file '{input_filepath}' not found."
        if log_callback:
            log_callback("epub_input_file_not_found", err_msg)
        return False

    try:
        llm_client, context_manager = build_refine_client(job, log_callback=log_callback)
    except LLMClientConfigError:
        # Already reported to the job log by build_refine_client.
        return False

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_epub(input_filepath, temp_dir, log_callback)
            manifest_data = parse_epub_manifest(temp_dir, log_callback)

            content_files: List[str] = manifest_data['content_files']
            opf_dir: str = manifest_data['opf_dir']

            total_files = len(content_files)
            if log_callback:
                log_callback("epub_refine_start",
                             f"✨ Starting EPUB refine pass over {total_files} content files...")

            # Progress is counted in chunks across all files (never in files), so
            # numerator and denominator share one unit and cannot exceed 100%.
            file_chunk_counts: Dict[str, int] = {}
            for href in content_files:
                fp = os.path.join(opf_dir, href)
                n = 0
                if os.path.exists(fp):
                    try:
                        n = _count_file_chunks(
                            etree.parse(fp, etree.XMLParser(recover=True)).getroot(),
                            max_tokens_per_chunk,
                        )
                    except Exception:
                        n = 0
                file_chunk_counts[href] = n
            total_chunks = sum(file_chunk_counts.values())
            chunks_done = 0  # chunks of files already handled (refined, failed or skipped)
            failed_chunks = 0

            def _emit(done: int, failed_c: int) -> None:
                if stats_callback:
                    stats_callback({
                        'total_chunks': total_chunks,
                        'completed_chunks': min(done, total_chunks),
                        'failed_chunks': failed_c,
                    })

            if stats_callback:
                stats_callback({'total_chunks': total_chunks, 'completed_chunks': 0, 'failed_chunks': 0})

            completed, failed, interrupted = 0, 0, False
            for idx, href in enumerate(content_files):
                if check_interruption_callback and check_interruption_callback():
                    if log_callback:
                        log_callback("epub_refine_interrupted",
                                     f"Refinement interrupted at file {idx + 1}/{total_files}")
                    interrupted = True
                    break

                file_path = os.path.join(opf_dir, href)
                if not os.path.exists(file_path):
                    if log_callback:
                        log_callback("epub_refine_missing",
                                     f"⚠️ Content file missing in EPUB: {href}, skipping")
                    failed += 1
                    continue

                file_chunks = file_chunk_counts.get(href, 0)
                if log_callback:
                    log_callback("epub_refine_file",
                                 f"📄 Refining file {idx + 1}/{total_files}: {href}")

                try:
                    parser = etree.XMLParser(recover=True, remove_blank_text=False)
                    tree = etree.parse(file_path, parser)
                    doc_root = tree.getroot()
                except Exception as e:
                    if log_callback:
                        log_callback("epub_refine_parse_error",
                                     f"⚠️ Could not parse {href}: {e}")
                    failed += 1
                    continue

                ok = await _refine_one_xhtml(
                    doc_root=doc_root,
                    target_language=target_language,
                    model_name=model_name,
                    llm_client=llm_client,
                    max_tokens_per_chunk=max_tokens_per_chunk,
                    log_callback=log_callback,
                    context_manager=context_manager,
                    prompt_options=prompt_options,
                    check_interruption_callback=check_interruption_callback,
                    stats_callback=lambda st, base=chunks_done: _emit(
                        base + st.get('completed_chunks', 0), failed_chunks),
                )

                if ok:
                    try:
                        tree.write(file_path, xml_declaration=True,
                                   encoding='utf-8', method='xml')
                    except Exception as e:
                        if log_callback:
                            log_callback("epub_refine_write_error",
                                         f"⚠️ Could not write refined XHTML {href}: {e}")
                        failed += 1
                        failed_chunks += file_chunks
                        chunks_done += file_chunks
                        _emit(chunks_done, failed_chunks)
                        continue
                    completed += 1
                else:
                    failed += 1
                    failed_chunks += file_chunks

                chunks_done += file_chunks
                _emit(chunks_done, failed_chunks)

            from src.utils.file_utils import get_partial_output_path
            final_output = (
                get_partial_output_path(output_filepath) if interrupted
                else output_filepath
            )
            repackage_epub(temp_dir=temp_dir,
                            output_filepath=final_output,
                            log_callback=log_callback)

            if log_callback:
                log_callback("epub_refine_done",
                             f"✅ EPUB refine complete: {completed} files refined, "
                             f"{failed} failed, output: {final_output}")
            return not interrupted and failed == 0
    finally:
        if llm_client and hasattr(llm_client, 'close'):
            try:
                await llm_client.close()
            except Exception:
                pass
