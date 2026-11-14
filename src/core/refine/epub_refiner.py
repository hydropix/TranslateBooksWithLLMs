"""EPUB refine-only mode.

Walks each XHTML content file of an already-translated EPUB, refines its
body in place, and repackages. An interruption or a rate limit writes the
book as refined so far to a ``[partial ...]`` output; there is no chunk-level
resume.
"""

import os
import tempfile
from typing import Optional, Callable, Dict, Any, List
from lxml import etree

from src.config import (
    DEFAULT_MODEL, MAX_TOKENS_PER_CHUNK,
)
from src.core.epub.archive import (
    extract_epub, parse_epub_manifest, repackage_epub,
)
from src.core.epub.body_reassembly import escape_stray_markup, replace_body
from src.core.epub.chunk_refinement import refine_chunks
from src.core.epub.document_chunking import chunk_document, tag_preserver_for
from src.core.epub.placeholder_renumberer import globalize_chunks
from src.core.epub.container import TranslationContainer
from src.core.job import JobConfig, JobContext
from src.core.llm.exceptions import LLMClientConfigError, RateLimitError
from src.core.progress import Phase, ProgressReporter
from .client_setup import build_refine_client


def _count_file_chunks(doc_root, max_tokens_per_chunk: int) -> int:
    """Count the chunks a content file will be split into (no LLM call)."""
    try:
        body_element, prepared = chunk_document(doc_root, max_tokens_per_chunk)
        if body_element is None or prepared.is_empty:
            return 0
        return len(prepared.chunks)
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
    progress: Optional[ProgressReporter] = None,
) -> bool:
    """Refine a single parsed XHTML document in place.

    The body is chunked exactly like the translation chunked it (same
    technical content protection setting).
    """
    body_element, prepared = chunk_document(
        doc_root, max_tokens_per_chunk, container=container, log_callback=log_callback
    )
    if body_element is None or prepared.is_empty:
        if log_callback:
            log_callback("no_body", "No <body> element found")
        return False

    chunks = prepared.chunks
    global_tag_map = prepared.global_tag_map
    placeholder_format = prepared.placeholder_format
    if not chunks:
        if log_callback:
            log_callback("no_chunks", "No translatable chunks in this XHTML, skipping")
        return True

    # Fresh chunks carry local indices; refine_chunks expects the draft with
    # global ones (it re-localizes each chunk before sending it).
    draft_globalized = globalize_chunks(chunks, placeholder_format)

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

    if check_interruption_callback and check_interruption_callback():
        if log_callback:
            log_callback("refine_interrupted",
                         "Refinement interrupted before reconstruction")
        return False

    full_text = escape_stray_markup(''.join(refined_chunks))
    final_html = tag_preserver_for(placeholder_format, container).restore_tags(full_text, global_tag_map)

    return replace_body(body_element, final_html, log_callback)


def _save_partial_epub(temp_dir: str, output_filepath: str,
                       log_callback: Optional[Callable]) -> None:
    """Package the book as refined so far into the ``[partial ...]`` output."""
    from src.utils.file_utils import get_partial_output_path
    partial_path = get_partial_output_path(output_filepath)
    try:
        repackage_epub(temp_dir=temp_dir, output_filepath=partial_path, log_callback=log_callback)
    except Exception as e:
        if log_callback:
            log_callback("refine_partial_save_failed",
                         f"⚠️ Could not save the partial refined EPUB: {e}")
        return
    if log_callback:
        log_callback("refine_partial_saved",
                     f"💾 Partial refined output saved before rate-limit pause: '{partial_path}'")


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

            # Each refined chunk ticks the reporter; every file end sets the
            # book-level counts again, so a file whose actual chunks differ
            # from its precount cannot push the count past the total.
            progress = ProgressReporter(ctx.stats_callback, phase=Phase.REFINING)
            progress.begin(total_chunks)
            progress.emit()

            def _emit_file_done() -> None:
                progress.set_counts(min(chunks_done, total_chunks), failed=failed_chunks)
                progress.emit()

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

                try:
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
                        progress=progress,
                    )
                except RateLimitError:
                    # Keep the files refined so far, exactly as an interruption
                    # does (the file in flight keeps its draft), then let the
                    # error propagate: the job runner drives the pause.
                    _save_partial_epub(temp_dir, output_filepath, log_callback)
                    raise

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
                        _emit_file_done()
                        continue
                    completed += 1
                else:
                    failed += 1
                    failed_chunks += file_chunks

                chunks_done += file_chunks
                _emit_file_done()

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
