"""EPUB translation entry point: one book, start to finish.

``translate_epub_file`` reads as the pipeline's step list; each step lives in
its own module:

- ``archive``: extraction, OPF discovery, spine reading, repackaging;
- ``llm_setup``: LLM client and context sizing;
- ``book_loop``: chunk precount and the per-file loop;
- ``book_resume`` / ``book_checkpoint``: resume planning, checkpoint I/O;
- ``book_state``: the pass state and its stats payload;
- ``toc_labels``, ``opf_metadata``, ``attribution_page``,
  ``book_postprocess``: what runs on the book once its files are translated.
"""
import os
import tempfile

from src.config import DEFAULT_MODEL, MAX_TRANSLATION_ATTEMPTS, resolve_parallel_workers

from ..job import JobConfig, JobContext, JobResult
from ..llm.exceptions import LLMClientConfigError
from .archive import extract_epub, parse_epub_manifest, resolve_content_path
from .attribution_page import add_attribution_page
from .book_checkpoint import restore_checkpoint_files
from .book_loop import precount_chunks, process_content_files
from .book_postprocess import (
    apply_text_direction,
    log_completion,
    normalize_source_script,
    save_translated_files,
    toc_source_docs,
    translate_packaging_metadata,
    write_output_epub,
)
from .book_state import BookRunState
from .head_titles import SourceTitles, capture_source_titles, localize_head_titles
from .lang_support import apply_target_language_to_xhtml_directory
from .llm_setup import (
    create_context_manager,
    create_llm_client,
    initial_context_size,
    job_max_tokens_per_chunk,
)
from .opf_metadata import update_epub_metadata
from .run_context import EpubRun
from .toc_labels import (
    update_nav_toc_labels_from_translated_docs,
    update_ncx_toc_labels_from_translated_docs,
)


async def translate_epub_file(job: JobConfig, ctx: JobContext) -> JobResult:
    """Translate an EPUB file.

    Args:
        job: The job. Reads ``input_path``, ``output_path``, the languages,
            the provider, ``context_window`` (2048 when unset),
            ``auto_adjust_context``, ``max_tokens_per_chunk`` (the current
            ``MAX_TOKENS_PER_CHUNK`` when unset), ``prompt_options``,
            ``bilingual_output``, ``parallel_workers`` and
            ``retry_token_aligned`` (explicit, user-initiated retry of the
            chunks token alignment translated with approximate tag
            positions; the only thing that widens the work set to them, D3).
        ctx: Log/stats/interruption hooks, checkpoint manager, job id and
            ``resume_from_index`` (file index to resume from).

    Returns:
        A :class:`JobResult`. COMPLETED only when the whole book was
        translated and written: no failed file, no chunk left unfinished, not
        interrupted. FAILED when the input is missing, the LLM client cannot
        be created or an error aborts the job. PARTIAL / INTERRUPTED
        otherwise (a partial EPUB and the checkpoint are still written as
        before, so the job can be resumed or retried); an interrupted run's
        ``output_path`` is the ``[partial ...]`` file it wrote. Documents
        whose translation failed are listed in ``failed_documents``.

    Raises:
        RateLimitError: propagated unchanged so the caller can auto-pause.

    Over 80 lines: it is the documented step list, and most of its length is
    the ordering rationale of steps 4.5..7, kept next to the calls it orders.
    """
    ctx, seen_stats = ctx.tap_stats()
    if not job.input_path or not os.path.exists(job.input_path):
        err_msg = f"ERROR: Input EPUB file '{job.input_path}' not found."
        ctx.log("epub_input_file_not_found", err_msg)
        return JobResult.failed(err_msg)

    # A configuration error stops the job (no fallback client).
    try:
        run = _build_run(job, ctx)
    except LLMClientConfigError as e:
        return JobResult.failed(f"Could not create the LLM client: {e}")

    output_filepath = job.output_path
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 1-4. Extract, parse the manifest, translate every file (resuming
            # from the checkpoint) and write the kept documents back.
            manifest_data, book = await _translate_book(run, temp_dir)
            opf_dir, opf_tree = manifest_data['opf_dir'], manifest_data['opf_tree']

            # 4.5. Update NCX table-of-contents labels from translated XHTML
            # headings. This preserves <content src="..."> jump targets while
            # localizing the reader's side-panel TOC for EPUB2 books. Files
            # whose translation failed keep the source text, so their TOC
            # entries keep the original label.
            toc_docs = toc_source_docs(book)
            update_ncx_toc_labels_from_translated_docs(
                opf_dir=opf_dir, parsed_xhtml_docs=toc_docs, log_callback=run.log_callback)

            # 4.6. Update the EPUB3 navigation document (nav.xhtml) TOC links
            # the same way. EPUB3 readers build their TOC from this document
            # rather than the NCX, so without this step the side-panel keeps the
            # source-language titles even though the body is translated.
            update_nav_toc_labels_from_translated_docs(
                opf_dir=opf_dir, opf_tree=opf_tree, parsed_xhtml_docs=toc_docs,
                log_callback=run.log_callback)

            # 4.7. Append the attribution page to the end of the spine. Runs before
            # the metadata write (step 5) so the manifest and spine edits are
            # persisted by that single OPF write, and before RTL (6) and the lang
            # pass (6.5) so the new page inherits RTL CSS and the target lang
            # attribute like any other document in the book.
            add_attribution_page(opf_tree=opf_tree, opf_dir=opf_dir,
                                 log_callback=run.log_callback)

            # 5. Update metadata
            update_epub_metadata(opf_tree=opf_tree, opf_path=manifest_data['opf_path'],
                                 target_language=run.target_language)

            # 5.5. Localize the packaging metadata (OPF dc:title/dc:description
            # and every NCX docTitle), which readers show on the library shelf
            # and in the book-information panel.
            #
            # Ordering rationale — DO NOT REORDER THIS AND STEP 6.6:
            #   - after 5 (update_epub_metadata): that call appends the
            #     attribution signature to dc:description, and this pass has to
            #     strip it before sending and re-append it afterwards, so the
            #     signature must already be there.
            #   - BEFORE 6.6 (apply_script_normalization_to_epub_directory):
            #     both passes write the OPF, from two *different* trees — 6.6
            #     re-parses the OPF from disk. Writing our in-memory
            #     `manifest_data['opf_tree']` after 6.6 has written its own tree
            #     would silently revert 6.6's font-override meta removal and its
            #     page-progression-direction reset. Running first means 6.6
            #     re-reads a file that already carries the translated metadata,
            #     and its NCX xml:lang write preserves our docTitle for the same
            #     reason.
            #
            # Guarded like 6.6: metadata localization is a nice-to-have relative
            # to the translated text and must never fail an otherwise-successful
            # job.
            await translate_packaging_metadata(run, manifest_data)

            # 6. Apply RTL/LTR layout based on source and target languages.
            apply_text_direction(run, temp_dir)

            # 6.5. Update <html lang="..."> on every XHTML to the target language
            # so that e-readers apply the correct hyphenation, dictionary and TTS.
            # Runs after RTL apply so it is the final authority on lang attributes.
            apply_target_language_to_xhtml_directory(
                temp_dir, run.target_language, log_callback=run.log_callback)

            # 6.6. Neutralize source-script (CJK) typography left over from the
            # original stylesheet/packaging (font stacks, indent, leading,
            # writing mode, reader-specific font-override metas).
            #
            # Ordering rationale:
            #   - after 5 (update_epub_metadata): that call performs the
            #     pipeline's single OPF write; running later means our OPF
            #     edits are not overwritten.
            #   - after 5.5 (translate_opf_metadata), and this must not be
            #     swapped: that pass writes the *in-memory* opf_tree while this
            #     one re-parses the OPF from disk, so running it first would
            #     have its meta removal and progression reset overwritten. See
            #     the longer note at step 5.5.
            #   - after 6 (apply_rtl_to_epub_directory): that pass injects
            #     <style> blocks and may set page-progression-direction.
            #     Running later makes this pass the final authority on
            #     writing-mode and progression direction, and lets it
            #     neutralize anything the RTL pass injected.
            #   - after 6.5 (apply_target_language_to_xhtml_directory): that
            #     pass rewrites every XHTML root's lang attributes; running
            #     later avoids two passes re-serializing the same files in an
            #     interleaved way.
            #   - before 7 (repackage): obviously.
            #
            # Guarded: cosmetic relative to the translated text.
            normalize_source_script(run, temp_dir)

            # 7. Repackage EPUB (a `[partial] ` file when interrupted).
            output_filepath = write_output_epub(run, temp_dir, output_filepath,
                                                book.was_interrupted)
            log_completion(run, book)

        except Exception as e_epub:
            # Re-raise RateLimitError to trigger auto-pause
            from src.core.llm.exceptions import RateLimitError
            if isinstance(e_epub, RateLimitError):
                raise
            err_msg = f"MAJOR ERROR processing EPUB '{job.input_path}': {e_epub}"
            if ctx.log_callback:
                ctx.log("epub_major_error", err_msg)
                import traceback
                ctx.log("epub_major_error_traceback", traceback.format_exc())
            return JobResult.failed(err_msg, stats=seen_stats)

    return JobResult.from_run(
        seen_stats, output_filepath,
        engine_ok=book.book_complete,
        interrupted=book.was_interrupted,
        failed_documents=book.failed_doc_paths,
    )


def _build_run(job: JobConfig, ctx: JobContext) -> EpubRun:
    """The LLM client, context manager and chunking settings of the job.

    Raises:
        LLMClientConfigError: the client cannot be created.
    """
    prompt_options = job.engine_prompt_options()
    if job.bilingual_output:
        prompt_options['bilingual'] = True

    initial_context, is_known_thinking_model = initial_context_size(
        job.model or DEFAULT_MODEL, job.auto_adjust_context, job.context_window or 2048)
    llm_client = create_llm_client(job, initial_context, ctx.log_callback)

    # Resolve effective parallel workers (local providers are forced back to 1).
    # translate_file() already logged the effective count; this re-resolve is
    # idempotent and covers the direct-call path.
    parallel_workers = resolve_parallel_workers(job.llm_provider, job.parallel_workers)

    context_manager = create_context_manager(
        llm_provider=job.llm_provider,
        auto_adjust_context=job.auto_adjust_context,
        initial_context=initial_context,
        is_thinking_model=is_known_thinking_model,
        log_callback=ctx.log_callback,
    )
    return EpubRun(
        job=job,
        ctx=ctx,
        llm_client=llm_client,
        context_manager=context_manager,
        max_tokens_per_chunk=job_max_tokens_per_chunk(job),
        prompt_options=prompt_options,
        parallel_workers=parallel_workers,
        max_retries=MAX_TRANSLATION_ATTEMPTS,
    )


async def _translate_book(run: EpubRun, temp_dir: str):
    """Steps 1-4: ``(manifest_data, BookRunState)`` after the files are written."""
    # 1. Extract EPUB
    extract_epub(run.job.input_path, temp_dir, run.log_callback)

    # 2. Parse manifest
    manifest_data = parse_epub_manifest(temp_dir, run.log_callback)
    resume_from_index = run.ctx.resume_from_index
    book = BookRunState(manifest_data['content_files'], manifest_data['opf_dir'], temp_dir)

    # 2.1. Read the source <title>s, first headings and TOC labels while the
    # extracted files are still the source ones (the checkpoint restore below
    # overwrites the finished documents with their translation). Cosmetic:
    # a failure here only leaves every title unchanged.
    try:
        source_titles = capture_source_titles(
            [resolve_content_path(book.opf_dir, href) for href in book.content_files],
            book.opf_dir, manifest_data['opf_tree'], book_root=temp_dir)
    except Exception:
        source_titles = SourceTitles()

    # 2.5. Count the chunks, then restore the checkpoint if resuming. The
    # chunks are counted first: the restore overwrites every finished file
    # with its translation, and counting those instead of the source gave a
    # resumed job a different total than the run it continues. The prepared
    # chunks are only kept for the files at or above the resume pointer,
    # which the restore never touches.
    precount = await precount_chunks(
        book.content_files, book.opf_dir, run.max_tokens_per_chunk, run.log_callback,
        plain_text_mode=run.plain_text_mode, keep_prepared_from=resume_from_index,
        keep_emphasis=run.plain_text_keep_emphasis,
    )
    book.total_chunks, book.chunks_per_file, book.prepared = precount[:3]
    book.skipped_chunks = precount.text_free_chunks
    if run.can_checkpoint and resume_from_index > 0:
        book.parsed_docs = await restore_checkpoint_files(
            run.checkpoint_manager, run.translation_id, temp_dir,
            resume_from_index, book.opf_dir, run.log_callback)
        book.completed_files = len(book.parsed_docs)

    # 3. Translate all files
    await process_content_files(run, book)

    # 3.5. Localize each document's <head><title> from its translated first
    # heading or TOC label, when the source title is that same text (no LLM
    # call). Cosmetic: must never fail the job.
    try:
        localize_head_titles(toc_source_docs(book), source_titles, run.log_callback)
    except Exception as e_title:
        run.log("epub_head_titles_failed",
                f"⚠️ Document title localization failed and was skipped: {e_title}")

    # 4. Save translated files
    await save_translated_files(parsed_xhtml_docs=book.parsed_docs,
                                log_callback=run.log_callback)
    return manifest_data, book
