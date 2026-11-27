"""The per-file loop of a book pass.

``precount_chunks`` counts (and prepares, once) every file's chunks before the
first one is translated. ``process_content_files`` walks the spine, skips
what earlier passes finished, translates each remaining file through
``document_translation.translate_document``, folds its outcome into the
``BookRunState`` and checkpoints it.
"""

import os
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from lxml import etree

from src.core.progress import ProgressReporter

from .archive import parse_xhtml, read_text, resolve_content_path
from .body_serializer import find_body
from .book_checkpoint import save_file_checkpoint
from .book_resume import add_file_prior_counts, begin_pass, chunk_projections, load_partial_state
from .book_state import BookRunState
from .document_translation import prepare_document, translate_document
from .html_utils import is_text_free_chunk
from .run_context import EpubRun, FileTask, PreparedDocument


class ChunkPrecount(NamedTuple):
    """Chunk precount of a book's source files.

    Attributes:
        total_chunks: Chunks of the whole book.
        chunks_per_file: Chunks per spine index (0 for a missing or broken file).
        prepared: Spine index -> PreparedDocument, for the files asked for.
        text_free_chunks: Chunks with no translatable character, which the
            chunk loop passes through without an LLM call (placeholder mode
            only; 0 in Plain Text Mode). The job verdict needs them to tell
            "nothing could be translated" from "some units needed no work".
    """

    total_chunks: int
    chunks_per_file: List[int]
    prepared: Dict[int, PreparedDocument]
    text_free_chunks: int = 0


class FileResult(NamedTuple):
    """Outcome of one file: the translated tree, success, metrics, source text."""

    doc_root: Optional[etree._Element]
    success: bool
    stats: Any
    source_text: Optional[str]


async def precount_chunks(
    content_files: list,
    opf_dir: str,
    max_tokens_per_chunk: int,
    log_callback: Optional[Callable] = None,
    plain_text_mode: bool = False,
    keep_prepared_from: Optional[int] = None,
    keep_emphasis: bool = False,
) -> ChunkPrecount:
    """Count chunks across all XHTML files for accurate progress tracking.

    In placeholder mode each file goes through ``prepare_document`` on its own
    parse, so the chunks are computed once: the result is kept for the files at
    index ``keep_prepared_from`` or above (None keeps none). Files below it are
    the ones a resume restores from the checkpoint, whose copies on disk are
    translations by the time they are processed, so their source chunks must
    never be handed to the translator.

    When plain_text_mode is True, counts chunks using the plain-text pipeline
    (paragraphs joined by \\n\\n then chunked by TokenChunker) instead of the
    HTML-aware chunker, and keeps nothing. ``keep_emphasis`` (Plain Text
    Mode only) extracts the paragraphs with their emphasis markers, exactly
    as plain_text_document sends them, so the counts match what is sent.
    """
    chunks_per_file: List[int] = []
    prepared: Dict[int, PreparedDocument] = {}
    total_chunks = 0
    text_free_chunks = 0

    if log_callback:
        log_callback("epub_precount_start", f"📊 Analyzing {len(content_files)} files for progress tracking...")

    for idx, content_href in enumerate(content_files):
        file_path = resolve_content_path(opf_dir, content_href)
        chunk_count, document = await _count_file(
            file_path, max_tokens_per_chunk, plain_text_mode, keep_emphasis)
        chunks_per_file.append(chunk_count)
        total_chunks += chunk_count
        if document is not None:
            text_free_chunks += sum(
                1 for chunk in document.chunks
                if is_text_free_chunk(chunk.get('text', ''), document.placeholder_format))
        if document is not None and keep_prepared_from is not None and idx >= keep_prepared_from:
            prepared[idx] = document

    if log_callback:
        log_callback("epub_precount_complete",
                     f"📊 Found {total_chunks} total chunks across {len(content_files)} files")

    return ChunkPrecount(total_chunks, chunks_per_file, prepared, text_free_chunks)


async def _count_file(file_path: str, max_tokens_per_chunk: int, plain_text_mode: bool,
                      keep_emphasis: bool = False) -> Tuple[int, Optional[PreparedDocument]]:
    """One file's chunk count (0 on any failure) and, in placeholder mode, its chunks."""
    if not os.path.exists(file_path):
        return 0, None
    try:
        doc_root = parse_xhtml(await read_text(file_path))
        if plain_text_mode:
            return _precount_chunks_plain_text(doc_root, max_tokens_per_chunk, keep_emphasis), None
        document = prepare_document(doc_root, max_tokens_per_chunk)
        if document is None:
            return 0, None
        return document.chunk_count, document
    except Exception:
        return 0, None


def _precount_chunks_plain_text(doc_root, max_tokens_per_chunk: int,
                                keep_emphasis: bool = False) -> int:
    """Count chunks for one XHTML file using the plain-text-mode pipeline.

    Returns 0 on any failure (matches the normal-path behavior).
    """
    try:
        from .plain_extractor import extract_plain_paragraphs
        from src.core.common.plain_text_pipeline import build_plain_segments

        body = find_body(doc_root)
        if body is None:
            return 0

        paragraphs, _, _, _ = extract_plain_paragraphs(body, keep_emphasis=keep_emphasis)
        if not paragraphs:
            return 0

        return len(build_plain_segments(paragraphs, max_tokens_per_chunk,
                                        keep_emphasis=keep_emphasis))
    except Exception:
        return 0


async def translate_xhtml_file(run: EpubRun, task: FileTask) -> FileResult:
    """Read, parse and translate one XHTML file, resuming from its partial state."""
    if not os.path.exists(task.file_path):
        run.log("epub_file_not_found", f"WARNING: File '{task.href}' not found, skipped.")
        return FileResult(None, False, None, None)

    if run.can_checkpoint:
        task.resume_state = run.checkpoint_manager.load_xhtml_partial_state(
            run.translation_id, task.href)
        if task.resume_state:
            run.log("xhtml_resume_detected",
                    f"📂 Resuming '{task.href}' from chunk "
                    f"{task.resume_state.current_chunk_index}/{len(task.resume_state.chunks)}")

    try:
        source_text = await read_text(task.file_path)
        doc_root = parse_xhtml(source_text)
        success, stats = await translate_document(doc_root, run, task)
        return FileResult(doc_root, success, stats, source_text)

    except etree.XMLSyntaxError as e:
        run.log("epub_xml_error", f"XML error in '{task.href}': {e}")
        return FileResult(None, False, None, None)
    except Exception as e:
        # Re-raise RateLimitError to trigger auto-pause
        from src.core.llm.exceptions import RateLimitError
        if isinstance(e, RateLimitError):
            raise
        run.log("epub_file_error", f"Error processing '{task.href}': {e}")
        return FileResult(None, False, None, None)


async def process_content_files(run: EpubRun, book: BookRunState) -> BookRunState:
    """Translate every spine file this pass owes, in reading order.

    ``book`` arrives with the precount and the restored documents; it leaves
    with the outcome of the pass (documents to write, counters, the chunk
    maps, ``was_interrupted``).
    """
    book.progress = ProgressReporter(run.ctx.stats_callback)
    begin_pass(run, book)
    resume_from_index = run.ctx.resume_from_index

    for file_idx, content_href in enumerate(book.content_files):
        if run.interrupted():
            book.was_interrupted = True
            run.log("epub_translation_interrupted",
                    f"Translation interrupted at file {file_idx + 1}/{book.total_files}")
            break

        # Skip if already processed (resume), unless the file still holds
        # unfinished chunks and has a partial state to re-enter it with.
        retry_indices = book.retry_tickets.get(content_href)
        if file_idx < resume_from_index and not retry_indices:
            # A restored file already seeded completed_files; counting it again
            # here reported twice the book's file count after a resume.
            if resolve_content_path(book.opf_dir, content_href) not in book.parsed_docs:
                book.completed_files += 1
            continue

        await _process_file(run, book, file_idx, content_href, retry_indices)

    _log_emphasis_summary(run)
    return book


def _log_emphasis_summary(run: EpubRun) -> None:
    """One book-level line for Plain Text Mode italics (issue #352)."""
    totals = run.emphasis_totals
    if not (run.plain_text_keep_emphasis and totals.get("source_spans")):
        return
    run.log("plain_text_emphasis_summary",
            f"Italics kept: {totals.get('kept_spans', 0)}/{totals['source_spans']} spans "
            f"({totals.get('rejected_paragraphs', 0)} paragraphs fell back to plain)")


async def _process_file(run: EpubRun, book: BookRunState, file_idx: int,
                        content_href: str, retry_indices: Optional[List[int]]) -> None:
    if retry_indices:
        run.log("epub_retry_file",
                f"Re-entering {content_href} to retry "
                f"{len(retry_indices)} unfinished chunk(s): {retry_indices}")

    file_path = resolve_content_path(book.opf_dir, content_href)
    chunks_in_this_file = book.chunks_in_file(file_idx)
    # A re-entered file is already in parsed_docs (it came from the restored
    # documents, which seeded completed_files), so its successful translation
    # must not be counted a second time.
    already_translated = file_path in book.parsed_docs

    # Fold this file's already-counted work into the per-run baseline before
    # it is translated (the chunk loop overwrites its partial state as it
    # goes, so it has to be read now). Only files that were entered in an
    # earlier pass can have one: a file the loop completed had its state
    # deleted, and files above the pointer were never touched.
    if run.ctx.resume_from_index > 0:
        add_file_prior_counts(book.run_prior_counts, run.checkpoint_manager,
                              run.translation_id, content_href)

    run.log("epub_file_translate_start",
            f"Translating file {file_idx + 1}/{book.total_files}: {content_href} ({chunks_in_this_file} chunks)")

    task = FileTask(
        href=content_href,
        file_path=file_path,
        stats_callback=_file_stats_hook(run, book),
        prepared=book.take_prepared(file_idx),
        global_total_chunks=book.total_chunks,
        # Chunks from ALL previous files (not including the current one).
        global_completed_chunks=book.completed_chunks,
    )
    result = await translate_xhtml_file(run, task)

    _advance_chunk_progress(run, book, chunks_in_this_file, result.stats)
    if result.stats:
        book.accumulated.merge(result.stats)

    # What this file still owes, read from the partial state the chunk loop
    # just wrote. This has to happen BEFORE the post-file stats report below
    # (so 'unfinished_chunks'/'unfinished_files' reflect this file's own
    # outcome instead of lagging by one file) and BEFORE save_file_checkpoint,
    # which deletes that state when the file comes back clean.
    state_after = load_partial_state(run.checkpoint_manager, run.translation_id, content_href)
    file_unfinished, file_untranslated, file_degraded = chunk_projections(state_after)
    book.record_file_units(content_href, file_unfinished, file_untranslated, file_degraded)

    if result.stats:
        book.emit()

    _record_outcome(run, book, file_idx, content_href, file_path, result, already_translated)

    if run.can_checkpoint and result.success and result.doc_root is not None:
        book.last_completed_file_idx = max(book.last_completed_file_idx, file_idx)
        await save_file_checkpoint(
            run.checkpoint_manager, run.translation_id, book.last_completed_file_idx,
            content_href, result.doc_root, file_path, book.temp_dir, run.log_callback,
            total_chunks=book.total_chunks,
            completed_chunks=book.completed_chunks,
            failed_chunks=book.accumulated.failed_chunks,
            epub_accumulated_stats=book.accumulated.snapshot_accumulated(),
            unfinished_units=book.unfinished_units,
            file_unfinished=file_unfinished,
            degraded_units=book.degraded_units,
            file_degraded=file_degraded
        )


def _file_stats_hook(run: EpubRun, book: BookRunState) -> Callable[[Dict], None]:
    """Turn the current file's stats into book stats (accumulated + this file)."""
    def file_stats_wrapper(file_stats_dict: Dict) -> None:
        if not run.ctx.stats_callback:
            return
        # The current file's completed chunks (its progress payload, see
        # TranslationMetrics.report_progress). `book.completed_chunks` still
        # excludes this file: it is only advanced once the file returns.
        current_file_completed = file_stats_dict.get('completed_chunks', 0)
        # The fallback counters are included so the Fallbacks stat card
        # updates live. `unfinished_units` still reflects the picture as of the
        # *previous* file at this point (the current file's own outcome is only
        # known once it returns) - an accepted one-file lag for this live
        # mid-file callback; the post-file emit is exact.
        book.emit(book.completed_chunks + current_file_completed, file_stats_dict)
    return file_stats_wrapper


def _advance_chunk_progress(run: EpubRun, book: BookRunState,
                            chunks_in_this_file: int, file_stats) -> None:
    """Update the global chunk counter after a file.

    A fully-translated file contributes all its chunks. On interruption the
    file stopped early, so count only the chunks actually processed —
    otherwise the bar jumps to 100% at the moment of pausing and then drops on
    resume. Clean runs are unaffected (processed == chunks_in_this_file when
    the file completes).
    """
    if run.interrupted() and file_stats is not None:
        book.completed_chunks += min(chunks_in_this_file, file_stats.processed_chunks)
    else:
        book.completed_chunks += chunks_in_this_file


def _record_outcome(run: EpubRun, book: BookRunState, file_idx: int, content_href: str,
                    file_path: str, result: FileResult, already_translated: bool) -> None:
    """Keep the file's document for the write-back and count it."""
    if result.success and result.doc_root is not None:
        book.parsed_docs[file_path] = result.doc_root
        book.failed_doc_paths.discard(file_path)
        if not already_translated:
            book.completed_files += 1
    elif not result.success and result.doc_root is not None:
        # A failed (or interrupted) file is written back exactly as it was
        # read: the translated tree went through boilerplate removal and ruby
        # folding, so writing it would drop content the source still has. For
        # a re-entered file the text read is the restored translated copy,
        # which is what has to be kept.
        book.parsed_docs[file_path] = parse_xhtml(result.source_text)
        book.failed_doc_paths.add(file_path)
        book.failed_files += 1
        run.log("epub_file_translate_failed",
                f"Failed to translate file {file_idx + 1}/{book.total_files}: {content_href}")
    else:
        book.failed_files += 1
