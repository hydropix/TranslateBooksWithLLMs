"""Book-level checkpoint storage of an EPUB run.

``restore_checkpoint_files`` puts the files a previous pass finished back into
the extracted book; ``save_file_checkpoint`` stores one finished file and the
job progress next to it. The row layout (``chunk_index`` = last completed file
index, ``epub_accumulated_stats``, ``epub_unfinished_units``,
``epub_degraded_units``) is what an older checkpoint was written with, so it
must not change.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from lxml import etree

from .archive import parse_xhtml, read_text, serialize_xhtml


async def restore_checkpoint_files(
    checkpoint_manager,
    translation_id: str,
    temp_dir: str,
    resume_from_index: int,
    opf_dir: str,
    log_callback: Optional[Callable] = None
) -> Dict[str, etree._Element]:
    """Restore previously translated files from the checkpoint.

    Returns:
        Dictionary of file_path -> doc_root for the restored files.
    """
    restored_docs: Dict[str, etree._Element] = {}

    if log_callback:
        log_callback("epub_restore_checkpoint",
                     f"Restoring {resume_from_index} previously translated files from checkpoint...")

    restore_success = checkpoint_manager.restore_epub_files(
        translation_id=translation_id,
        work_dir=Path(temp_dir)
    )

    if not restore_success:
        if log_callback:
            log_callback("epub_restore_warning",
                         "Warning: Could not restore all files from checkpoint. Translation will continue from scratch.")
        return restored_docs

    checkpoint_files_dir = checkpoint_manager.uploads_dir / translation_id / "translated_files"

    if not checkpoint_files_dir.exists():
        if log_callback:
            log_callback("epub_restore_no_files", "⚠️ No translated files found in checkpoint")
        return restored_docs

    restored_count = 0
    for saved_file in checkpoint_files_dir.rglob('*'):
        if not saved_file.is_file():
            continue
        rel_path_str = str(saved_file.relative_to(checkpoint_files_dir)).replace('\\', '/')
        file_path_abs = _restored_file_path(temp_dir, opf_dir, rel_path_str, log_callback)
        try:
            restored_docs[file_path_abs] = parse_xhtml(await read_text(file_path_abs))
            restored_count += 1
            if log_callback:
                log_callback("epub_restore_file_parsed",
                             f"📄 Restored file {restored_count}: {rel_path_str}")
        except Exception as e:
            if log_callback:
                log_callback("epub_restore_parse_error",
                             f"⚠️ Warning: Could not parse restored file {rel_path_str}: {e}")

    if log_callback:
        log_callback("epub_restore_success",
                     f"✅ Successfully restored {len(restored_docs)} files from checkpoint")

    return restored_docs


def _restored_file_path(temp_dir: str, opf_dir: str, rel_path_str: str,
                        log_callback: Optional[Callable]) -> str:
    """Where a stored file lives in the extracted book.

    Stored paths are relative to the book root; old checkpoints stored them
    relative to the OPF directory, hence the fallback.
    """
    file_path_abs = os.path.normpath(os.path.join(temp_dir, rel_path_str))
    if not os.path.exists(file_path_abs):
        file_path_abs = os.path.normpath(os.path.join(opf_dir, rel_path_str))
        if log_callback:
            log_callback("epub_restore_fallback",
                         f"🔄 Using fallback path for old checkpoint: {rel_path_str}")
    return file_path_abs


async def save_file_checkpoint(
    checkpoint_manager,
    translation_id: str,
    file_idx: int,
    content_href: str,
    doc_root: etree._Element,
    file_path: str,
    temp_dir: str,
    log_callback: Optional[Callable] = None,
    total_chunks: int = 0,
    completed_chunks: int = 0,
    failed_chunks: int = 0,
    epub_accumulated_stats: Optional[Dict] = None,
    unfinished_units: Optional[Dict[str, List[int]]] = None,
    file_unfinished: Optional[List[int]] = None,
    degraded_units: Optional[Dict[str, List[int]]] = None,
    file_degraded: Optional[List[int]] = None
) -> None:
    """Save checkpoint for a translated file.

    Args:
        file_idx: The LAST COMPLETED file index (monotonic, see the book loop).
        unfinished_units: Job-level index of the chunks still to translate
            ({file_href: [chunk_index, ...]}, issue #261). Stored verbatim in
            the job progress so the next resume knows which files to re-enter.
        file_unfinished: The chunk indices this file still owes. When it is
            empty the per-file partial state is deleted (the file is done);
            when it is not, the state is KEPT, because it is the only place
            that records which chunk is still in the source language.
        degraded_units: Job-level index of the chunks that are translated but
            only approximately tagged ({file_href: [chunk_index, ...]}). Stored
            verbatim next to `unfinished_units`, never merged into it.
        file_degraded: The approximately-tagged chunk indices of this file. Like
            `file_unfinished` it KEEPS the per-file partial state alive, because
            re-entering a file without its state would re-chunk an
            already-translated body (D7) - so the state is the only thing that
            makes an explicit retry of those chunks possible at all. It does NOT
            make the file unfinished: no automatic pass will ever look at it.
    """
    try:
        file_content = serialize_xhtml(doc_root)
        file_rel_path = os.path.relpath(file_path, temp_dir).replace('\\', '/')

        save_result = checkpoint_manager.save_epub_file(
            translation_id=translation_id,
            file_href=file_rel_path,
            file_content=file_content
        )

        if not save_result:
            if log_callback:
                log_callback("epub_checkpoint_save_error",
                             f"⚠️ Warning: Could not save file to checkpoint storage: {content_href}")
            return

        # Delete partial state AFTER successful file save (atomicity guarantee),
        # and ONLY when the file has nothing left to translate. A file with
        # unfinished chunks keeps its state: that state is what records which
        # chunk is still in the source language, and deleting it would make
        # the chunk unrecoverable all over again (issue #261).
        #
        # `content_href` is the authoritative partial-state key: it is what
        # `chunk_loop.ChunkSlots.save_state` saves under and what
        # `translate_xhtml_file` loads with. `file_rel_path` (the
        # temp-dir-relative path) is a different string and is only the key
        # used by `save_epub_file` above; deleting with it here never matched
        # the state actually written, which is what left stale "finished"
        # partial states on disk and blocked retries (issue #261).
        _settle_partial_state(checkpoint_manager, translation_id, content_href,
                              file_unfinished, file_degraded, log_callback)

        # Update checkpoint progress with chunk statistics. The
        # `epub_accumulated_stats` snapshot is what rehydrates the
        # Fallbacks stat card on resume — without it the cross-file
        # counters reset to zero after a pause (issue #180).
        checkpoint_manager.save_checkpoint(
            translation_id=translation_id,
            # Uniform convention: store the LAST COMPLETED file index
            # (resume adds +1), matching TXT/SRT. load_checkpoint maps it
            # back via the 'resume_index_semantics' marker.
            chunk_index=file_idx,
            original_text=content_href,
            translated_text=content_href,
            chunk_data={'last_file': content_href, 'file_type': 'epub_xhtml'},
            total_chunks=total_chunks,
            completed_chunks=completed_chunks,
            failed_chunks=failed_chunks,
            epub_accumulated_stats=epub_accumulated_stats,
            unfinished_units=unfinished_units,
            degraded_units=degraded_units
        )

        if log_callback:
            log_callback("epub_checkpoint_file_saved",
                         f"💾 Checkpoint saved: {file_rel_path} ({len(file_content)} bytes)")
    except Exception as e:
        if log_callback:
            log_callback("epub_checkpoint_save_error",
                         f"⚠️ Warning: Could not save checkpoint: {content_href}: {e}")


def _settle_partial_state(checkpoint_manager, translation_id: str, content_href: str,
                          file_unfinished: Optional[List[int]],
                          file_degraded: Optional[List[int]],
                          log_callback: Optional[Callable]) -> None:
    """Keep the file's partial state while it still owes work, else delete it."""
    if file_unfinished:
        if log_callback:
            log_callback("xhtml_partial_state_kept_unfinished",
                         f"📌 Partial state kept for {content_href}: "
                         f"chunk(s) {file_unfinished} still untranslated")
    elif file_degraded:
        if log_callback:
            log_callback("xhtml_partial_state_kept_degraded",
                         f"📌 Partial state kept for {content_href}: "
                         f"chunk(s) {file_degraded} are translated with "
                         f"approximate tag positions and can be retried on "
                         f"explicit request")
    else:
        checkpoint_manager.delete_xhtml_partial_state(translation_id, content_href)
        if log_callback:
            log_callback("xhtml_partial_state_deleted_after_save",
                         f"🗑️ Partial state deleted for {content_href} (file saved successfully)")
