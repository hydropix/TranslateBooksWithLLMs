"""
Checkpoint manager for translation job persistence and resume functionality.
"""

import os
import logging
import re
import shutil
import time
from typing import TYPE_CHECKING, Optional, Dict, List, Any, Tuple
from pathlib import Path
from .database import Database
from src.utils.security import SecurityError, resolve_within

if TYPE_CHECKING:
    from src.core.epub.xhtml_translation_state import XHTMLTranslationState


# Uploaded files are stored as "<16 hex chars>_<original name>"; the prefix is
# an implementation detail that must never reach the user interface.
_UPLOAD_PREFIX_RE = re.compile(r'^[0-9a-f]{16}_')


_logger = logging.getLogger(__name__)

def _strip_upload_prefix(filename: str) -> str:
    """Remove the 16-hex upload prefix from a stored filename, if present."""
    return _UPLOAD_PREFIX_RE.sub('', filename)


class CheckpointManager:
    """
    Manages translation job checkpoints including database persistence
    and file storage for uploaded files.
    """

    # Age below which cleanup_orphan_uploads never deletes an item.
    ORPHAN_GRACE_SECONDS = 3600

    def __init__(self, db_path: str = "data/jobs.db", server_session_id: Optional[str] = None):
        """
        Initialize checkpoint manager.

        Args:
            db_path: Path to SQLite database
            server_session_id: Unique identifier for the current server session
        """
        self.db = Database(db_path)
        self.uploads_dir = Path("data/uploads")
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.server_session_id = server_session_id

    def start_job(
        self,
        translation_id: str,
        file_type: str,
        config: Dict[str, Any],
        input_file_path: Optional[str] = None
    ) -> bool:
        """
        Start tracking a new translation job.

        Args:
            translation_id: Unique job identifier
            file_type: Type of file (txt, srt, epub)
            config: Full translation configuration. API keys are stripped at
                persistence (issue #213) — resume re-resolves them from .env
                or the resume request, never from the database.
            input_file_path: Path to input file (will be preserved if it's a temp file)

        Returns:
            True if started successfully
        """
        # Preserve input file first (updates config with preserved_input_path)
        if input_file_path:
            self._preserve_input_file(translation_id, input_file_path, config)

        # Create job in database with updated config and server session ID
        success = self.db.create_job(
            translation_id, file_type, config, self.server_session_id,
            owner_pid=os.getpid(),
        )

        return success

    def get_job(self, translation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job information by translation ID.

        Args:
            translation_id: Job identifier

        Returns:
            Job dictionary or None if not found
        """
        return self.db.get_job(translation_id)

    def update_job_config(self, translation_id: str, config: Dict[str, Any]) -> bool:
        """
        Update the configuration of an existing job.

        Args:
            translation_id: Job identifier
            config: New configuration dictionary

        Returns:
            True if updated successfully
        """
        return self.db.update_job_config(translation_id, config)

    def _preserve_input_file(
        self,
        translation_id: str,
        input_file_path: str,
        config: Dict[str, Any]
    ):
        """
        Preserve the input file for resume capability.

        Args:
            translation_id: Job identifier
            input_file_path: Original input file path
            config: Translation configuration (will be updated with preserved path)
        """
        # Check if file is in temp directory
        input_path = Path(input_file_path)

        # Only preserve if file exists
        if not input_path.exists():
            _logger.warning(f"Warning: Input file does not exist: {input_file_path}")
            return

        # Always preserve uploaded files for web interface
        # For CLI, only preserve if explicitly needed
        job_upload_dir = self.uploads_dir / translation_id
        job_upload_dir.mkdir(parents=True, exist_ok=True)

        # Keep the original filename (including any hash prefix)
        preserved_path = job_upload_dir / input_path.name

        try:
            shutil.copy2(input_file_path, preserved_path)
            # Update config with preserved path (stored in DB)
            config['preserved_input_path'] = str(preserved_path)
            _logger.debug(f"Input file preserved: {preserved_path}")
        except Exception as e:
            _logger.warning(f"Warning: Could not preserve input file: {e}")

    def save_checkpoint(
        self,
        translation_id: str,
        chunk_index: int,
        original_text: str,
        translated_text: Optional[str],
        chunk_data: Optional[Dict[str, Any]] = None,
        translation_context: Optional[Dict[str, Any]] = None,
        total_chunks: Optional[int] = None,
        completed_chunks: Optional[int] = None,
        failed_chunks: Optional[int] = None,
        epub_accumulated_stats: Optional[Dict[str, Any]] = None,
        unfinished_units: Optional[Dict[str, List[int]]] = None,
        degraded_units: Optional[Dict[str, List[int]]] = None,
        phase: str = 'translate'
    ) -> bool:
        """
        Save a checkpoint after translating a chunk.

        Args:
            translation_id: Job identifier
            chunk_index: Index of the chunk
            original_text: Original chunk text
            translated_text: Translated text (None if failed)
            chunk_data: Additional chunk metadata
            translation_context: LLM context for continuity
            total_chunks: Total number of chunks
            completed_chunks: Number of completed chunks
            failed_chunks: Number of failed chunks
            unfinished_units: EPUB per-file index of the chunks still to
                translate ({file_href: [chunk_index, ...]}, issue #261).
                Stored verbatim in the progress JSON so the next resume knows
                which files still have to be re-entered.
            degraded_units: EPUB per-file index of the chunks that are
                translated but carry only approximate tag positions (token
                alignment, {file_href: [chunk_index, ...]}). Stored verbatim
                alongside the map above, and kept separate from it on purpose:
                those chunks are never retried automatically (D3), so they must
                not reach the unfinished index, which drives the verdict.
            phase: Checkpoint namespace. 'refine' units are stored apart from
                the translation chunks and never move the job's progress
                pointer, which belongs to the translation.

        Returns:
            True if saved successfully
        """
        # Save chunk
        chunk_status = 'completed' if translated_text else 'failed'
        chunk_saved = self.db.save_chunk(
            translation_id,
            chunk_index,
            original_text,
            translated_text,
            chunk_data,
            chunk_status,
            phase=phase,
        )
        if phase != 'translate':
            return chunk_saved

        # Update job progress
        progress_saved = self.db.update_job_progress(
            translation_id,
            current_chunk_index=chunk_index,
            total_chunks=total_chunks,
            completed_chunks=completed_chunks,
            failed_chunks=failed_chunks,
            epub_accumulated_stats=epub_accumulated_stats,
            unfinished_units=unfinished_units,
            degraded_units=degraded_units
        )

        # Update translation context if provided
        if translation_context:
            self.db.update_translation_context(translation_id, translation_context)

        return chunk_saved and progress_saved

    def load_checkpoint(self, translation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data for a job.

        Args:
            translation_id: Job identifier

        Returns:
            Dictionary containing:
                - job: Job metadata and config
                - chunks: List of completed chunks
                - resume_from_index: Index to resume from
        """
        # Get job data
        job = self.db.get_job(translation_id)
        if not job:
            return None

        # Get chunks
        chunks = self.db.get_chunks(translation_id)

        # Determine resume point.
        #
        # New (uniform) convention: every format stores current_chunk_index as
        # the LAST COMPLETED unit, so resume is always current_chunk_index + 1.
        # New checkpoints carry the 'resume_index_semantics' = 'completed' marker
        # (set at job creation).
        #
        # Legacy fallback (pre-migration checkpoints, no marker): EPUB used to
        # store file_idx + 1 (the next file), so it must NOT add +1; TXT/SRT
        # stored the last completed chunk, so they add +1.
        progress = job['progress']
        file_type = job.get('file_type', 'txt')

        if progress.get('resume_index_semantics') == 'completed':
            resume_from_index = progress['current_chunk_index'] + 1
        elif file_type == 'epub':
            resume_from_index = max(0, progress['current_chunk_index'])
        else:
            resume_from_index = progress['current_chunk_index'] + 1

        failed_chunk_indices = [
            c['chunk_index'] for c in chunks if c.get('status') == 'failed'
        ]

        return {
            'job': job,
            'chunks': chunks,
            'resume_from_index': resume_from_index,
            'failed_chunk_indices': failed_chunk_indices,
            'translation_context': job.get('translation_context'),
            'job_phase': job.get('job_phase') or 'translate',
            'refine_resume_index': self.refine_resume_index(translation_id),
        }

    def refine_resume_index(self, translation_id: str) -> int:
        """Index the refine pass of this job resumes from (0 when none ran)."""
        done = [c['chunk_index'] for c in self.db.get_chunks(translation_id, phase='refine')
                if c.get('status') == 'completed']
        return (max(done) + 1) if done else 0

    def set_job_phase(self, translation_id: str, phase: str) -> bool:
        """Persist the phase the job is entering ('translate' or 'refine')."""
        return self.db.set_job_phase(translation_id, phase)

    def touch_owner(self, translation_id: str) -> bool:
        """Heartbeat: this process still runs the job (see owner_still_running)."""
        return self.db.touch_owner(translation_id, os.getpid())

    def get_resumable_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all jobs that can be resumed.

        Returns:
            List of job summaries with progress information
        """
        jobs = self.db.get_resumable_jobs()

        # Enrich with additional info
        for job in jobs:
            progress = job['progress']
            total = progress.get('total_chunks', 0)
            completed = progress.get('completed_chunks', 0)

            if total > 0:
                job['progress_percentage'] = int((completed / total) * 100)
            else:
                job['progress_percentage'] = 0

            # Get input and output file names from config
            config = job['config']

            # Extract input filename (use file_path, then preserved_input_path as fallback)
            input_path = config.get('file_path') or config.get('preserved_input_path', 'unknown')
            if input_path != 'unknown':
                job['input_filename'] = Path(input_path).name
            else:
                job['input_filename'] = 'unknown'

            # Extract output filename
            output_filename = config.get('output_filename', 'unknown')
            job['output_filename'] = output_filename if output_filename != 'unknown' else 'unknown'

        return jobs

    def get_job_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get the most recently completed jobs, newest first.

        Args:
            limit: Maximum number of rows to return (clamped by the database
                layer to 1..200)

        Returns:
            List of completed job summaries enriched with display fields.
            The raw 'config' stays attached; the HTTP layer strips it.
        """
        jobs = self.db.get_job_history(limit)

        for job in jobs:
            config = job.get('config') or {}
            progress = job.get('progress') or {}

            # Prefer the filename persisted with the job. Rows created before
            # that field existed only carry the on-disk path, whose basename
            # still wears the 16-hex upload prefix.
            input_filename = config.get('input_filename')
            if not input_filename:
                input_path = config.get('file_path') or config.get('preserved_input_path')
                if input_path:
                    input_filename = _strip_upload_prefix(Path(input_path).name)
                else:
                    input_filename = 'unknown'

            job['input_filename'] = input_filename
            job['output_filename'] = config.get('output_filename', 'unknown')
            job['total_chunks'] = progress.get('total_chunks', 0)
            job['completed_chunks'] = progress.get('completed_chunks', 0)

        return jobs

    def reset_running_jobs_on_startup(self) -> int:
        """
        Reset jobs with 'running' status from previous server sessions to 'interrupted'.

        Only resets jobs that have a different server_session_id, preserving
        jobs that are actually running in the current session. This prevents
        browser refreshes from interrupting active translations.

        This should be called on server startup to handle jobs that were
        interrupted by a server crash or restart. These jobs will then
        appear in the resumable jobs list.

        Returns:
            Number of jobs reset
        """
        if not self.server_session_id:
            # Fallback: if no session ID, don't reset anything to be safe
            return 0
        return self.db.reset_running_jobs(self.server_session_id)

    def cleanup_old_jobs(self, max_age_days: int = 30) -> Tuple[int, int]:
        """
        Clean up old jobs and their associated files.

        This removes jobs older than max_age_days and cleans up their
        upload directories to prevent database and disk bloat.

        Args:
            max_age_days: Maximum age in days for jobs to keep (default 30)

        Returns:
            Tuple of (jobs_deleted, files_cleaned)
        """
        # Get list of old job IDs before deletion (for file cleanup)
        old_jobs = []
        try:
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(days=max_age_days)

            # Get jobs that will be deleted
            all_jobs = self.db.get_resumable_jobs(max_age_days=9999)  # Get all
            for job in all_jobs:
                created_str = job.get('created_at', '')
                if created_str:
                    try:
                        created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                        if created.replace(tzinfo=None) < cutoff:
                            old_jobs.append(job['translation_id'])
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            _logger.warning(f"Warning: Error getting old job list: {e}")

        # Delete from database
        jobs_deleted = self.db.cleanup_old_jobs(max_age_days)

        # Clean up upload directories for deleted jobs
        files_cleaned = 0
        for job_id in old_jobs:
            job_upload_dir = self.uploads_dir / job_id
            if job_upload_dir.exists():
                try:
                    shutil.rmtree(job_upload_dir)
                    files_cleaned += 1
                except Exception as e:
                    _logger.warning(f"Warning: Could not delete upload directory for {job_id}: {e}")

        return jobs_deleted, files_cleaned

    def cleanup_orphan_uploads(self) -> int:
        """
        Clean up upload files/directories that don't have corresponding jobs in the database.

        These are "orphan" items left behind from previous incomplete cleanups.
        Handles:
        - job folders (trans_xxx from the web UI, cli_xxx from the CLI, ...):
          deleted only when no job row carries that ID
        - hash_filename files (legacy upload files)

        Items modified within ORPHAN_GRACE_SECONDS are kept: another process
        creates a job folder just before it inserts the job row.

        Returns:
            Number of orphan items deleted
        """
        orphans_deleted = 0

        if not self.uploads_dir.exists():
            return 0

        # Get all job IDs and preserved file paths from database
        try:
            import sqlite3
            import json
            conn = sqlite3.connect(self.db.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT translation_id, config FROM translation_jobs")
            db_job_ids = set()
            preserved_files = set()  # Full file paths that are referenced
            for row in cursor.fetchall():
                db_job_ids.add(row['translation_id'])
                config = json.loads(row['config'])
                preserved_path = config.get('preserved_input_path', '')
                if preserved_path:
                    # Store the filename to check against orphan files
                    preserved_files.add(Path(preserved_path).name)
            conn.close()
        except Exception as e:
            _logger.warning(f"Warning: Error getting job IDs: {e}")
            return 0

        # Check each item in uploads directory
        for item in self.uploads_dir.iterdir():
            item_name = item.name

            # Skip test folders
            if item_name.startswith('test_'):
                continue

            is_orphan = True

            try:
                if time.time() - item.stat().st_mtime < self.ORPHAN_GRACE_SECONDS:
                    continue
            except OSError:
                continue

            if item.is_dir():
                # A job folder is named after its job ID, whatever the prefix
                # (trans_ for web jobs, cli_ for CLI jobs). Only a folder no
                # job row points to is an orphan.
                if item_name in db_job_ids:
                    is_orphan = False
            else:
                # It's a file - check if it's referenced by any job
                if item_name in preserved_files:
                    is_orphan = False

            if is_orphan:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    orphans_deleted += 1
                except Exception as e:
                    _logger.warning(f"Warning: Could not delete orphan {item_name}: {e}")

        return orphans_deleted

    def mark_paused(self, translation_id: str) -> bool:
        """
        Mark a job as paused (user-initiated stop).

        Args:
            translation_id: Job identifier

        Returns:
            True if updated successfully
        """
        return self.db.update_job_progress(translation_id, status='paused')

    def mark_interrupted(self, translation_id: str) -> bool:
        """
        Mark a job as interrupted (unexpected stop/error).

        Args:
            translation_id: Job identifier

        Returns:
            True if updated successfully
        """
        return self.db.update_job_progress(translation_id, status='interrupted')

    def mark_partial(self, translation_id: str) -> bool:
        """
        Mark a job as 'partial' — the translation loop finished but some chunks
        remain in failed state. The job stays resumable so the user can retry
        those chunks without re-running the whole file.
        """
        return self.db.update_job_progress(translation_id, status='partial')

    def mark_error(self, translation_id: str) -> bool:
        """
        Mark a job as errored. The job stays resumable (get_resumable_jobs
        includes 'error'), so the checkpoint must NOT be cleaned up.
        """
        return self.db.update_job_progress(translation_id, status='error')

    def mark_completed(self, translation_id: str) -> bool:
        """
        Mark a job as completed.

        Args:
            translation_id: Job identifier

        Returns:
            True if updated successfully
        """
        return self.db.update_job_progress(translation_id, status='completed')

    def mark_running(self, translation_id: str) -> bool:
        """
        Mark a job as running (resumed).

        Args:
            translation_id: Job identifier

        Returns:
            True if updated successfully
        """
        return self.db.update_job_progress(translation_id, status='running')

    def delete_checkpoint(self, translation_id: str) -> bool:
        """
        Delete a job checkpoint completely (user-initiated cleanup).

        Args:
            translation_id: Job identifier

        Returns:
            True if deleted successfully
        """
        # Delete from database (chunks deleted via CASCADE)
        db_deleted = self.db.delete_job(translation_id)

        # Delete preserved files
        job_upload_dir = self.uploads_dir / translation_id
        if job_upload_dir.exists():
            try:
                shutil.rmtree(job_upload_dir)
            except Exception as e:
                _logger.warning(f"Warning: Could not delete upload directory: {e}")

        return db_deleted

    def prune_job_data(self, translation_id: str) -> bool:
        """
        Drop the bulky resume data of a job while keeping its database row.

        Called on the success path: the translation_jobs row survives so the
        job can be listed by the history endpoint (and is swept later by
        cleanup_old_jobs), but the checkpoint chunks and the preserved upload
        directory are removed because nothing can resume from them any more.

        This method NEVER touches translation_jobs. Use delete_checkpoint for
        the user-initiated "forget this job entirely" path.

        Args:
            translation_id: Job identifier

        Returns:
            True if the chunks were deleted successfully
        """
        ok = self.db.delete_chunks(translation_id)

        job_upload_dir = self.uploads_dir / translation_id
        if job_upload_dir.exists():
            try:
                shutil.rmtree(job_upload_dir)
            except Exception as e:
                _logger.warning(f"Warning: Could not delete upload directory: {e}")

        return ok

    def get_preserved_input_path(self, translation_id: str) -> Optional[str]:
        """
        Get the preserved input file path for a job.

        Args:
            translation_id: Job identifier

        Returns:
            Path to preserved input file or None
        """
        job = self.db.get_job(translation_id)
        if not job:
            return None

        config = job['config']
        preserved_path = config.get('preserved_input_path')

        if preserved_path and Path(preserved_path).exists():
            return preserved_path

        return None

    def save_epub_file(
        self,
        translation_id: str,
        file_href: str,
        file_content: bytes
    ) -> bool:
        """
        Save a translated XHTML file for EPUB reconstruction.

        Args:
            translation_id: Job identifier
            file_href: Relative path within EPUB (e.g., "OEBPS/chapter1.xhtml")
            file_content: Raw file content (bytes)

        Returns:
            True if saved successfully
        """
        job_dir = self.uploads_dir / translation_id / "translated_files"
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Preserve directory structure, but never outside the job directory:
            # file_href comes from the EPUB manifest and is attacker-controlled
            file_path = resolve_within(job_dir, file_href)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'wb') as f:
                f.write(file_content)
            return True
        except SecurityError:
            _logger.error(f"Refusing unsafe EPUB file href: {file_href}")
            return False
        except Exception as e:
            _logger.error(f"Error saving EPUB file {file_href}: {e}")
            return False

    def restore_epub_files(
        self,
        translation_id: str,
        work_dir: Path
    ) -> bool:
        """
        Restore translated XHTML files from checkpoint to work_dir.

        Args:
            translation_id: Job identifier
            work_dir: Work directory where files should be restored

        Returns:
            True if restore successful
        """
        translated_files_dir = self.uploads_dir / translation_id / "translated_files"
        if not translated_files_dir.exists():
            return False

        try:
            for file_path in translated_files_dir.rglob('*'):
                if file_path.is_file():
                    rel_path = file_path.relative_to(translated_files_dir)
                    try:
                        dest_path = resolve_within(
                            work_dir, str(rel_path).replace('\\', '/')
                        )
                    except SecurityError:
                        _logger.warning(f"Skipping unsafe restored EPUB path: {rel_path}")
                        continue
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_path)
            return True
        except Exception as e:
            _logger.error(f"Error restoring EPUB files: {e}")
            return False

    def _xhtml_state_path(self, translation_id: str, file_href: str) -> Path:
        """
        Compute the on-disk path of an XHTML partial-state file for `file_href`.

        This is the single authority for that location. Issue #261 was caused
        by save/load/delete each recomputing their own sanitised filename
        inline: the file loop deleted with a different key (the temp-dir-
        relative path) than the one `xhtml_translator` saved under (the
        `content_href`), so the delete silently never matched and stale
        "finished" states blocked every retry. Routing every caller through
        this one helper makes that kind of divergence impossible.

        The sanitisation rule (`/` and `\\` replaced with `_`) is kept
        byte-for-byte identical to the original inline version so that state
        files already on disk keep resolving to the same path.

        Does NOT create the parent directory; callers that write must call
        `mkdir(parents=True, exist_ok=True)` on the returned path's parent
        themselves.
        """
        safe_filename = file_href.replace('/', '_').replace('\\', '_')
        return self.uploads_dir / translation_id / "xhtml_states" / f"{safe_filename}.json"

    def save_xhtml_partial_state(
        self,
        translation_id: str,
        file_href: str,
        state: 'XHTMLTranslationState'
    ) -> bool:
        """
        Save partial translation state for an XHTML file (chunk-level checkpoint).

        This enables interruption and resume at the chunk level within a single
        XHTML file, rather than only at the file level.

        Args:
            translation_id: Job identifier
            file_href: Relative path in EPUB (e.g., "OEBPS/chapter1.xhtml")
            state: XHTMLTranslationState instance to save

        Returns:
            True if saved successfully
        """
        from datetime import datetime
        import json

        state_file = self._xhtml_state_path(translation_id, file_href)
        state_file.parent.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        from datetime import timezone
        state.updated_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        try:
            # Serialize and save
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

            _logger.debug(f"Partial state saved: {state_file} (chunk {state.current_chunk_index}/{len(state.chunks)})")

            # Update main checkpoint progress with global_stats if available
            # This ensures the UI shows correct progress across all XHTML files
            if state.global_stats:
                self.db.update_job_progress(
                    translation_id=translation_id,
                    current_chunk_index=None,  # Don't update chunk index
                    total_chunks=state.global_stats.get('total_chunks'),
                    completed_chunks=state.global_stats.get('completed_chunks'),
                    failed_chunks=state.global_stats.get('failed_chunks')
                )
                _logger.debug(f"Updated main checkpoint with global stats: {state.global_stats.get('completed_chunks')}/{state.global_stats.get('total_chunks')} chunks")

            return True
        except Exception as e:
            _logger.error(f"Error saving partial state: {e}")
            return False

    def load_xhtml_partial_state(
        self,
        translation_id: str,
        file_href: str
    ) -> Optional['XHTMLTranslationState']:
        """
        Load partial translation state for an XHTML file.

        Args:
            translation_id: Job identifier
            file_href: Relative path in EPUB (e.g., "OEBPS/chapter1.xhtml")

        Returns:
            XHTMLTranslationState instance or None if not found
        """
        import json
        from src.core.epub.xhtml_translation_state import XHTMLTranslationState

        state_file = self._xhtml_state_path(translation_id, file_href)

        if not state_file.exists():
            return None

        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            state = XHTMLTranslationState.from_dict(data)

            # Validate the loaded state
            if not state.validate():
                _logger.warning(f"Warning: Loaded state is invalid, ignoring: {state_file}")
                return None

            _logger.debug(f"Partial state loaded: {state_file} (resuming from chunk {state.current_chunk_index}/{len(state.chunks)})")
            return state
        except Exception as e:
            _logger.error(f"Error loading partial state: {e}")
            return None

    def delete_xhtml_partial_state(
        self,
        translation_id: str,
        file_href: str
    ) -> bool:
        """
        Delete partial state after successful completion of XHTML file translation.

        Args:
            translation_id: Job identifier
            file_href: Relative path in EPUB

        Returns:
            True if deleted successfully or file didn't exist
        """
        state_file = self._xhtml_state_path(translation_id, file_href)

        if state_file.exists():
            try:
                state_file.unlink()
                return True
            except Exception as e:
                _logger.warning(f"Warning: Could not delete partial state: {e}")
                return False
        return True

    def list_xhtml_partial_states(self, translation_id: str) -> List[str]:
        """
        List all partial states for a translation job.

        Args:
            translation_id: Job identifier

        Returns:
            List of file_href strings that have partial states
        """
        import json

        states_dir = self.uploads_dir / translation_id / "xhtml_states"
        if not states_dir.exists():
            return []

        states = []
        for state_file in states_dir.glob("*.json"):
            # The sanitised filename cannot be reliably reversed (it is lossy
            # for any href containing an underscore), so read the file_href
            # that was saved inside the payload instead. Fall back to the
            # file stem only if the payload is missing or unreadable.
            file_href = state_file.stem
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                file_href = data.get('file_href') or file_href
            except Exception as e:
                _logger.warning(f"Warning: Could not read partial state {state_file} for listing: {e}")
            states.append(file_href)
        return states

    def close(self):
        """Close database connection."""
        self.db.close()
