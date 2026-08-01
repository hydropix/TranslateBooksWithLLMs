"""
Checkpoint manager for translation job persistence and resume functionality.
"""

import os
import shutil
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from .database import Database


class CheckpointManager:
    """
    Manages translation job checkpoints including database persistence
    and file storage for uploaded files.
    """

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
            translation_id, file_type, config, self.server_session_id
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
            print(f"Warning: Input file does not exist: {input_file_path}")
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
            print(f"Input file preserved: {preserved_path}")
        except Exception as e:
            print(f"Warning: Could not preserve input file: {e}")

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
        epub_accumulated_stats: Optional[Dict[str, Any]] = None
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
            chunk_status
        )

        # Update job progress
        progress_saved = self.db.update_job_progress(
            translation_id,
            current_chunk_index=chunk_index,
            total_chunks=total_chunks,
            completed_chunks=completed_chunks,
            failed_chunks=failed_chunks,
            epub_accumulated_stats=epub_accumulated_stats
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
            'translation_context': job.get('translation_context')
        }

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
            print(f"Warning: Error getting old job list: {e}")

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
                    print(f"Warning: Could not delete upload directory for {job_id}: {e}")

        return jobs_deleted, files_cleaned

    def cleanup_orphan_uploads(self) -> int:
        """
        Clean up upload files/directories that don't have corresponding jobs in the database.

        These are "orphan" items left behind from previous incomplete cleanups.
        Handles:
        - trans_xxx folders (job ID folders)
        - hash_filename files (legacy upload files)

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
            print(f"Warning: Error getting job IDs: {e}")
            return 0

        # Check each item in uploads directory
        for item in self.uploads_dir.iterdir():
            item_name = item.name

            # Skip test folders
            if item_name.startswith('test_'):
                continue

            is_orphan = True

            if item.is_dir():
                # It's a folder - check if it's a job ID folder
                if item_name.startswith('trans_'):
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
                    print(f"Warning: Could not delete orphan {item_name}: {e}")

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
                print(f"Warning: Could not delete upload directory: {e}")

        return db_deleted

    def cleanup_completed_job(self, translation_id: str) -> bool:
        """
        Automatically clean up a completed job (immediate cleanup).

        Args:
            translation_id: Job identifier

        Returns:
            True if cleaned up successfully
        """
        return self.delete_checkpoint(translation_id)

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

    def build_translated_output(
        self,
        translation_id: str,
        file_type: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Build the complete translated output from saved chunks.

        Now uses the adapter pattern for all file formats, providing
        consistent reconstruction logic across TXT, SRT, and EPUB.

        Args:
            translation_id: Job identifier
            file_type: Type of file (txt, srt, epub)

        Returns:
            Tuple of (translated_text, error_message)
        """
        # Use the new adapter-based reconstruction
        import asyncio
        from src.core.adapters import build_translated_output as adapter_build_output

        try:
            output_bytes, error = asyncio.run(
                adapter_build_output(
                    translation_id=translation_id,
                    checkpoint_manager=self
                )
            )

            if error:
                return None, error

            if output_bytes:
                # For EPUB, return as base64-encoded string for consistency with legacy code
                if file_type == 'epub':
                    import base64
                    return base64.b64encode(output_bytes).decode('utf-8'), None
                else:
                    # For TXT/SRT, decode bytes to string
                    return output_bytes.decode('utf-8'), None

            return None, "No output generated"

        except Exception as e:
            # Fallback to legacy reconstruction if adapter fails
            return self._build_translated_output_legacy(translation_id, file_type)

    def _build_translated_output_legacy(
        self,
        translation_id: str,
        file_type: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Legacy build method - kept as fallback.

        This is the original implementation, preserved for backward compatibility
        in case the adapter-based reconstruction fails.

        Args:
            translation_id: Job identifier
            file_type: Type of file (txt, srt, epub)

        Returns:
            Tuple of (translated_text, error_message)
        """
        chunks = self.db.get_chunks(translation_id)

        if not chunks:
            return None, "No chunks found for this job"

        if file_type in ['txt', 'epub_simple']:
            # Simple concatenation for text-based formats
            translated_parts = []
            for chunk in chunks:
                if chunk['status'] == 'completed' and chunk['translated_text']:
                    translated_parts.append(chunk['translated_text'])
                else:
                    # Use original text if translation failed
                    translated_parts.append(chunk['original_text'])

            return '\n'.join(translated_parts), None

        elif file_type == 'srt':
            # SRT needs special handling to reconstruct from blocks
            # Each chunk contains block_translations dict mapping subtitle index to translated text
            job = self.db.get_job(translation_id)
            if not job:
                return None, "Job not found"

            # Build a complete translations dictionary from all blocks
            all_translations = {}
            for chunk in chunks:
                block_translations = chunk.get('chunk_data', {}).get('block_translations', {})
                for idx_str, trans_text in block_translations.items():
                    idx = int(idx_str)
                    all_translations[idx] = trans_text

            if not all_translations:
                return None, "No translations found in checkpoint"

            # Now we need to reconstruct the SRT file
            # We need the original subtitle structure (timing, numbering)
            # This should be stored in the config or we need to re-parse the original file
            config = job['config']
            preserved_input_path = config.get('preserved_input_path')

            if not preserved_input_path or not Path(preserved_input_path).exists():
                return None, "Original SRT file not found, cannot reconstruct"

            # Re-parse the original SRT to get structure
            from src.core.srt_processor import SRTProcessor
            srt_processor = SRTProcessor()

            with open(preserved_input_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            subtitles = srt_processor.parse_srt(original_content)

            # Update subtitles with translations
            updated_subtitles = srt_processor.update_translated_subtitles(
                subtitles, all_translations
            )

            # Reconstruct SRT
            translated_srt = srt_processor.reconstruct_srt(updated_subtitles)

            return translated_srt, None

        elif file_type == 'epub':
            # EPUB reconstruction from checkpoint
            # Extract original EPUB, restore translated files, and repackage
            job = self.db.get_job(translation_id)
            if not job:
                return None, "Job not found"

            config = job['config']
            preserved_input_path = config.get('preserved_input_path')

            if not preserved_input_path or not Path(preserved_input_path).exists():
                return None, "Original EPUB file not found, cannot reconstruct"

            try:
                import tempfile
                import zipfile
                from lxml import etree

                # Create temporary directory for reconstruction
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    # Extract original EPUB
                    with zipfile.ZipFile(preserved_input_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_path)

                    # Restore translated files from checkpoint
                    restore_success = self.restore_epub_files(translation_id, temp_path)

                    if not restore_success:
                        return None, "Failed to restore translated files from checkpoint"

                    # Repackage EPUB
                    output_path = Path(tempfile.mktemp(suffix='.epub'))
                    try:
                        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as epub_zip:
                            # Add mimetype first (uncompressed)
                            mimetype_path = temp_path / 'mimetype'
                            if mimetype_path.exists():
                                epub_zip.write(
                                    mimetype_path,
                                    'mimetype',
                                    compress_type=zipfile.ZIP_STORED
                                )

                            # Add all other files
                            for file_path in temp_path.rglob('*'):
                                if file_path.is_file() and file_path.name != 'mimetype':
                                    arcname = file_path.relative_to(temp_path)
                                    epub_zip.write(file_path, arcname)

                        # Read as string (will be written as binary by caller)
                        with open(output_path, 'rb') as f:
                            epub_bytes = f.read()

                        # Return as base64-encoded string for storage consistency
                        import base64
                        return base64.b64encode(epub_bytes).decode('utf-8'), None

                    finally:
                        if output_path.exists():
                            output_path.unlink()

            except Exception as e:
                return None, f"Error reconstructing EPUB: {str(e)}"

        else:
            return None, f"Unknown file type: {file_type}"

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

        # Preserve directory structure
        file_path = job_dir / file_href
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'wb') as f:
                f.write(file_content)
            return True
        except Exception as e:
            print(f"Error saving EPUB file {file_href}: {e}")
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
                    dest_path = work_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_path)
            return True
        except Exception as e:
            print(f"Error restoring EPUB files: {e}")
            return False

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

        # Create states directory
        states_dir = self.uploads_dir / translation_id / "xhtml_states"
        states_dir.mkdir(parents=True, exist_ok=True)

        # Generate safe filename (replace / and \ with _)
        safe_filename = file_href.replace('/', '_').replace('\\', '_')
        state_file = states_dir / f"{safe_filename}.json"

        # Update timestamp
        from datetime import timezone
        state.updated_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        try:
            # Serialize and save
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

            print(f"Partial state saved: {state_file} (chunk {state.current_chunk_index}/{len(state.chunks)})")

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
                print(f"Updated main checkpoint with global stats: {state.global_stats.get('completed_chunks')}/{state.global_stats.get('total_chunks')} chunks")

            return True
        except Exception as e:
            print(f"Error saving partial state: {e}")
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

        states_dir = self.uploads_dir / translation_id / "xhtml_states"
        safe_filename = file_href.replace('/', '_').replace('\\', '_')
        state_file = states_dir / f"{safe_filename}.json"

        if not state_file.exists():
            return None

        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            state = XHTMLTranslationState.from_dict(data)

            # Validate the loaded state
            if not state.validate():
                print(f"Warning: Loaded state is invalid, ignoring: {state_file}")
                return None

            print(f"Partial state loaded: {state_file} (resuming from chunk {state.current_chunk_index}/{len(state.chunks)})")
            return state
        except Exception as e:
            print(f"Error loading partial state: {e}")
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
        states_dir = self.uploads_dir / translation_id / "xhtml_states"
        safe_filename = file_href.replace('/', '_').replace('\\', '_')
        state_file = states_dir / f"{safe_filename}.json"

        if state_file.exists():
            try:
                state_file.unlink()
                return True
            except Exception as e:
                print(f"Warning: Could not delete partial state: {e}")
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
        states_dir = self.uploads_dir / translation_id / "xhtml_states"
        if not states_dir.exists():
            return []

        states = []
        for state_file in states_dir.glob("*.json"):
            # Reconstruct original file_href (reverse the safe filename transformation)
            file_href = state_file.stem.replace('_', '/')
            states.append(file_href)
        return states

    def close(self):
        """Close database connection."""
        self.db.close()
