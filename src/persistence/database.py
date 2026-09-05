"""
SQLite database manager for translation job persistence.
"""

import sqlite3
import json
import os
import time
from typing import Optional, Dict, List, Any
from datetime import datetime
import threading


def sanitize_config_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of a job config with every API key removed.

    The jobs database must never be a secret-bearing artifact (issue #213):
    it lives on disk unencrypted and is bind-mounted by docker-compose. Keys
    are therefore stripped before persistence and re-resolved at resume time
    from the environment (the LLM factory falls back to <PROVIDER>_API_KEY
    when the config value is empty) or from the resume request body.
    """
    return {
        k: v for k, v in config.items()
        if not (k == 'api_key' or k.endswith('_api_key'))
    }


class Database:
    """
    Manages SQLite database for translation job checkpoints.
    Thread-safe for concurrent access.
    """

    def __init__(self, db_path: str = "data/jobs.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.RLock()

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize schema
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            # SQLite scopes foreign_keys per connection, not per database, so it
            # must be enabled here (on every connection we hand out) rather than
            # once in _initialize_schema. Without it, the ON DELETE CASCADE
            # declared on checkpoint_chunks is silently ignored.
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.connection = conn
        return self._local.connection

    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Translation jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS translation_jobs (
                    translation_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    config JSON NOT NULL,
                    progress JSON NOT NULL,
                    translation_context JSON,
                    server_session_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    paused_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)

            # Add server_session_id column if it doesn't exist (migration for existing DBs)
            cursor.execute("PRAGMA table_info(translation_jobs)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'server_session_id' not in columns:
                cursor.execute("ALTER TABLE translation_jobs ADD COLUMN server_session_id TEXT")

            # Checkpoint chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_chunks (
                    translation_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    original_text TEXT NOT NULL,
                    translated_text TEXT,
                    chunk_data JSON,
                    status TEXT NOT NULL,
                    completed_at TIMESTAMP,
                    PRIMARY KEY (translation_id, chunk_index),
                    FOREIGN KEY (translation_id) REFERENCES translation_jobs(translation_id)
                        ON DELETE CASCADE
                )
            """)

            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status
                ON translation_jobs(status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_translation
                ON checkpoint_chunks(translation_id)
            """)

            # Reclaim chunk rows orphaned by deletions performed while foreign
            # key enforcement was off. Idempotent by construction: on a clean
            # database it deletes zero rows, so it can run on every startup and
            # needs no schema-version table.
            cursor.execute("""
                DELETE FROM checkpoint_chunks
                WHERE translation_id NOT IN (SELECT translation_id FROM translation_jobs)
            """)
            if cursor.rowcount > 0:
                print(f"Reclaimed {cursor.rowcount} orphaned checkpoint chunk row(s)")

            conn.commit()

    def create_job(
        self,
        translation_id: str,
        file_type: str,
        config: Dict[str, Any],
        server_session_id: Optional[str] = None
    ) -> bool:
        """
        Create a new translation job record.

        Args:
            translation_id: Unique job identifier
            file_type: Type of file (txt, srt, epub)
            config: Full translation configuration
            server_session_id: Unique identifier for the current server session

        Returns:
            True if created successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                progress = {
                    'current_chunk_index': -1,
                    'total_chunks': 0,
                    'completed_chunks': 0,
                    'failed_chunks': 0,
                    'start_time': time.time(),  # Use timestamp for compatibility with existing code
                    # Marks the uniform checkpoint convention: current_chunk_index
                    # is the LAST COMPLETED unit for every format (resume = +1).
                    # Absent on pre-migration checkpoints, which load_checkpoint
                    # still handles via the legacy per-format branch.
                    'resume_index_semantics': 'completed',
                }

                cursor.execute("""
                    INSERT INTO translation_jobs
                    (translation_id, status, file_type, config, progress, server_session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    translation_id,
                    'running',
                    file_type,
                    json.dumps(sanitize_config_secrets(config)),
                    json.dumps(progress),
                    server_session_id
                ))

                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Job already exists
                return False
            except Exception as e:
                print(f"Error creating job: {e}")
                return False

    def update_job_progress(
        self,
        translation_id: str,
        current_chunk_index: Optional[int] = None,
        total_chunks: Optional[int] = None,
        completed_chunks: Optional[int] = None,
        failed_chunks: Optional[int] = None,
        status: Optional[str] = None,
        epub_accumulated_stats: Optional[Dict[str, Any]] = None,
        unfinished_units: Optional[Dict[str, List[int]]] = None,
        degraded_units: Optional[Dict[str, List[int]]] = None
    ) -> bool:
        """
        Update job progress information.

        Args:
            translation_id: Job identifier
            current_chunk_index: Current chunk being processed
            total_chunks: Total number of chunks
            completed_chunks: Number of completed chunks
            failed_chunks: Number of failed chunks
            status: Job status (running, paused, completed, error)
            epub_accumulated_stats: Snapshot of cross-file accumulated EPUB
                fallback counters. Stored verbatim in the progress JSON so the
                resume path can rehydrate counters that live above the
                per-file checkpoint (token_alignment_used, fallback_used, ...).
            unfinished_units: EPUB index of the chunks still to translate
                ({file_href: [chunk_index, ...]}, issue #261). Stored verbatim
                in the progress JSON under 'epub_unfinished_units'; it is the
                complete current picture, so it replaces the stored value
                instead of merging into it.
            degraded_units: EPUB index of the chunks that were translated but
                whose inline tags were only approximately repositioned by token
                alignment ({file_href: [chunk_index, ...]}). Stored verbatim
                under 'epub_degraded_units', replaced as a whole exactly like
                the map above. Deliberately a SEPARATE map: those chunks are
                translated, they are never part of the automatic work set
                (design decision D3) and must never make a job 'partial'; only
                an explicit user-initiated retry reads this index.

        Returns:
            True if updated successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Get current progress
                cursor.execute(
                    "SELECT progress FROM translation_jobs WHERE translation_id = ?",
                    (translation_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return False

                progress = json.loads(row['progress'])

                # Update fields
                if current_chunk_index is not None:
                    progress['current_chunk_index'] = current_chunk_index
                if total_chunks is not None:
                    progress['total_chunks'] = total_chunks
                if completed_chunks is not None:
                    progress['completed_chunks'] = completed_chunks
                if failed_chunks is not None:
                    progress['failed_chunks'] = failed_chunks
                if epub_accumulated_stats is not None:
                    progress['epub_accumulated_stats'] = epub_accumulated_stats
                if unfinished_units is not None:
                    progress['epub_unfinished_units'] = unfinished_units
                if degraded_units is not None:
                    progress['epub_degraded_units'] = degraded_units

                # Build update query
                updates = ["progress = ?", "updated_at = CURRENT_TIMESTAMP"]
                params = [json.dumps(progress)]

                if status:
                    updates.append("status = ?")
                    params.append(status)

                    if status == 'paused':
                        updates.append("paused_at = CURRENT_TIMESTAMP")
                    elif status == 'completed':
                        updates.append("completed_at = CURRENT_TIMESTAMP")

                params.append(translation_id)

                cursor.execute(
                    f"UPDATE translation_jobs SET {', '.join(updates)} WHERE translation_id = ?",
                    params
                )

                conn.commit()
                return True
            except Exception as e:
                print(f"Error updating job progress: {e}")
                return False

    def save_chunk(
        self,
        translation_id: str,
        chunk_index: int,
        original_text: str,
        translated_text: Optional[str] = None,
        chunk_data: Optional[Dict[str, Any]] = None,
        status: str = 'completed'
    ) -> bool:
        """
        Save a translated chunk to database.

        Args:
            translation_id: Job identifier
            chunk_index: Index of the chunk
            original_text: Original text
            translated_text: Translated text (if completed)
            chunk_data: Additional chunk metadata (context_before, context_after, etc.)
            status: Chunk status (completed, failed)

        Returns:
            True if saved successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO checkpoint_chunks
                    (translation_id, chunk_index, original_text, translated_text,
                     chunk_data, status, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    translation_id,
                    chunk_index,
                    original_text,
                    translated_text,
                    json.dumps(chunk_data) if chunk_data else None,
                    status
                ))

                conn.commit()
                return True
            except Exception as e:
                print(f"Error saving chunk: {e}")
                return False

    def get_job(self, translation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve job information.

        Args:
            translation_id: Job identifier

        Returns:
            Job data dictionary or None if not found
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT * FROM translation_jobs WHERE translation_id = ?",
                    (translation_id,)
                )
                row = cursor.fetchone()

                if not row:
                    return None

                return {
                    'translation_id': row['translation_id'],
                    'status': row['status'],
                    'file_type': row['file_type'],
                    'config': json.loads(row['config']),
                    'progress': json.loads(row['progress']),
                    'translation_context': json.loads(row['translation_context']) if row['translation_context'] else None,
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'paused_at': row['paused_at'],
                    'completed_at': row['completed_at']
                }
            except Exception as e:
                print(f"Error getting job: {e}")
                return None

    def update_job_config(self, translation_id: str, config: Dict[str, Any]) -> bool:
        """
        Update the configuration of an existing job.

        Args:
            translation_id: Job identifier
            config: New configuration dictionary

        Returns:
            True if updated successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    "UPDATE translation_jobs SET config = ?, updated_at = CURRENT_TIMESTAMP WHERE translation_id = ?",
                    (json.dumps(sanitize_config_secrets(config)), translation_id)
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                print(f"Error updating job config: {e}")
                return False

    def get_chunks(self, translation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a job.

        Args:
            translation_id: Job identifier

        Returns:
            List of chunk dictionaries ordered by chunk_index
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM checkpoint_chunks
                    WHERE translation_id = ?
                    ORDER BY chunk_index
                """, (translation_id,))

                chunks = []
                for row in cursor.fetchall():
                    chunks.append({
                        'chunk_index': row['chunk_index'],
                        'original_text': row['original_text'],
                        'translated_text': row['translated_text'],
                        'chunk_data': json.loads(row['chunk_data']) if row['chunk_data'] else None,
                        'status': row['status'],
                        'completed_at': row['completed_at']
                    })

                return chunks
            except Exception as e:
                print(f"Error getting chunks: {e}")
                return []

    def get_failed_chunk_indices(self, translation_id: str) -> List[int]:
        """
        Return the chunk_index of every chunk currently marked 'failed'.

        Used by the resume path to retry chunks that previously errored,
        instead of leaving them as untranslated holes in the output.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT chunk_index FROM checkpoint_chunks
                    WHERE translation_id = ? AND status = 'failed'
                    ORDER BY chunk_index
                """, (translation_id,))
                return [row['chunk_index'] for row in cursor.fetchall()]
            except Exception as e:
                print(f"Error getting failed chunks: {e}")
                return []

    def get_resumable_jobs(self, max_age_days: int = 30) -> List[Dict[str, Any]]:
        """
        Get all jobs that can be resumed (status = paused or interrupted).

        Args:
            max_age_days: Maximum age in days for resumable jobs (default 30)

        Returns:
            List of job dictionaries
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Only return jobs created within max_age_days
                cursor.execute("""
                    SELECT * FROM translation_jobs
                    WHERE status IN ('paused', 'interrupted', 'error', 'partial')
                    AND created_at > datetime('now', ? || ' days')
                    ORDER BY updated_at DESC
                """, (f'-{max_age_days}',))

                jobs = []
                for row in cursor.fetchall():
                    jobs.append({
                        'translation_id': row['translation_id'],
                        'status': row['status'],
                        'file_type': row['file_type'],
                        'config': json.loads(row['config']),
                        'progress': json.loads(row['progress']),
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'paused_at': row['paused_at']
                    })

                return jobs
            except Exception as e:
                print(f"Error getting resumable jobs: {e}")
                return []

    def get_job_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get the most recently completed jobs, newest first.

        Only 'completed' rows are returned: unfinished work already has its own
        listing through get_resumable_jobs, and showing a job in both lists
        would be confusing. The rows are swept by cleanup_old_jobs, so the
        history is implicitly bounded by the same 30-day retention.

        Args:
            limit: Maximum number of rows to return (clamped to 1..200)

        Returns:
            List of job dictionaries (same shape as get_resumable_jobs plus
            'completed_at'), or an empty list on error.
        """
        limit = max(1, min(int(limit), 200))

        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT translation_id, status, file_type, config, progress,
                           created_at, updated_at, completed_at
                    FROM translation_jobs
                    WHERE status = 'completed'
                    ORDER BY COALESCE(completed_at, updated_at) DESC
                    LIMIT ?
                """, (limit,))

                jobs = []
                for row in cursor.fetchall():
                    jobs.append({
                        'translation_id': row['translation_id'],
                        'status': row['status'],
                        'file_type': row['file_type'],
                        'config': json.loads(row['config']),
                        'progress': json.loads(row['progress']),
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'completed_at': row['completed_at']
                    })

                return jobs
            except Exception as e:
                print(f"Error getting job history: {e}")
                return []

    def cleanup_old_jobs(self, max_age_days: int = 30) -> int:
        """
        Delete old jobs that are no longer relevant.

        Removes jobs older than max_age_days that are in resumable states
        (paused, interrupted, error) to prevent database bloat.

        Args:
            max_age_days: Maximum age in days for jobs to keep (default 30)

        Returns:
            Number of jobs deleted
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Get IDs of jobs to delete (for logging and file cleanup)
                cursor.execute("""
                    SELECT translation_id FROM translation_jobs
                    WHERE status IN ('paused', 'interrupted', 'error', 'partial', 'completed')
                    AND created_at <= datetime('now', ? || ' days')
                """, (f'-{max_age_days}',))

                job_ids = [row['translation_id'] for row in cursor.fetchall()]

                if not job_ids:
                    return 0

                # Delete the chunks explicitly (defence in depth: the ON DELETE
                # CASCADE only fires while PRAGMA foreign_keys is on). The
                # placeholder string is built from len(job_ids); the values
                # themselves are always bound as parameters.
                placeholders = ','.join('?' * len(job_ids))
                cursor.execute(
                    f"DELETE FROM checkpoint_chunks WHERE translation_id IN ({placeholders})",
                    job_ids
                )

                # Delete old jobs (chunks already deleted explicitly above)
                cursor.execute("""
                    DELETE FROM translation_jobs
                    WHERE status IN ('paused', 'interrupted', 'error', 'partial', 'completed')
                    AND created_at <= datetime('now', ? || ' days')
                """, (f'-{max_age_days}',))

                deleted_count = cursor.rowcount
                conn.commit()
                return deleted_count
            except Exception as e:
                print(f"Error cleaning up old jobs: {e}")
                return 0

    def reset_running_jobs(self, current_session_id: str) -> int:
        """
        Reset jobs with 'running' status from previous server sessions to 'interrupted'.

        Only resets jobs that have a different server_session_id than the current one,
        preserving jobs that are actually running in the current session.

        This should be called on server startup to handle jobs that were
        interrupted by a server crash or restart.

        Args:
            current_session_id: The current server's session ID

        Returns:
            Number of jobs reset
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Reset jobs that are 'running' but from a different session
                # (or have no session_id, meaning they're from before this feature)
                cursor.execute("""
                    UPDATE translation_jobs
                    SET status = 'interrupted',
                        updated_at = CURRENT_TIMESTAMP,
                        paused_at = CURRENT_TIMESTAMP
                    WHERE status = 'running'
                    AND (server_session_id IS NULL OR server_session_id != ?)
                """, (current_session_id,))

                affected_rows = cursor.rowcount
                conn.commit()
                return affected_rows
            except Exception as e:
                print(f"Error resetting running jobs: {e}")
                return 0

    def update_translation_context(
        self,
        translation_id: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Update translation context for continuity.

        Args:
            translation_id: Job identifier
            context: Context data (last_llm_context, context_accumulator, etc.)

        Returns:
            True if updated successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE translation_jobs
                    SET translation_context = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE translation_id = ?
                """, (json.dumps(context), translation_id))

                conn.commit()
                return True
            except Exception as e:
                print(f"Error updating translation context: {e}")
                return False

    def delete_chunks(self, translation_id: str) -> bool:
        """
        Delete only the checkpoint chunks of a job, keeping its row.

        Used by the success path: a finished job keeps its translation_jobs row
        so it can be listed as history, but its chunks are dead weight once the
        output file has been written.

        Args:
            translation_id: Job identifier

        Returns:
            True if deleted successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    "DELETE FROM checkpoint_chunks WHERE translation_id = ?",
                    (translation_id,)
                )

                conn.commit()
                return True
            except Exception as e:
                print(f"Error deleting chunks: {e}")
                return False

    def delete_job(self, translation_id: str) -> bool:
        """
        Delete a job and all its chunks (CASCADE).

        Args:
            translation_id: Job identifier

        Returns:
            True if deleted successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Defence in depth: the ON DELETE CASCADE below only fires while
                # PRAGMA foreign_keys is on for this connection.
                cursor.execute(
                    "DELETE FROM checkpoint_chunks WHERE translation_id = ?",
                    (translation_id,)
                )

                cursor.execute(
                    "DELETE FROM translation_jobs WHERE translation_id = ?",
                    (translation_id,)
                )

                conn.commit()
                return True
            except Exception as e:
                print(f"Error deleting job: {e}")
                return False

    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
