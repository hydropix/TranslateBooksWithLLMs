"""
SQLite CRUD layer for glossaries and glossary terms.

The glossary lives in its own SQLite file (``data/glossaries.db``), separate
from ``data/jobs.db`` (translation checkpoints) to avoid write-lock contention
between the two domains. A one-shot migration on init copies any pre-existing
glossary tables from ``jobs.db`` if present.
"""

import logging
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple

from src.core.glossary.models import (
    BulkReplaceResult,
    Glossary,
    GlossaryTerm,
    normalize_gender,
)

logger = logging.getLogger("glossary.store")

LEGACY_DB_PATH = "data/jobs.db"
DEFAULT_DB_PATH = "data/glossaries.db"

# Backoff retry on transient SQLite "database is locked" errors. Total budget
# is short (~750ms across 4 attempts).
_LOCK_RETRY_ATTEMPTS = 4
_LOCK_RETRY_BASE_DELAY = 0.05


def _parse_timestamp(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


def _row_to_term(row: sqlite3.Row) -> GlossaryTerm:
    return GlossaryTerm(
        id=row["id"],
        source_term=row["source_term"],
        translated_term=row["translated_term"],
        category=row["category"],
        gender=_row_value(row, "gender"),
    )


def _row_value(row: sqlite3.Row, column: str):
    """Read an optional column, tolerating rows selected before it existed."""
    try:
        return row[column]
    except (IndexError, KeyError):
        return None


def _row_to_glossary(row: sqlite3.Row, terms: Optional[List[GlossaryTerm]] = None) -> Glossary:
    return Glossary(
        id=row["id"],
        name=row["name"],
        source_language=row["source_language"] or "",
        target_language=row["target_language"] or "",
        terms=terms if terms is not None else [],
        created_at=_parse_timestamp(row["created_at"]),
        updated_at=_parse_timestamp(row["updated_at"]),
    )


def _is_lock_error(exc: BaseException) -> bool:
    """True when an exception is SQLite's transient 'database is locked'."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return "database is locked" in msg or "database table is locked" in msg


class GlossaryStore:
    """Thread-safe SQLite-backed store for glossaries and their terms."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize the store and ensure schema exists."""
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.RLock()
        # _local.connection is per-thread and unreachable from other threads;
        # this list lets close_all() drop every connection on shutdown.
        self._all_connections: list = []
        self._connections_lock = threading.RLock()

        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self._initialize_schema()
        self._migrate_from_legacy_db_if_needed()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.connection = conn
            with self._connections_lock:
                self._all_connections.append(conn)
            try:
                self._self_heal_legacy_schema(conn)
            except Exception as e:
                logger.warning(f"Schema self-heal skipped on new connection: {e}")
        return self._local.connection

    def _run_write(self, fn, *args, **kwargs):
        """Run a write callable, retrying briefly on transient SQLite locks."""
        delay = _LOCK_RETRY_BASE_DELAY
        for attempt in range(_LOCK_RETRY_ATTEMPTS):
            try:
                return fn(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                if not _is_lock_error(exc) or attempt == _LOCK_RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    def _self_heal_legacy_schema(self, conn: sqlite3.Connection) -> None:
        """Inspect the connection's view of the schema and drop any legacy
        columns (e.g. `notes`) we don't want around. Idempotent."""
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(glossaries)")
        gloss_cols = [row[1] for row in cursor.fetchall()]
        if gloss_cols:
            self._migrate_drop_glossaries_legacy_columns(cursor)
        cursor.execute("PRAGMA table_info(glossary_terms)")
        term_cols = [row[1] for row in cursor.fetchall()]
        if term_cols:
            self._migrate_drop_notes_column(cursor)
            self._migrate_add_gender_column(cursor)
        conn.commit()

    def _initialize_schema(self):
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS glossaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    source_language TEXT NOT NULL DEFAULT '',
                    target_language TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS glossary_terms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    glossary_id INTEGER NOT NULL,
                    source_term TEXT NOT NULL,
                    translated_term TEXT NOT NULL,
                    category TEXT,
                    gender TEXT,
                    FOREIGN KEY (glossary_id) REFERENCES glossaries(id) ON DELETE CASCADE,
                    UNIQUE (glossary_id, source_term)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_glossary_terms_glossary
                ON glossary_terms(glossary_id)
            """)

            self._migrate_drop_notes_column(cursor)
            self._migrate_add_gender_column(cursor)
            self._migrate_drop_glossaries_legacy_columns(cursor)

            conn.commit()

    def _migrate_from_legacy_db_if_needed(self) -> None:
        """Copy glossary tables from data/jobs.db on first start (idempotent).

        Only runs against the default DB path so tests passing a custom path
        stay isolated. A `_migration_state` flag prevents re-importing legacy
        rows after the user has deleted them — without it, deleting the last
        glossary would resurrect everything on the next restart.
        """
        if os.path.abspath(self.db_path) != os.path.abspath(DEFAULT_DB_PATH):
            return

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS _migration_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            cursor.execute(
                "SELECT 1 FROM _migration_state WHERE key = 'legacy_migration_done'"
            )
            if cursor.fetchone():
                return

            cursor.execute("SELECT COUNT(*) FROM glossaries")
            if (cursor.fetchone()[0] or 0) > 0:
                cursor.execute(
                    "INSERT OR REPLACE INTO _migration_state (key, value) VALUES (?, ?)",
                    ("legacy_migration_done", "1"),
                )
                conn.commit()
                return

            if not os.path.exists(LEGACY_DB_PATH):
                cursor.execute(
                    "INSERT OR REPLACE INTO _migration_state (key, value) VALUES (?, ?)",
                    ("legacy_migration_done", "1"),
                )
                conn.commit()
                return

            try:
                legacy = sqlite3.connect(LEGACY_DB_PATH, timeout=10.0)
                legacy.row_factory = sqlite3.Row
                try:
                    legacy_cur = legacy.cursor()
                    legacy_cur.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='glossaries'"
                    )
                    if legacy_cur.fetchone() is None:
                        cursor.execute(
                            "INSERT OR REPLACE INTO _migration_state (key, value) VALUES (?, ?)",
                            ("legacy_migration_done", "1"),
                        )
                        conn.commit()
                        return
                    legacy_cur.execute(
                        "SELECT id, name, source_language, target_language, "
                        "       created_at, updated_at FROM glossaries"
                    )
                    glossary_rows = legacy_cur.fetchall()
                    if not glossary_rows:
                        cursor.execute(
                            "INSERT OR REPLACE INTO _migration_state (key, value) VALUES (?, ?)",
                            ("legacy_migration_done", "1"),
                        )
                        conn.commit()
                        return

                    legacy_cur.execute(
                        "SELECT id, glossary_id, source_term, translated_term, category "
                        "FROM glossary_terms"
                    )
                    term_rows = legacy_cur.fetchall()
                finally:
                    legacy.close()
            except sqlite3.DatabaseError as exc:
                logger.warning(
                    "Could not read legacy glossary tables from %s: %s",
                    LEGACY_DB_PATH, exc,
                )
                return

            try:
                cursor.execute("BEGIN")
                for row in glossary_rows:
                    cursor.execute(
                        "INSERT INTO glossaries "
                        "(id, name, source_language, target_language, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            row["id"],
                            row["name"],
                            row["source_language"] or "",
                            row["target_language"] or "",
                            row["created_at"],
                            row["updated_at"],
                        ),
                    )
                for row in term_rows:
                    cursor.execute(
                        "INSERT INTO glossary_terms "
                        "(id, glossary_id, source_term, translated_term, category) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            row["id"],
                            row["glossary_id"],
                            row["source_term"],
                            row["translated_term"],
                            row["category"],
                        ),
                    )
                cursor.execute(
                    "INSERT OR REPLACE INTO _migration_state (key, value) VALUES (?, ?)",
                    ("legacy_migration_done", "1"),
                )
                conn.commit()
                logger.info(
                    "Migrated %d glossaries (%d terms) from %s to %s",
                    len(glossary_rows), len(term_rows), LEGACY_DB_PATH, self.db_path,
                )
            except Exception:
                conn.rollback()
                raise

    def _migrate_drop_notes_column(self, cursor: sqlite3.Cursor):
        """Drop the legacy `notes` column from glossary_terms if present."""
        cursor.execute("PRAGMA table_info(glossary_terms)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "notes" not in columns:
            return
        try:
            cursor.execute("ALTER TABLE glossary_terms DROP COLUMN notes")
        except sqlite3.OperationalError:
            # Old SQLite without DROP COLUMN support — full rebuild. `gender`
            # is carried over when the source table already has it, so a
            # rebuild never silently discards the genders a user has entered.
            has_gender = "gender" in columns
            gender_select = "gender" if has_gender else "NULL"
            cursor.execute("""
                CREATE TABLE glossary_terms_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    glossary_id INTEGER NOT NULL,
                    source_term TEXT NOT NULL,
                    translated_term TEXT NOT NULL,
                    category TEXT,
                    gender TEXT,
                    FOREIGN KEY (glossary_id) REFERENCES glossaries(id) ON DELETE CASCADE,
                    UNIQUE (glossary_id, source_term)
                )
            """)
            cursor.execute(f"""
                INSERT INTO glossary_terms_new
                    (id, glossary_id, source_term, translated_term, category, gender)
                SELECT id, glossary_id, source_term, translated_term, category, {gender_select}
                FROM glossary_terms
            """)
            cursor.execute("DROP TABLE glossary_terms")
            cursor.execute("ALTER TABLE glossary_terms_new RENAME TO glossary_terms")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_glossary_terms_glossary
                ON glossary_terms(glossary_id)
            """)

    def _migrate_add_gender_column(self, cursor: sqlite3.Cursor):
        """Add the `gender` column to glossary_terms if missing. Idempotent.

        Additive migration: existing rows get NULL, which means "no gender
        information" and keeps the injected prompt byte-identical for every
        glossary predating this column.
        """
        cursor.execute("PRAGMA table_info(glossary_terms)")
        columns = [row["name"] for row in cursor.fetchall()]
        if not columns or "gender" in columns:
            return
        try:
            cursor.execute("ALTER TABLE glossary_terms ADD COLUMN gender TEXT")
            logger.info("Added `gender` column to glossary_terms")
        except sqlite3.OperationalError as exc:
            logger.warning(f"Could not add `gender` column to glossary_terms: {exc}")

    def _migrate_drop_glossaries_legacy_columns(self, cursor: sqlite3.Cursor):
        """Drop legacy columns from `glossaries` if present. Idempotent."""
        cursor.execute("PRAGMA table_info(glossaries)")
        columns = [row["name"] for row in cursor.fetchall()]
        canonical = {"id", "name", "source_language", "target_language", "created_at", "updated_at"}
        extras = [c for c in columns if c not in canonical]
        if not extras:
            return

        logger.info(f"Healing legacy columns on `glossaries`: dropping {extras}")

        for col in extras:
            try:
                cursor.execute(f"ALTER TABLE glossaries DROP COLUMN {col}")
            except sqlite3.OperationalError:
                # Old SQLite without DROP COLUMN support — full rebuild.
                cursor.execute("""
                    CREATE TABLE glossaries_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        source_language TEXT NOT NULL DEFAULT '',
                        target_language TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    INSERT INTO glossaries_new
                        (id, name, source_language, target_language, created_at, updated_at)
                    SELECT id, name,
                           COALESCE(source_language, ''),
                           COALESCE(target_language, ''),
                           created_at, updated_at
                    FROM glossaries
                """)
                cursor.execute("DROP TABLE glossaries")
                cursor.execute("ALTER TABLE glossaries_new RENAME TO glossaries")
                return

    def _fetch_terms(self, cursor: sqlite3.Cursor, glossary_id: int) -> List[GlossaryTerm]:
        cursor.execute(
            "SELECT id, source_term, translated_term, category, gender "
            "FROM glossary_terms WHERE glossary_id = ? ORDER BY id",
            (glossary_id,),
        )
        return [_row_to_term(row) for row in cursor.fetchall()]

    def create_glossary(
        self,
        name: str,
        source_language: str = "",
        target_language: str = "",
    ) -> Glossary:
        """Create a new glossary and return it. Raises ValueError if name exists."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO glossaries (name, source_language, target_language) "
                    "VALUES (?, ?, ?)",
                    (name, source_language or "", target_language or ""),
                )
            except sqlite3.IntegrityError as exc:
                # Close the implicit transaction the failed write opened: the
                # connection is long-lived and per-thread, and an open write
                # transaction would block every other writer.
                conn.rollback()
                raise ValueError(f"Glossary name already exists: {name}") from exc

            new_id = cursor.lastrowid
            cursor.execute(
                "SELECT id, name, source_language, target_language, created_at, updated_at "
                "FROM glossaries WHERE id = ?",
                (new_id,),
            )
            row = cursor.fetchone()
            conn.commit()
            return _row_to_glossary(row, terms=[])

    def get_glossary(self, glossary_id: int) -> Optional[Glossary]:
        """Return the glossary with its terms loaded, or None if missing."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, source_language, target_language, created_at, updated_at "
                "FROM glossaries WHERE id = ?",
                (glossary_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            terms = self._fetch_terms(cursor, glossary_id)
            return _row_to_glossary(row, terms=terms)

    def get_glossary_by_name(self, name: str) -> Optional[Glossary]:
        """Return the glossary with its terms by unique name, or None."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, source_language, target_language, created_at, updated_at "
                "FROM glossaries WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            terms = self._fetch_terms(cursor, row["id"])
            return _row_to_glossary(row, terms=terms)

    def list_glossaries(self) -> List[Glossary]:
        """Return all glossaries without their terms (for index views)."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, source_language, target_language, created_at, updated_at "
                "FROM glossaries ORDER BY name"
            )
            return [_row_to_glossary(row, terms=[]) for row in cursor.fetchall()]

    def list_glossaries_with_counts(self) -> List[tuple]:
        """Return [(Glossary, term_count), ...] for index views."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT g.id, g.name, g.source_language, g.target_language, "
                "       g.created_at, g.updated_at, "
                "       COALESCE(COUNT(t.id), 0) AS term_count "
                "FROM glossaries g "
                "LEFT JOIN glossary_terms t ON t.glossary_id = g.id "
                "GROUP BY g.id "
                "ORDER BY g.name"
            )
            results = []
            for row in cursor.fetchall():
                glossary = _row_to_glossary(row, terms=[])
                results.append((glossary, int(row["term_count"])))
            return results

    def update_glossary(
        self,
        glossary_id: int,
        name: Optional[str] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> Optional[Glossary]:
        """Patch provided fields on a glossary and return the updated record."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM glossaries WHERE id = ?", (glossary_id,))
            if not cursor.fetchone():
                return None

            assignments = []
            params: List = []
            if name is not None:
                assignments.append("name = ?")
                params.append(name)
            if source_language is not None:
                assignments.append("source_language = ?")
                params.append(source_language)
            if target_language is not None:
                assignments.append("target_language = ?")
                params.append(target_language)

            assignments.append("updated_at = CURRENT_TIMESTAMP")
            params.append(glossary_id)

            try:
                cursor.execute(
                    f"UPDATE glossaries SET {', '.join(assignments)} WHERE id = ?",
                    params,
                )
            except sqlite3.IntegrityError as exc:
                # Close the implicit transaction the failed write opened: the
                # connection is long-lived and per-thread, and an open write
                # transaction would block every other writer.
                conn.rollback()
                raise ValueError(f"Glossary name already exists: {name}") from exc

            conn.commit()
            return self.get_glossary(glossary_id)

    def delete_glossary(self, glossary_id: int) -> bool:
        """Delete a glossary (and its terms via CASCADE). Returns True if deleted."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM glossaries WHERE id = ?", (glossary_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def add_term(self, glossary_id: int, term: GlossaryTerm) -> GlossaryTerm:
        """Insert a term for the given glossary and return it with id populated."""
        return self._run_write(self._add_term_impl, glossary_id, term)

    def _add_term_impl(self, glossary_id: int, term: GlossaryTerm) -> GlossaryTerm:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO glossary_terms "
                    "(glossary_id, source_term, translated_term, category, gender) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        glossary_id,
                        term.source_term,
                        term.translated_term,
                        term.category,
                        term.gender,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                # Close the implicit transaction the failed write opened: the
                # connection is long-lived and per-thread, and an open write
                # transaction would block every other writer.
                conn.rollback()
                raise ValueError(
                    f"Term already exists in glossary {glossary_id}: {term.source_term}"
                ) from exc

            new_id = cursor.lastrowid
            cursor.execute(
                "UPDATE glossaries SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (glossary_id,),
            )
            conn.commit()
            return GlossaryTerm(
                id=new_id,
                source_term=term.source_term,
                translated_term=term.translated_term,
                category=term.category,
                gender=term.gender,
            )

    def bulk_add_terms(
        self, glossary_id: int, terms: List[GlossaryTerm]
    ) -> Tuple[int, int, int]:
        """Insert many terms in one transaction. Returns ``(added, conflicts, skipped_empty)``.

        Conflicts (existing source_term, UNIQUE violation) and empty sources
        are skipped without aborting the batch.
        """
        return self._run_write(self._bulk_add_terms_impl, glossary_id, terms)

    def _bulk_add_terms_impl(
        self, glossary_id: int, terms: List[GlossaryTerm]
    ) -> Tuple[int, int, int]:
        added = 0
        conflicts = 0
        skipped_empty = 0
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM glossaries WHERE id = ?", (glossary_id,))
            if not cursor.fetchone():
                raise ValueError(f"Glossary {glossary_id} not found")

            try:
                cursor.execute("BEGIN")
                seen: set[str] = set()
                for term in terms:
                    source = (term.source_term or "").strip()
                    if not source:
                        skipped_empty += 1
                        continue
                    if source in seen:
                        conflicts += 1
                        continue
                    seen.add(source)
                    try:
                        cursor.execute(
                            "INSERT INTO glossary_terms "
                            "(glossary_id, source_term, translated_term, category, gender) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (
                                glossary_id,
                                source,
                                term.translated_term,
                                term.category,
                                term.gender,
                            ),
                        )
                        added += 1
                    except sqlite3.IntegrityError:
                        conflicts += 1

                if added:
                    cursor.execute(
                        "UPDATE glossaries SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (glossary_id,),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return added, conflicts, skipped_empty

    def update_term(
        self,
        term_id: int,
        source_term: Optional[str] = None,
        translated_term: Optional[str] = None,
        category: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> Optional[GlossaryTerm]:
        """Patch provided fields on a term and return the updated record.

        ``gender`` follows the same convention as the other fields: ``None``
        means "leave unchanged". Pass the empty string to clear a gender back
        to unknown.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id, glossary_id, source_term, translated_term, category, gender "
                "FROM glossary_terms WHERE id = ?",
                (term_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            assignments = []
            params: List = []
            if source_term is not None:
                assignments.append("source_term = ?")
                params.append(source_term)
            if translated_term is not None:
                assignments.append("translated_term = ?")
                params.append(translated_term)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if gender is not None:
                assignments.append("gender = ?")
                params.append(normalize_gender(gender))

            if assignments:
                params.append(term_id)
                try:
                    cursor.execute(
                        f"UPDATE glossary_terms SET {', '.join(assignments)} WHERE id = ?",
                        params,
                    )
                except sqlite3.IntegrityError as exc:
                    # Close the implicit transaction the failed write opened: the
                    # connection is long-lived and per-thread, and an open write
                    # transaction would block every other writer.
                    conn.rollback()
                    raise ValueError(
                        f"Term already exists in glossary {row['glossary_id']}: {source_term}"
                    ) from exc

                cursor.execute(
                    "UPDATE glossaries SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (row["glossary_id"],),
                )
                conn.commit()

            cursor.execute(
                "SELECT id, source_term, translated_term, category, gender "
                "FROM glossary_terms WHERE id = ?",
                (term_id,),
            )
            updated_row = cursor.fetchone()
            return _row_to_term(updated_row) if updated_row else None

    def delete_term(self, term_id: int) -> bool:
        """Delete a term by id. Returns True if a row was removed."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT glossary_id FROM glossary_terms WHERE id = ?",
                (term_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False
            glossary_id = row["glossary_id"]

            cursor.execute("DELETE FROM glossary_terms WHERE id = ?", (term_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                cursor.execute(
                    "UPDATE glossaries SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (glossary_id,),
                )
            conn.commit()
            return deleted

    def duplicate_glossary(
        self,
        glossary_id: int,
        new_name: Optional[str] = None,
    ) -> Optional[Glossary]:
        """Clone a glossary (with all its terms) under a new unique name.

        If ``new_name`` is omitted, append " (copy)" — and an integer suffix if
        a "(copy)" name already exists. Returns None if the source glossary
        does not exist.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id, name, source_language, target_language "
                "FROM glossaries WHERE id = ?",
                (glossary_id,),
            )
            src = cursor.fetchone()
            if not src:
                return None

            cursor.execute("SELECT name FROM glossaries")
            existing = {row["name"] for row in cursor.fetchall()}

            base = (new_name or f"{src['name']} (copy)").strip() or f"{src['name']} (copy)"
            candidate = base
            i = 2
            while candidate in existing:
                candidate = f"{base} {i}"
                i += 1
                if i > 1000:
                    candidate = f"{base} {int(datetime.now().timestamp())}"
                    break

            try:
                cursor.execute("BEGIN")
                cursor.execute(
                    "INSERT INTO glossaries (name, source_language, target_language) "
                    "VALUES (?, ?, ?)",
                    (candidate, src["source_language"] or "", src["target_language"] or ""),
                )
                new_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO glossary_terms "
                    "(glossary_id, source_term, translated_term, category, gender) "
                    "SELECT ?, source_term, translated_term, category, gender "
                    "FROM glossary_terms WHERE glossary_id = ?",
                    (new_id, glossary_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

            return self.get_glossary(new_id)

    def bulk_delete_terms(self, glossary_id: int, term_ids: List[int]) -> int:
        """Delete several terms from one glossary. Returns the number deleted."""
        if not term_ids:
            return 0
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in term_ids)
            params = [glossary_id, *term_ids]
            cursor.execute(
                f"DELETE FROM glossary_terms "
                f"WHERE glossary_id = ? AND id IN ({placeholders})",
                params,
            )
            deleted = cursor.rowcount
            if deleted:
                cursor.execute(
                    "UPDATE glossaries SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (glossary_id,),
                )
            conn.commit()
            return int(deleted or 0)

    def bulk_set_category(
        self,
        glossary_id: int,
        term_ids: List[int],
        category: Optional[str],
    ) -> int:
        """Set the category for several terms at once. Returns the number updated."""
        if not term_ids:
            return 0
        category_value = (category or "").strip() or None
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in term_ids)
            params = [category_value, glossary_id, *term_ids]
            cursor.execute(
                f"UPDATE glossary_terms SET category = ? "
                f"WHERE glossary_id = ? AND id IN ({placeholders})",
                params,
            )
            updated = cursor.rowcount
            if updated:
                cursor.execute(
                    "UPDATE glossaries SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (glossary_id,),
                )
            conn.commit()
            return int(updated or 0)

    def bulk_set_gender(
        self,
        glossary_id: int,
        term_ids: List[int],
        gender: Optional[str],
    ) -> int:
        """Set the gender for several terms at once. Returns the number updated.

        An unrecognized or empty ``gender`` clears the field back to NULL,
        which reads as "unknown" everywhere downstream.
        """
        if not term_ids:
            return 0
        gender_value = normalize_gender(gender)
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in term_ids)
            params = [gender_value, glossary_id, *term_ids]
            cursor.execute(
                f"UPDATE glossary_terms SET gender = ? "
                f"WHERE glossary_id = ? AND id IN ({placeholders})",
                params,
            )
            updated = cursor.rowcount
            if updated:
                cursor.execute(
                    "UPDATE glossaries SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (glossary_id,),
                )
            conn.commit()
            return int(updated or 0)

    def bulk_replace_terms(
        self, glossary_id: int, terms: List[GlossaryTerm]
    ) -> BulkReplaceResult:
        """Atomically replace all terms for a glossary.

        Returns a BulkReplaceResult breakdown so callers can report not just
        how many rows were inserted, but also how many were dropped because
        their source was empty or because an earlier row in the same batch
        already used that source.
        """
        total_input = len(terms)

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM glossaries WHERE id = ?", (glossary_id,))
            if not cursor.fetchone():
                return BulkReplaceResult(total_input=total_input)

            try:
                cursor.execute("BEGIN")
                cursor.execute(
                    "DELETE FROM glossary_terms WHERE glossary_id = ?",
                    (glossary_id,),
                )

                inserted = 0
                skipped_empty = 0
                skipped_duplicate = 0
                seen_sources = set()
                for term in terms:
                    source = (term.source_term or "").strip()
                    if not source:
                        skipped_empty += 1
                        continue
                    if source in seen_sources:
                        skipped_duplicate += 1
                        continue
                    seen_sources.add(source)
                    cursor.execute(
                        "INSERT INTO glossary_terms "
                        "(glossary_id, source_term, translated_term, category, gender) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            glossary_id,
                            source,
                            term.translated_term,
                            term.category,
                            term.gender,
                        ),
                    )
                    inserted += 1

                cursor.execute(
                    "UPDATE glossaries SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (glossary_id,),
                )
                conn.commit()
                return BulkReplaceResult(
                    inserted=inserted,
                    skipped_empty=skipped_empty,
                    skipped_duplicate=skipped_duplicate,
                    total_input=total_input,
                )
            except Exception:
                conn.rollback()
                raise

    def close(self):
        """Close this thread's database connection if open."""
        if hasattr(self._local, "connection") and self._local.connection:
            conn = self._local.connection
            self._local.connection = None
            with self._connections_lock:
                try:
                    self._all_connections.remove(conn)
                except ValueError:
                    pass
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def close_all(self):
        """Close every per-thread connection ever opened by this store.

        Required on server shutdown because _local.connection is per-thread
        and unreachable from other threads, so a plain close() would leave
        all worker-thread connections (and their file handles) dangling.
        """
        with self._connections_lock:
            connections = list(self._all_connections)
            self._all_connections.clear()
        for conn in connections:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        # Reset thread-local for the calling thread; other threads' locals
        # will lazily reopen on next use, which is the desired behavior if
        # the server somehow keeps running after a partial shutdown.
        self._local = threading.local()
