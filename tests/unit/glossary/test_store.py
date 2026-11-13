"""
Unit tests for GlossaryStore.

Verifies SQLite CRUD operations for glossaries and glossary terms,
including unique constraints, cascade deletes, and bulk replacement.
"""
import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.glossary.store import GlossaryStore
from src.core.glossary.models import BulkReplaceResult, GlossaryTerm


@pytest.fixture
def store():
    """Per-test temporary GlossaryStore on an isolated SQLite file."""
    db = os.path.join(
        tempfile.gettempdir(),
        f"glossary_test_{os.getpid()}_{id(object())}.db",
    )
    if os.path.exists(db):
        os.remove(db)
    s = GlossaryStore(db_path=db)
    try:
        yield s
    finally:
        s.close()
        try:
            os.remove(db)
        except OSError:
            pass


class TestGlossaryStoreCRUD:
    """Tests for glossary-level CRUD operations."""

    def test_create_and_get_roundtrip(self, store):
        """Create a glossary and retrieve it preserves the core fields."""
        created = store.create_glossary(
            name="Fantasy Series",
            source_language="English",
            target_language="French",
        )
        fetched = store.get_glossary(created.id)
        assert fetched is not None
        assert fetched.name == "Fantasy Series"
        assert fetched.source_language == "English"
        assert fetched.target_language == "French"

    def test_unique_name_conflict_raises(self, store):
        """Creating two glossaries with the same name raises ValueError."""
        store.create_glossary(name="Unique", source_language="en", target_language="fr")
        with pytest.raises(ValueError):
            store.create_glossary(name="Unique", source_language="en", target_language="fr")

    def test_list_glossaries_returns_all_without_terms(self, store):
        """list_glossaries returns all glossaries, with terms list empty."""
        g1 = store.create_glossary(name="A", source_language="en", target_language="fr")
        g2 = store.create_glossary(name="B", source_language="en", target_language="es")
        store.add_term(g1.id, GlossaryTerm(source_term="hello", translated_term="bonjour"))
        store.add_term(g2.id, GlossaryTerm(source_term="hello", translated_term="hola"))

        listed = store.list_glossaries()
        assert len(listed) == 2
        for glossary in listed:
            # Terms are NOT loaded by list_glossaries
            assert glossary.terms == []

    def test_list_glossaries_with_counts_returns_term_counts(self, store):
        """list_glossaries_with_counts returns (Glossary, count) tuples."""
        g1 = store.create_glossary(name="A", source_language="en", target_language="fr")
        g2 = store.create_glossary(name="B", source_language="en", target_language="es")
        store.create_glossary(name="C", source_language="en", target_language="de")
        store.add_term(g1.id, GlossaryTerm(source_term="hello", translated_term="bonjour"))
        store.add_term(g1.id, GlossaryTerm(source_term="cat", translated_term="chat"))
        store.add_term(g2.id, GlossaryTerm(source_term="hello", translated_term="hola"))
        # g3 has no terms

        results = store.list_glossaries_with_counts()
        by_name = {g.name: count for g, count in results}
        assert by_name == {"A": 2, "B": 1, "C": 0}

    def test_update_glossary_patches_only_provided_fields(self, store):
        """update_glossary changes only provided fields, leaves others alone."""
        g = store.create_glossary(name="Original", source_language="en", target_language="fr")
        updated = store.update_glossary(g.id, name="Renamed")
        assert updated is not None
        assert updated.name == "Renamed"
        # Other fields preserved
        assert updated.source_language == "en"
        assert updated.target_language == "fr"

    def test_delete_glossary_cascades_to_terms(self, store):
        """delete_glossary removes the glossary and its terms (CASCADE)."""
        g = store.create_glossary(name="ToDelete", source_language="en", target_language="fr")
        store.add_term(g.id, GlossaryTerm(source_term="cat", translated_term="chat"))

        deleted = store.delete_glossary(g.id)
        assert deleted is True

        # Re-creating with the same name must succeed (proves the glossary row is gone)
        recreated = store.create_glossary(
            name="ToDelete", source_language="en", target_language="fr"
        )
        # And the new glossary has no terms (proves the old terms were also wiped)
        fetched = store.get_glossary(recreated.id)
        assert fetched.terms == []

    def test_get_glossary_missing_returns_none(self, store):
        """get_glossary on a missing id returns None instead of raising."""
        assert store.get_glossary(9999) is None


class TestGlossaryStoreTerms:
    """Tests for term-level CRUD operations."""

    def test_add_term_then_get_includes_term(self, store):
        """add_term inserts a term and get_glossary returns it."""
        g = store.create_glossary(name="G", source_language="en", target_language="fr")
        store.add_term(g.id, GlossaryTerm(source_term="dog", translated_term="chien"))

        fetched = store.get_glossary(g.id)
        assert len(fetched.terms) == 1
        assert fetched.terms[0].source_term == "dog"
        assert fetched.terms[0].translated_term == "chien"

    def test_add_term_duplicate_source_raises(self, store):
        """Adding a duplicate (glossary_id, source_term) raises ValueError."""
        g = store.create_glossary(name="G", source_language="en", target_language="fr")
        store.add_term(g.id, GlossaryTerm(source_term="dog", translated_term="chien"))
        with pytest.raises(ValueError):
            store.add_term(g.id, GlossaryTerm(source_term="dog", translated_term="other"))

    def test_update_term_patches_subset_of_fields(self, store):
        """update_term changes only provided fields."""
        g = store.create_glossary(name="G", source_language="en", target_language="fr")
        term = store.add_term(
            g.id,
            GlossaryTerm(source_term="dog", translated_term="chien", category="animal"),
        )

        updated = store.update_term(term.id, translated_term="canin")
        assert updated is not None
        assert updated.translated_term == "canin"
        # Other fields untouched
        assert updated.source_term == "dog"
        assert updated.category == "animal"

    def test_delete_term_removes_only_that_term(self, store):
        """delete_term removes one term and leaves the rest intact."""
        g = store.create_glossary(name="G", source_language="en", target_language="fr")
        t1 = store.add_term(g.id, GlossaryTerm(source_term="dog", translated_term="chien"))
        t2 = store.add_term(g.id, GlossaryTerm(source_term="cat", translated_term="chat"))

        assert store.delete_term(t1.id) is True

        fetched = store.get_glossary(g.id)
        remaining_sources = [t.source_term for t in fetched.terms]
        assert "dog" not in remaining_sources
        assert "cat" in remaining_sources
        # Verify t2 still exists
        assert any(t.id == t2.id for t in fetched.terms)


class TestGlossaryStoreBulkReplace:
    """Tests for bulk_replace_terms."""

    def test_bulk_replace_wipes_and_inserts(self, store):
        """bulk_replace_terms replaces all terms atomically."""
        g = store.create_glossary(name="G", source_language="en", target_language="fr")
        store.add_term(g.id, GlossaryTerm(source_term="old1", translated_term="vieux1"))
        store.add_term(g.id, GlossaryTerm(source_term="old2", translated_term="vieux2"))

        new_terms = [
            GlossaryTerm(source_term="new1", translated_term="nouveau1"),
            GlossaryTerm(source_term="new2", translated_term="nouveau2"),
            GlossaryTerm(source_term="new3", translated_term="nouveau3"),
        ]
        result = store.bulk_replace_terms(g.id, new_terms)
        assert isinstance(result, BulkReplaceResult)
        assert result.inserted == 3
        assert result.skipped_empty == 0
        assert result.skipped_duplicate == 0
        assert result.total_input == 3

        fetched = store.get_glossary(g.id)
        sources = [t.source_term for t in fetched.terms]
        assert sources == ["new1", "new2", "new3"]

    def test_bulk_replace_with_empty_list_clears_all(self, store):
        """bulk_replace_terms with an empty list clears all existing terms."""
        g = store.create_glossary(name="G", source_language="en", target_language="fr")
        store.add_term(g.id, GlossaryTerm(source_term="old1", translated_term="vieux1"))
        store.add_term(g.id, GlossaryTerm(source_term="old2", translated_term="vieux2"))

        result = store.bulk_replace_terms(g.id, [])
        assert result.inserted == 0
        assert result.total_input == 0

        fetched = store.get_glossary(g.id)
        assert fetched.terms == []

    def test_bulk_replace_reports_skipped_empty_and_duplicate(self, store):
        """The result breaks down inserted vs empty vs duplicate sources."""
        g = store.create_glossary(name="G", source_language="en", target_language="fr")

        terms = [
            GlossaryTerm(source_term="alpha", translated_term="A"),
            GlossaryTerm(source_term="", translated_term="empty"),
            GlossaryTerm(source_term="   ", translated_term="whitespace"),
            GlossaryTerm(source_term="alpha", translated_term="dup"),
            GlossaryTerm(source_term="beta", translated_term="B"),
            GlossaryTerm(source_term="beta", translated_term="dup2"),
        ]
        result = store.bulk_replace_terms(g.id, terms)

        assert result.total_input == 6
        assert result.inserted == 2
        assert result.skipped_empty == 2
        assert result.skipped_duplicate == 2

        fetched = store.get_glossary(g.id)
        sources = sorted(t.source_term for t in fetched.terms)
        assert sources == ["alpha", "beta"]

    def test_bulk_replace_unknown_glossary_returns_empty_result(self, store):
        """Unknown glossary id returns a zeroed result with the input count."""
        result = store.bulk_replace_terms(
            9999,
            [GlossaryTerm(source_term="x", translated_term="y")],
        )
        assert result.inserted == 0
        assert result.skipped_empty == 0
        assert result.skipped_duplicate == 0
        assert result.total_input == 1


class TestGlossaryStoreCloseAll:
    """Tests for the close_all() shutdown helper."""

    def test_close_all_drops_per_thread_connections(self, store):
        """close_all() closes existing connections; next query reopens fresh."""
        import threading

        # Open a connection on a worker thread that the main thread cannot
        # see via _local — this is the leak case close_all() must handle.
        worker_conn_holder = {}

        def _open_on_worker():
            conn = store._get_connection()
            worker_conn_holder['conn'] = conn

        t = threading.Thread(target=_open_on_worker)
        t.start()
        t.join()

        # And one on the main thread, to exercise both paths.
        main_conn = store._get_connection()
        assert main_conn is not None
        assert worker_conn_holder['conn'] is not None
        assert len(store._all_connections) >= 2

        store.close_all()

        # Main thread's _local was reset; next call yields a brand-new conn.
        assert not hasattr(store._local, "connection") or store._local.connection is None
        new_conn = store._get_connection()
        assert new_conn is not main_conn

        # The previously tracked worker conn is closed: any operation on it
        # should fail with ProgrammingError ("closed database").
        import sqlite3
        with pytest.raises(sqlite3.ProgrammingError):
            worker_conn_holder['conn'].execute("SELECT 1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
