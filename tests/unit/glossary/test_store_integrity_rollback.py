"""A write rejected by a UNIQUE constraint must not leave a transaction open.

GlossaryStore keeps one long-lived connection per thread. An IntegrityError
raised by a write left the implicit transaction open on it, holding the write
lock and blocking every other writer until the next commit on that thread.
"""

import sqlite3

import pytest

from src.core.glossary.models import GlossaryTerm
from src.core.glossary.store import GlossaryStore


@pytest.fixture
def store(tmp_path):
    return GlossaryStore(db_path=str(tmp_path / "glossaries.db"))


def _other_writer_can_write(db_path):
    conn = sqlite3.connect(db_path, timeout=0.2)
    try:
        conn.execute("UPDATE glossaries SET name = name")
        conn.commit()
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def test_duplicate_term_rolls_back(store):
    glossary = store.create_glossary("g")
    store.add_term(glossary.id, GlossaryTerm(source_term="Fan", translated_term="Fan"))
    with pytest.raises(ValueError):
        store.add_term(glossary.id, GlossaryTerm(source_term="Fan", translated_term="X"))
    assert not store._get_connection().in_transaction
    assert _other_writer_can_write(store.db_path)


def test_update_term_conflict_rolls_back(store):
    glossary = store.create_glossary("g")
    store.add_term(glossary.id, GlossaryTerm(source_term="A", translated_term="a"))
    b = store.add_term(glossary.id, GlossaryTerm(source_term="B", translated_term="b"))
    with pytest.raises(ValueError):
        store.update_term(b.id, source_term="A")
    assert not store._get_connection().in_transaction
    assert _other_writer_can_write(store.db_path)


def test_duplicate_glossary_name_rolls_back(store):
    store.create_glossary("g")
    with pytest.raises(ValueError):
        store.create_glossary("g")
    assert not store._get_connection().in_transaction
