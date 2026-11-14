"""Regression tests for issue #180: fallback counters reset on resume.

The cross-file `accumulated_stats` in the EPUB pipeline used to be
re-initialized to a fresh `TranslationMetrics()` on every entry into
`_process_all_content_files`, so any token-alignment / Phase-3 fallbacks
that happened in previously translated files were lost when the job was
resumed from checkpoint. These tests cover the snapshot / restore wiring
end-to-end at the persistence layer.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.epub.translation_metrics import TranslationMetrics
from src.persistence.checkpoint_manager import CheckpointManager


@pytest.fixture
def checkpoint_manager(tmp_path):
    """Isolated CheckpointManager backed by a temporary DB."""
    db_path = str(tmp_path / "jobs.db")
    mgr = CheckpointManager(db_path=db_path)
    yield mgr
    mgr.close()


def _make_populated_metrics() -> TranslationMetrics:
    metrics = TranslationMetrics()
    metrics.token_alignment_used = 7
    metrics.token_alignment_success = 5
    metrics.fallback_used = 3
    metrics.failed_chunks = 1
    metrics.placeholder_errors = 4
    metrics.processed_chunks = 20
    metrics.successful_first_try = 12
    metrics.successful_after_retry = 4
    metrics.retry_attempts = 9
    metrics.quality_warning_fired = True
    metrics.total_tokens_processed = 1500
    metrics.total_tokens_generated = 1400
    metrics.refinement_chunks_completed = 6
    return metrics


def test_snapshot_round_trip_preserves_cross_file_counters():
    metrics = _make_populated_metrics()
    snapshot = metrics.snapshot_accumulated()

    restored = TranslationMetrics()
    restored.restore_accumulated(snapshot)

    assert restored.token_alignment_used == 7
    assert restored.token_alignment_success == 5
    assert restored.fallback_used == 3
    assert restored.failed_chunks == 1
    assert restored.placeholder_errors == 4
    assert restored.processed_chunks == 20
    assert restored.successful_first_try == 12
    assert restored.successful_after_retry == 4
    assert restored.retry_attempts == 9
    assert restored.quality_warning_fired is True
    assert restored.total_tokens_processed == 1500
    assert restored.total_tokens_generated == 1400
    assert restored.refinement_chunks_completed == 6


def test_restore_on_empty_snapshot_is_a_noop():
    """No snapshot (legacy checkpoint) must leave the fresh metrics zeroed."""
    metrics = TranslationMetrics()
    metrics.restore_accumulated(None)
    metrics.restore_accumulated({})

    assert metrics.token_alignment_used == 0
    assert metrics.fallback_used == 0
    assert metrics.placeholder_errors == 0


def test_save_checkpoint_persists_accumulated_stats(checkpoint_manager):
    translation_id = "trans_test_180"
    checkpoint_manager.start_job(translation_id, "epub", {"some": "config"})

    snapshot = _make_populated_metrics().snapshot_accumulated()
    checkpoint_manager.save_checkpoint(
        translation_id=translation_id,
        chunk_index=1,
        original_text="file1.xhtml",
        translated_text="file1.xhtml",
        chunk_data={"last_file": "file1.xhtml", "file_type": "epub_xhtml"},
        total_chunks=100,
        completed_chunks=20,
        failed_chunks=1,
        epub_accumulated_stats=snapshot,
    )

    job = checkpoint_manager.get_job(translation_id)
    persisted = job["progress"].get("epub_accumulated_stats")
    assert persisted is not None
    assert persisted["token_alignment_used"] == 7
    assert persisted["fallback_used"] == 3
    assert persisted["placeholder_errors"] == 4
    assert persisted["processed_chunks"] == 20
    assert persisted["quality_warning_fired"] is True


def test_subsequent_progress_update_keeps_accumulated_stats(checkpoint_manager):
    """save_xhtml_partial_state and similar callers update progress without
    touching epub_accumulated_stats — make sure the snapshot survives."""
    translation_id = "trans_test_180_b"
    checkpoint_manager.start_job(translation_id, "epub", {})

    checkpoint_manager.save_checkpoint(
        translation_id=translation_id,
        chunk_index=1,
        original_text="file1.xhtml",
        translated_text="file1.xhtml",
        total_chunks=100,
        completed_chunks=20,
        failed_chunks=0,
        epub_accumulated_stats=_make_populated_metrics().snapshot_accumulated(),
    )

    # Simulate save_xhtml_partial_state's progress nudge (no stats arg).
    checkpoint_manager.db.update_job_progress(
        translation_id=translation_id,
        completed_chunks=25,
    )

    job = checkpoint_manager.get_job(translation_id)
    persisted = job["progress"].get("epub_accumulated_stats")
    assert persisted is not None, "Progress update wiped out the fallback snapshot"
    assert persisted["fallback_used"] == 3
    assert job["progress"]["completed_chunks"] == 25


# The exact `epub_accumulated_stats` row shape written before the snapshot
# moved onto TranslationMetrics. Old checkpoints carry it verbatim.
_LEGACY_ROW = {
    'token_alignment_used': 7, 'token_alignment_success': 5, 'fallback_used': 3,
    'failed_chunks': 2, 'placeholder_errors': 11, 'processed_chunks': 40,
    'successful_first_try': 30, 'successful_after_retry': 4, 'retry_attempts': 9,
    'quality_warning_fired': True, 'fallback_warning_fired': True,
    'paragraph_count_mismatches': 1, 'paragraph_retry_recovered': 1,
    'paragraph_repair_failed': 0, 'correction_attempts': 6, 'correction_success': 2,
    'total_tokens_processed': 1000, 'total_tokens_generated': 900,
    'refinement_chunks_completed': 0,
}


def test_legacy_row_round_trips_with_the_same_keys():
    metrics = TranslationMetrics()
    metrics.restore_accumulated(dict(_LEGACY_ROW))
    assert metrics.snapshot_accumulated() == _LEGACY_ROW


def test_partial_legacy_row_defaults_missing_counters():
    metrics = TranslationMetrics()
    metrics.fallback_used = 99
    metrics.quality_warning_fired = True
    metrics.restore_accumulated({'token_alignment_used': 2})
    assert metrics.token_alignment_used == 2
    assert metrics.fallback_used == 0
    assert metrics.quality_warning_fired is False
