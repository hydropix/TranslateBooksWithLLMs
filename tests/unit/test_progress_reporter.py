"""Unit tests for ProgressReporter, the engine side of progress reporting."""

import pytest

from src.core.progress import COUNT_FIELDS, Phase, ProgressReporter, progress_payload


def test_counts_come_from_the_tracker():
    seen = []
    reporter = ProgressReporter(seen.append)
    reporter.begin(4)
    reporter.emit({"unfinished_chunks": 4})
    reporter.record()
    reporter.record(succeeded=False)
    reporter.emit({"unfinished_chunks": 3})
    assert seen == [
        {"total_chunks": 4, "completed_chunks": 0, "failed_chunks": 0, "unfinished_chunks": 4},
        {"total_chunks": 4, "completed_chunks": 2, "failed_chunks": 1, "unfinished_chunks": 3},
    ]
    assert (reporter.total, reporter.done, reporter.failed, reporter.succeeded) == (4, 2, 1, 1)


def test_begin_restores_done_units():
    reporter = ProgressReporter()
    reporter.begin(10, done=4)
    assert (reporter.done, reporter.succeeded, reporter.failed) == (4, 4, 0)


def test_set_counts_is_absolute():
    reporter = ProgressReporter()
    reporter.begin(10)
    reporter.set_counts(7, failed=2)
    assert reporter.payload() == {"total_chunks": 10, "completed_chunks": 7, "failed_chunks": 2}
    reporter.set_counts(8, total=12)
    assert reporter.payload() == {"total_chunks": 12, "completed_chunks": 8, "failed_chunks": 2}


def test_unchanged_payload_is_not_emitted_twice():
    seen = []
    reporter = ProgressReporter(seen.append)
    reporter.begin(2)
    reporter.emit()
    reporter.emit()
    reporter.record()
    reporter.emit({"maps": {"a": [1]}})
    reporter.emit({"maps": {"a": [1]}})
    reporter.emit({"maps": {}})
    assert [s["completed_chunks"] for s in seen] == [0, 1, 1]


def test_emit_without_callback_returns_the_payload():
    reporter = ProgressReporter(phase=Phase.REFINING)
    reporter.begin(3)
    reporter.record()
    assert reporter.emit({"x": 1}) == {
        "total_chunks": 3, "completed_chunks": 1, "failed_chunks": 0, "x": 1}
    assert reporter.snapshot().phase is Phase.REFINING


@pytest.mark.parametrize("key", COUNT_FIELDS)
def test_extra_may_not_set_count_fields(key):
    reporter = ProgressReporter()
    with pytest.raises(ValueError):
        reporter.emit({key: 1})


def test_progress_payload_is_the_one_builder():
    reporter = ProgressReporter()
    reporter.begin(5, done=2, failed=1)
    assert progress_payload(reporter.snapshot(), {"y": 2}) == reporter.payload({"y": 2})
