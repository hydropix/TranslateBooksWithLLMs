"""Unit tests for the job completion classifier."""
from src.api.completion_status import classify_completion


def _output_file(tmp_path, content="translated text"):
    """Create an output file under tmp_path and return its path as a string."""
    path = tmp_path / "output.txt"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_fallback_only_is_partial(tmp_path):
    """A chunk left as source text must not be reported as completed."""
    verdict = classify_completion(
        {'fallback_used': 1, 'failed_chunks': 0, 'total_chunks': 10},
        _output_file(tmp_path),
    )
    assert verdict.status == 'partial'
    assert verdict.fallback_chunks == 1
    assert verdict.failed_chunks == 0
    assert verdict.error is None


def test_failed_only_is_partial(tmp_path):
    """Failed chunks alone still produce a partial verdict."""
    verdict = classify_completion(
        {'failed_chunks': 2, 'total_chunks': 10},
        _output_file(tmp_path),
    )
    assert verdict.status == 'partial'
    assert verdict.failed_chunks == 2
    assert verdict.fallback_chunks == 0


def test_failed_and_fallback_are_kept_distinct(tmp_path):
    """The two counters are reported separately and never summed."""
    verdict = classify_completion(
        {'failed_chunks': 1, 'fallback_used': 3},
        _output_file(tmp_path),
    )
    assert verdict.status == 'partial'
    assert verdict.failed_chunks == 1
    assert verdict.fallback_chunks == 3


def test_token_alignment_is_not_a_failure(tmp_path):
    """token_alignment_used and placeholder_errors never degrade the verdict."""
    verdict = classify_completion(
        {'token_alignment_used': 5, 'placeholder_errors': 2},
        _output_file(tmp_path),
    )
    assert verdict.status == 'completed'
    assert verdict.error is None


def test_clean_run_is_completed(tmp_path):
    """A run with every chunk translated is completed."""
    verdict = classify_completion(
        {'total_chunks': 10, 'completed_chunks': 10},
        _output_file(tmp_path),
    )
    assert verdict.status == 'completed'
    assert verdict.failed_chunks == 0
    assert verdict.fallback_chunks == 0


def test_missing_stats_keys_default_to_zero(tmp_path):
    """An empty stats payload must not crash and must classify as completed."""
    verdict = classify_completion({}, _output_file(tmp_path))
    assert verdict.status == 'completed'
    assert verdict.failed_chunks == 0
    assert verdict.fallback_chunks == 0


def test_missing_output_file_is_error(tmp_path):
    """Issue #246: a job with no output file on disk can never be completed."""
    missing = str(tmp_path / "never_written.txt")
    verdict = classify_completion({'total_chunks': 10}, missing)
    assert verdict.status == 'error'
    assert verdict.error is not None
    assert missing in verdict.error


def test_empty_output_file_is_error(tmp_path):
    """Issue #246: a 0-byte output for a job that had chunks is an error."""
    verdict = classify_completion(
        {'total_chunks': 10},
        _output_file(tmp_path, content=""),
    )
    assert verdict.status == 'error'
    assert verdict.error is not None
    assert "0 bytes" in verdict.error


def test_empty_output_from_empty_source_is_completed(tmp_path):
    """An empty source file legitimately produces an empty output."""
    verdict = classify_completion(
        {'total_chunks': 0},
        _output_file(tmp_path, content=""),
    )
    assert verdict.status == 'completed'
    assert verdict.error is None


def test_none_output_path_is_error():
    """A job that never got an output path at all is an error."""
    verdict = classify_completion({'total_chunks': 1}, None)
    assert verdict.status == 'error'
    assert verdict.error is not None
    assert '(none)' in verdict.error
