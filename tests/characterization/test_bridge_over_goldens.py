"""Run the sink's progress derivation over the *real* captured engine traces.

The unit tests for ``snapshot_from_job_stats`` use synthetic dicts. This test
feeds every stats dict actually emitted by every format (as captured in the
golden files, tagged the way the job runner tags a single-phase run) through
it and pins the shape every engine now shares through ``ProgressReporter``:

* the derived percent is monotonic and bounded;
* it reaches 100% on the last emit of the run, and only there (no duplicate
  tick at the end);
* the unit total is the real unit count: it never changes during the run,
  and every emit after the first advances ``completed_chunks`` by exactly one
  unit, so the run makes ``total_chunks`` ticks.
"""

import json
from pathlib import Path

import pytest

from src.core.progress import snapshot_from_job_stats

GOLDEN_DIR = Path(__file__).parent / "golden"

# Unit counts of the fixtures: SRT translates per 20-cue block and refines per
# cue; DOCX/EPUB count HTML chunks.
_UNITS = {
    "translate_txt": 6,
    "translate_srt": 3,
    "translate_docx": 22,
    "translate_epub": 65,
    "refine_txt": 6,
    "refine_srt": 50,
    "refine_docx": 22,
    "refine_epub": 65,
}

_ALL = sorted(_UNITS)


def _load_sequence(name: str):
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))[
        "sequence"
    ]


def _percents(name: str):
    # The runner's tags for a single-phase run of that mode.
    meta = {"refine_only": True} if name.startswith("refine_") else {}
    return [snapshot_from_job_stats({**stats, **meta}).percent
            for stats in _load_sequence(name)]


@pytest.mark.parametrize("name", _ALL)
def test_percent_is_monotonic_and_bounded(name):
    percents = _percents(name)
    assert percents == sorted(percents), f"{name}: derived percent regressed"
    assert all(0.0 <= p <= 100.0 for p in percents), f"{name}: percent out of range"


@pytest.mark.parametrize("name", _ALL)
def test_reaches_100_only_on_the_last_emit(name):
    percents = _percents(name)
    assert percents[-1] == 100.0, f"{name}: expected final percent 100, got {percents[-1]}"
    assert all(p < 100.0 for p in percents[:-1]), f"{name}: 100% before the last emit"


@pytest.mark.parametrize("name", _ALL)
def test_unit_total_is_the_real_unit_count(name):
    sequence = _load_sequence(name)
    assert {stats["total_chunks"] for stats in sequence} == {_UNITS[name]}
    done = [stats["completed_chunks"] for stats in sequence]
    assert done == list(range(len(done))), f"{name}: not one tick per unit"
    assert done[-1] == _UNITS[name]
    assert all(stats["failed_chunks"] == 0 for stats in sequence)
