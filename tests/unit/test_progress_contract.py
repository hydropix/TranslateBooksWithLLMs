"""Unit tests for the unified progress contract (src/core/progress).

The tracker is the single percent authority; these tests pin its math,
its monotonicity guarantee, the two-phase segment mapping, honest
success/failure accounting, and the snapshot serialization.
"""

import pytest

from src.core.progress import Phase, ProgressSnapshot, ProgressTracker


class FakeClock:
    """Deterministic monotonic clock; advance() controls elapsed time."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def percents(tracker, succeeded=True, steps=None):
    """Drive a phase and collect the percent after each recorded unit."""
    out = [tracker.snapshot().percent]
    for _ in range(steps):
        tracker.record_unit(succeeded=succeeded)
        out.append(tracker.snapshot().percent)
    return out


# --------------------------------------------------------------------------
# Single-phase
# --------------------------------------------------------------------------

def test_single_phase_linear_percent():
    t = ProgressTracker(clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 4)
    assert percents(t, steps=4) == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_single_phase_complete_flag():
    t = ProgressTracker(clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 2)
    t.record_unit()
    assert not t.snapshot().is_complete
    t.record_unit()
    assert t.snapshot().is_complete


# --------------------------------------------------------------------------
# Two-phase segment mapping
# --------------------------------------------------------------------------

def test_two_phase_default_split_50():
    t = ProgressTracker(refinement_enabled=True, clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 2)
    assert [s for s in percents(t, steps=2)] == [0.0, 25.0, 50.0]
    t.begin_phase(Phase.REFINING, 2)
    # Refine segment is [50, 100].
    assert [s for s in percents(t, steps=2)] == [50.0, 75.0, 100.0]


def test_two_phase_custom_split():
    t = ProgressTracker(refinement_enabled=True, refine_split=80.0, clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 4)
    assert percents(t, steps=4) == [0.0, 20.0, 40.0, 60.0, 80.0]
    t.begin_phase(Phase.REFINING, 1)
    assert percents(t, steps=1) == [80.0, 100.0]


def test_two_phase_differing_totals_srt_like():
    # SRT translates per block (3) but refines per subtitle (6): each phase
    # scales into its own segment independently.
    t = ProgressTracker(refinement_enabled=True, clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 3)
    for _ in range(3):
        t.record_unit()
    assert t.snapshot().percent == 50.0
    t.begin_phase(Phase.REFINING, 6)
    seen = [t.snapshot().percent]
    for _ in range(6):
        t.record_unit()
        seen.append(t.snapshot().percent)
    assert seen[0] == 50.0 and seen[-1] == 100.0
    assert seen == sorted(seen)  # monotonic


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

def test_zero_total_phase_no_division_error():
    t = ProgressTracker(clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 0)
    assert t.snapshot().percent == 0.0


def test_done_exceeds_total_clamps():
    t = ProgressTracker(clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 2)
    for _ in range(5):
        t.record_unit()
    assert t.snapshot().percent == 100.0


def test_percent_is_monotonic_even_when_driven_backwards():
    # Re-beginning a phase resets per-phase counters; the global percent must
    # still never regress thanks to the monotonic floor.
    t = ProgressTracker(clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 4)
    for _ in range(3):
        t.record_unit()
    high = t.snapshot().percent
    assert high == 75.0
    t.begin_phase(Phase.TRANSLATING, 4)  # reset to 0/4
    assert t.snapshot().percent == high  # does not drop back to 0


def test_phase_transition_does_not_regress():
    t = ProgressTracker(refinement_enabled=True, clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 3)
    t.record_unit()  # 1/3 -> 16.67
    mid = t.snapshot().percent
    t.begin_phase(Phase.REFINING, 3)  # translation ended early at 1/3
    assert t.snapshot().percent >= mid
    assert t.snapshot().percent == 50.0  # jumps forward to segment boundary


# --------------------------------------------------------------------------
# Honest success/failure accounting
# --------------------------------------------------------------------------

def test_failed_units_advance_done_but_flag_quality():
    t = ProgressTracker(clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 4)
    t.record_unit(succeeded=True)
    t.record_unit(succeeded=False)
    snap = t.snapshot()
    assert snap.units_done == 2
    assert snap.units_succeeded == 1
    assert snap.units_failed == 1
    assert snap.percent == 50.0
    assert snap.quality_degraded is True


def test_clean_run_not_degraded():
    t = ProgressTracker(clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 1)
    t.record_unit()
    assert t.snapshot().quality_degraded is False


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_single_phase_refine_fills_the_whole_bar():
    # A refine-only run: REFINING on a single-phase tracker spans [0, 100].
    t = ProgressTracker(refinement_enabled=False, clock=FakeClock())
    t.begin_phase(Phase.REFINING, 4)
    t.record_unit(count=2)
    snap = t.snapshot()
    assert snap.phase is Phase.REFINING
    assert snap.percent == 50.0


def test_set_counts_overwrites_absolutely():
    t = ProgressTracker(clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 4)
    t.set_counts(done=3, failed=1, total=6)
    snap = t.snapshot()
    assert (snap.units_total, snap.units_done, snap.units_failed, snap.units_succeeded) == (6, 3, 1, 2)
    t.set_counts(done=4)
    snap = t.snapshot()
    assert (snap.units_total, snap.units_done, snap.units_failed) == (6, 4, 1)


def test_set_counts_rejects_negative_values():
    t = ProgressTracker(clock=FakeClock())
    with pytest.raises(ValueError):
        t.set_counts(done=-1)


@pytest.mark.parametrize("bad", [0.0, 100.0, -5.0, 150.0])
def test_invalid_split_raises(bad):
    with pytest.raises(ValueError):
        ProgressTracker(refinement_enabled=True, refine_split=bad)


def test_negative_counts_raise():
    t = ProgressTracker(clock=FakeClock())
    with pytest.raises(ValueError):
        t.begin_phase(Phase.TRANSLATING, -1)
    t.begin_phase(Phase.TRANSLATING, 1)
    with pytest.raises(ValueError):
        t.record_unit(count=-1)


# --------------------------------------------------------------------------
# Snapshot serialization & timing
# --------------------------------------------------------------------------

def test_snapshot_to_dict_shape():
    t = ProgressTracker(refinement_enabled=True, clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 3)
    t.record_unit()
    d = t.snapshot().to_dict()
    assert d == {
        "phase": "translating",
        "percent": 16.67,
        "units_done": 1,
        "units_total": 3,
        "units_succeeded": 1,
        "units_failed": 0,
        "refinement_enabled": True,
        "quality_degraded": False,
        "elapsed_seconds": 0.0,
        "eta_seconds": 0.0,  # elapsed is 0 (clock not advanced) -> eta 0
    }


def test_elapsed_and_eta_use_injected_clock():
    clock = FakeClock()
    t = ProgressTracker(clock=clock)
    t.begin_phase(Phase.TRANSLATING, 4)
    clock.advance(10.0)
    t.record_unit()  # 25%
    snap = t.snapshot()
    assert snap.elapsed_seconds == 10.0
    # eta = elapsed * (100 - pct) / pct = 10 * 75 / 25 = 30
    assert snap.eta_seconds == pytest.approx(30.0)


def test_eta_none_at_zero_and_hundred():
    t = ProgressTracker(clock=FakeClock())
    t.begin_phase(Phase.TRANSLATING, 1)
    assert t.snapshot().eta_seconds is None  # 0%
    t.record_unit()
    assert t.snapshot().eta_seconds is None  # 100%


def test_snapshot_is_immutable():
    snap = ProgressSnapshot(
        phase=Phase.TRANSLATING,
        units_done=1,
        units_total=2,
        units_succeeded=1,
        units_failed=0,
        percent=50.0,
        refinement_enabled=False,
    )
    with pytest.raises(Exception):
        snap.percent = 99.0  # frozen dataclass
