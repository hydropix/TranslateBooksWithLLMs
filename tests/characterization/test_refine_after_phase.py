"""Characterize the refine-after phase-2 emission as the UI sees it.

The handlers orchestration runs translate_file then refine_file, injecting
`_workflow_meta = {enable_refinement: True, current_phase: 2}` into every
stats emit and resetting the counters at the phase boundary. This test
reproduces that exact merge + the Step 2 bridge over a *multi-chunk* refine
and asserts the bar climbs 50 -> 100 with per-chunk feedback.

This pins the behavior probed during the Step 3 smoke test: a single-chunk
file shows no movement in the refine segment (it sits at 50% during the one
LLM call), whereas a multi-chunk file advances through the [50, 100] segment.
"""

import asyncio
import os
from pathlib import Path

import pytest

from src.core.adapters import refine_file
from src.core.progress import snapshot_from_legacy_stats
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fake_llm, fixtures
from tests.job_helpers import make_ctx, make_job


def _run_refine_after(work_dir: Path, input_path: Path, output_path: Path):
    """Drive refine_file through the same merge the handlers refine-after path
    uses, returning the sequence of emitted global percents."""
    # Mirrors handlers._workflow_meta for the refine-after phase 2.
    workflow_meta = {"enable_refinement": True, "current_phase": 2}
    percents = []

    def emit(stats):
        merged = {**stats, **workflow_meta}
        percents.append(snapshot_from_legacy_stats(merged).percent)

    # handlers emits a counter reset at the phase-1 -> phase-2 boundary.
    emit({"completed_chunks": 0, "failed_chunks": 0})

    checkpoint_manager = CheckpointManager(db_path=str(work_dir / "jobs.db"))

    async def factory():
        await refine_file(
            make_job(
                str(input_path), str(output_path), operation='refine',
                provider="poe", model="fake-echo", api_key="UNUSED_FAKE_KEY",
                source_language="French", target_language="French",
                max_tokens_per_chunk=60,
                context_window=4096,
                auto_adjust_context=False,
            ),
            make_ctx(translation_id="char_refine_after",
                     checkpoint_manager=checkpoint_manager,
                     stats_callback=emit),
        )

    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        asyncio.run(factory())
    finally:
        os.chdir(cwd)
    return percents


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    fake_llm.install(monkeypatch)


def test_multi_chunk_refine_after_climbs_50_to_100(tmp_path):
    input_path = fixtures.build_txt(tmp_path)  # 6 chunks
    percents = _run_refine_after(tmp_path, input_path, tmp_path / "refined.txt")

    assert percents[0] == 50.0, "refine phase must start at the 50% boundary"
    assert percents == sorted(percents), "refine-after percent must be monotonic"
    assert percents[-1] == 100.0, "completed refine must reach 100%"
    # The whole point: a multi-chunk refine produces intermediate feedback
    # strictly between 50 and 100 (the bug the smoke test surfaced only on a
    # single-chunk file, where no such intermediate value can exist).
    intermediate = [p for p in percents if 50.0 < p < 100.0]
    assert len(intermediate) >= 2, f"expected per-chunk feedback, got {percents}"
