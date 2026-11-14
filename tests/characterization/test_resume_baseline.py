"""Resume safety net for EPUB.

The progress-system goldens cover full runs, but not interrupt/resume — which
is exactly the path the checkpoint `chunk_index` convention and the EPUB
cross-file accumulator drive. This test interrupts an EPUB translation partway,
resumes it from the checkpoint, and asserts the resumed output is byte-for-byte
identical (content-wise) to an uninterrupted run.

It is the prerequisite net for any change to the EPUB checkpoint/accumulator
logic: if resume stops producing the same output, this fails.
"""

import asyncio
import os
from pathlib import Path

import pytest

from src.core.adapters import translate_file
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fake_llm, fixtures
from tests.characterization.recorder import fingerprint_output
from tests.job_helpers import make_ctx, make_job

_COMMON = dict(
    source_language="English",
    target_language="French",
    model="fake-echo",
    provider="poe",
    api_key="UNUSED_FAKE_KEY",
    max_tokens_per_chunk=60,
    context_window=4096,
    auto_adjust_context=False,
)


def _run(coro_factory, work_dir: Path):
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        asyncio.run(coro_factory())
    finally:
        os.chdir(cwd)


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    fake_llm.install(monkeypatch)


def test_epub_interrupt_resume_matches_clean_run(tmp_path):
    input_path = fixtures.build_epub(tmp_path)

    # 1) Clean, uninterrupted reference run.
    clean_out = tmp_path / "clean.epub"
    clean_cm = CheckpointManager(db_path=str(tmp_path / "jobs_clean.db"))

    def clean_factory():
        return translate_file(
            make_job(str(input_path), str(clean_out), **_COMMON),
            make_ctx(checkpoint_manager=clean_cm, translation_id="clean"),
        )

    _run(clean_factory, tmp_path)
    clean_fp = fingerprint_output(clean_out)

    # 2) Interrupted run: stop once a couple of chunks have been processed
    #    (file 0 checkpointed, into a later file).
    resume_out = tmp_path / "resumed.epub"
    cm = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    tid = "resume_job"
    cm.start_job(tid, "epub", {}, str(input_path))

    state = {"completed": 0, "stop": False}

    def stats_cb(stats):
        state["completed"] = stats.get("completed_chunks", 0)
        if state["completed"] >= 2:
            state["stop"] = True

    def interrupted_factory():
        return translate_file(
            make_job(str(input_path), str(resume_out), **_COMMON),
            make_ctx(checkpoint_manager=cm, translation_id=tid,
                     stats_callback=stats_cb,
                     check_interruption=lambda: state["stop"]),
        )

    _run(interrupted_factory, tmp_path)

    checkpoint = cm.load_checkpoint(tid)
    resume_from_index = checkpoint["resume_from_index"] if checkpoint else 0
    assert state["completed"] >= 2, "interrupt did not fire as expected"
    assert resume_from_index >= 1, "expected at least one file checkpointed before interrupt"

    # 3) Resume to completion.
    def resume_factory():
        return translate_file(
            make_job(str(input_path), str(resume_out), **_COMMON),
            make_ctx(checkpoint_manager=cm, translation_id=tid,
                     resume_from_index=resume_from_index,
                     check_interruption=lambda: False),
        )

    _run(resume_factory, tmp_path)
    resumed_fp = fingerprint_output(resume_out)

    assert resumed_fp == clean_fp, (
        "resumed EPUB output differs from the uninterrupted run"
    )
