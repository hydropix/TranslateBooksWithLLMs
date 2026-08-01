"""Issue #225 — a tab may only react to translation jobs it owns.

Before the ownership set, a terminal `translation_update` with no current job
reset the batch UI unconditionally. That fired for another tab's job, and for a
straggler duplicate arriving in the window where `finishCurrentFileTranslation`
has nulled `currentJob` but the queue has not advanced yet.

These are the plan's manual protocol items 1 to 5, automated.
"""
import time

import pytest

from .helpers import (
    TERMINAL_STATUSES,
    emit_translation_update,
    force_busy_ui,
    open_app,
    queue_files,
    register_owned_job,
    start_batch,
    state,
    ui,
)

pytestmark = pytest.mark.e2e


def test_foreign_terminal_event_is_ignored_when_idle(page):
    """A job this tab never started must not touch the idle UI."""
    before = ui(page)
    emit_translation_update(page, "not-mine-123", "completed")
    page.wait_for_timeout(800)
    assert ui(page) == before


def test_owned_and_unowned_terminal_events_on_a_busy_ui(page):
    """Only an owned id may reset a busy UI to idle."""
    force_busy_ui(page)

    emit_translation_update(page, "unowned-xyz", "completed")
    page.wait_for_timeout(600)
    busy = ui(page)
    assert busy["progress"] is True and busy["interrupt"] is True, busy

    register_owned_job(page, "owned-abc")
    emit_translation_update(page, "owned-abc", "completed")
    page.wait_for_timeout(1000)
    idle = ui(page)
    assert idle["progress"] is False and idle["interrupt"] is False, idle


def test_batch_of_three_survives_a_foreign_terminal_event(page, input_files):
    """A three-file batch runs to the end with no UI reset, foreign event included."""
    queue_files(page, input_files["short"])
    assert state(page)["queue"] == ["Queued"] * 3

    page.click("#translateBtn")
    page.wait_for_timeout(1500)

    idle_flashes = 0
    injected = False
    jobs_seen = set()
    deadline = time.time() + 240

    while time.time() < deadline:
        current_state, current_ui = state(page), ui(page)

        if current_state["currentJob"]:
            jobs_seen.add(current_state["currentJob"])
        # The progress panel disappearing mid-batch is what a reset looks like.
        if current_state["isBatchActive"] and not current_ui["progress"]:
            idle_flashes += 1
        if not current_state["isBatchActive"] and current_ui["cards"] >= 3:
            break

        if not injected and current_state["currentJob"]:
            emit_translation_update(page, "foreign-during-batch-999", "completed")
            injected = True
            page.wait_for_timeout(600)
            mid_state, mid_ui = state(page), ui(page)
            assert mid_state["currentJob"] is not None, mid_state
            assert mid_state["isBatchActive"] is True, mid_state
            assert mid_ui["progress"] is True, mid_ui

        page.wait_for_timeout(1000)

    page.wait_for_timeout(2000)
    final_state, final_ui = state(page), ui(page)

    assert injected, "the foreign event was never injected during the run"
    assert idle_flashes == 0, "the progress panel vanished mid-batch"
    assert len(jobs_seen) == 3, jobs_seen
    assert final_ui["cards"] == 3, final_ui
    assert final_state["isBatchActive"] is False and not final_ui["interrupt"]
    assert final_state["ownedJobIds"] == [], "finished ids must be retired"


def test_reload_mid_job_keeps_ownership(page, input_files, api, live_server):
    """After a reload the tab still owns, and still reacts to, its running job."""
    base_url, _ = live_server

    queue_files(page, [input_files["long"]])
    translation_id = start_batch(page)
    assert translation_id, "no job id was registered"
    assert translation_id in state(page)["ownedJobIds"]

    page.wait_for_timeout(2000)
    assert api.job_status(translation_id) == "running", (
        "the job finished before the reload; the test window was too short")

    page.reload(wait_until="networkidle")
    page.wait_for_timeout(4000)

    restored = state(page)
    assert translation_id in (restored["ownedJobIds"] or []), restored
    assert restored["isBatchActive"] is True and restored["currentJob"] == translation_id

    api.wait_for_terminal(translation_id)
    page.wait_for_timeout(6000)
    final_ui = ui(page)
    assert final_ui["cards"] >= 1 and not final_ui["interrupt"], final_ui


def test_resume_registers_the_job_as_owned(page, input_files, api):
    """Resuming a checkpointed job goes through resume-manager's registration."""
    queue_files(page, [input_files["long"]])
    translation_id = start_batch(page)
    assert translation_id
    page.wait_for_timeout(4000)

    page.click("#interruptBtn")
    deadline = time.time() + 120
    while time.time() < deadline and api.job_status(translation_id) not in TERMINAL_STATUSES:
        page.wait_for_timeout(2000)
    assert api.job_status(translation_id) == "interrupted"

    page.wait_for_timeout(3000)
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(4000)

    assert translation_id not in (state(page)["ownedJobIds"] or []), (
        "a fresh page must not own the job before it is resumed")
    listed = page.evaluate("""(translationId) => {
        const list = document.getElementById('resumableJobsList');
        return list ? list.innerHTML.includes(translationId) : false;
    }""", translation_id)
    assert listed, "the interrupted job is missing from the resumable list"

    page.evaluate("(translationId) => window.resumeJob(translationId)", translation_id)
    page.wait_for_timeout(6000)

    resumed = state(page)
    assert translation_id in (resumed["ownedJobIds"] or []), resumed
    assert resumed["currentJob"] == translation_id, resumed

    api.wait_for_terminal(translation_id)
    page.wait_for_timeout(6000)
    assert not ui(page)["interrupt"]
