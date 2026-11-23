"""Issue #349 — closing the tab must not interrupt the running job.

A job belongs to the server, not to the tab that started it: on a self-hosted
server the expected workflow is to start a book, close the browser and come
back later. A `pagehide` handler used to POST `/interrupt` on every page
close, reload or mobile background unload, which stopped the job after the
current chunk.
"""
import time

import pytest

from .helpers import queue_files, start_batch

pytestmark = pytest.mark.e2e


def test_closing_the_tab_leaves_the_job_running(page, input_files, api):
    queue_files(page, [input_files["long"]])
    translation_id = start_batch(page)
    assert translation_id, "no job id was registered"

    page.wait_for_timeout(2000)
    assert api.job_status(translation_id) == "running", (
        "the job finished before the tab was closed; the test window was too short")

    # run_before_unload fires the real unload sequence (pagehide included).
    page.close(run_before_unload=True)

    # An interrupt request lands within a chunk; give it time to show up.
    deadline = time.time() + 15
    while time.time() < deadline:
        assert api.job_status(translation_id) != "interrupted", (
            "closing the tab interrupted the job")
        time.sleep(1)

    assert api.wait_for_terminal(translation_id) == "completed"
