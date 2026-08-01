"""Issue #224 — desync recovery must actually resync the UI.

`checkStateConsistency` used to react by dispatching `translationUpdate` and
`resetUIToIdle` CustomEvents that nothing listened to, so after a WebSocket
drop the tab stayed stuck on a job the server had already finished. The
handlers are now injected from index.js and called directly.

`tests/unit/test_no_dead_custom_events.py` guards the static half of this
(no dispatch without a listener); this test covers the behaviour.
"""
import time

import pytest

from .helpers import TERMINAL_STATUSES, queue_files, start_batch, state, ui

pytestmark = pytest.mark.e2e


def _instrument_desync_handlers(page):
    """Count handler invocations so a resync can be attributed to this path.

    Without it a socket reconnect could resync the UI too, and the test would
    pass for the wrong reason.
    """
    page.evaluate("""async () => {
        const lm = await import('/static/js/utils/lifecycle-manager.js');
        const wired = lm.LifecycleManager._desyncHandlers;
        if (!wired) throw new Error('index.js never called setDesyncHandlers');
        window.__desyncCalls = { terminal: 0, missing: 0 };
        lm.LifecycleManager.setDesyncHandlers({
            onTerminalStatus: (data) => {
                window.__desyncCalls.terminal++;
                return wired.onTerminalStatus(data);
            },
            onJobMissing: () => {
                window.__desyncCalls.missing++;
                return wired.onJobMissing();
            },
        });
    }""")


def test_desync_handlers_are_wired_at_startup(page):
    """The composition root must inject the handlers before anything can fire."""
    assert page.evaluate("""async () => {
        const lm = await import('/static/js/utils/lifecycle-manager.js');
        const h = lm.LifecycleManager._desyncHandlers;
        return !!h && typeof h.onTerminalStatus === 'function'
                   && typeof h.onJobMissing === 'function';
    }""")


def test_ui_resyncs_after_a_socket_drop(page, context, input_files, api):
    """Kill the socket, let the job finish, come back: the UI must catch up."""
    _instrument_desync_handlers(page)

    queue_files(page, [input_files["long"]])
    translation_id = start_batch(page)
    assert translation_id, "no job id was registered"
    page.wait_for_timeout(2000)

    context.set_offline(True)  # drops the WebSocket
    deadline = time.time() + 300
    while time.time() < deadline and api.job_status(translation_id) not in TERMINAL_STATUSES:
        time.sleep(2)
    assert api.job_status(translation_id) in TERMINAL_STATUSES, "the job never finished"

    offline_ui = ui(page)
    assert offline_ui["interrupt"] is True and offline_ui["progress"] is True, (
        f"the desync was not reproduced: {offline_ui}")

    context.set_offline(False)
    page.wait_for_timeout(1500)
    # Emulation.setPageVisibilityState is unavailable in the headless shell, so
    # fire the event the app actually listens for. The 10s interval in
    # startStateConsistencyCheck reaches the same code path on its own.
    page.evaluate("() => document.dispatchEvent(new Event('visibilitychange'))")
    page.wait_for_timeout(8000)

    calls = page.evaluate("() => window.__desyncCalls")
    final_ui, final_state = ui(page), state(page)

    assert calls and calls["terminal"] >= 1, (
        f"the recovery handler never fired, so the resync is not attributable "
        f"to the desync path: {calls}")
    assert final_ui["interrupt"] is False and final_ui["progress"] is False, final_ui
    assert final_ui["cards"] >= 1, "no completion card was rendered by the recovery"
    assert final_state["currentJob"] is None, final_state
