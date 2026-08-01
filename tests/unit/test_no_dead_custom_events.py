"""Guard against dispatching CustomEvents that nothing listens to.

`window.dispatchEvent(new CustomEvent('X'))` fails silently when no
`addEventListener('X')` exists: the call succeeds, the intended side effect
never happens, and nothing is logged. Issue #224 is exactly that failure mode,
so the whole class is covered by a static scan of the frontend sources.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
JS_DIR = ROOT / "src" / "web" / "static" / "js"

DISPATCH_RE = re.compile(r"""dispatchEvent\(\s*new\s+CustomEvent\(\s*['"]([A-Za-z0-9_]+)['"]""")
LISTENER_RE = re.compile(r"""addEventListener\(\s*['"]([A-Za-z0-9_]+)['"]""")

# Escape hatch, not a parking lot: an event belongs here only when its
# listener genuinely lives outside `src/web/static/js` (inline template
# script, browser extension, third-party embed) or when it is a known dead
# dispatch already filed for triage with a TODO below. Adding an entry to
# silence a new failure without triaging it defeats the purpose of the test.
KNOWN_DEAD_EVENTS = frozenset({
    # TODO(ttsChanged): dispatched at src/web/static/js/ui/form-manager.js:316
    # when the TTS checkbox toggles; no listener anywhere. Pre-existing, found
    # by this test, out of scope for issue #224 — triage separately (wire a
    # listener or drop the dispatch).
    "ttsChanged",
    # TODO(formReset): dispatched at src/web/static/js/ui/form-manager.js:672
    # after the form is reset; no listener anywhere. Same triage as above.
    "formReset",
})


def _iter_js_files():
    for path in sorted(JS_DIR.rglob("*.js")):
        if "vendor" in path.parts:
            continue
        yield path


def test_every_dispatched_custom_event_has_a_listener():
    """Every window.dispatchEvent(new CustomEvent('X')) under
    src/web/static/js must have at least one addEventListener('X') in the same
    tree. Issue #224: the desync recovery path dispatched 'translationUpdate'
    and 'resetUIToIdle', which nothing listened to, so recovery was a no-op."""
    assert JS_DIR.is_dir(), f"JS directory not found at {JS_DIR}"

    dispatched: dict[str, list[str]] = {}
    listened: set[str] = set()

    for js_path in _iter_js_files():
        src = js_path.read_text(encoding="utf-8")
        rel = js_path.relative_to(ROOT).as_posix()

        for match in DISPATCH_RE.finditer(src):
            line = src.count("\n", 0, match.start()) + 1
            dispatched.setdefault(match.group(1), []).append(f"{rel}:{line}")

        listened.update(match.group(1) for match in LISTENER_RE.finditer(src))

    dead = set(dispatched) - listened - KNOWN_DEAD_EVENTS

    details = "\n".join(
        f"  - '{event}' dispatched at {', '.join(dispatched[event])} but never listened to"
        for event in sorted(dead)
    )
    assert not dead, (
        "CustomEvent(s) dispatched with no addEventListener anywhere under "
        f"{JS_DIR.relative_to(ROOT).as_posix()}:\n{details}\n"
        "Either wire a listener, call the target module directly, or delete "
        "the dispatch. Do not add the event to KNOWN_DEAD_EVENTS without "
        "triaging it first."
    )
