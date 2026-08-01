"""Page-driving helpers shared by the e2e tests.

The SPA hides its <select>s behind a SearchableSelect widget, so provider and
model are set on the underlying element plus an explicit change event rather
than through Playwright's select_option.

Reaching the app's live singletons: `import()` of the same absolute URL the SPA
imported resolves to the same module instance (the ES module registry is keyed
by resolved URL), so these helpers observe and drive the real objects, not
copies.
"""
from pathlib import Path

TERMINAL_STATUSES = ("completed", "partial", "error", "interrupted")

# The translate button is disabled whenever the queue is empty
# (batch-controller.js), so idleness is asserted on the progress panel and the
# interrupt button, never on the button's disabled flag.
UI_SNAPSHOT_JS = """() => {
    const visible = (id) => {
        const el = document.getElementById(id);
        return el ? getComputedStyle(el).display !== 'none' : null;
    };
    return {
        progress: visible('progressSection'),
        interrupt: visible('interruptBtn'),
        translateDisabled: document.getElementById('translateBtn')?.disabled,
        cards: document.getElementById('completionCardsContainer')?.children.length ?? 0,
    };
}"""

TRANSLATION_STATE_JS = """async () => {
    const sm = await import('/static/js/core/state-manager.js');
    const t = sm.StateManager.getState('translation');
    return {
        currentJob: t.currentJob?.translationId ?? null,
        ownedJobIds: t.ownedJobIds,
        isBatchActive: t.isBatchActive,
        queue: (sm.StateManager.getState('files.toProcess') || []).map(f => f.status),
    };
}"""


def _paragraph(n):
    return (f"Paragraph {n}. The quick brown fox jumps over the lazy dog while "
            "the patient translator carefully preserves the meaning of every clause.")


def build_input_files(dest_dir: Path) -> dict:
    """Write the TXT inputs the suite uses and return them by role.

    `short` files finish in a couple of seconds (batch progression); `long` is
    big enough to still be running several seconds in, which is what makes the
    mid-job reload and the offline window reachable.
    """
    dest_dir = Path(dest_dir)
    files = {"short": [], "long": None}

    for i in (1, 2, 3):
        path = dest_dir / f"batch{i}.txt"
        path.write_text("\n\n".join(_paragraph(n) for n in range(1, 4)), encoding="utf-8")
        files["short"].append(path)

    long_path = dest_dir / "long.txt"
    long_path.write_text("\n\n".join(_paragraph(n) for n in range(1, 201)), encoding="utf-8")
    files["long"] = long_path

    return files


def open_app(page, base_url, provider=None, model=None):
    """Load the SPA and select the LLM provider and model."""
    from .conftest import E2E_MODEL, E2E_PROVIDER

    page.goto(base_url, wait_until="networkidle")
    page.wait_for_timeout(1500)

    page.evaluate("""(provider) => {
        const el = document.getElementById('llmProvider');
        el.value = provider;
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }""", provider or E2E_PROVIDER)
    # The model list is fetched from the provider after the change event.
    page.wait_for_timeout(3500)

    page.evaluate("""(model) => {
        const el = document.getElementById('model');
        el.value = model;
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }""", model or E2E_MODEL)
    page.wait_for_timeout(500)


def ui(page):
    """Snapshot of the parts of the UI these tests assert on."""
    return page.evaluate(UI_SNAPSHOT_JS)


def state(page):
    """Snapshot of the app's translation state slice."""
    return page.evaluate(TRANSLATION_STATE_JS)


def queue_files(page, paths):
    page.set_input_files("#fileInput", [str(p) for p in paths])
    page.wait_for_timeout(2500)


def start_batch(page, timeout_ms=20000):
    """Click Translate and return the first job id the tab registers."""
    page.click("#translateBtn")
    waited = 0
    while waited < timeout_ms:
        translation_id = state(page)["currentJob"]
        if translation_id:
            return translation_id
        page.wait_for_timeout(500)
        waited += 500
    return None


def emit_translation_update(page, translation_id, status):
    """Deliver a translation_update to the tracker as the socket would."""
    page.evaluate("""async ([translationId, status]) => {
        const m = await import('/static/js/translation/translation-tracker.js');
        m.TranslationTracker.handleTranslationUpdate({
            translation_id: translationId, status: status,
        });
    }""", [translation_id, status])


def register_owned_job(page, translation_id):
    page.evaluate("""async (translationId) => {
        const m = await import('/static/js/translation/translation-tracker.js');
        m.TranslationTracker.registerOwnedJob(translationId);
    }""", translation_id)


def force_busy_ui(page):
    """Put the UI in a mid-batch look without spending a real translation."""
    page.evaluate("""async () => {
        const sm = await import('/static/js/core/state-manager.js');
        const dh = await import('/static/js/ui/dom-helpers.js');
        sm.StateManager.setState('translation.isBatchActive', true);
        dh.DomHelpers.show('progressSection');
        dh.DomHelpers.show('interruptBtn');
    }""")
    page.wait_for_timeout(300)
