# End-to-end tests

These tests run the whole stack: a real server process, a real LLM, and — for
the frontend ones — a real Chromium via Playwright. They exist because the
project has no JavaScript test runner, so behaviour like "the tab must ignore
another tab's terminal event" had no automated coverage at all.

They are **deselected by default** (see `pytest.ini`), because they are slow
and they spend real API tokens.

## Running them

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium

pytest tests/e2e -m e2e
```

Individual files:

```bash
pytest tests/e2e/test_job_ownership.py -m e2e          # issue #225
pytest tests/e2e/test_desync_recovery.py -m e2e        # issue #224
pytest tests/e2e/test_completion_classification.py -m e2e
```

## Requirements

| Requirement | Effect when missing |
| --- | --- |
| `playwright` + its Chromium | the whole directory skips |
| `GEMINI_API_KEY` (env or `.env`) | the whole directory skips |
| a free port (default `5099`) | the whole directory skips |

Nothing here fails for a missing prerequisite — it skips, so a normal
`pytest` run on a machine without a key stays green.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `E2E_PORT` | `5099` | port the test server binds to |
| `E2E_MODEL` | `gemini-2.5-flash` | model used for every run |
| `E2E_HEADED` | unset | set to `1` to watch the browser |

## What each file covers

- **`test_job_ownership.py`** — issue #225. A foreign terminal event must never
  reset the batch UI; an owned one still must. Covers a three-file batch, a
  mid-job reload, and the resume path.
- **`test_desync_recovery.py`** — issue #224. After a WebSocket drop, the
  visibility/interval consistency check must resync the UI without a reload.
  The injected handlers are instrumented so a passing resync cannot be
  attributed to a socket reconnect instead.
- **`test_completion_classification.py`** — no browser. Every format must still
  end `completed` on a clean run, a run whose chunks fell back to source text
  must end `partial` with its checkpoint kept, and TXT reassembly must preserve
  paragraph breaks across chunk seams (issue #208).

## How the fixtures work

`conftest.py` starts the server itself, on its own port, through
`_server_runner.py` (which skips `start_server()` so no browser tab opens) and
reads the per-process API token the runner writes to a temp file.

Two autouse fixtures clean up after the run: one deletes any checkpoint a test
created (and only those — a developer's own in-flight job in the same database
is never touched), the other removes the output, upload and thumbnail files the
session produced. The file cleanup is a before/after diff, because uploads and
thumbnails are named by content hash — so a translation started by hand *while
the suite runs* would be swept up too. Run e2e on a quiet instance.

## Writing new ones

`helpers.py` holds the page-driving primitives. Two things to know:

1. The SPA hides its `<select>`s behind a SearchableSelect widget, so provider
   and model are set on the underlying element plus an explicit `change` event,
   not through `select_option`.
2. `import()` of the same absolute URL the SPA imported returns the same module
   instance, so tests observe and drive the app's live singletons rather than
   copies. That is how `state()` and `emit_translation_update()` work.

Two traps worth remembering, both of which produced false failures while these
tests were written:

- The translate button is disabled whenever the queue is empty, so it is not a
  usable "is the UI idle?" signal. Assert on `progressSection` and
  `interruptBtn` instead.
- Anything testing a mid-job window needs the `long` input. With a short file
  the job finishes before the reload completes, `reconcileStateWithServer`
  correctly resets the UI, and the test fails for the wrong reason.
