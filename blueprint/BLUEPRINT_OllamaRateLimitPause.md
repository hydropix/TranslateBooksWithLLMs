# BLUEPRINT — Ollama: pause the translation on HTTP 429 instead of silently falling back to source text

Issue: [#279](https://github.com/hydropix/TranslateBooksWithLLMs/issues/279)
Status: ready for `/implement-blueprint`
Scope decided with the maintainer: **429 handling only**. Ollama API-key support and
multi-key rotation are explicitly **out of scope** (see "Out of scope" below).

---

## 1. Summary

When a user runs an Ollama Cloud model through their local Ollama daemon
(`gemma4:31b-cloud` and friends, authenticated on the daemon side via `ollama signin`),
the daemon relays the cloud quota error as a plain **HTTP 429**:

```
Status: 429
Error: you (<user>) have reached your monthly usage limit, upgrade for higher limits:
       https://ollama.com/upgrade or add usage credits: https://ollama.com/settings
```

`OllamaProvider` treats that 429 as a generic HTTP error: it retries twice, gives up, and
returns `None`. The pipeline then does what it does for any failed chunk — it keeps the
source text and moves on. The reporter's EPUB finished with **271 chunks left in the
source language** after a 2h38 run, with no pause and no actionable stop.

Every cloud provider in this repo already raises `RateLimitError` on an exhausted 429,
which the pipeline turns into a checkpointed auto-pause (or a timed auto-resume when the
user disabled auto-pause). Ollama is the only provider that never raises it.

**The fix is one branch in one file**: raise `RateLimitError` from `OllamaProvider` when
the server answers 429. Every consumer downstream already exists, is provider-agnostic,
and is already covered by the existing UI toggle.

---

## 2. Project grounding

### Stack

Python 3.11 async backend (httpx, Flask + Socket.IO), vanilla-JS frontend under
`src/web/static/js/`, pytest for tests. No framework magic in the LLM layer: providers are
plain classes deriving from `LLMProvider`.

### Conventions that constrain this change (from `CLAUDE.md` + surrounding code)

- **English only** in everything committed: code, comments, docstrings, commit messages,
  PR text, issue replies, docs.
- **No credentials anywhere.** This change introduces no key, no token, no new `.env`
  variable; if any phase feels like it needs one, the phase is wrong — stop and re-read
  the scope.
- **Frontend i18n rules do not apply**: this change adds no user-facing frontend string.
  The log lines it emits go through the existing backend `log_callback`, same as the
  neighbouring branches in the same file. Do **not** add a locale key.
- **No ad-hoc scripts at the repo root.** Tests go under `tests/unit/`.
- Provider code style in `src/core/llm/providers/ollama.py`: ANSI colour constants declared
  locally inside each `except` block, `self.log_callback(event_key, message)` when a
  callback exists, `print(...)` fallback otherwise. Match it exactly; do not refactor the
  surrounding handler.

### Verification gate (quote verbatim)

```bash
python -m pytest tests/unit -q
```

```bash
python -m pytest
```

Note for the implementer: `pytest.ini` already excludes `integration` and `e2e` markers by
default, so the full run needs no live Ollama server. **Two EPUB characterization goldens
are known-stale on `main`** and have been failing since v1.5.3, unrelated to this change.
The gate is therefore: *no new failures compared to a baseline run on the same tree*. Run
the baseline first (`git stash` or a run before editing) if in doubt.

### Sacred / high-risk surfaces involved

1. **`OllamaProvider.generate()` error handling** — the hot path of every local
   translation. A regression here breaks the default provider for every user. The change
   must be strictly additive: one new branch that only fires on `status_code == 429`.
2. **The pipeline pause/resume contract** (`RateLimitError` → checkpoint → `rate_limited`
   status). Already implemented and tested; this phase only becomes a new *producer* of
   that exception. Do not touch the consumers.
3. **Outward-facing communication** — the GitHub reply on issue #279 is published under
   the maintainer's account. Phase 5 covers it; it is the only phase with external
   side-effects.

### Companion docs that must stay in sync

`docs/API_KEY_ROTATION.md:168` currently states:

> **Ollama** (local) has no rate limits, so it has no key pool. Rotation is irrelevant.

The first half becomes false the moment Ollama relays cloud models. That line is corrected
in the same change as the behavior (Phase 3), not in a trailing docs pass.

---

## 3. Architecture / integration notes

### Exact integration points

| Where | Signature / contract |
|---|---|
| `src/core/llm/providers/ollama.py:236` | `async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT, system_prompt: Optional[str] = None) -> Optional[LLMResponse]` |
| `src/core/llm/providers/ollama.py:537` | `except httpx.HTTPStatusError as e:` — the handler to extend. Already extracts `error_message` from `e.response.json()["error"]`, falling back to `e.response.text`. |
| `src/core/llm/exceptions.py:31` | `RateLimitError(message, retry_after: int = None, provider: str = None, partial_result: Any = None)` |
| `src/core/llm/rate_limit_handler.py:46` | `compute_wait_time(headers: Mapping[str, str], attempt: int) -> int` — priority: `Retry-After` (s), then `X-RateLimit-Reset` (UTC ms), then exponential fallback `min(2 ** (attempt + 2), 60)` |
| `src/core/llm_client.py:65` | `await provider.generate(...)` — pass-through, catches nothing |
| `src/core/adapters/generic_translator.py:484`, `src/core/epub/translator.py:421`, `src/core/epub/translator.py:814`, `src/core/epub/xhtml_translator.py:377`, `src/core/common/plain_text_pipeline.py:522` | already re-raise `RateLimitError` instead of degrading to source text |
| `src/api/handlers.py:1093` | `except RateLimitError as e:` — auto-pause + checkpoint, or timed auto-resume when `config['auto_pause_on_rate_limit']` is false. Uses `e.retry_after` when set, else `RATE_LIMIT_AUTO_RESUME_DELAY` (60s, `src/config.py:308`). |

### Design decisions (made here, not by the implementer)

- **D1 — Raise immediately, do not retry.** Cloud providers retry/rotate because they own
  a key pool and their 429s are per-minute throttles. Ollama's is a monthly or credit
  quota; a 2-second retry cannot clear it and only produces the duplicated noise visible
  in the reporter's log. The new branch raises on the first 429, before the attempt loop
  sleeps.
- **D2 — Do not call `handle_rate_limit()`.** It requires a `KeyPool`, and `OllamaProvider`
  has none (no key). Construct and raise `RateLimitError` directly.
- **D3 — `retry_after` is passed only when the server actually hinted one.** If the 429
  carries `Retry-After` or `X-RateLimit-Reset`, use `compute_wait_time(headers, 0)`.
  Otherwise pass `None`, so `handlers.py` falls back to `RATE_LIMIT_AUTO_RESUME_DELAY`
  (60s). Rationale: `compute_wait_time`'s header-less fallback at `attempt=0` is 4 seconds,
  which would make auto-resume hammer an exhausted monthly quota. The existing
  "auto-resume looped 3 times without progress" warning (`handlers.py:1126`) then does its
  job on a sane cadence.
- **D4 — `provider="ollama"`.** It is what the pause message shows the user
  (`"⏸️ Rate limited by ollama."`). Do not derive it from the endpoint.
- **D5 — Scope is `status_code == 429` and nothing else.** No new handling for 401/402/403/404,
  no `is_retryable_http_status()` adoption in this provider. That is a separate, behavior-
  changing cleanup; keep this change reviewable.
- **D6 — No message-content matching.** Do not grep the body for "usage limit", "quota" or
  a URL. The status code is the contract; body text is provider-version-dependent and would
  rot.
- **D7 — Ordering inside the handler.** The 429 branch goes *before* the existing
  context-overflow keyword check (`"context" / "truncate" / "length" / "too long"`). A 429 is
  never a context overflow, and the quota message contains no such keyword today — but the
  ordering must not depend on that.

### Out of scope (decided, document in the issue reply)

- `OLLAMA_API_KEY` support, `Authorization: Bearer` headers, Ollama entries in
  `PROVIDER_ENV_VARS`, an Ollama key field in the settings UI, multi-key rotation for
  Ollama. Users who want to talk to `https://ollama.com` directly, or chain several
  accounts, can already do it today via the **OpenAI-compatible** provider (endpoint
  `https://ollama.com/v1`, key in the existing OpenAI key field), which gives them both key
  rotation and auto-pause.
- Version bump / release. `src/__version__.py` is bumped at release time only, and CI
  guards a tag/version mismatch.

---

## 4. Phases

### Phase 1 — Raise `RateLimitError` on HTTP 429 in `OllamaProvider`

| Field | Value |
|---|---|
| **Goal** | `OllamaProvider.generate()` surfaces a 429 as `RateLimitError` so the pipeline pauses and checkpoints instead of returning `None`. |
| **Files touched** | `src/core/llm/providers/ollama.py` |
| **Dependency** | Sequential (root phase) |
| **Risk surface** | **HIGH — sacred surface #1 and #2**: the default provider's hot error path, and a new producer of the pause contract. Strictly additive change; no existing branch may be reordered, reworded or removed apart from the insertion point described below. |

**Concrete deliverables**

1. Extend the imports at the top of the file:
   - `from ..exceptions import ContextOverflowError, RepetitionLoopError` → add `RateLimitError`.
   - add `from ..rate_limit_handler import compute_wait_time`.
2. Inside `except httpx.HTTPStatusError as e:` (currently line 537), **after** `error_message`
   has been extracted from the response and **before** the context-overflow keyword check,
   insert the 429 branch.

**Contract**

```
Input:  an httpx.HTTPStatusError raised by response.raise_for_status() inside generate()
Guard:  e.response is not None and e.response.status_code == 429

Behavior when the guard holds:
  headers     = e.response.headers
  hinted      = any of "Retry-After", "retry-after", "X-RateLimit-Reset", "x-ratelimit-reset"
                is present in headers
  retry_after = compute_wait_time(headers, 0) if hinted else None

  log (once, before raising), following the file's existing style:
    self.log_callback("llm_rate_limit", "<YELLOW>⚠️ Ollama rate limit (HTTP 429)…<RESET>")
    when self.log_callback is set, else the equivalent print(...)
    The message MUST include error_message verbatim (it carries the upgrade/credits URL
    the user needs) and MUST state that the translation is being paused.

  raise RateLimitError(error_message, retry_after=retry_after, provider="ollama")

Behavior when the guard does not hold: byte-for-byte the current behavior.

Invariants:
  I1  generate() never returns None for a 429; it raises.
  I2  the 429 path performs no asyncio.sleep and consumes no retry attempt: exactly one
      HTTP request is issued for a chunk whose first response is 429.
  I3  no API key, no KeyPool, no new config variable, no new locale key, no UI change.
  I4  non-429 HTTP statuses keep the existing retry-then-return-None behavior.
  I5  ContextOverflowError / RepetitionLoopError propagation is unchanged.
```

**Validation criteria**

- Phase 2's tests pass.
- `python -m pytest tests/unit -q` — no new failures.
- `python -m pytest` — no new failures versus a baseline run on the unmodified tree
  (2 EPUB characterization goldens are stale on `main`; they must be the *only* failures,
  and identical before/after).
- Manual read-through: `git diff src/core/llm/providers/ollama.py` shows only the two import
  lines and one contiguous inserted block.

**Ambiguity flag** — NONE.

---

### Phase 2 — Unit tests for the Ollama 429 contract

| Field | Value |
|---|---|
| **Goal** | Lock Phase 1's contract, including the "no retry, no None" regression that caused #279. |
| **Files touched** | `tests/unit/test_ollama_rate_limit.py` (new) |
| **Dependency** | Sequential (asserts Phase 1's behavior) |
| **Risk surface** | NONE |

**Concrete deliverables**

A pytest module following the conventions of `tests/unit/test_key_pool_and_rate_limit.py`
(`@pytest.mark.asyncio`, `monkeypatch`, local fake client classes, a module docstring
naming issue #279).

Test harness requirements, stated so they need not be inferred:

- `OllamaProvider.generate()` first calls `_detect_thinking_behavior()`. Bypass it by
  setting `provider._thinking_behavior = ThinkingBehavior.STANDARD` (import from
  `src.core.llm.thinking.behavior`) before calling `generate()`.
- The request is issued through `client.stream("POST", url, json=..., timeout=...)` used as
  an **async context manager**. The fake client must therefore return an object implementing
  `__aenter__` / `__aexit__`, whose yielded response exposes `is_error`, `await aread()`,
  `raise_for_status()`, `aiter_lines()`, `headers`, `status_code`, `json()`, `text`.
  Build the error case around a real `httpx.Response(429, headers=..., request=httpx.Request("POST", url))`
  so `raise_for_status()` raises a real `httpx.HTTPStatusError`.
- Patch the provider's `_get_client` with an async function returning the fake, as
  `TestRotationDoesNotConsumeAttempts._make_provider` does.

**Contract — the cases to cover**

```
T1  429 + {"Retry-After": "30"}
      -> pytest.raises(RateLimitError); exc.retry_after == 30; exc.provider == "ollama"
T2  429 with no rate-limit headers
      -> pytest.raises(RateLimitError); exc.retry_after is None
T3  429 whose body is {"error": "you (x) have reached your monthly usage limit, ..."}
      -> "monthly usage limit" in str(exc)          (the actionable text survives)
T4  429, counting requests issued
      -> exactly 1 stream() call (invariant I2: no retry burned, no sleep)
T5  500 on every attempt (regression guard for invariant I4)
      -> generate() returns None after MAX_TRANSLATION_ATTEMPTS calls, raises nothing
T6  200 happy path with a minimal streamed NDJSON body
      -> returns an LLMResponse; asserts the new branch did not hijack the success path
```

**Validation criteria**

- `python -m pytest tests/unit/test_ollama_rate_limit.py -q` passes.
- Each test fails on a tree where Phase 1 is reverted (T1–T4 especially). The implementer
  must verify this explicitly by stashing Phase 1 once, running the module, and restoring —
  a test that passes without the fix is worthless here.
- `python -m pytest tests/unit -q` — no new failures.

**Ambiguity flag** — NONE.

---

### Phase 3 — Documentation sync

| Field | Value |
|---|---|
| **Goal** | Stop the docs from claiming Ollama cannot be rate-limited, and tell users what now happens. |
| **Files touched** | `docs/API_KEY_ROTATION.md`, `docs/PROVIDERS.md` |
| **Dependency** | Parallelizable (touches neither `ollama.py` nor the test file) |
| **Risk surface** | NONE |

**Concrete deliverables**

1. `docs/API_KEY_ROTATION.md`, in "When rotation is skipped", replace the line
   `- **Ollama** (local) has no rate limits, so it has no key pool. Rotation is irrelevant.`
   with a correction that says, in English, in the document's existing voice:
   - a purely local Ollama model has no quota, so Ollama has no key pool and rotation is
     irrelevant;
   - a `-cloud` model relayed by the local Ollama daemon *can* return HTTP 429 when the
     account's usage limit is reached, and TBL now pauses the translation and checkpoints it
     instead of continuing (there is still no key rotation for Ollama);
   - users who want several Ollama accounts chained should use the OpenAI-compatible
     provider against `https://ollama.com/v1`, which does support multi-key rotation.
2. `docs/PROVIDERS.md`, in the `## Ollama (Local)` section (line 7), add a short subsection
   — suggested title `### Cloud models` — stating that `-cloud` models are relayed by the
   local daemon using the account signed in with `ollama signin`, that hitting the account
   quota produces HTTP 429, and that TBL pauses and checkpoints the job so it can be resumed
   after the quota resets or credits are added. Mention the existing
   "Don't auto-pause on rate limit" setting as the opt-out that waits and auto-resumes instead.

**Contract**

```
- English only. No new .env variable, no key, no UI reference that does not already exist.
- Do not document an OLLAMA_API_KEY: it does not exist and this change does not add it.
- Do not promise key rotation for Ollama.
- Keep both files' existing heading levels, tone and table formatting.
```

**Validation criteria**

- `grep -n "has no rate limits" docs/API_KEY_ROTATION.md` returns nothing.
- `grep -rn "OLLAMA_API_KEY" docs/` returns nothing.
- Both files still render as valid Markdown (headings nest correctly, tables intact).

**Ambiguity flag** — NONE.

---

### Phase 4 — Delivery: branch, commit, PR

| Field | Value |
|---|---|
| **Goal** | Land phases 1–3 as one reviewable PR against `main`. |
| **Files touched** | none (git operations only) |
| **Dependency** | Sequential (after 1, 2, 3) |
| **Risk surface** | **Outward-facing** — pushing a branch and opening a PR is public. The implementer MUST show the diff summary and the proposed commit message to the maintainer and get an explicit go-ahead before `git push`. |

**Concrete deliverables**

- Branch `fix/ollama-429-auto-pause` created from `main`.
- One commit, English, conventional-commit style matching recent history
  (`fix(ollama): ...`), body explaining the 429-to-source-text failure mode and referencing
  `#279`, ending with the attribution line required by this session:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- PR against `main`, English title and description, description ending with
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Pre-commit secret check per `CLAUDE.md`: confirm the diff contains no key-shaped literal.
  (Expected: it contains none; the diff is an exception branch, tests and docs.)

**Validation criteria**

- `git diff main --stat` lists exactly: `src/core/llm/providers/ollama.py`,
  `tests/unit/test_ollama_rate_limit.py`, `docs/API_KEY_ROTATION.md`, `docs/PROVIDERS.md`,
  and this blueprint file if the maintainer wants it committed.
- Verification gate re-run on the branch before pushing.
- PR URL reported back to the maintainer.

**Ambiguity flag** — Whether to commit `blueprint/BLUEPRINT_OllamaRateLimitPause.md` itself
is the maintainer's call; ask, do not decide.

---

### Phase 5 — Reply on issue #279

| Field | Value |
|---|---|
| **Goal** | Answer the reporter: what was actually broken, what is fixed, and why the API-key half of the request is not being implemented. |
| **Files touched** | none (GitHub comment) |
| **Dependency** | Sequential (after Phase 4, so the reply can reference the PR) |
| **Risk surface** | **Outward-facing, sacred surface #3.** Published under the maintainer's account. The orchestrator posts it — never a sub-agent — and shows the full text in chat first. The maintainer already authorized posting directly. |

**Concrete deliverables**

One comment posted with:

```bash
gh issue comment 279 --repo hydropix/TranslateBooksWithLLMs --body-file <file>
```

**Contract — what the reply must and must not contain**

```
MUST (content):
  - confirm the diagnosis from the reporter's own log: a plain HTTP 429 relayed by the
    local daemon for the -cloud model, which TBL retried twice and then degraded to source
    text, which is what left 271 chunks untranslated;
  - state that Ollama now raises the same rate-limit signal as the cloud providers, so the
    job pauses and checkpoints and can be resumed once the quota resets or credits are
    added, and that the existing "don't auto-pause on rate limit" setting still applies;
  - explain that the API-key half is not needed for their setup, since the daemon is
    already authenticated, and that chaining several Ollama accounts is already possible
    today through the OpenAI-compatible provider pointed at https://ollama.com/v1 with the
    keys in the existing API key field, which also gives them rotation;
  - reference the PR from Phase 4.

MUST (form) — the maintainer's standing style rule for human-facing replies:
  - English;
  - plain prose, flowing paragraphs;
  - no em-dashes, no bullet lists, no emojis, no section headers;
  - no invented facts: do not promise a release date, do not claim the reporter's file will
    be re-translated automatically, do not state a version number that has not been tagged.

MUST NOT:
  - paste any key, token or endpoint containing a secret;
  - close the issue (leave that to the maintainer once the PR merges).
```

**Validation criteria**

- The draft is shown in chat before posting.
- After posting, the comment URL is reported back.
- Re-read the posted text against the four form rules above; a stray bullet or em-dash is a
  defect to fix by editing the comment.

**Ambiguity flag** — NONE.

---

## 5. Risks and open questions

- **R1 — Local Ollama returning 429 for a non-quota reason.** A reverse proxy or a gateway
  in front of Ollama could emit 429 under load. With this change the job pauses instead of
  retrying twice. That is still the better failure mode (the work is checkpointed, nothing
  is silently untranslated), and users who prefer to keep going already have the
  "don't auto-pause on rate limit" toggle, which waits 60s and resumes from the checkpoint.
  Accepted, not mitigated further.
- **R2 — No `Retry-After` from Ollama.** Expected; D3 handles it by falling back to the
  pipeline's own 60s delay, and the existing stuck-loop warning covers the case where the
  quota will not reset for hours.
- **R3 — Stale EPUB characterization goldens on `main`** will make a naive "all green" check
  fail. The gate is explicitly "no *new* failures"; the implementer must baseline first.
- **Q1 — Should the blueprint file itself be committed?** Maintainer's call (Phase 4).

---

## 6. Effort estimate

Phase 1 is roughly fifteen lines. Phase 2 is the real work, perhaps sixty lines of fake
client. Phases 3 to 5 are text. No UI, no i18n, no config, no migration.
