# CLAUDE.md

Instructions for Claude Code when working in this repo.

## Language policy

Everything that lives on GitHub must be written in English: code, identifiers, code comments, docstrings, commit messages, PR titles and descriptions, issue text, documentation files (README, CLAUDE.md, etc.), and user-facing strings in committed files.

Replies to the user in chat may follow the language the user writes in (typically French), but any artifact that ends up committed to the repo must be English-only. If the user asks for a comment, doc, or commit message in another language, push back and write it in English.

## Security rules — API keys and secrets

**Absolutely forbidden** to hardcode an API key, token, password, URL containing a secret, or any other credential in a source file, a test, an ad-hoc script, a comment, a doc example, or a commit message. No exceptions, even temporary, even "just for debugging".

**Always** load credentials from `.env` (already gitignored) through the project's existing mechanism. If a variable is missing from `.env`, do not fabricate it or invent a value — ask the user, or fail cleanly with a clear message.

**For examples** in docs or tests that need to illustrate a key format, use an explicit, inert placeholder: `YOUR_API_KEY_HERE`, `sk-xxxxxxxx`, `<REDACTED>`. Never a real key, even expired or disabled — git history keeps it anyway.

**Before each commit**, if the diff includes a literal that looks like a key (long base64/hex string, prefix `sk-`/`Bearer`/`xoxp-`/`ghp_`/`AKIA`/etc., or a variable named `api_key`/`token`/`secret` with a non-empty value), stop and alert the user before proposing the commit.

**If a key is found in existing code** (old hardcoding, forgotten test file), flag it immediately to the user, recommend revoking it on the provider side, and propose moving it to `.env` plus removing the literal.

## Frontend i18n rules — every user-facing string must be reactive

Every user-facing string in the web frontend (`src/web/templates/`, `src/web/static/js/`) MUST be translatable AND update immediately when the user switches the UI language. The i18n machinery lives in `src/web/static/js/i18n/i18n.js`; locales live under `src/web/static/locales/<lang>/<namespace>.json` and must stay in sync across all 7 supported locales (`en`, `fr`, `es`, `de`, `zh-CN`, `ja`, `ko`).

**For static HTML strings**, use one of the i18n marker attributes — never write raw human text:

- `data-i18n="ns:key"` — sets `textContent`
- `data-i18n-html="ns:key"` — sets `innerHTML` (use sparingly; the key value must be trusted)
- `data-i18n-attr="placeholder:ns:key;title:ns:key"` — sets attributes (`placeholder`, `title`, `aria-label`, `alt`, optgroup `label`, …)
- `data-i18n-params='{"name":"foo"}'` — interpolation params

`applyToDOM(document.body)` re-runs on the `languageChanged` event and re-translates every marked element — so a `data-i18n*` attribute is enough to be reactive.

**For JS-rendered strings**, always wrap the literal with `t('ns:key', { params })` AND make sure the surrounding render runs again on locale switch. The `t()` call returns the current locale's string at call time — calling it once at module load freezes the string at boot. Two correct patterns:

1. The render function listens to the `localeChanged` window event:

   ```js
   window.addEventListener('localeChanged', () => render());
   ```

2. The injected DOM carries `data-i18n*` attributes (in which case `applyToDOM` handles it automatically — no explicit listener needed). Helper example: `setPlaceholderOption()` in `provider-manager.js`.

**Forbidden patterns** (any of these is a bug that must be fixed in the same PR):

- A raw English (or any-language) string literal written to `textContent`, `innerHTML`, `insertAdjacentHTML`, `.title`, `.placeholder`, `.setAttribute('aria-label'|'title'|'placeholder'|'alt', …)`, or passed to `alert()` / `confirm()` / a toast / `MessageLogger.showMessage` / `MessageLogger.addLog`.
- A `t(...)` call cached into a `const`/module-scope variable at module load, then reused forever.
- A `data-i18n` key that doesn't exist in **all 7** locale files.
- A translation that drops a `{{placeholder}}` token present in the English source — interpolation keys must match across locales.

**When adding a new locale key**: add it to all 7 `<lang>/<namespace>.json` files in the same commit. The English version goes first and is the source of truth; the other six are translations of it.

**Before declaring an i18n change done**, smoke-test it by toggling the UI language selector and confirming the string updates without a page reload. If it doesn't, the surrounding render is not reactive — fix it before merging.

## Script organization rules

No ad-hoc test or debug scripts at the repo root. The patterns `check_*.py`, `test_*.py`, `debug_*.py`, `scratch_*.py`, `tmp_*.py` at the root are gitignored — this is intentional.

The real entrypoints at the root are `translate.py`, `translation_api.py`, `launcher.py`. Real tests live under `tests/` (pytest) or `tests/standalone/` (manual scripts). If an ad-hoc script becomes useful long-term, move it into `tests/standalone/` or `scripts/` before committing it.

## Prompt change rules — every addition costs the rest of the prompt

Changes to LLM prompts (`src/prompts/`, and any instruction text injected into translation, refinement, glossary or style prompts) do not behave like ordinary code. A system prompt is a shared attention budget: each new instruction, example or warning dilutes the weight of every other one. A fix that solves one failure case can silently degrade quality elsewhere (formatting, tag preservation, tone, glossary adherence, output completeness), especially on small local models.

Rules:

- **Do not patch a prompt by appending a new instruction for each bug.** First ask whether the problem can be solved in code (pre/post-processing, validation, retry, chunking) instead of in the prompt. Prefer the code fix when it exists.
- **When a prompt change is necessary, keep it minimal.** Rephrase or merge with an existing instruction rather than adding a new one. Remove or shorten something if you add something. Never let the prompt grow without a reason that is worth its cost.
- **Prove there is no regression.** A prompt change is not done because the targeted case now works. The primary tool is the prompt regression harness (`tests/prompt_eval/README.md`): run `python -m tests.prompt_eval diff` first to see which corpus cases the change touches, then `python -m tests.prompt_eval run` with at least one small local model (where dilution effects show up first), and paste the generated `report.md` into the PR. Also run the existing prompt tests (`tests/unit/test_plain_text_prompt_structure.py`, `tests/unit/style/`, `tests/unit/glossary/`, and related) and the real-LLM integration tests (`pytest tests/integration/test_real_llm_translation.py -v`, uses `.env`).
- **Report the evidence.** In the PR description, state which tests and samples were run before and after, and what changed in the outputs. If the regression check could not be run (no LLM available, etc.), say so explicitly instead of claiming the change is safe.
