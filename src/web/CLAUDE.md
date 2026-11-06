# CLAUDE.md

Instructions for Claude Code when working under `src/web/`.

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

