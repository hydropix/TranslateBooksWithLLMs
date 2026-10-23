/**
 * i18n bootstrap — initializes i18next (loaded via CDN UMD) and exposes
 * helpers used across modules. Designed to be imported once from index.js
 * before any module that renders user-facing text.
 *
 * The i18next + i18nextHttpBackend globals are expected to be present on
 * window (see <script> tags in translation_interface.html).
 */

export const SUPPORTED_LOCALES = ['en', 'fr', 'es', 'de', 'zh-CN', 'ja', 'ko'];
export const NAMESPACES = ['common', 'translation', 'settings', 'glossary', 'files', 'tts', 'errors', 'sample', 'style', 'ask'];

let ready = false;
let readyPromise = null;

function resolveInitialLocale() {
    const initial = window.__INITIAL_LOCALE__;
    if (initial && SUPPORTED_LOCALES.includes(initial)) {
        return initial;
    }
    return 'en';
}

/**
 * Initialize i18next. Safe to call multiple times — subsequent calls return
 * the same promise. Awaits the underlying i18next.init().
 */
export function initI18n() {
    if (readyPromise) return readyPromise;

    if (typeof window.i18next === 'undefined' || typeof window.i18nextHttpBackend === 'undefined') {
        console.warn('[i18n] i18next or i18nextHttpBackend not loaded — using identity translator');
        ready = true;
        readyPromise = Promise.resolve();
        return readyPromise;
    }

    const version = window.__APP_VERSION__ || 'dev';
    const initial = resolveInitialLocale();

    readyPromise = window.i18next
        .use(window.i18nextHttpBackend)
        .init({
            lng: initial,
            fallbackLng: 'en',
            supportedLngs: SUPPORTED_LOCALES,
            ns: NAMESPACES,
            defaultNS: 'common',
            backend: {
                loadPath: `/static/locales/{{lng}}/{{ns}}.json?v=${encodeURIComponent(version)}`,
            },
            interpolation: { escapeValue: false },
            returnEmptyString: false,
            load: 'currentOnly',
        })
        .then(() => {
            ready = true;
            applyToDOM(document.body);
            window.i18next.on('languageChanged', () => {
                document.documentElement.setAttribute('lang', window.i18next.language);
                applyToDOM(document.body);
                window.dispatchEvent(new CustomEvent('localeChanged', {
                    detail: { locale: window.i18next.language }
                }));
            });
        })
        .catch((err) => {
            console.error('[i18n] init failed', err);
            ready = true; // mark as ready so t() degrades to the key
        });

    return readyPromise;
}

/**
 * Translate a key. Falls back to the key itself if i18next isn't ready
 * (e.g. during very early boot or if the CDN failed to load).
 */
export function t(key, options) {
    if (ready && window.i18next && typeof window.i18next.t === 'function') {
        return window.i18next.t(key, options);
    }
    if (options && typeof options === 'object' && 'defaultValue' in options) {
        return options.defaultValue;
    }
    return key;
}

/**
 * Change UI locale. Persists via /api/ui-locale (cookie), then asks i18next
 * to reload — i18next emits 'languageChanged' which re-applies the DOM.
 */
export async function setLocale(locale) {
    if (!SUPPORTED_LOCALES.includes(locale)) {
        throw new Error(`Unsupported locale: ${locale}`);
    }
    try {
        await fetch('/api/ui-locale', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ locale }),
            credentials: 'same-origin',
        });
    } catch (err) {
        console.warn('[i18n] failed to persist locale', err);
    }
    if (window.i18next && typeof window.i18next.changeLanguage === 'function') {
        await window.i18next.changeLanguage(locale);
    }
}

export function getCurrentLocale() {
    if (window.i18next && window.i18next.language) {
        return window.i18next.language;
    }
    return resolveInitialLocale();
}

/**
 * Apply translations to every element under `root` carrying a data-i18n
 * (text content) or data-i18n-attr (attributes like placeholder/title/aria-label)
 * marker. Call after dynamic DOM insertions.
 *
 * data-i18n            => element textContent
 * data-i18n-html       => element innerHTML (use sparingly — keys must be trusted)
 * data-i18n-attr       => "attr1:key1;attr2:key2" pairs
 */
export function applyToDOM(root) {
    if (!root || typeof root.querySelectorAll !== 'function') return;

    root.querySelectorAll('[data-i18n]').forEach((el) => {
        const key = el.getAttribute('data-i18n');
        if (!key) return;
        const params = parseParams(el.getAttribute('data-i18n-params'));
        el.textContent = t(key, params);
    });

    root.querySelectorAll('[data-i18n-html]').forEach((el) => {
        const key = el.getAttribute('data-i18n-html');
        if (!key) return;
        const params = parseParams(el.getAttribute('data-i18n-params'));
        el.innerHTML = t(key, params);
    });

    root.querySelectorAll('[data-i18n-attr]').forEach((el) => {
        const spec = el.getAttribute('data-i18n-attr');
        if (!spec) return;
        const params = parseParams(el.getAttribute('data-i18n-params'));
        spec.split(';').forEach((pair) => {
            const idx = pair.indexOf(':');
            if (idx < 0) return;
            const attr = pair.slice(0, idx).trim();
            const key = pair.slice(idx + 1).trim();
            if (attr && key) {
                el.setAttribute(attr, t(key, params));
            }
        });
    });
}

function parseParams(raw) {
    if (!raw) return undefined;
    try {
        return JSON.parse(raw);
    } catch {
        return undefined;
    }
}

/** Convenience: await initI18n then call cb. Useful for late-bound modules. */
export function whenReady(cb) {
    initI18n().then(cb);
}
