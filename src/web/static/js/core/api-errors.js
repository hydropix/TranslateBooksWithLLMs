/**
 * Map known API error responses to localized, readable messages.
 *
 * ApiClient errors carry `status` (HTTP status), `code` (body.error_code) and
 * `body` (parsed JSON body, or null for a non-JSON answer such as Werkzeug's
 * HTML 413 page). Matching uses the status and the stable body fields, never
 * the localized UI text. Anything unknown falls back to the server message.
 */

import { t } from '../i18n/i18n.js';

// Stable `error` labels returned by the server's credential guard
// (src/api/services/credential_guard.py).
const GUARD_OVERRIDE_NEEDS_KEY = 'Endpoint override requires its own API key';
const GUARD_ENDPOINT_NOT_ALLOWED = 'Endpoint not allowed';

function bodyField(err, name) {
    const body = err && err.body;
    return body && typeof body === 'object' && typeof body[name] === 'string' ? body[name] : '';
}

/**
 * Localized message for an API error, or `null` when it is not a known case.
 * @param {Error & {status?: number, code?: string, body?: object|null}} err
 * @param {{context?: 'open'|'resume'}} [options] - Call-site hints for
 *   statuses whose meaning depends on the route (403, 409).
 * @returns {string|null}
 */
export function knownApiErrorMessage(err, options = {}) {
    if (!err) return null;
    const status = err.status;
    const label = bodyField(err, 'error');
    const message = bodyField(err, 'message');

    if (status === 413) {
        return t('errors:request_too_large');
    }

    if (err.code === 'endpoint_rejected') {
        // Settings save: the body's `error` holds the guard's explanation.
        if (/own (api )?key/i.test(label) || /stored key will not be used/i.test(label)) {
            return t('errors:endpoint_override_needs_key');
        }
        return t('errors:endpoint_rejected', { detail: label });
    }

    if (label === GUARD_OVERRIDE_NEEDS_KEY
        || (status === 400 && /stored key will not be used/i.test(label || message))) {
        // Translate, sample columns, glossary and custom-instruction routes.
        return t('errors:endpoint_override_needs_key');
    }
    if (label === GUARD_ENDPOINT_NOT_ALLOWED) {
        return t('errors:endpoint_rejected', { detail: message || label });
    }

    if (status === 409 && options.context === 'resume') {
        if (/already running/i.test(label)) {
            return t('errors:resume_already_running');
        }
        return t('errors:resume_active_conflict');
    }

    if (status === 403 && options.context === 'open') {
        return t('errors:file_type_not_allowed');
    }

    return null;
}

/**
 * Localized message for an API error, falling back to its raw message.
 * @param {Error} err
 * @param {{context?: 'open'|'resume'}} [options]
 * @returns {string}
 */
export function describeApiError(err, options = {}) {
    return knownApiErrorMessage(err, options) || (err && err.message) || String(err);
}
