/**
 * Model Override Panel - shared "run this job on a different model" panel.
 *
 * Two places let the user point an existing job at another provider/model:
 *   - the resumable-job card ("Change model" then "Resume with this model"),
 *   - the completion card of a `partial` job ("Fix these N chunks"), which
 *     retranslates only the chunks that stayed in the source language (#261).
 *
 * Both need the exact same three things - the panel markup, a lazily built
 * provider+model picker, and the picker-config to backend-override mapping - so
 * they live here once instead of being duplicated per call site. This module
 * owns:
 *   - the live picker instances (created on first open, destroyed on teardown),
 *   - the `{provider, model, api_endpoint, api_key}` to `{llm_provider, model,
 *     llm_api_endpoint, api_key}` translation (the backend override schema),
 *   - the "no model selected" guard and its user-facing error.
 *
 * It deliberately knows nothing about resuming: callers decide what to do with
 * the overrides object (ResumeManager.resumeJob / window.resumeJob).
 */

import { DomHelpers } from '../ui/dom-helpers.js';
import { MessageLogger } from '../ui/message-logger.js';
import { t } from '../i18n/i18n.js';
import { createProviderModelPicker } from '../providers/provider-model-picker.js';

/**
 * Live picker instances, keyed by their panel ELEMENT (not by translation id).
 *
 * The same job can legitimately own two panels at once - a `partial` job shows
 * up both in the completion card and in the resumable-jobs list - so a
 * tid-keyed map would make the second panel silently reuse (and then fight
 * over) the first one's picker. The element is the only identity that is unique
 * per panel, and it also makes teardown a subtree question
 * (`destroyOverridePickers(card)`) instead of bookkeeping over ids.
 */
const overridePickers = new Map();

/**
 * Build the markup for an override panel. The panel starts hidden; call
 * `toggleOverridePanel` / `openOverridePanel` to reveal it (and to create the
 * picker on first open).
 *
 * @param {Object} opts
 * @param {string} opts.tid - Translation id the panel acts on.
 * @param {string} [opts.provider] - Original provider, seeds the picker.
 * @param {string} [opts.model] - Original model, seeds the picker.
 * @param {string} [opts.endpoint] - Original API endpoint, seeds the picker.
 * @param {string} [opts.applyLabelKey] - i18n key for the apply button label.
 * @param {string} [opts.panelClass] - Outer class, used by the caller as its
 *   own query selector (`resume-override`, `completion-override`, ...).
 * @returns {string} HTML for the panel
 */
export function overridePanelHtml({
    tid,
    provider = 'ollama',
    model = '',
    endpoint = '',
    applyLabelKey = 'translation:resume_apply_btn',
    panelClass = 'resume-override',
} = {}) {
    const safeTid = DomHelpers.escapeHtml(tid || '');
    const safeProvider = DomHelpers.escapeHtml(provider || 'ollama');
    const safeModel = DomHelpers.escapeHtml(model || '');
    const safeEndpoint = DomHelpers.escapeHtml(endpoint || '');
    const safeApplyKey = DomHelpers.escapeHtml(applyLabelKey);

    return `
        <div class="${panelClass} model-override-panel" data-tid="${safeTid}"
             data-provider="${safeProvider}"
             data-model="${safeModel}"
             data-endpoint="${safeEndpoint}"
             style="display: none;">
            <div class="model-override-panel__original">
                <span data-i18n="translation:resume_original_model">${DomHelpers.escapeHtml(t('translation:resume_original_model'))}</span>:
                <strong>${safeProvider} / ${safeModel || '&mdash;'}</strong>
            </div>
            <div class="resume-picker-mount"></div>
            <div class="resume-style-warning" style="display: none;">
                &#9888;&#65039; <span data-i18n="translation:resume_style_warning">${DomHelpers.escapeHtml(t('translation:resume_style_warning'))}</span>
            </div>
            <div class="model-override-panel__actions">
                <button type="button" class="btn btn-primary btn-launch resume-apply" data-tid="${safeTid}" data-i18n="${safeApplyKey}">${DomHelpers.escapeHtml(t(applyLabelKey))}</button>
            </div>
        </div>
    `;
}

/**
 * Lazily build the provider+model picker the first time a panel is opened.
 * Seeds it from the panel's data-* attributes and toggles the style-break
 * warning whenever the chosen model/provider diverges from the original.
 *
 * @param {HTMLElement} panel - Panel element built by `overridePanelHtml`
 */
export function openOverridePanel(panel) {
    if (!panel || overridePickers.has(panel)) return;

    const mount = panel.querySelector('.resume-picker-mount');
    const warning = panel.querySelector('.resume-style-warning');
    if (!mount) return;

    const origProvider = panel.dataset.provider || 'ollama';
    const origModel = panel.dataset.model || '';
    const origEndpoint = panel.dataset.endpoint || '';

    const picker = createProviderModelPicker(mount, {
        config: { provider: origProvider, model: origModel, api_endpoint: origEndpoint },
        onChange: (cfg) => {
            const changed = (cfg.provider !== origProvider) || (cfg.model && cfg.model !== origModel);
            if (warning) warning.style.display = changed ? 'block' : 'none';
        },
    });
    overridePickers.set(panel, picker);
}

/**
 * Show/hide a panel, creating its picker on first reveal.
 *
 * @param {HTMLElement} panel - Panel element built by `overridePanelHtml`
 * @returns {boolean} True when the panel is now open
 */
export function toggleOverridePanel(panel) {
    if (!panel) return false;
    if (panel.style.display !== 'none') {
        panel.style.display = 'none';
        return false;
    }
    panel.style.display = 'block';
    openOverridePanel(panel);
    return true;
}

/**
 * Read a panel's picker and map it onto the backend override schema.
 *
 * Three-valued on purpose, so a caller can tell "keep the job's own config"
 * apart from "the user must fix something first":
 *   - Object    -> send these overrides,
 *   - null      -> nothing to override (panel never opened), run as configured,
 *   - undefined -> ABORT; the user has already been told what is wrong
 *                  (currently: a picker with no model selected).
 *
 * @param {HTMLElement} panel - Panel element built by `overridePanelHtml`
 * @returns {Object|null|undefined} Overrides, no-override, or abort
 */
export function readOverrideConfig(panel) {
    const picker = panel ? overridePickers.get(panel) : null;
    const cfg = picker ? picker.getConfig() : null;
    if (!cfg) return null;

    if (!cfg.model) {
        MessageLogger.showMessage(t('translation:resume_no_model_selected'), 'error');
        return undefined;
    }

    // Map the picker's generic field names onto the backend override schema
    // (llm_provider / llm_api_endpoint). Single place in the frontend where
    // this translation happens.
    const overrides = { model: cfg.model, llm_provider: cfg.provider };
    if (cfg.api_endpoint) overrides.llm_api_endpoint = cfg.api_endpoint;
    if (cfg.api_key) overrides.api_key = cfg.api_key;
    return overrides;
}

/**
 * Destroy picker instances so their SearchableSelect registrations are released
 * before the surrounding DOM is wiped or removed.
 *
 * Always drops pickers whose panel has left the document (a dismissed card, a
 * re-rendered list), plus - when `root` is given - every panel inside it. Call
 * it right before replacing the markup that holds a panel: without it a locale
 * switch, which rebuilds the completion card, would leave the old picker
 * registered against detached nodes.
 *
 * Deliberately NOT a destroy-all when `root` is omitted: the two owners share
 * this map, and the resumable list re-renders on both `localeChanged` and
 * `translation.hasActive`. A bare call would then tear down the completion
 * card's *open* picker, leaving a visible panel whose picker is gone - and
 * `readOverrideConfig` would silently fall back to "no override" and resume on
 * the old model. Omitting `root` prunes detached panels only.
 *
 * @param {HTMLElement} [root] - Subtree about to be rebuilt/removed. Omit to
 *   prune detached panels only.
 */
export function destroyOverridePickers(root) {
    overridePickers.forEach((picker, panel) => {
        const inRoot = !!root && (panel === root
            || (typeof root.contains === 'function' && root.contains(panel)));
        if (panel.isConnected && !inRoot) return;
        picker.destroy?.();
        overridePickers.delete(panel);
    });
}
