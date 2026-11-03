/**
 * Cost Estimator - estimate translation cost in USD before launching.
 *
 * Renders a per-file cost badge inside each <li> of the Selected Files list
 * (slot: <div class="cost-badge file-cost-badge" data-cost-badge-for="...">).
 *
 * Triggers on model change, file added/removed, and language/options changes.
 * Pricing for OpenRouter/Poe is read from the model option's pricing data
 * already returned by their APIs. For other paid providers, defaults come
 * from the backend; users can override per model via the Edit Prices modal
 * (overrides persist in localStorage).
 */

import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { StateManager } from '../core/state-manager.js';
import { t } from '../i18n/i18n.js';

const STORAGE_KEY = 'tbl_pricing_overrides_v1';

const LOCAL_PROVIDERS = new Set(['ollama']);
const API_PRICING_PROVIDERS = new Set(['openrouter', 'poe']);

let pricingDefaults = null;
let pricingLastUpdated = null;
let listenersAttached = false;

// Per-badge AbortControllers so a fresh refresh cancels stale in-flight calls.
const inFlightByBadge = new WeakMap();

// Cache last estimate per (file, provider, model, langs, options). Keyed by a
// string so it survives when updateFileDisplay rebuilds the badge element —
// the next refresh restores the result without re-hitting the API.
const estimateCache = new Map();

function makeCacheKey(file, ctx) {
    const op = file.operation || 'translate';
    const refineAfter = !!file.refineAfter;
    return [
        ctx.provider,
        ctx.model,
        file.filePath,
        ctx.src,
        ctx.tgt,
        op,
        refineAfter ? 1 : 0,
        ctx.options.text_cleanup ? 1 : 0,
    ].join('|');
}

function invalidateCacheFor(provider, model) {
    const prefix = `${provider}|${model}|`;
    for (const key of estimateCache.keys()) {
        if (key.startsWith(prefix)) estimateCache.delete(key);
    }
}

function loadOverrides() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        return (parsed && typeof parsed === 'object') ? parsed : {};
    } catch {
        return {};
    }
}

function saveOverrides(overrides) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
    } catch {
        /* ignore quota errors */
    }
}

function getOverride(provider, model) {
    const all = loadOverrides();
    return all?.[provider]?.[model] || null;
}

function setOverride(provider, model, pricing) {
    const all = loadOverrides();
    if (!all[provider]) all[provider] = {};
    all[provider][model] = pricing;
    saveOverrides(all);
}

function clearOverride(provider, model) {
    const all = loadOverrides();
    if (all?.[provider]?.[model]) {
        delete all[provider][model];
        saveOverrides(all);
    }
}

function getCurrentProvider() {
    return DomHelpers.getValue('llmProvider');
}

function getCurrentModel() {
    return DomHelpers.getValue('model');
}

function getLanguagePair() {
    const src = DomHelpers.getValue('sourceLang') || '';
    const tgt = DomHelpers.getValue('targetLang') || '';
    return { src, tgt };
}

function getOptions() {
    return {
        text_cleanup: !!DomHelpers.getElement('textCleanup')?.checked,
    };
}

function fileOptions(file, baseOptions) {
    const op = file.operation || 'translate';
    const refine = op === 'refine' || !!file.refineAfter;
    return { ...baseOptions, refine };
}

function readPricingFromModelOption() {
    const select = DomHelpers.getElement('model');
    if (!select) return null;
    const opt = select.selectedOptions?.[0];
    if (!opt) return null;
    const inAttr = opt.getAttribute('data-pricing-input');
    const outAttr = opt.getAttribute('data-pricing-output');
    if (inAttr === null || outAttr === null) return null;
    const input = parseFloat(inAttr);
    const output = parseFloat(outAttr);
    if (!Number.isFinite(input) || !Number.isFinite(output)) return null;
    return { input, output };
}

function resolvePricing(provider, model) {
    if (!provider || !model) return { pricing: null, source: 'unknown' };

    const override = getOverride(provider, model);
    if (override) return { pricing: override, source: 'user_override' };

    if (API_PRICING_PROVIDERS.has(provider)) {
        const fromOption = readPricingFromModelOption();
        if (fromOption) return { pricing: fromOption, source: 'provider_api' };
    }

    if (pricingDefaults && pricingDefaults[provider]) {
        const provData = pricingDefaults[provider];
        if (provData[model]) {
            const { input, output } = provData[model];
            return {
                pricing: { input, output },
                source: 'default_table',
            };
        }
        const lower = model.toLowerCase();
        for (const knownModel of Object.keys(provData)) {
            if (knownModel.toLowerCase() === lower) {
                const { input, output } = provData[knownModel];
                return { pricing: { input, output }, source: 'default_table' };
            }
        }
    }

    return { pricing: null, source: 'unknown' };
}

function formatUSD(amount) {
    if (amount === 0) return '$0.00';
    if (Math.abs(amount) < 0.01) return t('settings:cost_format_lt_001');
    if (Math.abs(amount) < 1) return `$${amount.toFixed(3)}`;
    return `$${amount.toFixed(2)}`;
}

function sourceLabel(source, lastUpdated) {
    switch (source) {
        case 'user_override': return t('settings:cost_source_user_override');
        case 'provider_api':  return t('settings:cost_source_provider_api');
        case 'default_table': return t('settings:cost_source_default_table', { date: lastUpdated || pricingLastUpdated || '' });
        default: return '';
    }
}

function renderBadge(badge, state) {
    if (!badge) return;

    badge.classList.remove(
        'cost-free',
        'cost-estimated',
        'cost-unknown',
        'cost-loading',
        'cost-error',
    );

    if (state.kind === 'hidden') {
        badge.style.display = 'none';
        badge.textContent = '';
        badge.title = '';
        return;
    }

    badge.style.display = 'flex';

    if (state.kind === 'free') {
        badge.classList.add('cost-free');
        badge.innerHTML = `
            <span class="cost-badge-icon material-symbols-outlined">verified</span>
            <span class="cost-badge-text">${state.message || t('settings:cost_free_local')}</span>
        `;
        badge.title = '';
        return;
    }

    if (state.kind === 'loading') {
        badge.classList.add('cost-loading');
        badge.innerHTML = `
            <span class="cost-badge-icon material-symbols-outlined cost-spinning">progress_activity</span>
            <span class="cost-badge-text">${t('settings:cost_estimating')}</span>
        `;
        badge.title = '';
        return;
    }

    if (state.kind === 'unknown') {
        badge.classList.add('cost-unknown');
        const provider = state.provider || '';
        const model = state.model || '';
        badge.innerHTML = `
            <span class="cost-badge-icon material-symbols-outlined">help</span>
            <span class="cost-badge-text">${t('settings:cost_unknown')}</span>
            <button type="button" class="cost-badge-edit" data-action="edit"
                title="${t('settings:cost_set_prices_title', { provider, model })}">${t('settings:cost_set_prices')}</button>
        `;
        badge.title = `${provider} / ${model}`;
        return;
    }

    if (state.kind === 'no_content') {
        badge.classList.add('cost-unknown');
        badge.innerHTML = `
            <span class="cost-badge-icon material-symbols-outlined">description</span>
            <span class="cost-badge-text">${t('settings:cost_no_content')}</span>
        `;
        badge.title = '';
        return;
    }

    if (state.kind === 'error') {
        badge.classList.add('cost-error');
        badge.innerHTML = `
            <span class="cost-badge-icon material-symbols-outlined">error</span>
            <span class="cost-badge-text">${t('settings:cost_error')}</span>
        `;
        badge.title = state.message || '';
        return;
    }

    badge.classList.add('cost-estimated');
    const min = formatUSD(state.total_cost_min);
    const max = formatUSD(state.total_cost_max);
    const display = (min === max)
        ? t('settings:cost_estimated', { value: min })
        : t('settings:cost_estimated_range', { min, max });

    const passesNote = state.passes && state.passes > 1
        ? t('settings:cost_passes_suffix', { count: state.passes })
        : '';
    const tokensNote = state.input_tokens
        ? `${t('settings:cost_tokens_chunks', { tokens: state.input_tokens.toLocaleString(), chunks: state.n_chunks })}${passesNote}`
        : '';
    const sourceNote = sourceLabel(state.pricing_source, state.pricing_last_updated);

    badge.innerHTML = `
        <span class="cost-badge-icon material-symbols-outlined">payments</span>
        <span class="cost-badge-text">${display}</span>
        <button type="button" class="cost-badge-edit" data-action="edit" title="${t('settings:cost_edit_title')}">${t('settings:cost_edit_btn')}</button>
    `;
    badge.title = [tokensNote, sourceNote].filter(Boolean).join(' • ');
}

async function ensureDefaults() {
    if (pricingDefaults) return;
    try {
        const data = await ApiClient.getPricingDefaults();
        pricingDefaults = data?.pricing || {};
        pricingLastUpdated = data?.last_updated || null;
    } catch {
        pricingDefaults = {};
    }
}

function findFileForBadge(badge, files) {
    const key = badge.getAttribute('data-cost-badge-for');
    if (!key) return null;
    return files.find(f => f.filePath === key) || files.find(f => f.name === key) || null;
}

async function estimateOne(badge, file, ctx) {
    const cacheKey = makeCacheKey(file, ctx);

    // If we already estimated for these exact params, restore instantly.
    const cached = estimateCache.get(cacheKey);
    if (cached) {
        renderBadge(badge, cached);
        return;
    }

    const prev = inFlightByBadge.get(badge);
    if (prev) prev.abort();

    const controller = new AbortController();
    inFlightByBadge.set(badge, controller);

    renderBadge(badge, { kind: 'loading' });

    const payload = {
        provider: ctx.provider,
        model: ctx.model,
        src_lang: ctx.src,
        tgt_lang: ctx.tgt,
        options: fileOptions(file, ctx.options),
        pricing: ctx.pricing,
        file_path: file.filePath,
    };

    try {
        const data = await ApiClient.estimateCost(payload, { signal: controller.signal });
        let state;
        if (data.free) {
            state = { kind: 'free', message: data.message };
        } else if (data.unknown) {
            state = { kind: 'unknown', provider: ctx.provider, model: ctx.model };
        } else if (data.no_content) {
            state = { kind: 'no_content' };
        } else {
            state = { kind: 'estimated', ...data, pricing_source: ctx.source };
        }
        estimateCache.set(cacheKey, state);
        renderBadge(badge, state);
    } catch (error) {
        if (error?.name === 'AbortError') return;
        renderBadge(badge, { kind: 'error', message: error?.message });
    } finally {
        if (inFlightByBadge.get(badge) === controller) {
            inFlightByBadge.delete(badge);
        }
    }
}

let refreshScheduled = false;
function scheduleRefresh() {
    if (refreshScheduled) return;
    refreshScheduled = true;
    // queueMicrotask coalesces multiple synchronous triggers (file upload may
    // run updateFileDisplay several times in one tick — language auto-detect
    // + notifyFileListChanged) so we re-estimate against the FINAL DOM.
    queueMicrotask(() => {
        refreshScheduled = false;
        CostEstimator.refresh();
    });
}

export const CostEstimator = {
    initialize() {
        this.attachListeners();
        ensureDefaults().then(() => this.refresh());
    },

    attachListeners() {
        if (listenersAttached) return;
        listenersAttached = true;

        window.addEventListener('modelChanged', scheduleRefresh);
        window.addEventListener('fileListChanged', scheduleRefresh);
        window.addEventListener('translationOptionsChanged', scheduleRefresh);
        // Rebuild badge innerHTML on language switch: renderBadge re-runs t()
        // each call, so a plain refresh() (which restores from cache) is enough.
        window.addEventListener('localeChanged', scheduleRefresh);

        ['textCleanup', 'sourceLang', 'targetLang'].forEach((id) => {
            const el = DomHelpers.getElement(id);
            if (el) el.addEventListener('change', scheduleRefresh);
        });

        // Safety net: observe the file list for any <li> appearing or
        // disappearing. updateFileDisplay rebuilds <li>s and the badge slots
        // get recreated; the observer catches any case the event listeners
        // miss. IMPORTANT: childList only (no subtree) — otherwise renderBadge
        // mutating badge.innerHTML would re-trigger refresh, infinite loop.
        const container = DomHelpers.getElement('fileListContainer');
        if (container) {
            const observer = new MutationObserver(scheduleRefresh);
            observer.observe(container, { childList: true });
        }

        // Delegate Edit-price button clicks for all per-file badges.
        document.addEventListener('click', (event) => {
            const editBtn = event.target.closest('.cost-badge [data-action="edit"]');
            if (!editBtn) return;
            event.preventDefault();
            event.stopPropagation();
            this.openEditModal();
        });
    },

    refresh() {
        const badges = document.querySelectorAll('[data-cost-badge-for]');
        if (badges.length === 0) return;

        const provider = getCurrentProvider();
        const model = getCurrentModel();

        if (!provider || !model) {
            badges.forEach(b => renderBadge(b, { kind: 'hidden' }));
            return;
        }

        if (LOCAL_PROVIDERS.has(provider)) {
            badges.forEach(b => renderBadge(b, { kind: 'free', message: t('settings:cost_free_local_model') }));
            return;
        }

        const { pricing, source } = resolvePricing(provider, model);
        if (!pricing) {
            badges.forEach(b => renderBadge(b, { kind: 'unknown', provider, model }));
            return;
        }

        const { src, tgt } = getLanguagePair();
        const ctx = {
            provider,
            model,
            pricing,
            source,
            src,
            tgt,
            options: getOptions(),
        };

        const files = StateManager.getState('files.toProcess') || [];

        badges.forEach((badge) => {
            const file = findFileForBadge(badge, files);
            if (file && file.operation === 'term_fix') {
                // The estimate prices a full pass; a term fix only sends the
                // sentences that hold a match, so any figure would mislead.
                renderBadge(badge, { kind: 'hidden' });
                return;
            }
            if (!file || !file.filePath) {
                renderBadge(badge, { kind: 'no_content' });
                return;
            }
            estimateOne(badge, file, ctx);
        });
    },

    openEditModal() {
        const provider = getCurrentProvider();
        const model = getCurrentModel();
        if (!provider || !model) return;

        const existing = document.getElementById('costPricingModal');
        if (existing) existing.remove();

        const current = resolvePricing(provider, model).pricing
            || readPricingFromModelOption()
            || { input: 0, output: 0 };

        const override = getOverride(provider, model);

        const modal = document.createElement('div');
        modal.id = 'costPricingModal';
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content cost-pricing-modal">
                <div class="modal-header">
                    <h3>${t('settings:cost_edit_modal_title')}</h3>
                    <button class="close-btn" data-action="close">&times;</button>
                </div>
                <div class="modal-body">
                    <p class="cost-pricing-subtitle">
                        ${DomHelpers.escapeHtml(provider)} / <strong>${DomHelpers.escapeHtml(model)}</strong>
                    </p>
                    <p class="cost-pricing-help">
                        ${t('settings:cost_edit_modal_help')}
                    </p>
                    <div class="cost-pricing-grid">
                        <div class="form-group">
                            <label for="costPriceInput">${t('settings:cost_input_label')}</label>
                            <div class="neu-inset-light">
                                <input type="number" min="0" step="0.001"
                                    id="costPriceInput" class="form-control"
                                    value="${current.input}">
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="costPriceOutput">${t('settings:cost_output_label')}</label>
                            <div class="neu-inset-light">
                                <input type="number" min="0" step="0.001"
                                    id="costPriceOutput" class="form-control"
                                    value="${current.output}">
                            </div>
                        </div>
                    </div>
                    ${override ? `<p class="cost-pricing-note">${t('settings:cost_override_note')}</p>` : ''}
                </div>
                <div class="modal-footer">
                    ${override ? `<button class="btn btn-secondary" data-action="reset">${t('settings:cost_reset_default')}</button>` : ''}
                    <button class="btn btn-secondary" data-action="close">${t('common:cancel')}</button>
                    <button class="btn btn-primary" data-action="save">${t('common:save')}</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        const close = () => modal.remove();

        modal.addEventListener('click', (event) => {
            if (event.target === modal) close();
            const action = event.target.closest('[data-action]')?.dataset.action;
            if (action === 'close') close();
            if (action === 'save') {
                const inputEl = modal.querySelector('#costPriceInput');
                const outputEl = modal.querySelector('#costPriceOutput');
                const inputVal = parseFloat(inputEl.value);
                const outputVal = parseFloat(outputEl.value);
                if (!Number.isFinite(inputVal) || inputVal < 0 ||
                    !Number.isFinite(outputVal) || outputVal < 0) {
                    inputEl.focus();
                    return;
                }
                setOverride(provider, model, { input: inputVal, output: outputVal });
                invalidateCacheFor(provider, model);
                close();
                this.refresh();
            }
            if (action === 'reset') {
                clearOverride(provider, model);
                invalidateCacheFor(provider, model);
                close();
                this.refresh();
            }
        });

        const onEsc = (e) => {
            if (e.key === 'Escape') {
                close();
                document.removeEventListener('keydown', onEsc);
            }
        };
        document.addEventListener('keydown', onEsc);
    },
};
