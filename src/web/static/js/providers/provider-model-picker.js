/**
 * Reusable provider + model picker.
 *
 * A self-contained {provider, model, endpoint, optional API key} picker that
 * any feature can drop into a container. It owns NO per-provider knowledge:
 * the provider list, logos, model-list rendering and key handling all come from
 * provider-select-helpers.js and the generic /api/models endpoint — the same
 * single source the Settings panel and the Sample tab already use. Adding a new
 * provider therefore never touches this file.
 *
 * Used by the Resume dialog (issue #183) to let a paused job continue on a
 * different model/provider for the remaining chunks.
 */

import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { SearchableSelectFactory } from '../ui/searchable-select.js';
import { t, applyToDOM } from '../i18n/i18n.js';
import {
    PROVIDER_ORDER,
    PROVIDER_META,
    attachProviderSearchable,
    attachModelSearchable,
    populateModelSelectInto,
    setPlaceholderOption,
} from './provider-select-helpers.js';

// Providers that take a user-supplied endpoint; the rest use a built-in one.
const ENDPOINT_PROVIDERS = new Set(['ollama', 'openai']);

// Default endpoints, fetched once from Settings so a freshly picked
// ollama/openai provider starts from the same endpoint as the rest of the app.
let settingsEndpoints = { ollama: '', openai: '' };
let endpointsLoaded = false;

async function ensureSettingsEndpoints() {
    if (endpointsLoaded) return;
    try {
        const cfg = await ApiClient.getConfig();
        settingsEndpoints = {
            ollama: cfg.ollama_api_endpoint || cfg.api_endpoint || '',
            openai: cfg.openai_api_endpoint || '',
        };
    } catch (err) {
        console.warn('[picker] could not load default endpoints', err);
    } finally {
        endpointsLoaded = true;
    }
}

function endpointPlaceholder(provider) {
    if (provider === 'ollama') return settingsEndpoints.ollama || 'http://localhost:11434/api/generate';
    return settingsEndpoints.openai || 'https://api.openai.com/v1/chat/completions';
}

// Per-instance id counter so multiple pickers on one page never collide.
let uid = 0;

/**
 * Render a provider+model picker into `container`. The container must already
 * be in the DOM (SearchableSelect inserts its wrapper next to each <select>).
 *
 * @param {HTMLElement} container - Element to render into (its content is replaced).
 * @param {Object} [opts]
 * @param {Object} [opts.config] - Initial {provider, model, api_endpoint} to preselect.
 * @param {Function} [opts.onChange] - Called with the current config whenever the user edits anything.
 * @returns {{ getConfig: Function, destroy: Function }}
 */
export function createProviderModelPicker(container, { config = {}, onChange } = {}) {
    const n = ++uid;
    const providerId = `pmp-provider-${n}`;
    const modelId = `pmp-model-${n}`;
    const endpointId = `pmp-endpoint-${n}`;
    const keyId = `pmp-key-${n}`;

    const col = {
        provider: (config.provider || 'ollama').toLowerCase(),
        model: config.model || '',
        api_endpoint: config.api_endpoint || '',
        api_key: '', // never pre-filled; empty means "use the .env key"
    };

    const providerOptions = PROVIDER_ORDER.map((value) => {
        const meta = PROVIDER_META[value] || { name: value };
        const selected = col.provider === value ? 'selected' : '';
        return `<option value="${value}" ${selected}>${DomHelpers.escapeHtml(meta.name)}</option>`;
    }).join('');

    const showEndpoint = ENDPOINT_PROVIDERS.has(col.provider);
    const showKey = col.provider !== 'chatgpt';

    container.innerHTML = `
        <div class="provider-model-picker" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div class="form-group" style="margin-bottom: 0;">
                <label data-i18n="settings:ai_provider">Provider</label>
                <select id="${providerId}" class="form-control pmp-provider">${providerOptions}</select>
            </div>
            <div class="form-group" style="margin-bottom: 0;">
                <label data-i18n="settings:model">Model</label>
                <select id="${modelId}" class="form-control pmp-model">
                    <option value="" data-i18n="common:loading">Loading...</option>
                </select>
            </div>
        </div>
        <div class="form-group pmp-endpoint-wrap" style="margin: 10px 0 0; ${showEndpoint ? '' : 'display: none;'}">
            <label data-i18n="settings:api_endpoint">API Endpoint</label>
            <input type="text" id="${endpointId}" class="form-control pmp-endpoint"
                   value="${DomHelpers.escapeHtml(col.api_endpoint || '')}"
                   placeholder="${DomHelpers.escapeHtml(endpointPlaceholder(col.provider))}">
        </div>
        <div class="form-group pmp-key-wrap" style="margin: 10px 0 0; ${showKey ? '' : 'display: none;'}">
            <label data-i18n="settings:api_key_optional">API key (optional)</label>
            <input type="password" id="${keyId}" class="form-control pmp-key" autocomplete="off"
                   data-i18n-attr="placeholder:settings:api_key_env_placeholder"
                   placeholder="${DomHelpers.escapeHtml(t('settings:api_key_env_placeholder'))}">
        </div>
    `;

    // Translate the freshly injected data-i18n markup right away (it stays
    // reactive afterwards via the global languageChanged -> applyToDOM(body) hook).
    applyToDOM(container);

    const providerSelectEl = container.querySelector('.pmp-provider');
    const modelSelectEl = container.querySelector('.pmp-model');
    const endpointWrap = container.querySelector('.pmp-endpoint-wrap');
    const endpointInput = container.querySelector('.pmp-endpoint');
    const keyWrap = container.querySelector('.pmp-key-wrap');
    const keyInput = container.querySelector('.pmp-key');

    const emitChange = () => {
        if (typeof onChange === 'function') onChange(currentConfig());
    };

    /** Endpoint to send/use; undefined unless the provider actually takes one. */
    const effectiveEndpoint = () =>
        ENDPOINT_PROVIDERS.has(col.provider) ? (col.api_endpoint || undefined) : undefined;

    async function loadModels() {
        setPlaceholderOption(modelSelectEl, 'common:loading');
        try {
            const data = await ApiClient.getModels(col.provider, {
                apiKey: col.api_key || '__USE_ENV__',
                apiEndpoint: effectiveEndpoint(),
            });
            const models = data.models || [];
            if (!models.length) {
                setPlaceholderOption(modelSelectEl, 'settings:search_models_no_models_available');
                col.model = '';
                emitChange();
                return;
            }
            populateModelSelectInto(modelSelectEl, models, col.model || data.default || '', col.provider);
            col.model = modelSelectEl.value;
            emitChange();
        } catch (err) {
            console.error('[picker] model fetch failed', err);
            setPlaceholderOption(modelSelectEl, 'settings:search_models_error');
            col.model = '';
            emitChange();
        }
    }

    attachProviderSearchable(providerSelectEl, {
        onChange: async (newProvider) => {
            col.provider = newProvider;
            col.model = '';
            col.api_endpoint = settingsEndpoints[newProvider] || '';
            const takesEndpoint = ENDPOINT_PROVIDERS.has(newProvider);
            if (endpointWrap) endpointWrap.style.display = takesEndpoint ? '' : 'none';
            if (keyWrap) keyWrap.style.display = newProvider === 'chatgpt' ? 'none' : '';
            if (endpointInput) {
                endpointInput.value = col.api_endpoint;
                endpointInput.placeholder = endpointPlaceholder(newProvider);
            }
            await loadModels();
        },
    });

    attachModelSearchable(modelSelectEl, {
        onChange: (value) => {
            col.model = value;
            emitChange();
        },
    });

    if (endpointInput) {
        // Re-list models on commit (change), not per keystroke.
        endpointInput.addEventListener('change', async () => {
            col.api_endpoint = endpointInput.value.trim();
            await loadModels();
        });
    }

    if (keyInput) {
        keyInput.addEventListener('change', async () => {
            col.api_key = keyInput.value;
            // A freshly typed key may unlock the model list for a cloud provider.
            await loadModels();
        });
    }

    function currentConfig() {
        const out = { provider: col.provider, model: col.model };
        if (ENDPOINT_PROVIDERS.has(col.provider) && col.api_endpoint) {
            out.api_endpoint = col.api_endpoint;
        }
        if (col.api_key) out.api_key = col.api_key;
        return out;
    }

    // Seed default endpoints, then list models for the initial provider.
    ensureSettingsEndpoints().then(() => {
        if (!col.api_endpoint && ENDPOINT_PROVIDERS.has(col.provider)) {
            col.api_endpoint = settingsEndpoints[col.provider] || '';
            if (endpointInput) {
                endpointInput.value = col.api_endpoint;
                endpointInput.placeholder = endpointPlaceholder(col.provider);
            }
        }
        loadModels();
    });

    return {
        getConfig: currentConfig,
        destroy() {
            SearchableSelectFactory.destroy?.(providerId);
            SearchableSelectFactory.destroy?.(modelId);
        },
    };
}
