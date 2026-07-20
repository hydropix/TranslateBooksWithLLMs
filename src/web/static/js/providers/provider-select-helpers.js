/**
 * Provider/Model select helpers — shared between the Settings panel
 * (provider-manager.js) and the Sample tab columns (sample-manager.js).
 *
 * Holds the canonical provider metadata, the logos used in dropdown rows,
 * and the per-provider model-list rendering logic so both UIs stay in sync
 * (icons, pricing labels, optgroups, tooltips, …).
 */

import { DomHelpers } from '../ui/dom-helpers.js';
import { SearchableSelectFactory } from '../ui/searchable-select.js';
import { t } from '../i18n/i18n.js';

export const PROVIDER_LOGOS = {
    ollama: '/static/img/providers/ollama.png',
    poe: '/static/img/providers/poe.png',
    atlascloud: '/static/img/providers/openai.png',
    deepseek: '/static/img/providers/deepseek.png',
    mistral: '/static/img/providers/mistral.png',
    gemini: '/static/img/providers/gemini.png',
    openai: '/static/img/providers/openai.png',
    openrouter: '/static/img/providers/openrouter.png',
    nim: '/static/img/providers/nvidia.png',
};

export const PROVIDER_META = {
    ollama: { name: 'Ollama', description: 'Local' },
    poe: { name: 'Poe', description: 'Multi-Provider' },
    atlascloud: { name: 'Atlas Cloud', description: 'Cloud API' },
    deepseek: { name: 'DeepSeek', description: 'Cloud API' },
    mistral: { name: 'Mistral', description: 'Cloud API' },
    gemini: { name: 'Gemini', description: 'Cloud' },
    openai: { name: 'OpenAI', description: 'Compatible' },
    openrouter: { name: 'OpenRouter', description: '200+ models' },
    nim: { name: 'NVIDIA NIM', description: 'Cloud API' },
};

// Canonical A-Z order used everywhere a provider dropdown is built.
export const PROVIDER_ORDER = ['atlascloud', 'deepseek', 'gemini', 'mistral', 'nim', 'ollama', 'openai', 'openrouter', 'poe'];

/**
 * Replace the dropdown content with a single placeholder option whose text
 * comes from an i18n key. The data-i18n attribute keeps the placeholder in
 * sync when the UI locale changes.
 */
export function setPlaceholderOption(selectEl, i18nKey) {
    if (!selectEl) return;
    selectEl.innerHTML = `<option value="" data-i18n="${i18nKey}">${t(i18nKey)}</option>`;
}

function formatPrice(price) {
    if (price === 0) return t('settings:cost_format_free');
    if (price < 0.01) return t('settings:cost_format_lt_001');
    if (price < 1) return `$${price.toFixed(2)}`;
    return `$${price.toFixed(2)}`;
}

/**
 * Populate any <select> element with the right per-provider format:
 *   - Gemini  → displayName + token-limit tooltip
 *   - OpenAI  → curated label list
 *   - OpenRouter / Poe → optgroups (Poe) + pricing in the label
 *   - Mistral / DeepSeek / NIM → label + context-length tooltip
 *   - Ollama  → plain model names (strings)
 *
 * Returns true iff `defaultModel` was found among the options and selected.
 * Identical behavior to provider-manager.js' inline populateModelSelect but
 * targets the caller-supplied element so it can serve N column dropdowns
 * inside the Sample tab.
 */
export function populateModelSelectInto(selectEl, models, defaultModel = null, provider = 'ollama') {
    if (!selectEl) return false;
    selectEl.innerHTML = '';
    let defaultFound = false;
    const list = models || [];

    if (provider === 'gemini') {
        list.forEach((m) => {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = m.displayName || m.name;
            const parts = [];
            if (m.description) parts.push(m.description);
            parts.push(`Input: ${m.inputTokenLimit || 'N/A'} tokens, Output: ${m.outputTokenLimit || 'N/A'} tokens`);
            opt.title = parts.join(' | ');
            if (m.name === defaultModel) { opt.selected = true; defaultFound = true; }
            selectEl.appendChild(opt);
        });
    } else if (provider === 'openai') {
        list.forEach((m) => {
            // Accept both the normalized `{value, label}` shape (Settings panel)
            // and the backend's raw `{id, name}` shape (Sample tab passes the
            // /api/models payload straight through). Without this fallback the
            // OpenAI list renders as many blank-labelled rows.
            const value = m.value ?? m.id ?? '';
            const label = m.label ?? m.name ?? value;
            const opt = document.createElement('option');
            opt.value = value;
            opt.textContent = label;
            if (value === defaultModel) { opt.selected = true; defaultFound = true; }
            selectEl.appendChild(opt);
        });
    } else if (provider === 'openrouter' || provider === 'poe') {
        let currentGroup = null;
        let optgroup = null;
        list.forEach((m) => {
            if (provider === 'poe') {
                const groupKey = m.group || m.owned_by;
                if (groupKey && groupKey !== currentGroup) {
                    currentGroup = groupKey;
                    optgroup = document.createElement('optgroup');
                    optgroup.label = currentGroup;
                    selectEl.appendChild(optgroup);
                }
            }
            const opt = document.createElement('option');
            const modelId = m.id || m.value;
            opt.value = modelId;
            if (m.pricing && (m.pricing.prompt_per_million !== undefined || m.pricing.request)) {
                if (m.pricing.request && m.pricing.request > 0) {
                    opt.textContent = `${m.name || modelId} ($${m.pricing.request.toFixed(4)}/req)`;
                } else {
                    const inputPrice = formatPrice(m.pricing.prompt_per_million);
                    const outputPrice = formatPrice(m.pricing.completion_per_million);
                    opt.textContent = `${m.name || modelId} (In: ${inputPrice}/M, Out: ${outputPrice}/M)`;
                }
                if (m.pricing.prompt_per_million !== undefined) {
                    opt.dataset.pricingInput = m.pricing.prompt_per_million;
                }
                if (m.pricing.completion_per_million !== undefined) {
                    opt.dataset.pricingOutput = m.pricing.completion_per_million;
                }
            } else {
                opt.textContent = m.label || m.name || modelId;
            }
            const tip = [];
            if (m.context_length) tip.push(`Context: ${m.context_length} tokens`);
            if (m.description) tip.push(m.description);
            if (tip.length) opt.title = tip.join(' | ');
            if (modelId === defaultModel) { opt.selected = true; defaultFound = true; }
            (optgroup || selectEl).appendChild(opt);
        });
    } else if (provider === 'mistral' || provider === 'deepseek' || provider === 'atlascloud' || provider === 'nim') {
        list.forEach((m) => {
            // Same dual-shape tolerance as the openai branch: Settings sends
            // `{value, label}`, the Sample tab forwards the raw `{id, name}`.
            const value = m.value ?? m.id ?? '';
            const label = m.label ?? m.name ?? value;
            const opt = document.createElement('option');
            opt.value = value;
            opt.textContent = label;
            if (m.context_length) opt.title = `Context: ${m.context_length} tokens`;
            if (value === defaultModel) { opt.selected = true; defaultFound = true; }
            selectEl.appendChild(opt);
        });
    } else {
        // Ollama: plain strings
        list.forEach((name) => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            if (name === defaultModel) { opt.selected = true; defaultFound = true; }
            selectEl.appendChild(opt);
        });
    }

    return defaultFound;
}

/**
 * SearchableSelect renderOption used by both the Settings provider dropdown
 * and the Sample-tab per-column provider dropdown — same logo + name + tag.
 */
export function renderProviderOption(opt) {
    const logo = PROVIDER_LOGOS[opt.value] || '';
    const meta = PROVIDER_META[opt.value] || { name: opt.label, description: '' };
    const checkmark = opt.selected
        ? '<span class="option-check material-symbols-outlined">check</span>'
        : '<span class="option-check"></span>';
    return `
        ${checkmark}
        <span class="provider-option">
            <img src="${logo}" alt="" class="provider-logo" onerror="this.style.display='none'">
            <span class="provider-name">${DomHelpers.escapeHtml(meta.name)}</span>
            <span class="provider-description">${DomHelpers.escapeHtml(meta.description)}</span>
        </span>
    `;
}

/**
 * HTML used by both UIs for the selected-provider display chip (logo + name).
 */
export function providerDisplayHtml(providerValue) {
    const logo = PROVIDER_LOGOS[providerValue] || '';
    const meta = PROVIDER_META[providerValue] || { name: providerValue, description: '' };
    return `
        <span class="provider-option">
            <img src="${logo}" alt="" class="provider-logo" onerror="this.style.display='none'">
            <span class="provider-name">${DomHelpers.escapeHtml(meta.name)}</span>
        </span>
    `;
}

/**
 * Attach a SearchableSelect to a provider <select>, wiring the same logo +
 * name + description row + display-chip rendering as the Settings panel.
 *
 * `onChange(value)` fires whenever the user picks a different provider.
 */
export function attachProviderSearchable(selectEl, { placeholder, onChange } = {}) {
    if (!selectEl) return null;
    const id = selectEl.id;
    const instance = SearchableSelectFactory.create(selectEl, {
        placeholder: placeholder || t('settings:search_providers_placeholder'),
        showBadge: false,
        renderOption: renderProviderOption,
        onSelect: (option) => {
            const inst = SearchableSelectFactory.get(id);
            if (inst && inst.displayText) {
                inst.displayText.innerHTML = providerDisplayHtml(option.value);
            }
            if (typeof onChange === 'function') onChange(option.value);
        },
    });
    if (instance && instance.displayText && selectEl.value) {
        instance.displayText.innerHTML = providerDisplayHtml(selectEl.value);
    }
    return instance;
}

/**
 * Attach a SearchableSelect to a model <select> with the same UX as Settings:
 * search field, optgroup-aware filtering, allow free-form values (handy for
 * Poe bot names not in the curated list).
 */
export function attachModelSearchable(selectEl, { placeholder, onChange, allowCustomValue = true } = {}) {
    if (!selectEl) return null;
    return SearchableSelectFactory.create(selectEl, {
        placeholder: placeholder || t('settings:search_models_placeholder'),
        allowCustomValue,
        onSelect: (option) => {
            if (typeof onChange === 'function') onChange(option.value);
        },
    });
}
