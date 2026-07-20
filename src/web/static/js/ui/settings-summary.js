/**
 * Settings Summary - concise overview of LLM + active options.
 *
 * Reads form state and renders a summary under the Start Translation button:
 *  - LLM line: gray text (provider · model · src → tgt)
 *  - Options line: each active option as a small colored chip
 *
 * Each part is clickable: it navigates to the matching tab and opens the
 * corresponding collapsible section so the user can adjust the setting in one
 * click instead of hunting for it.
 */

import { DomHelpers } from './dom-helpers.js';
import { StateManager } from '../core/state-manager.js';
import { t } from '../i18n/i18n.js';

const PROVIDER_LABELS = {
    ollama: 'Ollama',
    gemini: 'Gemini',
    openai: 'OpenAI',
    openrouter: 'OpenRouter',
    mistral: 'Mistral',
    deepseek: 'DeepSeek',
    atlascloud: 'Atlas Cloud',
    poe: 'Poe',
    nim: 'NVIDIA NIM',
};

// Each chip carries its own tint. Background uses rgba so it stays readable
// on both light and dark themes; the text uses the same hue at full saturation.
const OPTION_STYLES = {
    bilingual:    { bg: 'rgba(59, 130, 246, 0.18)', fg: '#2563eb', border: 'rgba(59, 130, 246, 0.45)' },
    plainText:    { bg: 'rgba(245, 158, 11, 0.20)', fg: '#d97706', border: 'rgba(245, 158, 11, 0.45)' },
    ocr:          { bg: 'rgba(168, 85, 247, 0.18)', fg: '#9333ea', border: 'rgba(168, 85, 247, 0.45)' },
    noPause:      { bg: 'rgba(239, 68, 68, 0.18)',  fg: '#dc2626', border: 'rgba(239, 68, 68, 0.45)' },
    glossary:     { bg: 'rgba(99, 102, 241, 0.18)', fg: '#4f46e5', border: 'rgba(99, 102, 241, 0.45)' },
    instructions: { bg: 'rgba(20, 184, 166, 0.20)', fg: '#0d9488', border: 'rgba(20, 184, 166, 0.45)' },
    refineOnly:   { bg: 'rgba(34, 197, 94, 0.20)',  fg: '#16a34a', border: 'rgba(34, 197, 94, 0.55)' },
};

// Maps a summary item key to the tab + collapsible section it should reveal.
// `focus` is an optional element id to focus/scroll-to after switching.
// Also reused by the Fallbacks recommendation panel (progress-manager.js) to
// jump to the relevant setting when the user clicks a link in the panel.
const TARGETS = {
    provider:     { tab: 'settings', section: 'settings', focus: 'llmProvider' },
    model:        { tab: 'settings', section: 'settings', focus: 'model' },
    languages:    { tab: 'translate', section: null,      focus: 'sourceLang' },
    noPause:      { tab: 'settings', section: 'settings', focus: 'disableAutoPause' },
    bilingual:    { tab: 'settings', section: 'prompt',   focus: 'bilingualMode' },
    plainText:    { tab: 'settings', section: 'prompt',   focus: 'plainTextMode' },
    ocr:          { tab: 'settings', section: 'prompt',   focus: 'textCleanup' },
    glossary:     { tab: 'settings', section: 'prompt',   focus: 'glossarySelect' },
    instructions: { tab: 'settings', section: 'prompt',   focus: 'customInstructionSelect' },
    refineOnly:   { tab: 'files',    section: null,       focus: null },
};

const SECTION_IDS = {
    settings: { section: 'settingsOptionsSection',     icon: 'settingsOptionsIcon',     stateKey: 'ui.isSettingsOptionsOpen' },
    prompt:   { section: 'promptOptionsSection',       icon: 'promptOptionsIcon',       stateKey: 'ui.isPromptOptionsOpen' },
    notify:   { section: 'notificationOptionsSection', icon: 'notificationOptionsIcon', stateKey: null },
};

function getSelectText(id) {
    const el = DomHelpers.getElement(id);
    if (!el || el.selectedIndex < 0) return '';
    const opt = el.options[el.selectedIndex];
    return opt ? (opt.textContent || opt.value || '').trim() : '';
}

function getLanguage(selectId, customId) {
    const select = DomHelpers.getElement(selectId);
    if (!select) return '';
    if (select.value === 'Other') {
        const custom = DomHelpers.getElement(customId);
        return custom ? (custom.value || '').trim() : '';
    }
    return select.value || '';
}

function isChecked(id) {
    const el = DomHelpers.getElement(id);
    return !!(el && el.checked);
}

function queueOperation() {
    const files = StateManager.getState('files.toProcess') || [];
    const pending = files.find(f => f.status === 'Queued');
    if (!pending) return null;
    return pending.operation || 'translate';
}

function buildLlmLine() {
    const providerKey = (DomHelpers.getValue('llmProvider') || '').trim();
    const providerLabel = PROVIDER_LABELS[providerKey] || providerKey || '—';
    const modelLabel = getSelectText('model') || DomHelpers.getValue('model') || '—';
    const sourceLang = getLanguage('sourceLang', 'customSourceLang') || t('translation:summary_lang_auto_detect');
    const targetLang = getLanguage('targetLang', 'customTargetLang') || '—';
    if (queueOperation() === 'refine') {
        return [
            { key: 'provider',  label: providerLabel },
            { key: 'model',     label: modelLabel },
            { key: 'languages', label: t('translation:summary_refining_in', { lang: targetLang }) },
        ];
    }
    return [
        { key: 'provider',  label: providerLabel },
        { key: 'model',     label: modelLabel },
        { key: 'languages', label: `${sourceLang} → ${targetLang}` },
    ];
}

function buildChips() {
    const chips = [];

    const hasGlossary = !!(DomHelpers.getValue('glossarySelect') || '').trim();
    const hasInstructions = !!(DomHelpers.getValue('customInstructionSelect') || '').trim();

    if (queueOperation() === 'refine') {
        chips.push({ key: 'refineOnly', label: t('translation:summary_refine_only'), prominent: true });

        if (hasGlossary) {
            const name = getSelectText('glossarySelect').split('·')[0].trim();
            chips.push({ key: 'glossary', label: t('translation:summary_glossary', { name }) });
        }
        if (hasInstructions) {
            chips.push({ key: 'instructions', label: t('translation:summary_instructions', { name: getSelectText('customInstructionSelect') }) });
        }
        if (isChecked('disableAutoPause')) {
            chips.push({ key: 'noPause', label: t('translation:summary_no_auto_pause') });
        }
        return chips;
    }

    if (isChecked('bilingualMode'))     chips.push({ key: 'bilingual', label: t('translation:summary_bilingual') });
    if (isChecked('plainTextMode'))     chips.push({ key: 'plainText', label: t('translation:summary_plain_text_mode') });
    if (isChecked('textCleanup'))       chips.push({ key: 'ocr', label: t('translation:summary_ocr_cleanup') });
    if (isChecked('disableAutoPause'))  chips.push({ key: 'noPause', label: t('translation:summary_no_auto_pause') });

    if (hasGlossary) {
        const name = getSelectText('glossarySelect').split('·')[0].trim();
        chips.push({ key: 'glossary', label: t('translation:summary_glossary', { name }) });
    }

    if (hasInstructions) {
        chips.push({ key: 'instructions', label: t('translation:summary_instructions', { name: getSelectText('customInstructionSelect') }) });
    }

    return chips;
}

function renderLlmPart({ key, label }) {
    const style = [
        'cursor: pointer',
        'border-radius: 6px',
        'padding: 1px 4px',
        'transition: background 0.15s ease, color 0.15s ease',
    ].join('; ');
    return `<span class="summary-llm-part" data-summary-action="${key}" style="${style}">${DomHelpers.escapeHtml(label)}</span>`;
}

function renderChip({ key, label, prominent }) {
    const s = OPTION_STYLES[key] || OPTION_STYLES.bilingual;
    const style = [
        'display: inline-flex',
        'align-items: center',
        prominent ? 'padding: 6px 18px' : 'padding: 2px 10px',
        'border-radius: 999px',
        prominent ? 'font-size: 0.8125rem' : 'font-size: 0.75rem',
        'font-weight: 600',
        'line-height: 1.6',
        `background: ${s.bg}`,
        `color: ${s.fg}`,
        `border: ${prominent ? '1.5px' : '1px'} solid ${s.border}`,
        'cursor: pointer',
        'transition: transform 0.1s ease, filter 0.15s ease',
    ].join('; ');
    return `<span class="summary-chip" data-summary-action="${key}" style="${style}">${DomHelpers.escapeHtml(label)}</span>`;
}

function render() {
    const container = DomHelpers.getElement('settingsSummary');
    if (!container) return;

    const llmParts = buildLlmLine();
    const chips = buildChips();

    const sep = '<span style="opacity: 0.5; margin: 0 6px;">·</span>';
    const llmLine = llmParts.map(renderLlmPart).join(sep);

    const chipsHtml = chips.length
        ? `<div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; justify-content: center;">
               ${chips.map(renderChip).join('')}
           </div>`
        : '';

    container.innerHTML = `<div>${llmLine}</div>${chipsHtml}`;
}

function setSectionOpen(sectionKey, open) {
    const ids = SECTION_IDS[sectionKey];
    if (!ids) return;
    const section = DomHelpers.getElement(ids.section);
    const icon = DomHelpers.getElement(ids.icon);
    if (!section) return;
    const isHidden = section.classList.contains('hidden');
    if (open && isHidden) {
        section.classList.remove('hidden');
        if (icon) icon.style.transform = 'rotate(180deg)';
    } else if (!open && !isHidden) {
        section.classList.add('hidden');
        if (icon) icon.style.transform = 'rotate(0deg)';
    }
    if (ids.stateKey) {
        StateManager.setState(ids.stateKey, open);
    }
}

// Open the requested section and collapse the others, so only one is visible
// at a time — clicking a summary item should land the user on a clean view.
function openSection(sectionKey) {
    if (!SECTION_IDS[sectionKey]) return;
    for (const key of Object.keys(SECTION_IDS)) {
        setSectionOpen(key, key === sectionKey);
    }
}

function focusElement(id) {
    if (!id) return;
    const el = DomHelpers.getElement(id);
    if (!el) return;
    try {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (_) { /* older browsers */ }
    // Defer focus until the tab switch has settled; otherwise a hidden
    // ancestor will silently swallow the focus call.
    setTimeout(() => {
        try { el.focus({ preventScroll: true }); } catch (_) {
            try { el.focus(); } catch (_) { /* ignore */ }
        }
    }, 50);
}

function handleClick(event) {
    const target = event.target.closest('[data-summary-action]');
    if (!target) return;
    const action = target.getAttribute('data-summary-action');
    const dest = TARGETS[action];
    if (!dest) return;

    if (typeof window.switchTopTab === 'function') {
        window.switchTopTab(dest.tab);
    }
    if (dest.section) {
        openSection(dest.section);
    }
    if (dest.focus) {
        focusElement(dest.focus);
    }
}

function injectStyles() {
    if (document.getElementById('settings-summary-styles')) return;
    const style = document.createElement('style');
    style.id = 'settings-summary-styles';
    style.textContent = `
        #settingsSummary .summary-llm-part:hover {
            background: rgba(0, 0, 0, 0.06);
            color: var(--text-dark);
        }
        #settingsSummary .summary-chip:hover {
            transform: translateY(-1px);
            filter: brightness(0.95);
        }
        #settingsSummary [data-summary-action]:focus-visible {
            outline: 2px solid var(--primary-light, #3b82f6);
            outline-offset: 2px;
        }
    `;
    document.head.appendChild(style);
}

const WATCHED_IDS = [
    'llmProvider', 'model',
    'sourceLang', 'customSourceLang',
    'targetLang', 'customTargetLang',
    'bilingualMode', 'plainTextMode',
    'textCleanup', 'disableAutoPause',
    'glossarySelect', 'customInstructionSelect',
];

/**
 * Jump to the form control behind one of the TARGETS keys. Intended for reuse
 * by other modules (the Fallbacks recommendation panel) so they get the same
 * tab-switch + section-open + scroll-to-focus behaviour as the settings
 * summary chips.
 */
export function navigateToSetting(action) {
    const dest = TARGETS[action];
    if (!dest) return;
    if (typeof window.switchTopTab === 'function') {
        window.switchTopTab(dest.tab);
    }
    if (dest.section) {
        openSection(dest.section);
    }
    if (dest.focus) {
        focusElement(dest.focus);
    }
}

export const SettingsSummary = {
    initialize() {
        for (const id of WATCHED_IDS) {
            const el = DomHelpers.getElement(id);
            if (!el) continue;
            el.addEventListener('change', render);
            if (el.tagName === 'INPUT' && el.type === 'text') {
                el.addEventListener('input', render);
            }
        }
        // Several dropdowns are populated asynchronously (model list, custom
        // instructions). Those paths don't fire native change events, so we
        // also listen to the custom signals they emit after restoring state.
        window.addEventListener('modelChanged', render);
        window.addEventListener('customInstructionsLoaded', render);
        window.addEventListener('fileListChanged', render);
        window.addEventListener('localeChanged', render);

        const container = DomHelpers.getElement('settingsSummary');
        if (container) {
            container.addEventListener('click', handleClick);
            container.style.cursor = 'default';
        }
        injectStyles();

        render();
    },
    refresh: render,
};
