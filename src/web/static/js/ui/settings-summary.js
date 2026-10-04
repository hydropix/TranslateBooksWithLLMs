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
    poe: 'Poe',
    nim: 'NVIDIA NIM',
    opencode: 'Opencode',
};

// Every active option renders the same way: the chips all say "this run uses
// X", so hue carried no information and only added noise. They borrow
// .btn-primary's look (white on solid blue) at chip size. The colours live in
// style.css (--option-chip-*) with the rest of the theme, so they follow the
// active theme's primary without a dark-mode branch here.
const OPTION_STYLE = {
    bg: 'var(--option-chip-bg)',
    fg: 'var(--option-chip-fg)',
    border: 'var(--option-chip-border)',
};

// Provider / model / languages. Same chip, inked instead of blue: what the run
// *is* stays visually ahead of what the run *uses*.
const LLM_STYLE = {
    bg: 'var(--llm-chip-bg)',
    fg: 'var(--llm-chip-fg)',
    border: 'var(--llm-chip-bg)',
};

// Maps a summary item key to the tab + collapsible section it should reveal.
// `focus` is an optional element id to focus/scroll-to after switching.
// `panel: 'preflight'` marks the five per-run options that now live in the
// pre-flight panel on the Translate tab instead of a Settings accordion.
// Also reused by the Fallbacks recommendation panel (progress-manager.js) to
// jump to the relevant setting when the user clicks a link in the panel.
const TARGETS = {
    provider:     { tab: 'settings', section: 'settings', focus: 'llmProvider' },
    model:        { tab: 'settings', section: 'settings', focus: 'model' },
    languages:    { tab: 'translate', section: null,      focus: 'sourceLang' },
    noPause:      { tab: 'settings', section: 'settings', focus: 'disableAutoPause' },
    bilingual:    { tab: 'translate', section: null, focus: 'bilingualMode',           panel: 'preflight' },
    plainText:    { tab: 'translate', section: null, focus: 'plainTextMode',           panel: 'preflight' },
    ocr:          { tab: 'translate', section: null, focus: 'textCleanup',             panel: 'preflight' },
    glossary:     { tab: 'translate', section: null, focus: 'glossarySelect',          panel: 'preflight' },
    instructions: { tab: 'translate', section: null, focus: 'customInstructionSelect', panel: 'preflight' },
    refineOnly:   { tab: 'files',    section: null,       focus: null },
    // Not a summary chip: reached only through navigateToSetting(), by the
    // pre-flight zone's "Parallel requests" advice.
    parallelWorkers: { tab: 'settings', section: 'settings', focus: 'parallelWorkers' },
};

const SECTION_IDS = {
    settings: { section: 'settingsOptionsSection',     icon: 'settingsOptionsIcon',     stateKey: 'ui.isSettingsOptionsOpen' },
    notify:   { section: 'notificationOptionsSection', icon: 'notificationOptionsIcon', stateKey: null },
};

// Set by preflight-zone.js at init. Inverted rather than imported: this module
// is imported *by* preflight-zone.js, so a direct import would close a cycle.
let panelOpener = null;

/**
 * Register the callback that opens/closes the pre-flight panel.
 * @param {(open: boolean) => void} fn - Opener, typically PreflightZone.setPanelOpen
 */
export function registerPanelOpener(fn) {
    panelOpener = fn;
}

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

// Both rows are the same object at the same size, only the palette changes,
// so the geometry is written once. `prominent` (the refine-only chip) is the
// single deviation, and it deviates by size, never by colour.
function chipStyle({ bg, fg, border }, prominent = false) {
    return [
        'display: inline-flex',
        'align-items: center',
        prominent ? 'padding: 6px 18px' : 'padding: 2px 10px',
        'border-radius: 999px',
        prominent ? 'font-size: 0.8125rem' : 'font-size: 0.75rem',
        'font-weight: 600',
        'line-height: 1.6',
        `background: ${bg}`,
        `color: ${fg}`,
        `border: ${prominent ? '1.5px' : '1px'} solid ${border}`,
        'cursor: pointer',
        'transition: transform 0.1s ease, opacity 0.15s ease',
    ].join('; ');
}

// One row of chips, laid out like the other one.
function chipRow(html, marginTop) {
    return `<div style="margin-top: ${marginTop}; display: flex; flex-wrap: wrap; gap: 6px; justify-content: center;">${html}</div>`;
}

function renderLlmPart({ key, label }) {
    return `<span class="summary-llm-part" data-summary-action="${key}" style="${chipStyle(LLM_STYLE)}">${DomHelpers.escapeHtml(label)}</span>`;
}

function renderChip({ key, label, prominent }) {
    return `<span class="summary-chip" data-summary-action="${key}" style="${chipStyle(OPTION_STYLE, prominent)}">${DomHelpers.escapeHtml(label)}</span>`;
}

function render() {
    const container = DomHelpers.getElement('settingsSummary');
    if (!container) return;

    const llmParts = buildLlmLine();
    const chips = buildChips();

    // Provider, model and languages are three separate targets, each opening a
    // different setting, so they are three separate chips. The interpunct that
    // used to join them made them read as one sentence.
    const llmRow = chipRow(llmParts.map(renderLlmPart).join(''), '0');
    const chipsRow = chips.length ? chipRow(chips.map(renderChip).join(''), '8px') : '';

    container.innerHTML = `${llmRow}${chipsRow}`;
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

/**
 * Scroll a control into view, then focus it one macrotask later.
 * Exported so preflight-zone.js reuses the exact same sequence instead of
 * keeping a second copy of the deferred-focus workaround below.
 * @param {string} id - Element id to reveal
 */
export function focusElement(id) {
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

// The single implementation behind both the summary's own click handler and the
// exported navigateToSetting(): switch tab, reveal the container holding the
// control, then focus it.
function goToTarget(action) {
    const dest = TARGETS[action];
    if (!dest) return;

    if (typeof window.switchTopTab === 'function') {
        window.switchTopTab(dest.tab);
    }
    if (dest.section) {
        openSection(dest.section);
    }
    if (dest.panel === 'preflight') {
        // Guarded: the opener is only present once preflight-zone.js has run.
        panelOpener?.(true);
    }
    if (dest.focus) {
        focusElement(dest.focus);
    }
}

function handleClick(event) {
    const target = event.target.closest('[data-summary-action]');
    if (!target) return;
    goToTarget(target.getAttribute('data-summary-action'));
}

function injectStyles() {
    if (document.getElementById('settings-summary-styles')) return;
    const style = document.createElement('style');
    style.id = 'settings-summary-styles';
    style.textContent = `
        /* One hover for both rows. Opacity rather than brightness(): a
           multiplicative filter leaves a black chip black. */
        #settingsSummary .summary-chip:hover,
        #settingsSummary .summary-llm-part:hover {
            transform: translateY(-1px);
            opacity: 0.85;
        }
        #settingsSummary [data-summary-action]:focus-visible {
            outline: 2px solid var(--primary-light, #3b82f6);
            outline-offset: 2px;
        }
    `;
    document.head.appendChild(style);
}

// Exported so preflight-zone.js can subscribe to the exact same signals instead
// of keeping a second copy of this list, which would drift.
export const WATCHED_IDS = [
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
 * tab-switch + section-open + panel-open + scroll-to-focus behaviour as the
 * settings summary chips. Callers stay unchanged when a control moves between
 * containers: only its TARGETS entry does.
 * @param {string} action - TARGETS key
 */
export function navigateToSetting(action) {
    goToTarget(action);
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
