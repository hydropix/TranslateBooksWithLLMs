/**
 * Term Fix Panel - rules editor and match preview for the term-fix queue
 *
 * The panel is visible while the queue holds files dropped on the term-fix
 * zone. The rules and mode apply to every file of the batch; they are kept in
 * localStorage as a per-viewer convenience so a long rule list survives a
 * reload. "Check matches" asks the server to count the matches per rule in
 * each queued file (no LLM call), so a forgotten variant shows up as 0 before
 * any money is spent.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { t } from '../i18n/i18n.js';

const RULES_STORAGE_KEY = 'tbl.termFix.rules';
const MODE_STORAGE_KEY = 'tbl.termFix.mode';
const OPERATION = 'term_fix';

// Last scan, re-rendered on locale switch: [{ name, rules } | { name, error }]
let lastScan = null;
let scanning = false;

function storageGet(key) {
    try {
        return localStorage.getItem(key);
    } catch {
        return null;
    }
}

function storageSet(key, value) {
    try {
        localStorage.setItem(key, value);
    } catch {
        // Private mode or blocked storage: the panel still works, it just forgets.
    }
}

function queuedTermFixFiles() {
    const files = StateManager.getState('files.toProcess') || [];
    return files.filter(f => (f.operation || 'translate') === OPERATION && f.status === 'Queued');
}

function renderResults() {
    const container = DomHelpers.getElement('termFixScanResults');
    if (!container) return;
    container.innerHTML = '';

    if (scanning) {
        const line = document.createElement('div');
        line.textContent = t('translation:term_fix_scanning');
        container.appendChild(line);
        return;
    }
    if (!lastScan) return;

    for (const entry of lastScan) {
        const fileLine = document.createElement('div');
        fileLine.className = 'term-fix-file';
        fileLine.textContent = entry.name;
        container.appendChild(fileLine);

        if (entry.error) {
            const err = document.createElement('div');
            err.className = 'term-fix-rule term-fix-error';
            err.textContent = entry.error;
            container.appendChild(err);
            continue;
        }

        for (const rule of entry.rules) {
            const ruleLine = document.createElement('div');
            ruleLine.className = 'term-fix-rule';
            const counts = rule.alternatives
                .map(alt => t('translation:term_fix_scan_alternative', { form: alt.text, count: alt.count }))
                .join(', ');
            ruleLine.textContent = t('translation:term_fix_scan_rule', {
                replacement: rule.replacement,
                total: rule.total,
                counts,
            });
            if (rule.alternatives.some(alt => alt.count === 0)) {
                ruleLine.classList.add('term-fix-zero');
            }
            container.appendChild(ruleLine);

            for (const sample of rule.samples) {
                const sampleLine = document.createElement('div');
                sampleLine.className = 'term-fix-sample';
                sampleLine.textContent = sample;
                container.appendChild(sampleLine);
            }
        }
    }
}

async function scan() {
    const rulesText = getRulesText();
    const files = queuedTermFixFiles();
    if (!rulesText.trim() || files.length === 0) {
        lastScan = [{ name: t('translation:term_fix_panel_title'), error: t('translation:term_fix_no_rules') }];
        renderResults();
        return;
    }

    scanning = true;
    renderResults();
    const results = [];
    for (const file of files) {
        try {
            const data = await ApiClient.scanTermFix(file.filePath, rulesText);
            results.push({ name: file.name, rules: data.rules || [] });
        } catch (error) {
            results.push({ name: file.name, error: error.message });
        }
    }
    scanning = false;
    lastScan = results;
    renderResults();
}

/**
 * The rules typed in the panel.
 * @returns {string}
 */
export function getRulesText() {
    return DomHelpers.getValue('termFixRules') || '';
}

/**
 * The selected mode: 'smart' or 'literal'.
 * @returns {string}
 */
export function getMode() {
    return DomHelpers.getValue('termFixMode') || 'smart';
}

export const TermFixPanel = {
    initialize() {
        const rules = DomHelpers.getElement('termFixRules');
        const mode = DomHelpers.getElement('termFixMode');
        const scanBtn = DomHelpers.getElement('termFixScanBtn');
        if (!rules || !mode || !scanBtn) return;

        const savedRules = storageGet(RULES_STORAGE_KEY);
        if (savedRules) rules.value = savedRules;
        const savedMode = storageGet(MODE_STORAGE_KEY);
        if (savedMode === 'smart' || savedMode === 'literal') mode.value = savedMode;

        rules.addEventListener('input', () => {
            storageSet(RULES_STORAGE_KEY, rules.value);
            // A stale preview would describe rules that no longer exist.
            lastScan = null;
            renderResults();
        });
        mode.addEventListener('change', () => storageSet(MODE_STORAGE_KEY, mode.value));
        scanBtn.addEventListener('click', (e) => {
            e.preventDefault();
            if (!scanning) scan();
        });

        window.addEventListener('fileListChanged', () => this.refreshVisibility());
        window.addEventListener('localeChanged', renderResults);
        this.refreshVisibility();
    },

    /** Show the panel only while the queue holds term-fix files. */
    refreshVisibility() {
        const panel = DomHelpers.getElement('termFixPanel');
        if (!panel) return;
        const files = StateManager.getState('files.toProcess') || [];
        const hasTermFix = files.some(f => (f.operation || 'translate') === OPERATION);
        panel.classList.toggle('hidden', !hasTermFix);
        if (!hasTermFix) {
            lastScan = null;
            renderResults();
        }
    },
};
