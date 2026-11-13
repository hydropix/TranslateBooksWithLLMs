/**
 * Glossary dropdown on the Translate tab
 *
 * Fills the glossary <select>, renders the info card under it (with the
 * language-mismatch warning), and wires its refresh / manage buttons.
 */

import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { t } from '../i18n/i18n.js';
import { switchTopTab } from '../ui/tab-router.js';
import { $, STORAGE_KEY } from './glossary-shared.js';
import { openEditor } from './glossary-editor.js';

// ========================================
// Translate-tab dropdown
// ========================================

let _dropdownGlossaries = [];

export async function refreshDropdown() {
    const select = $('glossarySelect');
    if (!select) return;

    const previous = localStorage.getItem(STORAGE_KEY) || '';

    let glossaries = [];
    try {
        const resp = await ApiClient.getGlossaries();
        glossaries = (resp && resp.glossaries) || [];
    } catch (err) {
        console.error('Failed to load glossaries for dropdown:', err);
    }
    _dropdownGlossaries = glossaries;

    select.innerHTML = '';

    const noneOpt = document.createElement('option');
    noneOpt.value = '';
    noneOpt.textContent = t('settings:select_none');
    select.appendChild(noneOpt);

    const autoOpt = document.createElement('option');
    autoOpt.value = '__auto__';
    autoOpt.textContent = t('settings:glossary_auto_option');
    autoOpt.setAttribute('data-i18n', 'settings:glossary_auto_option');
    select.appendChild(autoOpt);

    for (const g of glossaries) {
        const opt = document.createElement('option');
        opt.value = String(g.id);
        const src = g.source_lang || '?';
        const tgt = g.target_lang || '?';
        const count = (g.term_count != null) ? g.term_count : 0;
        const termWord = count === 1 ? t('glossary:dropdown_term_singular') : t('glossary:dropdown_term_plural');
        opt.textContent = t('glossary:dropdown_summary', { name: g.name, count, termWord, src, tgt });
        select.appendChild(opt);
    }

    const stillExists = glossaries.some((g) => String(g.id) === previous);
    select.value = (previous === '__auto__' || stillExists) ? previous : '';
    refreshGlossaryInfoCard();
    select.dispatchEvent(new Event('change', { bubbles: true }));
}

function _normalizeLang(s) {
    return String(s || '').trim().toLowerCase();
}

function refreshGlossaryInfoCard() {
    const card = $('glossaryInfoCard');
    const select = $('glossarySelect');
    if (!card || !select) return;
    const id = select.value || '';
    if (!id || id === '__auto__') {
        card.style.display = 'none';
        card.innerHTML = '';
        card.classList.remove('is-warning');
        return;
    }
    const g = _dropdownGlossaries.find((x) => String(x.id) === String(id));
    if (!g) {
        card.style.display = 'none';
        return;
    }

    const count = (g.term_count != null) ? g.term_count : 0;
    const termWord = count === 1 ? t('glossary:dropdown_term_singular') : t('glossary:dropdown_term_plural');
    const src = g.source_lang || '?';
    const tgt = g.target_lang || '?';

    const sourceLangEl = $('sourceLang');
    const targetLangEl = $('targetLang');
    const reqSrc = sourceLangEl ? (sourceLangEl.value || '') : '';
    const reqTgt = targetLangEl ? (targetLangEl.value || '') : '';

    const srcMismatch = reqSrc && g.source_lang && _normalizeLang(reqSrc) !== _normalizeLang(g.source_lang);
    const tgtMismatch = reqTgt && g.target_lang && _normalizeLang(reqTgt) !== _normalizeLang(g.target_lang);
    const hasWarning = srcMismatch || tgtMismatch;

    card.classList.toggle('is-warning', hasWarning);
    card.style.display = '';
    card.innerHTML = '';

    const meta = document.createElement('span');
    meta.innerHTML = `<strong>${DomHelpers.escapeHtml(g.name)}</strong> · ${count} ${termWord} · <span class="glossary-info-meta">${DomHelpers.escapeHtml(src)} → ${DomHelpers.escapeHtml(tgt)}</span>`;
    card.appendChild(meta);

    if (hasWarning) {
        const warn = document.createElement('span');
        warn.className = 'glossary-info-warn';
        const parts = [];
        if (srcMismatch) parts.push(t('glossary:lang_mismatch_source', { glossary: DomHelpers.escapeHtml(g.source_lang), request: DomHelpers.escapeHtml(reqSrc) }));
        if (tgtMismatch) parts.push(t('glossary:lang_mismatch_target', { glossary: DomHelpers.escapeHtml(g.target_lang), request: DomHelpers.escapeHtml(reqTgt) }));
        warn.innerHTML = `<span class="material-symbols-outlined" aria-hidden="true">warning</span> ${t('glossary:lang_mismatch', { parts: parts.join(', ') })}`;
        card.appendChild(warn);
    }

    const link = document.createElement('a');
    link.className = 'glossary-info-link';
    link.href = '#';
    link.textContent = t('glossary:info_view_link');
    link.addEventListener('click', (e) => {
        e.preventDefault();
        switchTopTab('glossaries');
        openEditor(g.id);
    });
    card.appendChild(link);
}

export function wireTranslateTabDropdown() {
    const select = $('glossarySelect');
    const refreshBtn = $('glossaryRefreshBtn');
    const manageBtn = $('glossaryManageBtn');

    if (select) {
        select.addEventListener('change', () => {
            localStorage.setItem(STORAGE_KEY, select.value || '');
            refreshGlossaryInfoCard();
        });
    }
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => { refreshDropdown(); });
    }
    if (manageBtn) {
        manageBtn.addEventListener('click', () => switchTopTab('glossaries'));
    }

    // Re-evaluate the warning when the user changes language selectors.
    const sourceLang = $('sourceLang');
    const targetLang = $('targetLang');
    if (sourceLang) sourceLang.addEventListener('change', refreshGlossaryInfoCard);
    if (targetLang) targetLang.addEventListener('change', refreshGlossaryInfoCard);
}
