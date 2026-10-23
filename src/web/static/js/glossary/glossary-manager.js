/**
 * Glossary Manager
 *
 * Wires up the glossary UI: top-tab switching, the dropdown on the
 * Translate tab, the list/editor views on the Glossaries tab, the
 * inline term editing flow, import/export, and the NER auto-extract
 * modal. Talks to the backend through ApiClient and surfaces feedback
 * via the toast module.
 */

import { ApiClient } from '../core/api-client.js';
import { toast } from '../ui/toast.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import { t } from '../i18n/i18n.js';

// ========================================
// Module state
// ========================================

let currentGlossaryId = null;
let nerSelectedFile = null;
let nerLastCandidates = [];
const NER_ACCEPTED_EXTS = ['txt', 'srt', 'epub', 'docx', 'pdf'];

// In-memory cache of the terms for the currently open glossary so we can
// re-render after sort / filter / bulk operations without round-tripping.
let currentTerms = [];

// Last committed effective values for the lang selects (the visible select
// value can be "Other" while the real lang lives in the custom input, so we
// can't rely on per-element dataset.lastValue here).
const _lastMeta = { source_lang: '', target_lang: '' };

const STORAGE_KEY = 'selected_glossary_id';

const CATEGORY_OPTIONS = [
    { value: '',             labelKey: 'glossary:category_none' },
    { value: 'character',    labelKey: 'glossary:category_character' },
    { value: 'location',     labelKey: 'glossary:category_location' },
    { value: 'organization', labelKey: 'glossary:category_organization' },
    { value: 'item',         labelKey: 'glossary:category_item' },
    { value: 'title',        labelKey: 'glossary:category_title' },
    { value: 'other',        labelKey: 'glossary:category_other' },
];

// The empty value means "unknown": no gender information reaches the prompt,
// which is what every pre-existing glossary entry holds.
const GENDER_OPTIONS = [
    { value: '',       labelKey: 'glossary:gender_unknown' },
    { value: 'female', labelKey: 'glossary:gender_female' },
    { value: 'male',   labelKey: 'glossary:gender_male' },
];

// ========================================
// Helpers
// ========================================

function $(id) {
    return document.getElementById(id);
}

function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function isConflictError(err) {
    if (!err || !err.message) return false;
    const m = err.message.toLowerCase();
    return m.includes('already') || m.includes('conflict') || m.includes('unique') || m.includes('409');
}

function buildOptionSelect(field, options, currentValue) {
    const select = document.createElement('select');
    select.className = 'glossary-cell-input';
    select.dataset.field = field;
    for (const opt of options) {
        const o = document.createElement('option');
        o.value = opt.value;
        o.textContent = t(opt.labelKey);
        if ((currentValue || '') === opt.value) o.selected = true;
        select.appendChild(o);
    }
    return select;
}

function buildCategorySelect(currentValue) {
    return buildOptionSelect('category', CATEGORY_OPTIONS, currentValue);
}

function buildGenderSelect(currentValue) {
    return buildOptionSelect('gender', GENDER_OPTIONS, currentValue);
}

function genderLabel(value) {
    const opt = GENDER_OPTIONS.find((o) => o.value === (value || ''));
    return opt ? t(opt.labelKey) : (value || '');
}

function getRowFieldValues(tr) {
    const out = { source: '', target: '', category: '', gender: '' };
    const inputs = tr.querySelectorAll('.glossary-cell-input');
    inputs.forEach((el) => {
        const f = el.dataset.field;
        if (f) out[f] = el.value;
    });
    return out;
}

function flashRow(tr, color) {
    const prev = tr.style.backgroundColor;
    tr.style.backgroundColor = color;
    setTimeout(() => { tr.style.backgroundColor = prev; }, 800);
}

// Show "Saving..." (intermediate) / "Saved" (success) inside an indicator
// span next to a meta-field label.
function setSavedIndicator(indicatorEl, state) {
    if (!indicatorEl) return;
    const textEl = indicatorEl.querySelector('.text');
    const iconEl = indicatorEl.querySelector('.material-symbols-outlined');
    if (state === 'saving') {
        indicatorEl.classList.add('is-visible', 'is-saving');
        if (textEl) textEl.textContent = t('glossary:saving_indicator');
        if (iconEl) iconEl.textContent = 'progress_activity';
    } else if (state === 'saved') {
        indicatorEl.classList.add('is-visible');
        indicatorEl.classList.remove('is-saving');
        if (textEl) textEl.textContent = t('glossary:saved_indicator');
        if (iconEl) iconEl.textContent = 'check_circle';
        clearTimeout(indicatorEl._hideTimer);
        indicatorEl._hideTimer = setTimeout(() => {
            indicatorEl.classList.remove('is-visible');
        }, 1400);
    } else {
        indicatorEl.classList.remove('is-visible', 'is-saving');
    }
}

// ========================================
// Sort / filter / bulk-selection state (per-glossary, sessionStorage)
// ========================================

function _sortKey(gid) { return `glossary_sort_${gid}`; }
function _filterKey(gid) { return `glossary_filter_${gid}`; }

function getSortState(gid) {
    try {
        const raw = sessionStorage.getItem(_sortKey(gid));
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (parsed && parsed.key && (parsed.dir === 'asc' || parsed.dir === 'desc')) {
            return parsed;
        }
    } catch (_) { /* ignore */ }
    return null;
}

function setSortState(gid, state) {
    try {
        if (!state) sessionStorage.removeItem(_sortKey(gid));
        else sessionStorage.setItem(_sortKey(gid), JSON.stringify(state));
    } catch (_) { /* ignore */ }
}

function getFilterText(gid) {
    try {
        return sessionStorage.getItem(_filterKey(gid)) || '';
    } catch (_) { return ''; }
}

function setFilterText(gid, text) {
    try {
        if (!text) sessionStorage.removeItem(_filterKey(gid));
        else sessionStorage.setItem(_filterKey(gid), text);
    } catch (_) { /* ignore */ }
}

const _selectedTermIds = new Set();

function clearSelection() {
    _selectedTermIds.clear();
    refreshBulkBar();
    const selectAll = $('glossaryTermsSelectAll');
    if (selectAll) selectAll.checked = false;
    document.querySelectorAll('#glossaryTermsBody .glossary-row-check').forEach((cb) => {
        cb.checked = false;
    });
}

function refreshBulkBar() {
    const bar = $('glossaryBulkBar');
    const count = $('glossaryBulkCount');
    if (!bar) return;
    const n = _selectedTermIds.size;
    if (n > 0) {
        bar.style.display = '';
        if (count) count.textContent = String(n);
    } else {
        bar.style.display = 'none';
    }
}

// ========================================
// Top-tab switching
// ========================================

function switchTopTab(name) {
    const translateTab = $('tab-translate');
    const settingsTab = $('tab-settings');
    const glossariesTab = $('tab-glossaries');
    const filesTab = $('tab-files');
    const sampleTab = $('tab-sample');
    const stylesTab = $('tab-styles');
    const askTab = $('tab-ask');

    if (translateTab) translateTab.classList.toggle('hidden', name !== 'translate');
    if (settingsTab) settingsTab.classList.toggle('hidden', name !== 'settings');
    if (glossariesTab) glossariesTab.classList.toggle('hidden', name !== 'glossaries');
    if (filesTab) filesTab.classList.toggle('hidden', name !== 'files');
    if (sampleTab) sampleTab.classList.toggle('hidden', name !== 'sample');
    if (stylesTab) stylesTab.classList.toggle('hidden', name !== 'styles');
    if (askTab) askTab.classList.toggle('hidden', name !== 'ask');

    const buttons = document.querySelectorAll('#topTabNav .tab-btn');
    buttons.forEach((btn) => {
        const isActive = btn.dataset.tab === name;
        btn.classList.toggle('tab-btn-active', isActive);
    });

    if (name === 'glossaries') {
        loadList();
    } else if (name === 'files' && typeof window.refreshFileList === 'function') {
        window.refreshFileList();
    } else if (name === 'styles' && typeof window.refreshStyleList === 'function') {
        window.refreshStyleList();
    }
}

// ========================================
// Translate-tab dropdown
// ========================================

let _dropdownGlossaries = [];

async function refreshDropdown() {
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
    meta.innerHTML = `<strong>${escapeHtml(g.name)}</strong> · ${count} ${termWord} · <span class="glossary-info-meta">${escapeHtml(src)} → ${escapeHtml(tgt)}</span>`;
    card.appendChild(meta);

    if (hasWarning) {
        const warn = document.createElement('span');
        warn.className = 'glossary-info-warn';
        const parts = [];
        if (srcMismatch) parts.push(t('glossary:lang_mismatch_source', { glossary: escapeHtml(g.source_lang), request: escapeHtml(reqSrc) }));
        if (tgtMismatch) parts.push(t('glossary:lang_mismatch_target', { glossary: escapeHtml(g.target_lang), request: escapeHtml(reqTgt) }));
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

// ========================================
// Glossary list view
// ========================================

async function loadList() {
    const loading = $('glossaryListLoading');
    const empty = $('glossaryListEmpty');
    const table = $('glossaryListTable');
    const body = $('glossaryListBody');

    if (loading) loading.classList.remove('hidden');
    if (empty) empty.classList.add('hidden');
    if (table) table.classList.add('hidden');
    if (body) body.innerHTML = '';

    let resp;
    try {
        resp = await ApiClient.getGlossaries();
    } catch (err) {
        console.error('Failed to load glossary list:', err);
        if (loading) loading.classList.add('hidden');
        if (body) {
            body.innerHTML = `<tr><td colspan="5" style="color:#ef4444;">${t('glossary:load_failed_row', { error: escapeHtml(err.message || t('glossary:unknown_error')) })}</td></tr>`;
        }
        if (table) table.classList.remove('hidden');
        return;
    }

    const glossaries = (resp && resp.glossaries) || [];
    const count = (resp && resp.count != null) ? resp.count : glossaries.length;

    if (loading) loading.classList.add('hidden');

    if (count === 0) {
        if (empty) empty.classList.remove('hidden');
        if (table) table.classList.add('hidden');
        return;
    }

    if (table) table.classList.remove('hidden');
    if (empty) empty.classList.add('hidden');

    if (!body) return;
    body.innerHTML = '';
    const assignedId = localStorage.getItem(STORAGE_KEY) || '';
    for (const g of glossaries) {
        const tr = document.createElement('tr');
        tr.dataset.glossaryId = String(g.id);
        const isAssigned = String(g.id) === assignedId;

        const tdName = document.createElement('td');
        const link = document.createElement('a');
        link.href = '#';
        link.textContent = g.name;
        link.style.cursor = 'pointer';
        link.addEventListener('click', (e) => {
            e.preventDefault();
            openEditor(g.id);
        });
        tdName.appendChild(link);

        const tdSrc = document.createElement('td');
        tdSrc.textContent = g.source_lang || '—';

        const tdTgt = document.createElement('td');
        tdTgt.textContent = g.target_lang || '—';

        const tdCount = document.createElement('td');
        tdCount.className = 'col-center';
        tdCount.textContent = (g.term_count != null) ? String(g.term_count) : '—';

        const tdActions = document.createElement('td');
        tdActions.className = 'col-right';

        const actionsWrapper = document.createElement('div');
        actionsWrapper.style.display = 'inline-flex';
        actionsWrapper.style.gap = '0.25rem';
        actionsWrapper.style.alignItems = 'center';
        actionsWrapper.style.justifyContent = 'flex-end';

        const assignBtn = document.createElement('button');
        assignBtn.className = 'file-action-btn download';
        assignBtn.title = isAssigned ? t('glossary:unassign_title') : t('glossary:assign_title');
        assignBtn.setAttribute('aria-label', assignBtn.title);
        const assignIcon = isAssigned ? 'bookmark_added' : 'bookmark_add';
        assignBtn.innerHTML = `<span class="material-symbols-outlined" style="font-size: 0.875rem;">${assignIcon}</span>`;
        if (isAssigned) {
            assignBtn.style.color = '#10b981';
        }
        assignBtn.addEventListener('click', async () => {
            const select = $('glossarySelect');
            if (isAssigned) {
                localStorage.setItem(STORAGE_KEY, '');
                if (select) {
                    select.value = '';
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                }
                toast.success(t('glossary:unassigned_msg', { name: g.name }));
            } else {
                localStorage.setItem(STORAGE_KEY, String(g.id));
                if (select) {
                    const exists = Array.from(select.options).some((o) => o.value === String(g.id));
                    if (exists) {
                        select.value = String(g.id);
                        select.dispatchEvent(new Event('change', { bubbles: true }));
                    } else {
                        await refreshDropdown();
                    }
                } else {
                    await refreshDropdown();
                }
                toast.success(t('glossary:assigned_msg', { name: g.name }));
            }
            await loadList();
        });

        const editBtn = document.createElement('button');
        editBtn.className = 'file-action-btn download';
        editBtn.title = t('glossary:edit_glossary_title');
        editBtn.setAttribute('aria-label', editBtn.title);
        editBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">edit</span>';
        editBtn.addEventListener('click', () => openEditor(g.id));

        const dupBtn = document.createElement('button');
        dupBtn.className = 'file-action-btn download';
        dupBtn.title = t('glossary:duplicate_glossary_title');
        dupBtn.setAttribute('aria-label', dupBtn.title);
        dupBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">content_copy</span>';
        dupBtn.addEventListener('click', async () => {
            try {
                const dup = await ApiClient.duplicateGlossary(g.id);
                toast.success(t('glossary:duplicated_as', { name: dup.name }));
                await loadList();
                await refreshDropdown();
            } catch (err) {
                console.error('Duplicate glossary failed:', err);
                toast.error(t('glossary:duplicate_failed', { error: err.message || t('glossary:unknown_error') }));
            }
        });

        const delBtn = document.createElement('button');
        delBtn.className = 'file-action-btn delete';
        delBtn.title = t('glossary:delete_glossary_btn_title');
        delBtn.setAttribute('aria-label', delBtn.title);
        delBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">delete</span>';
        delBtn.addEventListener('click', async () => {
            if (!confirm(t('glossary:confirm_delete_with_undone', { name: g.name }))) return;
            try {
                await ApiClient.deleteGlossary(g.id);
                toast.success(t('glossary:deleted_glossary', { name: g.name }));
                await loadList();
                await refreshDropdown();
            } catch (err) {
                console.error('Delete glossary failed:', err);
                toast.error(t('glossary:delete_failed', { error: err.message || t('glossary:unknown_error') }));
            }
        });

        actionsWrapper.appendChild(assignBtn);
        actionsWrapper.appendChild(editBtn);
        actionsWrapper.appendChild(dupBtn);
        actionsWrapper.appendChild(delBtn);
        tdActions.appendChild(actionsWrapper);

        tr.appendChild(tdName);
        tr.appendChild(tdSrc);
        tr.appendChild(tdTgt);
        tr.appendChild(tdCount);
        tr.appendChild(tdActions);
        body.appendChild(tr);
    }
}

// ========================================
// New glossary
// ========================================

async function _generateUniqueDefaultName() {
    let existing = [];
    try {
        const resp = await ApiClient.getGlossaries();
        existing = ((resp && resp.glossaries) || []).map((g) => g.name || '');
    } catch (_) {
        existing = [];
    }
    const taken = new Set(existing);
    const base = t('glossary:new_glossary');
    if (!taken.has(base)) return base;
    for (let i = 2; i < 1000; i++) {
        const candidate = `${base} ${i}`;
        if (!taken.has(candidate)) return candidate;
    }
    return `${base} ${Date.now()}`;
}

async function handleNewGlossary() {
    let name;
    try {
        name = await _generateUniqueDefaultName();
    } catch (_) {
        name = t('glossary:new_glossary');
    }

    let created;
    try {
        created = await ApiClient.createGlossary({
            name,
            source_lang: '',
            target_lang: '',
        });
    } catch (err) {
        console.error('Create glossary failed:', err);
        toast.error(t('glossary:create_failed', { error: err.message || t('glossary:unknown_error') }));
        return;
    }

    // The create endpoint returns the full glossary dict — open the editor
    // directly from that data. Avoids a second GET that could fail on a
    // transient issue and would surface as a misleading "Failed to load"
    // toast even though the create itself succeeded.
    const data = (created && created.glossary) ? created.glossary : created;
    const newId = data && data.id;
    if (newId == null) {
        toast.warn(t('glossary:new_glossary_unexpected_response'));
        await loadList();
        refreshDropdown().catch(() => {});
        return;
    }

    _openEditorWithData(data);
    // Refresh the dropdown in the background; failures here are not fatal
    // for the UX (the user is already in the editor for the new glossary).
    refreshDropdown().catch((err) => {
        console.warn('Background dropdown refresh failed:', err);
    });

    const nameInput = $('glossaryEditorName');
    if (nameInput) {
        nameInput.focus();
        nameInput.select();
    }
    toast.success(t('glossary:new_glossary_created'));
}

// ========================================
// Editor view
// ========================================

async function openEditor(gid) {
    let g;
    try {
        g = await ApiClient.getGlossary(gid);
    } catch (err) {
        console.error('Failed to load glossary:', err);
        toast.error(t('glossary:load_failed', { error: err.message || t('glossary:unknown_error') }));
        backToList();
        return;
    }
    _openEditorWithData(g);
}

function _setLangSelect(selectId, customContainerId, customInputId, value) {
    const select = $(selectId);
    const customContainer = $(customContainerId);
    const customInput = $(customInputId);
    const v = value || '';

    if (!select) return;

    if (!v) {
        select.value = '';
        if (customContainer) customContainer.style.display = 'none';
        if (customInput) customInput.value = '';
        return;
    }

    const knownValues = Array.from(select.options)
        .map((o) => o.value)
        .filter((val) => val && val !== 'Other');

    if (knownValues.includes(v)) {
        select.value = v;
        if (customContainer) customContainer.style.display = 'none';
        if (customInput) customInput.value = '';
    } else {
        select.value = 'Other';
        if (customInput) customInput.value = v;
        if (customContainer) customContainer.style.display = 'block';
    }
}

function _resolveLangValue(selectId, customInputId) {
    const select = $(selectId);
    if (!select) return '';
    if (select.value === 'Other') {
        const c = $(customInputId);
        return c ? (c.value || '').trim() : '';
    }
    return select.value || '';
}

function _wireLangSelect(selectId, customContainerId, customInputId, fieldKey) {
    const select = $(selectId);
    const customContainer = $(customContainerId);
    const customInput = $(customInputId);

    if (select) {
        select.addEventListener('change', () => {
            if (select.value === 'Other') {
                if (customContainer) customContainer.style.display = 'block';
                if (customInput) customInput.focus();
                // Wait for the user to type — commit happens on blur of the
                // custom input. Committing now would save an empty value.
            } else {
                if (customContainer) customContainer.style.display = 'none';
                if (customInput) customInput.value = '';
                handleEditorMetaCommit(fieldKey);
            }
        });
    }

    if (customInput) {
        customInput.addEventListener('blur', () => handleEditorMetaCommit(fieldKey));
    }
}

function _openEditorWithData(g) {
    if (!g || g.id == null) return;
    currentGlossaryId = g.id;
    clearSelection();

    const listView = $('glossaryListView');
    const editorView = $('glossaryEditorView');
    if (listView) listView.classList.add('hidden');
    if (editorView) editorView.classList.remove('hidden');

    const title = $('glossaryEditorTitle');
    const nameInput = $('glossaryEditorName');

    if (title) title.textContent = g.name || '';
    if (nameInput) {
        nameInput.value = g.name || '';
        nameInput.dataset.lastValue = nameInput.value;
    }

    _setLangSelect(
        'glossaryEditorSourceLang',
        'glossaryEditorCustomSourceLangContainer',
        'glossaryEditorCustomSourceLang',
        g.source_lang,
    );
    _setLangSelect(
        'glossaryEditorTargetLang',
        'glossaryEditorCustomTargetLangContainer',
        'glossaryEditorCustomTargetLang',
        g.target_lang,
    );
    _lastMeta.source_lang = g.source_lang || '';
    _lastMeta.target_lang = g.target_lang || '';

    currentTerms = (g.terms || []).slice();

    const filterInput = $('glossaryTermsFilter');
    if (filterInput) {
        filterInput.value = getFilterText(g.id);
    }

    renderSortIndicators();
    rerenderTerms();
}

function _termSortValue(term, key) {
    if (key === 'source') return (term.source || '').toLowerCase();
    if (key === 'target') return (term.target || '').toLowerCase();
    if (key === 'category') return (term.category || '').toLowerCase();
    // Sorting on the raw value rather than the translated label keeps the
    // order stable across UI languages, and groups the blanks together so a
    // character missing a gender is easy to spot.
    if (key === 'gender') return (term.gender || '').toLowerCase();
    return '';
}

function _applySort(terms) {
    if (!currentGlossaryId) return terms.slice();
    const state = getSortState(currentGlossaryId);
    if (!state) return terms.slice();
    const dir = state.dir === 'desc' ? -1 : 1;
    return terms.slice().sort((a, b) => {
        const av = _termSortValue(a, state.key);
        const bv = _termSortValue(b, state.key);
        if (av < bv) return -1 * dir;
        if (av > bv) return 1 * dir;
        return 0;
    });
}

function _applyFilter(terms) {
    if (!currentGlossaryId) return terms;
    const q = (getFilterText(currentGlossaryId) || '').trim().toLowerCase();
    if (!q) return terms;
    return terms.filter((term) => {
        return (term.source || '').toLowerCase().includes(q)
            || (term.target || '').toLowerCase().includes(q)
            || (term.category || '').toLowerCase().includes(q)
            // Match both the canonical value and the localized label, so
            // "female" and "féminin" both find the same rows.
            || (term.gender || '').toLowerCase().includes(q)
            || (term.gender ? genderLabel(term.gender).toLowerCase().includes(q) : false);
    });
}

function rerenderTerms() {
    const empty = $('glossaryTermsEmpty');
    const table = $('glossaryTermsTable');
    const body = $('glossaryTermsBody');
    const countEl = $('glossaryTermsFilterCount');

    if (!body) return;

    const total = currentTerms.length;
    const visible = _applyFilter(_applySort(currentTerms));

    body.innerHTML = '';

    if (total === 0) {
        if (empty) empty.classList.remove('hidden');
        if (table) table.classList.add('hidden');
        if (countEl) countEl.textContent = '';
        return;
    }
    if (empty) empty.classList.add('hidden');
    if (table) table.classList.remove('hidden');

    for (const t of visible) {
        body.appendChild(buildTermRow(t));
    }

    if (countEl) {
        if (visible.length !== total) {
            countEl.textContent = t('glossary:term_filter_count', { visible: visible.length, total });
        } else {
            countEl.textContent = total === 1
                ? t('glossary:term_count_one', { count: total })
                : t('glossary:term_count_other', { count: total });
        }
    }

    refreshSelectAllCheckbox();
}

function refreshSelectAllCheckbox() {
    const selectAll = $('glossaryTermsSelectAll');
    if (!selectAll) return;
    const visibleBoxes = document.querySelectorAll('#glossaryTermsBody .glossary-row-check');
    if (visibleBoxes.length === 0) {
        selectAll.checked = false;
        selectAll.indeterminate = false;
        return;
    }
    let checked = 0;
    visibleBoxes.forEach((cb) => { if (cb.checked) checked += 1; });
    if (checked === 0) {
        selectAll.checked = false;
        selectAll.indeterminate = false;
    } else if (checked === visibleBoxes.length) {
        selectAll.checked = true;
        selectAll.indeterminate = false;
    } else {
        selectAll.checked = false;
        selectAll.indeterminate = true;
    }
}

function renderSortIndicators() {
    const headers = document.querySelectorAll('#glossaryTermsTable th.sortable');
    if (!headers || !headers.length) return;
    const state = currentGlossaryId ? getSortState(currentGlossaryId) : null;
    headers.forEach((th) => {
        const ind = th.querySelector('.sort-indicator');
        const isActive = state && state.key === th.dataset.sortKey;
        th.classList.toggle('sort-active', !!isActive);
        if (ind) {
            if (isActive) {
                ind.textContent = state.dir === 'desc' ? '↓' : '↑';
            } else {
                ind.textContent = '⇅';
            }
        }
    });
}

function buildTermRow(term) {
    const tr = document.createElement('tr');
    tr.dataset.termId = (term && term.id != null) ? String(term.id) : 'new';

    // Selection checkbox
    const tdSelect = document.createElement('td');
    tdSelect.className = 'col-center';
    if (term && term.id != null) {
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.className = 'glossary-row-check';
        cb.checked = _selectedTermIds.has(term.id);
        cb.addEventListener('change', () => {
            if (cb.checked) _selectedTermIds.add(term.id);
            else _selectedTermIds.delete(term.id);
            refreshBulkBar();
            refreshSelectAllCheckbox();
        });
        tdSelect.appendChild(cb);
    }

    const mkInputCell = (field, value) => {
        const td = document.createElement('td');
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.className = 'glossary-cell-input';
        inp.dataset.field = field;
        inp.value = value || '';
        inp.dataset.lastValue = inp.value;
        if (field === 'source') {
            inp.placeholder = t('glossary:term_source_placeholder');
            inp.title = t('glossary:term_source_title');
        }
        td.appendChild(inp);
        return td;
    };

    const tdSource = mkInputCell('source', term && term.source);
    const tdTarget = mkInputCell('target', term && term.target);

    const tdCategory = document.createElement('td');
    const catSelect = buildCategorySelect(term && term.category);
    catSelect.dataset.lastValue = catSelect.value;
    tdCategory.appendChild(catSelect);

    const tdGender = document.createElement('td');
    const genderSelect = buildGenderSelect(term && term.gender);
    genderSelect.dataset.lastValue = genderSelect.value;
    genderSelect.title = t('glossary:gender_cell_title');
    tdGender.appendChild(genderSelect);

    const tdActions = document.createElement('td');
    tdActions.className = 'col-center';
    const delBtn = document.createElement('button');
    delBtn.type = 'button';
    delBtn.className = 'file-action-btn delete';
    delBtn.title = t('glossary:delete_term_title');
    delBtn.setAttribute('aria-label', delBtn.title);
    delBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">delete</span>';
    delBtn.addEventListener('click', () => handleDeleteRow(tr));
    tdActions.appendChild(delBtn);

    tr.appendChild(tdSelect);
    tr.appendChild(tdSource);
    tr.appendChild(tdTarget);
    tr.appendChild(tdCategory);
    tr.appendChild(tdGender);
    tr.appendChild(tdActions);

    const fields = tr.querySelectorAll('.glossary-cell-input');
    fields.forEach((el) => {
        if (el.tagName === 'SELECT') {
            el.addEventListener('change', () => handleFieldCommit(tr, el));
        } else {
            el.addEventListener('blur', () => handleFieldCommit(tr, el));
        }
    });

    return tr;
}

/**
 * Serialize commits per row so two quick blurs can't race.
 */
function _enqueueRowCommit(tr, work) {
    const previous = tr._commitChain || Promise.resolve();
    const next = previous.catch(() => {}).then(work);
    tr._commitChain = next;
    return next;
}

function _updateLocalTerm(termId, patch) {
    const idx = currentTerms.findIndex((term) => term && term.id === termId);
    if (idx >= 0) currentTerms[idx] = Object.assign({}, currentTerms[idx], patch);
}

function _removeLocalTerm(termId) {
    currentTerms = currentTerms.filter((term) => !term || term.id !== termId);
}

async function _doFieldCommit(tr, el) {
    if (currentGlossaryId == null) return;

    const tid = tr.dataset.termId;
    const field = el.dataset.field;
    const newValue = el.value;
    const oldValue = el.dataset.lastValue || '';

    if (newValue === oldValue) return;

    const values = getRowFieldValues(tr);

    if (tid === 'new') {
        if (!values.source.trim()) {
            el.dataset.lastValue = newValue;
            return;
        }
        try {
            const created = await ApiClient.addGlossaryTerm(currentGlossaryId, {
                source: values.source,
                target: values.target,
                category: values.category,
                gender: values.gender,
            });
            if (created && created.id != null) {
                tr.dataset.termId = String(created.id);
                tr.querySelectorAll('.glossary-cell-input').forEach((f) => {
                    f.dataset.lastValue = f.value;
                });
                currentTerms.push({
                    id: created.id,
                    source: created.source || values.source,
                    target: created.target || values.target,
                    category: created.category || values.category || '',
                    gender: created.gender || values.gender || '',
                });
                flashRow(tr, 'rgba(34, 197, 94, 0.15)');
            }
        } catch (err) {
            console.error('Add term failed:', err);
            if (isConflictError(err)) {
                flashRow(tr, 'rgba(239, 68, 68, 0.25)');
                const srcEl = tr.querySelector('.glossary-cell-input[data-field="source"]');
                if (srcEl) {
                    srcEl.value = '';
                    srcEl.dataset.lastValue = '';
                }
                toast.error(t('glossary:term_add_conflict'));
            } else {
                toast.error(t('glossary:term_add_failed', { error: err.message || t('glossary:unknown_error') }));
            }
        }
        return;
    }

    if (field === 'source' && !newValue.trim()) {
        try {
            await ApiClient.deleteGlossaryTerm(currentGlossaryId, tid);
            const numId = parseInt(tid, 10);
            _removeLocalTerm(numId);
            _selectedTermIds.delete(numId);
            tr.remove();
            checkTermsEmptyState();
            refreshBulkBar();
        } catch (err) {
            console.error('Delete term failed:', err);
            toast.error(t('glossary:term_delete_failed', { error: err.message || t('glossary:unknown_error') }));
            el.value = oldValue;
        }
        return;
    }

    const payload = {};
    payload[field] = newValue;
    try {
        await ApiClient.updateGlossaryTerm(currentGlossaryId, tid, payload);
        el.dataset.lastValue = newValue;
        const numId = parseInt(tid, 10);
        _updateLocalTerm(numId, { [field]: newValue });
        flashRow(tr, 'rgba(34, 197, 94, 0.15)');
    } catch (err) {
        console.error('Update term failed:', err);
        if (isConflictError(err)) {
            el.value = oldValue;
            flashRow(tr, 'rgba(239, 68, 68, 0.25)');
            toast.error(t('glossary:term_update_conflict'));
        } else {
            el.value = oldValue;
            toast.error(t('glossary:term_update_failed', { error: err.message || t('glossary:unknown_error') }));
        }
    }
}

function handleFieldCommit(tr, el) {
    return _enqueueRowCommit(tr, () => _doFieldCommit(tr, el));
}

async function handleDeleteRow(tr) {
    const tid = tr.dataset.termId;
    if (tid === 'new') {
        tr.remove();
        checkTermsEmptyState();
        return;
    }
    if (!confirm(t('glossary:delete_term_confirm'))) return;
    return _enqueueRowCommit(tr, async () => {
        try {
            await ApiClient.deleteGlossaryTerm(currentGlossaryId, tid);
            const numId = parseInt(tid, 10);
            _removeLocalTerm(numId);
            _selectedTermIds.delete(numId);
            tr.remove();
            checkTermsEmptyState();
            refreshBulkBar();
        } catch (err) {
            console.error('Delete term failed:', err);
            toast.error(t('glossary:term_delete_failed', { error: err.message || t('glossary:unknown_error') }));
        }
    });
}

function checkTermsEmptyState() {
    const body = $('glossaryTermsBody');
    const empty = $('glossaryTermsEmpty');
    const table = $('glossaryTermsTable');
    if (!body) return;
    const hasRows = body.children.length > 0 || currentTerms.length > 0;
    if (empty) empty.classList.toggle('hidden', hasRows);
    if (table) table.classList.toggle('hidden', !hasRows);
}

function handleAddRow() {
    if (currentGlossaryId == null) return;
    const body = $('glossaryTermsBody');
    const empty = $('glossaryTermsEmpty');
    const table = $('glossaryTermsTable');
    if (!body) return;

    const tr = buildTermRow({ id: null, source: '', target: '', category: '', gender: '' });
    body.appendChild(tr);

    if (empty) empty.classList.add('hidden');
    if (table) table.classList.remove('hidden');

    const srcEl = tr.querySelector('.glossary-cell-input[data-field="source"]');
    if (srcEl) srcEl.focus();
}

// Serialize editor-meta commits the same way per-row commits are serialized.
let _editorMetaChain = Promise.resolve();

function handleEditorMetaCommit(fieldKey) {
    _editorMetaChain = _editorMetaChain.catch(() => {}).then(() => _doEditorMetaCommit(fieldKey));
    return _editorMetaChain;
}

function _indicatorFor(fieldKey) {
    if (fieldKey === 'name') return $('glossaryEditorNameSaved');
    if (fieldKey === 'source_lang') return $('glossaryEditorSourceLangSaved');
    if (fieldKey === 'target_lang') return $('glossaryEditorTargetLangSaved');
    return null;
}

async function _doEditorMetaCommit(fieldKey) {
    if (currentGlossaryId == null) return;

    const nameInput = $('glossaryEditorName');
    const newName = nameInput ? nameInput.value : '';
    const newSrc = _resolveLangValue('glossaryEditorSourceLang', 'glossaryEditorCustomSourceLang');
    const newTgt = _resolveLangValue('glossaryEditorTargetLang', 'glossaryEditorCustomTargetLang');

    let changed = false;
    if (fieldKey === 'name') {
        const oldName = (nameInput && nameInput.dataset.lastValue) || '';
        if (newName !== oldName) changed = true;
    } else if (fieldKey === 'source_lang') {
        if (newSrc !== _lastMeta.source_lang) changed = true;
    } else if (fieldKey === 'target_lang') {
        if (newTgt !== _lastMeta.target_lang) changed = true;
    }
    if (!changed) return;

    const indicator = _indicatorFor(fieldKey);

    // Show "saving" only if the request is slow (>150ms), otherwise jump
    // directly to "saved" — keeps quick LAN saves from flickering.
    let savingTimer = null;
    if (indicator) {
        savingTimer = setTimeout(() => setSavedIndicator(indicator, 'saving'), 150);
    }

    const payload = {
        name: newName,
        source_lang: newSrc,
        target_lang: newTgt,
    };

    try {
        await ApiClient.updateGlossary(currentGlossaryId, payload);
        if (fieldKey === 'name') {
            if (nameInput) nameInput.dataset.lastValue = newName;
            const title = $('glossaryEditorTitle');
            if (title) title.textContent = newName;
        } else if (fieldKey === 'source_lang') {
            _lastMeta.source_lang = newSrc;
        } else if (fieldKey === 'target_lang') {
            _lastMeta.target_lang = newTgt;
        }
        if (savingTimer) clearTimeout(savingTimer);
        if (indicator) setSavedIndicator(indicator, 'saved');
        refreshDropdown().catch(() => {});
    } catch (err) {
        console.error('Update glossary failed:', err);
        if (savingTimer) clearTimeout(savingTimer);
        if (indicator) setSavedIndicator(indicator, null);
        if (isConflictError(err) && fieldKey === 'name') {
            const oldName = (nameInput && nameInput.dataset.lastValue) || '';
            if (nameInput) nameInput.value = oldName;
            toast.error(t('glossary:name_conflict'));
        } else {
            toast.error(t('glossary:save_failed', { error: err.message || t('glossary:unknown_error') }));
        }
    }
}

function backToList() {
    const listView = $('glossaryListView');
    const editorView = $('glossaryEditorView');
    if (editorView) editorView.classList.add('hidden');
    if (listView) listView.classList.remove('hidden');
    currentGlossaryId = null;
    currentTerms = [];
    clearSelection();
    loadList();
    refreshDropdown().catch(() => {});
}

async function handleDeleteCurrent() {
    if (currentGlossaryId == null) return;
    if (!confirm(t('glossary:confirm_delete_current'))) return;
    try {
        await ApiClient.deleteGlossary(currentGlossaryId);
        toast.success(t('glossary:deleted_glossary_simple'));
        backToList();
    } catch (err) {
        console.error('Delete glossary failed:', err);
        toast.error(t('glossary:delete_failed', { error: err.message || t('glossary:unknown_error') }));
    }
}

// ========================================
// Sort / Filter wiring
// ========================================

function handleHeaderSortClick(th) {
    if (currentGlossaryId == null) return;
    const key = th.dataset.sortKey;
    if (!key) return;
    const current = getSortState(currentGlossaryId);
    let next;
    if (!current || current.key !== key) {
        next = { key, dir: 'asc' };
    } else if (current.dir === 'asc') {
        next = { key, dir: 'desc' };
    } else {
        next = null;
    }
    setSortState(currentGlossaryId, next);
    renderSortIndicators();
    rerenderTerms();
}

function handleFilterInput(e) {
    if (currentGlossaryId == null) return;
    const text = e.target.value || '';
    setFilterText(currentGlossaryId, text);
    rerenderTerms();
}

// ========================================
// Bulk actions
// ========================================

async function handleBulkDelete() {
    const ids = Array.from(_selectedTermIds);
    if (ids.length === 0) return;
    const confirmMsg = ids.length === 1
        ? t('glossary:bulk_delete_confirm_one')
        : t('glossary:bulk_delete_confirm_other', { count: ids.length });
    if (!confirm(confirmMsg)) return;
    try {
        const resp = await ApiClient.bulkGlossaryTerms(currentGlossaryId, {
            action: 'delete',
            term_ids: ids,
        });
        const n = (resp && resp.deleted) || 0;
        for (const id of ids) {
            _removeLocalTerm(id);
        }
        _selectedTermIds.clear();
        refreshBulkBar();
        rerenderTerms();
        toast.success(n === 1 ? t('glossary:bulk_deleted_one') : t('glossary:bulk_deleted_other', { count: n }));
    } catch (err) {
        console.error('Bulk delete failed:', err);
        toast.error(t('glossary:bulk_delete_failed', { error: err.message || t('glossary:unknown_error') }));
    }
}

async function handleBulkSetCategory() {
    const ids = Array.from(_selectedTermIds);
    if (ids.length === 0) return;
    const sel = $('glossaryBulkCategorySelect');
    const category = sel ? (sel.value || '') : '';
    try {
        const resp = await ApiClient.bulkGlossaryTerms(currentGlossaryId, {
            action: 'set_category',
            term_ids: ids,
            category,
        });
        const n = (resp && resp.updated) || 0;
        for (const id of ids) {
            _updateLocalTerm(id, { category: category || '' });
        }
        rerenderTerms();
        toast.success(n === 1 ? t('glossary:bulk_category_updated_one') : t('glossary:bulk_category_updated_other', { count: n }));
    } catch (err) {
        console.error('Bulk set-category failed:', err);
        toast.error(t('glossary:bulk_category_failed', { error: err.message || t('glossary:unknown_error') }));
    }
}

async function handleBulkSetGender() {
    const ids = Array.from(_selectedTermIds);
    if (ids.length === 0) return;
    const sel = $('glossaryBulkGenderSelect');
    const gender = sel ? (sel.value || '') : '';
    try {
        const resp = await ApiClient.bulkGlossaryTerms(currentGlossaryId, {
            action: 'set_gender',
            term_ids: ids,
            gender,
        });
        const n = (resp && resp.updated) || 0;
        for (const id of ids) {
            _updateLocalTerm(id, { gender: gender || '' });
        }
        rerenderTerms();
        toast.success(n === 1 ? t('glossary:bulk_gender_updated_one') : t('glossary:bulk_gender_updated_other', { count: n }));
    } catch (err) {
        console.error('Bulk set-gender failed:', err);
        toast.error(t('glossary:bulk_gender_failed', { error: err.message || t('glossary:unknown_error') }));
    }
}

// ========================================
// Import / Export
// ========================================

// Pick the singular or plural variant of a count-bearing key. The i18n layer
// leaves plural selection to the caller (see the bulk_* handlers), so this
// keeps the choice in one place instead of at every call site.
function pluralized(baseKey, count) {
    return count === 1
        ? t(`${baseKey}_one`)
        : t(`${baseKey}_other`, { count });
}

async function handleImportFile(file) {
    if (currentGlossaryId == null || !file) return;
    try {
        const result = await ApiClient.importGlossaryTerms(currentGlossaryId, file);
        const imported = (result && result.imported != null) ? result.imported : 0;
        const skippedEmpty = (result && result.skipped_empty) || 0;
        const skippedDuplicate = (result && result.skipped_duplicate) || 0;

        let message = pluralized('glossary:import_success', imported);
        if (skippedEmpty > 0 || skippedDuplicate > 0) {
            const parts = [];
            if (skippedEmpty > 0) {
                parts.push(pluralized('glossary:import_skipped_empty', skippedEmpty));
            }
            if (skippedDuplicate > 0) {
                parts.push(pluralized('glossary:import_skipped_duplicate', skippedDuplicate));
            }
            // The separator and the surrounding punctuation are localized too:
            // CJK uses '、' and fullwidth parentheses, not ASCII.
            message += t('glossary:import_skipped_suffix', {
                details: parts.join(t('glossary:list_separator')),
            });
        }
        toast.success(message);
        await openEditor(currentGlossaryId);
    } catch (err) {
        console.error('Import failed:', err);
        toast.error(t('glossary:import_failed', { error: err.message || t('glossary:unknown_error') }));
    }
}

function handleExport(format) {
    if (currentGlossaryId == null) return;
    const url = ApiClient.getGlossaryExportUrl(currentGlossaryId, format);
    if (url) window.location.href = url;
}

// ========================================
// Drag & drop import (editor + empty list)
// ========================================

function _wireDragDropImport(targetEl, onFile) {
    if (!targetEl) return;
    targetEl.classList.add('glossary-drop-target');
    let depth = 0;
    targetEl.addEventListener('dragenter', (e) => {
        if (!e.dataTransfer || !e.dataTransfer.types) return;
        if (Array.prototype.indexOf.call(e.dataTransfer.types, 'Files') < 0) return;
        e.preventDefault();
        depth += 1;
        targetEl.classList.add('is-drag-over');
    });
    targetEl.addEventListener('dragover', (e) => {
        if (!e.dataTransfer || !e.dataTransfer.types) return;
        if (Array.prototype.indexOf.call(e.dataTransfer.types, 'Files') < 0) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });
    targetEl.addEventListener('dragleave', () => {
        depth = Math.max(0, depth - 1);
        if (depth === 0) targetEl.classList.remove('is-drag-over');
    });
    targetEl.addEventListener('drop', (e) => {
        if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) return;
        const file = e.dataTransfer.files[0];
        const lower = (file.name || '').toLowerCase();
        if (!lower.endsWith('.csv') && !lower.endsWith('.json')) {
            toast.warn(t('glossary:drop_only_csv_json'));
            return;
        }
        e.preventDefault();
        depth = 0;
        targetEl.classList.remove('is-drag-over');
        onFile(file);
    });
}

// ========================================
// NER auto-extract modal
// ========================================

function _formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function _fileExt(name) {
    const dot = (name || '').lastIndexOf('.');
    return dot >= 0 ? name.slice(dot + 1).toLowerCase() : '';
}

function _setNerSelectedFile(file) {
    nerSelectedFile = file || null;
    const pill = $('ner-file-pill');
    const dropzone = $('ner-drop-zone');
    const nameEl = $('ner-file-pill-name');
    const sizeEl = $('ner-file-pill-size');
    const extractBtn = $('ner-extract-btn');

    if (file) {
        if (pill) pill.classList.remove('hidden');
        if (dropzone) dropzone.classList.add('hidden');
        if (nameEl) nameEl.textContent = file.name || '';
        if (sizeEl) sizeEl.textContent = _formatFileSize(file.size);
        if (extractBtn) extractBtn.disabled = false;
    } else {
        if (pill) pill.classList.add('hidden');
        if (dropzone) dropzone.classList.remove('hidden');
        if (nameEl) nameEl.textContent = '';
        if (sizeEl) sizeEl.textContent = '';
        if (extractBtn) extractBtn.disabled = true;
    }
}

function _validateNerFile(file) {
    if (!file || !file.name) return false;
    const ext = _fileExt(file.name);
    if (!NER_ACCEPTED_EXTS.includes(ext)) {
        toast.warn(t('glossary:ner_unsupported_ext', { ext, accepted: NER_ACCEPTED_EXTS.join(', ') }));
        return false;
    }
    return true;
}

function _syncNerLabels() {
    const budgetInput = $('ner-max-chars');
    const budgetLabel = $('ner-budget-label');
    if (budgetInput && budgetLabel) {
        const v = parseInt(budgetInput.value, 10);
        budgetLabel.textContent = Number.isFinite(v) && v > 0 ? String(v) : '6000';
    }
    const sampleInput = $('ner-sample-count');
    const sampleLabel = $('ner-samples-label');
    if (sampleInput && sampleLabel) {
        const v = parseInt(sampleInput.value, 10);
        sampleLabel.textContent = Number.isFinite(v) && v > 0 ? String(v) : '10';
    }
}

function openNerModal() {
    if (currentGlossaryId == null) return;
    const modal = $('ner-modal');
    const inputStep = $('ner-input-step');
    const resultsStep = $('ner-results-step');
    const extractBtn = $('ner-extract-btn');
    const addBtn = $('ner-add-selected-btn');
    const status = $('ner-status');
    const warnings = $('ner-warnings');
    const fileInput = $('ner-file-input');

    if (!modal) return;
    modal.classList.remove('hidden');
    if (inputStep) inputStep.classList.remove('hidden');
    if (resultsStep) resultsStep.classList.add('hidden');
    if (extractBtn) extractBtn.classList.remove('hidden');
    if (addBtn) {
        addBtn.classList.add('hidden');
        addBtn.disabled = false;
    }
    if (status) status.innerHTML = '';
    if (warnings) {
        warnings.classList.add('hidden');
        warnings.innerHTML = '';
    }
    if (fileInput) fileInput.value = '';

    const selectAll = $('ner-select-all');
    if (selectAll) selectAll.checked = false;

    const candidatesBody = $('ner-candidates-body');
    if (candidatesBody) candidatesBody.innerHTML = '';

    nerLastCandidates = [];

    _setNerSelectedFile(null);
    _syncNerLabels();
}

function closeNerModal() {
    const modal = $('ner-modal');
    if (modal) modal.classList.add('hidden');
    _setNerSelectedFile(null);
    const fileInput = $('ner-file-input');
    if (fileInput) fileInput.value = '';
}

function _setNerLoading(active) {
    const status = $('ner-status');
    if (!status) return;
    if (active) {
        status.innerHTML = `<span class="glossary-ner-loading"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span> ${t('glossary:ner_extracting')}`;
    } else {
        status.textContent = '';
    }
}

async function handleNerExtract() {
    if (currentGlossaryId == null) return;
    if (!nerSelectedFile) {
        toast.warn(t('glossary:ner_no_file_selected'));
        return;
    }

    const maxCharsEl = $('ner-max-chars');
    const sampleCountEl = $('ner-sample-count');
    const extractBtn = $('ner-extract-btn');

    if (extractBtn) extractBtn.disabled = true;
    _setNerLoading(true);

    const srcLang = _resolveLangValue('glossaryEditorSourceLang', 'glossaryEditorCustomSourceLang');
    const tgtLang = _resolveLangValue('glossaryEditorTargetLang', 'glossaryEditorCustomTargetLang');
    const maxChars = maxCharsEl ? parseInt(maxCharsEl.value, 10) : NaN;
    const sampleCount = sampleCountEl ? parseInt(sampleCountEl.value, 10) : NaN;

    // Reuse the provider/model/endpoint/key from the main translate form so
    // NER targets the same backend the user has configured for translation.
    const provider = (DomHelpers.getValue('llmProvider') || '').trim();
    const model = (DomHelpers.getValue('model') || '').trim();
    const apiEndpoint = (provider === 'openai'
        ? DomHelpers.getValue('openaiEndpoint')
        : DomHelpers.getValue('apiEndpoint')) || '';
    const apiKey = provider ? ApiKeyUtils.getValueForProvider(provider) : '';

    const payload = new FormData();
    payload.append('file', nerSelectedFile);
    if (srcLang) payload.append('source_lang', srcLang);
    if (tgtLang) payload.append('target_lang', tgtLang);
    if (Number.isFinite(maxChars) && maxChars > 0) {
        payload.append('max_chars', String(maxChars));
    }
    if (Number.isFinite(sampleCount) && sampleCount > 0) {
        payload.append('sample_count', String(sampleCount));
    }
    if (provider) payload.append('provider', provider);
    if (model) payload.append('model', model);
    if (apiEndpoint) payload.append('api_endpoint', apiEndpoint);
    if (apiKey) payload.append('api_key', apiKey);

    let resp;
    try {
        resp = await ApiClient.suggestGlossaryTerms(currentGlossaryId, payload);
    } catch (err) {
        console.error('NER extract failed:', err);
        toast.error(t('glossary:ner_extract_failed', { error: err.message || t('glossary:unknown_error') }));
        if (extractBtn) extractBtn.disabled = false;
        _setNerLoading(false);
        return;
    }

    const candidates = (resp && resp.candidates) || [];
    const warnings = (resp && resp.warnings) || [];
    const count = (resp && resp.count != null) ? resp.count : candidates.length;
    const newCount = (resp && resp.new_count != null) ? resp.new_count : count;
    const sampleFilename = (resp && resp.sample_filename) || null;
    const respSampleCount = (resp && resp.sample_count) || 1;
    const sampleChars = (resp && resp.sample_chars) || 0;
    const fullTextChars = (resp && resp.full_text_chars) || 0;

    const inputStep = $('ner-input-step');
    const resultsStep = $('ner-results-step');
    if (inputStep) inputStep.classList.add('hidden');
    if (resultsStep) resultsStep.classList.remove('hidden');

    const countEl = $('ner-results-count');
    if (countEl) countEl.textContent = String(count);
    const newCountEl = $('ner-results-new-count');
    if (newCountEl) {
        let suffix = '';
        if (count > 0 && newCount < count) {
            suffix = t('glossary:ner_results_with_new', { new: newCount, existing: count - newCount });
        } else if (count > 0) {
            suffix = t('glossary:ner_results_only_new', { count: newCount });
        }
        if (sampleFilename) {
            const samplesPart = respSampleCount > 1
                ? t('glossary:ner_results_sample_excerpts', { count: respSampleCount })
                : '';
            const charsPart = sampleChars
                ? t('glossary:ner_results_sample_chars', { chars: sampleChars.toLocaleString() })
                : '';
            const fullPart = fullTextChars
                ? t('glossary:ner_results_full_chars', { chars: fullTextChars.toLocaleString() })
                : '';
            suffix += ` — ${sampleFilename}${samplesPart}${charsPart}${fullPart}`;
        }
        newCountEl.textContent = suffix;
    }

    const warningsEl = $('ner-warnings');
    if (warningsEl) {
        if (warnings.length > 0) {
            warningsEl.classList.remove('hidden');
            const ul = document.createElement('ul');
            for (const w of warnings) {
                const li = document.createElement('li');
                li.textContent = String(w);
                ul.appendChild(li);
            }
            warningsEl.innerHTML = '';
            warningsEl.appendChild(ul);
        } else {
            warningsEl.classList.add('hidden');
            warningsEl.innerHTML = '';
        }
    }

    nerLastCandidates = candidates;
    const body = $('ner-candidates-body');
    if (body) {
        body.innerHTML = '';
        for (const c of candidates) {
            body.appendChild(buildNerRow(c));
        }
    }

    const selectAll = $('ner-select-all');
    if (selectAll) selectAll.checked = false;

    const addBtn = $('ner-add-selected-btn');
    if (extractBtn) extractBtn.classList.add('hidden');
    if (addBtn) {
        addBtn.classList.remove('hidden');
        addBtn.disabled = false;
    }
    _setNerLoading(false);
}

function _isIdenticalPair(c) {
    const s = (c && c.source ? String(c.source) : '').trim();
    const t = (c && c.target ? String(c.target) : '').trim();
    return s.length > 0 && s === t;
}

function buildNerRow(candidate) {
    const tr = document.createElement('tr');
    const isIdentical = _isIdenticalPair(candidate);
    if (isIdentical) tr.classList.add('ner-row-identical');

    const tdCheck = document.createElement('td');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.className = 'ner-row-check';
    // Dimmed (identical) and already-in-glossary rows are unchecked by default
    cb.checked = !candidate.already_in_glossary && !isIdentical;
    cb.addEventListener('change', () => {
        // When the user opts an identical row in, lift the dimmed styling so
        // they can clearly see it's now selected.
        if (cb.checked) tr.classList.remove('ner-row-identical');
        else if (isIdentical) tr.classList.add('ner-row-identical');
    });
    tdCheck.appendChild(cb);

    const tdSource = document.createElement('td');
    tdSource.textContent = candidate.source || '';
    if (candidate.already_in_glossary) {
        const tag = document.createElement('span');
        tag.textContent = t('glossary:ner_already_in_glossary');
        tag.style.color = 'var(--text-muted-light)';
        tag.style.fontSize = '0.85em';
        tdSource.appendChild(tag);
    } else if (isIdentical) {
        const tag = document.createElement('span');
        tag.textContent = t('glossary:ner_identical_pair');
        tag.style.color = 'var(--text-muted-light)';
        tag.style.fontSize = '0.85em';
        tdSource.appendChild(tag);
    }

    const tdTarget = document.createElement('td');
    const targetInput = document.createElement('input');
    targetInput.type = 'text';
    targetInput.className = 'glossary-cell-input ner-target-input';
    targetInput.value = candidate.target || '';
    tdTarget.appendChild(targetInput);

    const tdCategory = document.createElement('td');
    const catSelect = buildCategorySelect(candidate.category);
    catSelect.classList.add('ner-category-select');
    tdCategory.appendChild(catSelect);

    const tdGender = document.createElement('td');
    const genderSelect = buildGenderSelect(candidate.gender);
    genderSelect.classList.add('ner-gender-select');
    // A character the model could not gender from the sample is the one row
    // worth a human glance, so mark it instead of letting it read as settled.
    if ((candidate.category || '') === 'character' && !candidate.gender) {
        genderSelect.classList.add('ner-gender-missing');
        genderSelect.title = t('glossary:ner_gender_unknown_title');
    }
    tdGender.appendChild(genderSelect);

    tr.dataset.source = candidate.source || '';

    tr.appendChild(tdCheck);
    tr.appendChild(tdSource);
    tr.appendChild(tdTarget);
    tr.appendChild(tdCategory);
    tr.appendChild(tdGender);
    return tr;
}

async function handleNerAddSelected() {
    if (currentGlossaryId == null) return;
    const body = $('ner-candidates-body');
    const addBtn = $('ner-add-selected-btn');
    if (!body) return;

    const rows = Array.from(body.querySelectorAll('tr'));
    const selected = rows.filter((tr) => {
        const cb = tr.querySelector('.ner-row-check');
        return cb && cb.checked;
    });

    if (selected.length === 0) {
        toast.warn(t('glossary:ner_no_candidates_selected'));
        return;
    }

    // Snapshot term data before closeNerModal() swaps out the rows.
    const termsToAdd = selected.map((tr) => {
        const targetEl = tr.querySelector('.ner-target-input');
        const catEl = tr.querySelector('.ner-category-select');
        const genderEl = tr.querySelector('.ner-gender-select');
        return {
            source: tr.dataset.source || '',
            target: targetEl ? targetEl.value : '',
            category: catEl ? catEl.value : '',
            gender: genderEl ? genderEl.value : '',
        };
    });

    if (addBtn) addBtn.disabled = true;
    const gid = currentGlossaryId;
    closeNerModal();

    const progress = toast.info(
        termsToAdd.length === 1
            ? t('glossary:ner_adding_terms_one')
            : t('glossary:ner_adding_terms_other', { count: termsToAdd.length }),
        { duration: 0 },
    );

    let added = 0;
    let conflicts = 0;
    let failed = 0;

    try {
        const resp = await ApiClient.bulkGlossaryTerms(gid, {
            action: 'add',
            terms: termsToAdd,
        });
        added = (resp && resp.added) || 0;
        conflicts = (resp && resp.conflicts) || 0;
    } catch (err) {
        console.error('Bulk add failed:', err);
        failed = termsToAdd.length;
    }

    progress.dismiss();

    let summary = added === 1
        ? t('glossary:ner_added_summary_one')
        : t('glossary:ner_added_summary_other', { count: added });
    if (conflicts > 0) summary += t('glossary:ner_added_with_conflicts', { count: conflicts });
    if (failed > 0) {
        summary += t('glossary:ner_bulk_request_failed');
        toast.error(summary);
    } else if (added === 0 && conflicts === 0) {
        toast.warn(t('glossary:ner_no_terms_added'));
    } else {
        toast.success(summary);
    }

    if (currentGlossaryId === gid) {
        await openEditor(gid);
    }
}

// ========================================
// Preview prompt block modal
// ========================================

function openPreviewModal() {
    if (currentGlossaryId == null) return;
    const modal = $('glossary-preview-modal');
    const out = $('glossary-preview-output');
    const meta = $('glossary-preview-meta');
    if (!modal) return;
    if (out) out.textContent = t('glossary:preview_output_placeholder');
    if (meta) meta.textContent = '';
    modal.classList.remove('hidden');
}

function closePreviewModal() {
    const modal = $('glossary-preview-modal');
    if (modal) modal.classList.add('hidden');
}

async function handlePreviewRender() {
    if (currentGlossaryId == null) return;
    const input = $('glossary-preview-input');
    const out = $('glossary-preview-output');
    const meta = $('glossary-preview-meta');
    const text = input ? input.value : '';

    if (out) out.textContent = t('glossary:preview_loading');
    if (meta) meta.textContent = '';

    try {
        const resp = await ApiClient.previewGlossaryBlock(currentGlossaryId, text);
        const block = (resp && resp.block) || '';
        const matched = (resp && resp.matched_count) || 0;
        const total = (resp && resp.total_terms) || 0;
        const capped = !!(resp && resp.capped);

        if (out) {
            out.textContent = block || t('glossary:preview_no_match');
        }
        if (meta) {
            const parts = [t('glossary:preview_matched', { matched, total })];
            if (capped) parts.push(t('glossary:preview_capped'));
            meta.textContent = parts.join(' ');
        }
    } catch (err) {
        console.error('Preview block failed:', err);
        toast.error(t('glossary:preview_failed', { error: err.message || t('glossary:unknown_error') }));
        if (out) out.textContent = '';
    }
}

// ========================================
// Wire-up
// ========================================

function wireTranslateTabDropdown() {
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

function wireListView() {
    const newBtn = $('glossaryNewBtn');
    if (newBtn) newBtn.addEventListener('click', handleNewGlossary);

    const emptyNewBtn = $('glossaryEmptyNewBtn');
    if (emptyNewBtn) emptyNewBtn.addEventListener('click', handleNewGlossary);

    const emptyImportBtn = $('glossaryEmptyListImportBtn');
    const emptyImportFile = $('glossaryEmptyImportFile');
    if (emptyImportBtn && emptyImportFile) {
        emptyImportBtn.addEventListener('click', () => emptyImportFile.click());
        emptyImportFile.addEventListener('change', async () => {
            const file = emptyImportFile.files && emptyImportFile.files[0];
            if (!file) return;
            try {
                // Create a fresh glossary first, then import into it.
                const name = await _generateUniqueDefaultName();
                const created = await ApiClient.createGlossary({ name, source_lang: '', target_lang: '' });
                const newId = created && (created.id != null ? created.id : (created.glossary && created.glossary.id));
                if (newId == null) {
                    toast.error(t('glossary:create_for_import_failed'));
                    return;
                }
                currentGlossaryId = newId;
                await handleImportFile(file);
                await refreshDropdown();
            } catch (err) {
                console.error('Empty import failed:', err);
                toast.error(t('glossary:import_failed', { error: err.message || t('glossary:unknown_error') }));
            } finally {
                emptyImportFile.value = '';
            }
        });
    }

    // Drag & drop on the empty list area
    const emptyList = $('glossaryListEmpty');
    if (emptyList) {
        _wireDragDropImport(emptyList, async (file) => {
            try {
                const name = await _generateUniqueDefaultName();
                const created = await ApiClient.createGlossary({ name, source_lang: '', target_lang: '' });
                const newId = created && (created.id != null ? created.id : (created.glossary && created.glossary.id));
                if (newId == null) {
                    toast.error(t('glossary:create_for_import_failed'));
                    return;
                }
                currentGlossaryId = newId;
                await handleImportFile(file);
                await refreshDropdown();
            } catch (err) {
                console.error('Drop import failed:', err);
                toast.error(t('glossary:import_failed', { error: err.message || t('glossary:unknown_error') }));
            }
        });
    }
}

function wireEditorView() {
    const backBtn = $('glossaryEditorBackBtn');
    if (backBtn) backBtn.addEventListener('click', backToList);

    const delBtn = $('glossaryEditorDeleteBtn');
    if (delBtn) delBtn.addEventListener('click', handleDeleteCurrent);

    const nameInput = $('glossaryEditorName');
    if (nameInput) nameInput.addEventListener('blur', () => handleEditorMetaCommit('name'));

    _wireLangSelect(
        'glossaryEditorSourceLang',
        'glossaryEditorCustomSourceLangContainer',
        'glossaryEditorCustomSourceLang',
        'source_lang',
    );
    _wireLangSelect(
        'glossaryEditorTargetLang',
        'glossaryEditorCustomTargetLangContainer',
        'glossaryEditorCustomTargetLang',
        'target_lang',
    );

    const addRowBtn = $('glossaryAddRowBtn');
    if (addRowBtn) addRowBtn.addEventListener('click', handleAddRow);

    const previewBtn = $('glossaryPreviewBtn');
    if (previewBtn) previewBtn.addEventListener('click', openPreviewModal);

    const importBtn = $('glossaryImportBtn');
    const importFile = $('glossaryImportFile');
    if (importBtn && importFile) {
        importBtn.addEventListener('click', () => importFile.click());
        importFile.addEventListener('change', () => {
            const file = importFile.files && importFile.files[0];
            if (file) {
                handleImportFile(file).finally(() => {
                    importFile.value = '';
                });
            }
        });
    }

    const exportCsv = $('glossaryExportCsvBtn');
    if (exportCsv) exportCsv.addEventListener('click', () => handleExport('csv'));
    const exportJson = $('glossaryExportJsonBtn');
    if (exportJson) exportJson.addEventListener('click', () => handleExport('json'));

    const autoExtractBtn = $('glossaryAutoExtractBtn');
    if (autoExtractBtn) autoExtractBtn.addEventListener('click', openNerModal);

    // Filter
    const filterInput = $('glossaryTermsFilter');
    if (filterInput) filterInput.addEventListener('input', handleFilterInput);

    // Sort headers
    document.querySelectorAll('#glossaryTermsTable th.sortable').forEach((th) => {
        th.addEventListener('click', () => handleHeaderSortClick(th));
    });

    // Bulk selection
    const selectAll = $('glossaryTermsSelectAll');
    if (selectAll) {
        selectAll.addEventListener('change', () => {
            const visibleBoxes = document.querySelectorAll('#glossaryTermsBody .glossary-row-check');
            visibleBoxes.forEach((cb) => {
                cb.checked = selectAll.checked;
                const tr = cb.closest('tr');
                const tid = tr && tr.dataset.termId;
                const numId = tid && tid !== 'new' ? parseInt(tid, 10) : null;
                if (numId == null) return;
                if (selectAll.checked) _selectedTermIds.add(numId);
                else _selectedTermIds.delete(numId);
            });
            refreshBulkBar();
        });
    }

    const bulkDel = $('glossaryBulkDeleteBtn');
    if (bulkDel) bulkDel.addEventListener('click', handleBulkDelete);
    const bulkSet = $('glossaryBulkSetCategoryBtn');
    if (bulkSet) bulkSet.addEventListener('click', handleBulkSetCategory);
    const bulkSetGender = $('glossaryBulkSetGenderBtn');
    if (bulkSetGender) bulkSetGender.addEventListener('click', handleBulkSetGender);
    const bulkClear = $('glossaryBulkClearBtn');
    if (bulkClear) bulkClear.addEventListener('click', clearSelection);

    // Drag & drop import on the editor body
    const editorView = $('glossaryEditorView');
    if (editorView) {
        _wireDragDropImport(editorView, (file) => {
            handleImportFile(file);
        });
    }
}

function wireNerModal() {
    const closeBtn = $('ner-modal-close');
    const cancelBtn = $('ner-modal-cancel');
    const extractBtn = $('ner-extract-btn');
    const addSelectedBtn = $('ner-add-selected-btn');
    const selectAll = $('ner-select-all');
    const fileInput = $('ner-file-input');
    const pickBtn = $('ner-pick-file-btn');
    const dropzone = $('ner-drop-zone');
    const clearBtn = $('ner-file-clear-btn');
    const maxCharsInput = $('ner-max-chars');

    if (closeBtn) closeBtn.addEventListener('click', closeNerModal);
    if (cancelBtn) cancelBtn.addEventListener('click', closeNerModal);
    if (extractBtn) extractBtn.addEventListener('click', () => handleNerExtract());
    if (addSelectedBtn) addSelectedBtn.addEventListener('click', handleNerAddSelected);
    if (selectAll) {
        selectAll.addEventListener('change', () => {
            const body = $('ner-candidates-body');
            if (!body) return;
            const checked = selectAll.checked;
            body.querySelectorAll('.ner-row-check').forEach((cb) => { cb.checked = checked; });
        });
    }

    if (pickBtn && fileInput) {
        pickBtn.addEventListener('click', () => fileInput.click());
    }
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            const file = fileInput.files && fileInput.files[0];
            if (!file) return;
            if (!_validateNerFile(file)) {
                fileInput.value = '';
                return;
            }
            _setNerSelectedFile(file);
        });
    }
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            _setNerSelectedFile(null);
            if (fileInput) fileInput.value = '';
        });
    }
    if (dropzone) {
        let depth = 0;
        const hasFiles = (e) => e.dataTransfer && e.dataTransfer.types
            && Array.prototype.indexOf.call(e.dataTransfer.types, 'Files') >= 0;
        dropzone.addEventListener('dragenter', (e) => {
            if (!hasFiles(e)) return;
            e.preventDefault();
            depth += 1;
            dropzone.classList.add('is-drag-over');
        });
        dropzone.addEventListener('dragover', (e) => {
            if (!hasFiles(e)) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
        });
        dropzone.addEventListener('dragleave', () => {
            depth = Math.max(0, depth - 1);
            if (depth === 0) dropzone.classList.remove('is-drag-over');
        });
        dropzone.addEventListener('drop', (e) => {
            if (!hasFiles(e)) return;
            e.preventDefault();
            depth = 0;
            dropzone.classList.remove('is-drag-over');
            const file = e.dataTransfer.files && e.dataTransfer.files[0];
            if (!file) return;
            if (!_validateNerFile(file)) return;
            _setNerSelectedFile(file);
        });
    }
    if (maxCharsInput) {
        maxCharsInput.addEventListener('input', _syncNerLabels);
    }
    const sampleCountInput = $('ner-sample-count');
    if (sampleCountInput) {
        sampleCountInput.addEventListener('input', _syncNerLabels);
    }
}

function wirePreviewModal() {
    const closeBtn = $('glossary-preview-close');
    const cancelBtn = $('glossary-preview-cancel');
    const renderBtn = $('glossary-preview-render');
    if (closeBtn) closeBtn.addEventListener('click', closePreviewModal);
    if (cancelBtn) cancelBtn.addEventListener('click', closePreviewModal);
    if (renderBtn) renderBtn.addEventListener('click', handlePreviewRender);
}

// ========================================
// Public API
// ========================================

export const GlossaryManager = {
    initialize() {
        wireTranslateTabDropdown();
        wireListView();
        wireEditorView();
        wireNerModal();
        wirePreviewModal();
        refreshDropdown().catch((err) => {
            console.error('Initial glossary dropdown refresh failed:', err);
        });

        // Re-render dynamic glossary content when the UI locale changes:
        //  - the Translate-tab dropdown carries t()-built option labels,
        //  - the list view tbody is JS-rendered (term counts, "—" fallbacks),
        //  - an open editor has translated category dropdowns per row.
        // Each helper is a no-op if its container isn't currently mounted.
        window.addEventListener('localeChanged', () => {
            refreshDropdown().catch(() => { /* swallow — UI text only */ });
            const listTable = $('glossaryListTable');
            if (listTable && !listTable.classList.contains('hidden')) {
                loadList().catch(() => { /* swallow — UI text only */ });
            }
            const editorView = $('glossaryEditorView');
            if (editorView && !editorView.classList.contains('hidden') && currentGlossaryId) {
                openEditor(currentGlossaryId).catch(() => { /* swallow */ });
            }
        });
    },
    refreshDropdown() {
        return refreshDropdown();
    },
};

window.switchTopTab = switchTopTab;
