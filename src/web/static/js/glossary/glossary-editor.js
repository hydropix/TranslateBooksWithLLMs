/**
 * Glossary editor view (Glossaries tab)
 *
 * Name / language meta fields, the term table with inline add / edit /
 * delete, per-glossary sort and filter, bulk selection and bulk actions,
 * and import / export of terms.
 */

import { ApiClient } from '../core/api-client.js';
import { toast } from '../ui/toast.js';
import { t } from '../i18n/i18n.js';
import {
    $,
    glossaryState,
    isConflictError,
    buildCategorySelect,
    buildGenderSelect,
    genderLabel,
    getRowFieldValues,
    flashRow,
    setSavedIndicator,
    _wireDragDropImport,
} from './glossary-shared.js';
import { refreshDropdown } from './glossary-dropdown.js';
import { loadList } from './glossary-list.js';
import { openNerModal } from './glossary-ner-modal.js';
import { openPreviewModal } from './glossary-preview-modal.js';

// Last committed effective values for the lang selects (the visible select
// value can be "Other" while the real lang lives in the custom input, so we
// can't rely on per-element dataset.lastValue here).
const _lastMeta = { source_lang: '', target_lang: '' };

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
// Editor view
// ========================================

export async function openEditor(gid) {
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

export function _resolveLangValue(selectId, customInputId) {
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

export function _openEditorWithData(g) {
    if (!g || g.id == null) return;
    glossaryState.currentGlossaryId = g.id;
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

    glossaryState.currentTerms = (g.terms || []).slice();

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
    if (!glossaryState.currentGlossaryId) return terms.slice();
    const state = getSortState(glossaryState.currentGlossaryId);
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
    if (!glossaryState.currentGlossaryId) return terms;
    const q = (getFilterText(glossaryState.currentGlossaryId) || '').trim().toLowerCase();
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

    const total = glossaryState.currentTerms.length;
    const visible = _applyFilter(_applySort(glossaryState.currentTerms));

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
    const state = glossaryState.currentGlossaryId ? getSortState(glossaryState.currentGlossaryId) : null;
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
    const idx = glossaryState.currentTerms.findIndex((term) => term && term.id === termId);
    if (idx >= 0) glossaryState.currentTerms[idx] = Object.assign({}, glossaryState.currentTerms[idx], patch);
}

function _removeLocalTerm(termId) {
    glossaryState.currentTerms = glossaryState.currentTerms.filter((term) => !term || term.id !== termId);
}

async function _doFieldCommit(tr, el) {
    if (glossaryState.currentGlossaryId == null) return;

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
            const created = await ApiClient.addGlossaryTerm(glossaryState.currentGlossaryId, {
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
                glossaryState.currentTerms.push({
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
            await ApiClient.deleteGlossaryTerm(glossaryState.currentGlossaryId, tid);
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
        await ApiClient.updateGlossaryTerm(glossaryState.currentGlossaryId, tid, payload);
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
            await ApiClient.deleteGlossaryTerm(glossaryState.currentGlossaryId, tid);
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
    const hasRows = body.children.length > 0 || glossaryState.currentTerms.length > 0;
    if (empty) empty.classList.toggle('hidden', hasRows);
    if (table) table.classList.toggle('hidden', !hasRows);
}

function handleAddRow() {
    if (glossaryState.currentGlossaryId == null) return;
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
    if (glossaryState.currentGlossaryId == null) return;

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
        await ApiClient.updateGlossary(glossaryState.currentGlossaryId, payload);
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
    glossaryState.currentGlossaryId = null;
    glossaryState.currentTerms = [];
    clearSelection();
    loadList();
    refreshDropdown().catch(() => {});
}

async function handleDeleteCurrent() {
    if (glossaryState.currentGlossaryId == null) return;
    if (!confirm(t('glossary:confirm_delete_current'))) return;
    try {
        await ApiClient.deleteGlossary(glossaryState.currentGlossaryId);
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
    if (glossaryState.currentGlossaryId == null) return;
    const key = th.dataset.sortKey;
    if (!key) return;
    const current = getSortState(glossaryState.currentGlossaryId);
    let next;
    if (!current || current.key !== key) {
        next = { key, dir: 'asc' };
    } else if (current.dir === 'asc') {
        next = { key, dir: 'desc' };
    } else {
        next = null;
    }
    setSortState(glossaryState.currentGlossaryId, next);
    renderSortIndicators();
    rerenderTerms();
}

function handleFilterInput(e) {
    if (glossaryState.currentGlossaryId == null) return;
    const text = e.target.value || '';
    setFilterText(glossaryState.currentGlossaryId, text);
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
        const resp = await ApiClient.bulkGlossaryTerms(glossaryState.currentGlossaryId, {
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
        const resp = await ApiClient.bulkGlossaryTerms(glossaryState.currentGlossaryId, {
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
        const resp = await ApiClient.bulkGlossaryTerms(glossaryState.currentGlossaryId, {
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

export async function handleImportFile(file) {
    if (glossaryState.currentGlossaryId == null || !file) return;
    try {
        const result = await ApiClient.importGlossaryTerms(glossaryState.currentGlossaryId, file);
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
        await openEditor(glossaryState.currentGlossaryId);
    } catch (err) {
        console.error('Import failed:', err);
        toast.error(t('glossary:import_failed', { error: err.message || t('glossary:unknown_error') }));
    }
}

function handleExport(format) {
    if (glossaryState.currentGlossaryId == null) return;
    const url = ApiClient.getGlossaryExportUrl(glossaryState.currentGlossaryId, format);
    if (url) window.location.href = url;
}

export function wireEditorView() {
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
