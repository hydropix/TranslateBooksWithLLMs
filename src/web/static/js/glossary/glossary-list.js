/**
 * Glossary list view (Glossaries tab)
 *
 * Renders the table of glossaries with their assign / edit / duplicate /
 * delete actions, creates new glossaries, and handles import from the
 * empty-list state (button and drag & drop).
 */

import { ApiClient } from '../core/api-client.js';
import { toast } from '../ui/toast.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { t } from '../i18n/i18n.js';
import { $, glossaryState, STORAGE_KEY, _wireDragDropImport } from './glossary-shared.js';
import { refreshDropdown } from './glossary-dropdown.js';
import { openEditor, _openEditorWithData, handleImportFile } from './glossary-editor.js';

// ========================================
// Glossary list view
// ========================================

export async function loadList() {
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
            body.innerHTML = `<tr><td colspan="5" style="color:#ef4444;">${t('glossary:load_failed_row', { error: DomHelpers.escapeHtml(err.message || t('glossary:unknown_error')) })}</td></tr>`;
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

export function wireListView() {
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
                glossaryState.currentGlossaryId = newId;
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
                glossaryState.currentGlossaryId = newId;
                await handleImportFile(file);
                await refreshDropdown();
            } catch (err) {
                console.error('Drop import failed:', err);
                toast.error(t('glossary:import_failed', { error: err.message || t('glossary:unknown_error') }));
            }
        });
    }
}
