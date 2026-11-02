/**
 * Style Manager
 *
 * Wires up the Styles tab: the list/editor views for custom-instruction
 * "style" presets (extended-schema YAML under Custom_Instructions/), and
 * the multi-file extract-from-books modal. Talks to the backend through
 * ApiClient and surfaces feedback via the toast module.
 *
 * Mirrors the shape of glossary-manager.js (list view / editor view /
 * extract modal / public API). FormManager owns the option list of
 * #customInstructionSelect: this module never rebuilds it, it dispatches
 * 'customInstructionsChanged' after every create/update/delete/duplicate and
 * lets FormManager reload. It does set the select's *value* when the user
 * assigns a preset to the translation, exactly as glossary-manager.js drives
 * #glossarySelect.
 */

import { ApiClient } from '../core/api-client.js';
import { toast } from '../ui/toast.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import { FormManager } from '../ui/form-manager.js';
import { t } from '../i18n/i18n.js';

// ========================================
// Module state
// ========================================

const DIMENSIONS = [
    'register', 'narrative_voice', 'sentence_rhythm', 'lexicon',
    'imagery', 'dialogue', 'punctuation', 'formatting', 'other',
];

const EXTRACT_ACCEPTED_EXTS = ['txt', 'srt', 'epub', 'docx', 'pdf'];
const EXTRACT_MAX_FILES = 5;
const ASSEMBLE_DEBOUNCE_MS = 300;

// --- Editor view state ---
let currentEditorFilename = null;
let currentEditorIsTxt = false;
let manualDirty = false;
let _editorAssembleTimer = null;
let _editorAssembleReqId = 0;

// --- Extract modal state ---
let _extractFiles = [];
let _extractCandidates = [];
let _extractSummary = '';
let _extractAssembleTimer = null;
let _extractAssembleReqId = 0;

// ========================================
// Helpers
// ========================================

function $(id) {
    return document.getElementById(id);
}

function _isConflictError(err) {
    if (!err || !err.message) return false;
    const m = err.message.toLowerCase();
    return m.includes('already') || m.includes('conflict') || m.includes('unique') || m.includes('409');
}

function _fileExt(name) {
    const dot = (name || '').lastIndexOf('.');
    return dot >= 0 ? name.slice(dot + 1).toLowerCase() : '';
}

/**
 * Grow a rule textarea to fit its content. A style rule is a full sentence
 * and must be readable in one glance, so the box follows the text instead of
 * scrolling it. CSS keeps a min-height and allows a manual resize.
 */
function _autoGrow(textarea) {
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${textarea.scrollHeight}px`;
}

// ========================================
// Assignment to the translation
// ========================================

/**
 * The preset currently assigned to the translation, i.e. the value of
 * #customInstructionSelect. That select is auto-persisted by SettingsManager,
 * so there is no separate storage key to keep in sync (unlike glossaries,
 * which mirror their selection into localStorage).
 */
function _assignedFilename() {
    const select = DomHelpers.getElement('customInstructionSelect');
    return select ? select.value || '' : '';
}

/**
 * Assign `filename` to the translation, or clear the assignment when passed
 * an empty string. Dispatching 'change' is what makes SettingsManager persist
 * the choice and the settings summary refresh.
 */
async function _setAssignment(filename) {
    const select = DomHelpers.getElement('customInstructionSelect');
    if (!select) return;

    const wanted = filename || '';
    if (wanted) {
        const exists = Array.from(select.options).some((o) => o.value === wanted);
        // A preset created moments ago may not be in the dropdown yet; the
        // option list belongs to FormManager, so ask it to reload rather than
        // injecting the option here.
        if (!exists) await FormManager.loadCustomInstructions();
    }

    select.value = wanted;
    select.dispatchEvent(new Event('change', { bubbles: true }));
}

// ========================================
// List view
// ========================================

function _buildListRow(file) {
    const tr = document.createElement('tr');
    const isAssigned = file.filename === _assignedFilename();

    const tdName = document.createElement('td');
    const link = document.createElement('a');
    link.href = '#';
    link.textContent = file.display_name;
    link.style.cursor = 'pointer';
    link.addEventListener('click', (e) => {
        e.preventDefault();
        openEditor(file.filename);
    });
    tdName.appendChild(link);
    if (file.format === 'txt') {
        // Legacy plain-text preset — mark it with its real extension (read
        // from the filename, never a hardcoded literal) rather than a
        // translated word, since it's a file-format marker, not prose.
        const marker = document.createElement('em');
        marker.style.color = 'var(--text-muted-light)';
        marker.style.marginLeft = '6px';
        marker.style.fontSize = '0.8em';
        marker.textContent = file.filename.slice(file.filename.lastIndexOf('.'));
        tdName.appendChild(marker);
    }

    const tdDesc = document.createElement('td');
    tdDesc.textContent = file.description || '—';

    const tdPhases = document.createElement('td');
    tdPhases.className = 'col-center';
    const hasT = !!file.has_translation;
    const hasR = !!file.has_refinement;
    let phases = '—';
    if (hasT && hasR) phases = 'T+R';
    else if (hasT) phases = 'T';
    else if (hasR) phases = 'R';
    tdPhases.textContent = phases;

    const tdMode = document.createElement('td');
    tdMode.className = 'col-center';
    tdMode.textContent = file.mode === 'source'
        ? t('style:mode_source')
        : file.mode === 'model' ? t('style:mode_model') : '—';

    const tdActions = document.createElement('td');
    tdActions.className = 'col-right';
    const wrapper = document.createElement('div');
    wrapper.style.display = 'inline-flex';
    wrapper.style.gap = '0.25rem';
    wrapper.style.alignItems = 'center';
    wrapper.style.justifyContent = 'flex-end';

    const assignBtn = document.createElement('button');
    assignBtn.className = 'file-action-btn download';
    assignBtn.title = isAssigned ? t('style:unassign_title') : t('style:assign_title');
    assignBtn.setAttribute('aria-label', assignBtn.title);
    const assignIcon = isAssigned ? 'bookmark_added' : 'bookmark_add';
    assignBtn.innerHTML = `<span class="material-symbols-outlined" style="font-size: 0.875rem;">${assignIcon}</span>`;
    if (isAssigned) assignBtn.style.color = 'var(--success-light)';
    assignBtn.addEventListener('click', async () => {
        // The list is redrawn by the #customInstructionSelect 'change' handler
        // wired in wireListView(), so no explicit reload here.
        await _setAssignment(isAssigned ? '' : file.filename);
        toast.success(isAssigned
            ? t('style:unassigned_msg', { name: file.display_name })
            : t('style:assigned_msg', { name: file.display_name }));
    });

    const editBtn = document.createElement('button');
    editBtn.className = 'file-action-btn download';
    editBtn.title = t('style:edit');
    editBtn.setAttribute('aria-label', editBtn.title);
    editBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">edit</span>';
    editBtn.addEventListener('click', () => openEditor(file.filename));

    const dupBtn = document.createElement('button');
    dupBtn.className = 'file-action-btn download';
    dupBtn.title = t('style:duplicate');
    dupBtn.setAttribute('aria-label', dupBtn.title);
    dupBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">content_copy</span>';
    dupBtn.addEventListener('click', async () => {
        try {
            await ApiClient.duplicateCustomInstruction(file.filename);
            toast.success(t('style:created'));
            window.dispatchEvent(new CustomEvent('customInstructionsChanged'));
            await loadList();
        } catch (err) {
            console.error('Duplicate style failed:', err);
            toast.error(t('errors:operation_failed', { error: err.message || t('glossary:unknown_error') }));
        }
    });

    const exportBtn = document.createElement('button');
    exportBtn.className = 'file-action-btn download';
    exportBtn.title = t('style:export');
    exportBtn.setAttribute('aria-label', exportBtn.title);
    exportBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">download</span>';
    exportBtn.addEventListener('click', () => {
        const url = ApiClient.exportCustomInstructionUrl(file.filename);
        if (url) window.location.href = url;
    });

    const delBtn = document.createElement('button');
    delBtn.className = 'file-action-btn delete';
    delBtn.title = t('style:delete');
    delBtn.setAttribute('aria-label', delBtn.title);
    delBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">delete</span>';
    delBtn.addEventListener('click', async () => {
        if (!confirm(t('style:confirm_delete', { name: file.display_name }))) return;
        try {
            await ApiClient.deleteCustomInstruction(file.filename);
            toast.success(t('style:deleted'));
            window.dispatchEvent(new CustomEvent('customInstructionsChanged'));
            await loadList();
        } catch (err) {
            console.error('Delete style failed:', err);
            toast.error(t('errors:operation_failed', { error: err.message || t('glossary:unknown_error') }));
        }
    });

    wrapper.appendChild(assignBtn);
    wrapper.appendChild(editBtn);
    wrapper.appendChild(dupBtn);
    wrapper.appendChild(exportBtn);
    wrapper.appendChild(delBtn);
    tdActions.appendChild(wrapper);

    tr.appendChild(tdName);
    tr.appendChild(tdDesc);
    tr.appendChild(tdPhases);
    tr.appendChild(tdMode);
    tr.appendChild(tdActions);
    return tr;
}

async function loadList() {
    const loading = $('styleListLoading');
    const empty = $('styleListEmpty');
    const table = $('styleListTable');
    const body = $('styleListBody');

    if (loading) loading.classList.remove('hidden');
    if (empty) empty.classList.add('hidden');
    if (table) table.classList.add('hidden');
    if (body) body.innerHTML = '';

    let resp;
    try {
        resp = await ApiClient.getCustomInstructions();
    } catch (err) {
        console.error('Failed to load style list:', err);
        if (loading) loading.classList.add('hidden');
        if (body) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = 5;
            td.style.color = '#ef4444';
            td.textContent = t('style:load_failed', { error: err.message || t('glossary:unknown_error') });
            tr.appendChild(td);
            body.appendChild(tr);
        }
        if (table) table.classList.remove('hidden');
        return;
    }

    const files = (resp && resp.files) || [];
    if (loading) loading.classList.add('hidden');

    if (files.length === 0) {
        if (empty) empty.classList.remove('hidden');
        if (table) table.classList.add('hidden');
        return;
    }

    if (table) table.classList.remove('hidden');
    if (empty) empty.classList.add('hidden');
    if (!body) return;
    body.innerHTML = '';
    for (const file of files) {
        body.appendChild(_buildListRow(file));
    }
}

function wireListView() {
    // The preset can also be picked from the Translate tab, from Settings, or
    // restored with a saved configuration. Keep the bookmark icons in step
    // with whatever the dropdown currently holds.
    const instructionSelect = DomHelpers.getElement('customInstructionSelect');
    if (instructionSelect) {
        instructionSelect.addEventListener('change', () => {
            const table = $('styleListTable');
            if (table && !table.classList.contains('hidden')) {
                loadList().catch(() => { /* swallow — icon state only */ });
            }
        });
    }

    const openFolderBtn = $('styleOpenFolderBtn');
    if (openFolderBtn) {
        openFolderBtn.addEventListener('click', async () => {
            try {
                const resp = await ApiClient.openCustomInstructionsFolder();
                if (!resp || !resp.success) {
                    console.error('Failed to open Custom_Instructions folder:', resp && resp.error);
                }
            } catch (err) {
                console.error('Failed to open Custom_Instructions folder:', err);
            }
        });
    }
}

function _showListView() {
    const listView = $('styleListView');
    const editorView = $('styleEditorView');
    if (editorView) editorView.classList.add('hidden');
    if (listView) listView.classList.remove('hidden');
    clearTimeout(_editorAssembleTimer);
    currentEditorFilename = null;
    currentEditorIsTxt = false;
    manualDirty = false;
}

// ========================================
// Editor view
// ========================================

function _collectEditorRules() {
    const body = $('styleRulesBody');
    if (!body) return [];
    return Array.from(body.querySelectorAll('.style-rule-card')).map((card) => {
        const dimSelect = card.querySelector('select.style-rule-dimension');
        const instrInput = card.querySelector('textarea.style-rule-instruction');
        return {
            dimension: dimSelect ? dimSelect.value : 'other',
            instruction: instrInput ? instrInput.value : '',
        };
    });
}

function _buildEditorRuleCard(rule) {
    const card = document.createElement('div');
    card.className = 'style-rule-card';

    const head = document.createElement('div');
    head.className = 'style-rule-card-head';

    const dimSelect = document.createElement('select');
    dimSelect.className = 'form-control style-rule-dimension';
    DIMENSIONS.forEach((dim) => {
        const opt = document.createElement('option');
        opt.value = dim;
        opt.textContent = t(`style:dimension_${dim}`);
        if ((rule.dimension || 'other') === dim) opt.selected = true;
        dimSelect.appendChild(opt);
    });
    dimSelect.addEventListener('change', _scheduleEditorAssemble);

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'file-action-btn delete style-rule-remove';
    removeBtn.title = t('style:remove_rule');
    removeBtn.setAttribute('aria-label', removeBtn.title);
    removeBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">delete</span>';
    removeBtn.addEventListener('click', () => {
        card.remove();
        _scheduleEditorAssemble();
    });

    head.appendChild(dimSelect);
    head.appendChild(removeBtn);

    const instrInput = document.createElement('textarea');
    instrInput.className = 'form-control style-rule-instruction';
    instrInput.rows = 2;
    instrInput.value = rule.instruction || '';
    instrInput.addEventListener('input', () => {
        _autoGrow(instrInput);
        _scheduleEditorAssemble();
    });

    card.appendChild(head);
    card.appendChild(instrInput);
    return card;
}

function _applyTxtModeVisibility() {
    const rulesBody = $('styleRulesBody');
    const addRuleBtn = $('styleAddRuleBtn');
    if (rulesBody) rulesBody.classList.toggle('hidden', currentEditorIsTxt);
    if (addRuleBtn) addRuleBtn.classList.toggle('hidden', currentEditorIsTxt);
}

function _updateManualBadge() {
    const badge = $('styleManualBadge');
    if (badge) badge.classList.toggle('hidden', !manualDirty);
}

function _scheduleEditorAssemble() {
    if (manualDirty) return;
    clearTimeout(_editorAssembleTimer);
    _editorAssembleTimer = setTimeout(_runEditorAssemble, ASSEMBLE_DEBOUNCE_MS);
}

async function _runEditorAssemble() {
    if (manualDirty) return;
    const reqId = ++_editorAssembleReqId;
    const modeSelect = $('styleEditorMode');
    const mode = (modeSelect && modeSelect.value) || 'source';
    const rules = _collectEditorRules();
    const contextTextarea = $('styleEditorContext');
    const context = contextTextarea ? contextTextarea.value : '';

    let resp;
    try {
        resp = await ApiClient.assembleCustomInstruction(mode, rules, context);
    } catch (err) {
        console.error('Style editor assemble failed:', err);
        return;
    }

    // Out-of-order guard (plan risk R7): only the latest debounced call may
    // touch the DOM. A user may also have started a manual edit while this
    // request was in flight — respect that too.
    if (reqId !== _editorAssembleReqId || manualDirty) return;

    const transEl = $('styleTranslationText');
    const refEl = $('styleRefinementText');
    if (transEl) transEl.value = (resp && resp.translation) || '';
    if (refEl) refEl.value = (resp && resp.refinement) || '';
}

function handleReassemble() {
    if (!confirm(t('style:manual_edits_warning'))) return;
    manualDirty = false;
    _updateManualBadge();
    clearTimeout(_editorAssembleTimer);
    _runEditorAssemble();
}

/**
 * Mirror of the server's `slugify_preset_name`, lowercased: two names that
 * map to the same file compare equal. Comparing raw display names instead
 * misses that "New blank style" is saved as "New_blank_style.yaml".
 */
function _presetSlugKey(name) {
    return (name || '')
        .replace(/[^\p{L}\p{N}_-]+/gu, '_')
        .replace(/^_+|_+$/g, '')
        .slice(0, 64)
        .toLowerCase();
}

async function _generateDefaultPresetName() {
    let existing = [];
    try {
        const resp = await ApiClient.getCustomInstructions();
        existing = ((resp && resp.files) || []).map((f) => _presetSlugKey(f.display_name));
    } catch (_) {
        existing = [];
    }
    const base = t('style:new_blank');
    if (!existing.includes(_presetSlugKey(base))) return base;
    for (let i = 2; i < 1000; i++) {
        const candidate = `${base} ${i}`;
        if (!existing.includes(_presetSlugKey(candidate))) return candidate;
    }
    return `${base} ${Date.now()}`;
}

async function openEditor(filename) {
    currentEditorFilename = filename;
    manualDirty = false;
    _updateManualBadge();
    clearTimeout(_editorAssembleTimer);

    const listView = $('styleListView');
    const editorView = $('styleEditorView');
    if (listView) listView.classList.add('hidden');
    if (editorView) editorView.classList.remove('hidden');

    const nameInput = $('styleEditorName');
    const descTextarea = $('styleEditorDescription');
    const contextTextarea = $('styleEditorContext');
    const modeSelect = $('styleEditorMode');
    const rulesBody = $('styleRulesBody');
    const transTextarea = $('styleTranslationText');
    const refTextarea = $('styleRefinementText');

    if (filename) {
        let preset;
        try {
            preset = await ApiClient.getCustomInstruction(filename);
        } catch (err) {
            console.error('Failed to load style preset:', err);
            toast.error(t('style:load_failed', { error: err.message || t('glossary:unknown_error') }));
            _showListView();
            return;
        }

        currentEditorIsTxt = preset.format === 'txt';
        if (nameInput) nameInput.value = preset.display_name || filename;
        if (descTextarea) descTextarea.value = preset.description || '';
        if (contextTextarea) contextTextarea.value = preset.context || '';
        if (modeSelect) modeSelect.value = preset.mode || 'source';
        if (rulesBody) {
            rulesBody.innerHTML = '';
            (preset.rules || []).forEach((rule) => rulesBody.appendChild(_buildEditorRuleCard(rule)));
            // scrollHeight is only meaningful once the textarea is in the document.
            rulesBody.querySelectorAll('textarea.style-rule-instruction').forEach(_autoGrow);
        }
        if (transTextarea) transTextarea.value = preset.translation || '';
        if (refTextarea) refTextarea.value = preset.refinement || '';
    } else {
        // New unsaved preset: no server round-trip until the first Save
        // (POST). A free default name is pre-filled; the user can change it
        // before saving.
        currentEditorIsTxt = false;
        if (nameInput) nameInput.value = await _generateDefaultPresetName();
        if (descTextarea) descTextarea.value = '';
        if (contextTextarea) contextTextarea.value = '';
        if (modeSelect) modeSelect.value = 'source';
        if (rulesBody) rulesBody.innerHTML = '';
        if (transTextarea) transTextarea.value = '';
        if (refTextarea) refTextarea.value = '';
    }

    _applyTxtModeVisibility();
}

function handleNewBlank() {
    openEditor(null);
}

async function handleEditorSave() {
    const descTextarea = $('styleEditorDescription');
    const contextTextarea = $('styleEditorContext');
    const modeSelect = $('styleEditorMode');
    const transTextarea = $('styleTranslationText');
    const refTextarea = $('styleRefinementText');
    const saveBtn = $('styleEditorSaveBtn');
    const nameInput = $('styleEditorName');

    const name = (nameInput ? nameInput.value : '').trim();
    if (!name) {
        toast.warn(t('errors:field_required', { field: t('style:preset_name_label') }));
        if (nameInput) nameInput.focus();
        return;
    }

    const payload = {
        description: descTextarea ? descTextarea.value : '',
        context: contextTextarea ? contextTextarea.value : '',
        mode: (modeSelect && modeSelect.value) || 'source',
        rules: _collectEditorRules(),
        translation: transTextarea ? transTextarea.value : '',
        refinement: refTextarea ? refTextarea.value : '',
        manual: manualDirty,
    };

    if (saveBtn) saveBtn.disabled = true;
    try {
        let saved;
        if (currentEditorFilename) {
            // The server only renames when 'name' differs from the current
            // file stem, and may also turn a .txt into a .yaml: always adopt
            // the filename it returns.
            const wasAssigned = currentEditorFilename === _assignedFilename();
            saved = await ApiClient.updateCustomInstruction(currentEditorFilename, { ...payload, name });
            if (saved.filename !== currentEditorFilename && wasAssigned) {
                await FormManager.loadCustomInstructions();
                await _setAssignment(saved.filename);
            }
        } else {
            try {
                saved = await ApiClient.createCustomInstruction({ ...payload, name, overwrite: false });
            } catch (err) {
                // Never overwrite another preset silently: a taken name
                // needs an explicit confirmation, otherwise the user picks
                // another name.
                if (!_isConflictError(err)) throw err;
                if (!confirm(t('style:confirm_overwrite', { name }))) {
                    if (nameInput) nameInput.focus();
                    return;
                }
                saved = await ApiClient.createCustomInstruction({ ...payload, name, overwrite: true });
            }
        }
        currentEditorFilename = saved.filename;
        currentEditorIsTxt = _fileExt(saved.filename) === 'txt';
        _applyTxtModeVisibility();
        if (nameInput) nameInput.value = saved.display_name;
        toast.success(t('style:saved'));
        window.dispatchEvent(new CustomEvent('customInstructionsChanged'));
        await loadList();
    } catch (err) {
        console.error('Failed to save style preset:', err);
        toast.error(t('style:save_failed', { error: err.message || t('glossary:unknown_error') }));
    } finally {
        if (saveBtn) saveBtn.disabled = false;
    }
}

function handleEditorExport() {
    if (!currentEditorFilename) return;
    const url = ApiClient.exportCustomInstructionUrl(currentEditorFilename);
    if (url) window.location.href = url;
}

async function handleEditorDelete() {
    if (!currentEditorFilename) return;
    const name = currentEditorFilename.slice(0, currentEditorFilename.lastIndexOf('.'));
    if (!confirm(t('style:confirm_delete', { name }))) return;
    try {
        await ApiClient.deleteCustomInstruction(currentEditorFilename);
        toast.success(t('style:deleted'));
        window.dispatchEvent(new CustomEvent('customInstructionsChanged'));
        _showListView();
        await loadList();
    } catch (err) {
        console.error('Failed to delete style preset:', err);
        toast.error(t('errors:operation_failed', { error: err.message || t('glossary:unknown_error') }));
    }
}

function _refreshEditorRuleLabels() {
    const body = $('styleRulesBody');
    if (!body) return;
    body.querySelectorAll('select.style-rule-dimension').forEach((select) => {
        const current = select.value;
        Array.from(select.options).forEach((opt) => {
            opt.textContent = t(`style:dimension_${opt.value}`);
        });
        select.value = current;
    });
    body.querySelectorAll('button').forEach((btn) => {
        btn.title = t('style:remove_rule');
        btn.setAttribute('aria-label', btn.title);
    });
}

function wireEditorView() {
    const backBtn = $('styleEditorBack');
    if (backBtn) backBtn.addEventListener('click', _showListView);

    const newBlankBtn = $('styleNewBlankBtn');
    if (newBlankBtn) newBlankBtn.addEventListener('click', handleNewBlank);

    const addRuleBtn = $('styleAddRuleBtn');
    if (addRuleBtn) {
        addRuleBtn.addEventListener('click', () => {
            const body = $('styleRulesBody');
            if (body) {
                const card = _buildEditorRuleCard({ dimension: 'other', instruction: '' });
                body.appendChild(card);
                const textarea = card.querySelector('textarea.style-rule-instruction');
                if (textarea) textarea.focus();
            }
            _scheduleEditorAssemble();
        });
    }

    const modeSelect = $('styleEditorMode');
    if (modeSelect) modeSelect.addEventListener('change', _scheduleEditorAssemble);

    const contextTextarea = $('styleEditorContext');
    if (contextTextarea) contextTextarea.addEventListener('input', _scheduleEditorAssemble);

    const transTextarea = $('styleTranslationText');
    if (transTextarea) {
        transTextarea.addEventListener('input', () => {
            manualDirty = true;
            _updateManualBadge();
        });
    }
    const refTextarea = $('styleRefinementText');
    if (refTextarea) {
        refTextarea.addEventListener('input', () => {
            manualDirty = true;
            _updateManualBadge();
        });
    }

    const reassembleBtn = $('styleReassembleBtn');
    if (reassembleBtn) reassembleBtn.addEventListener('click', handleReassemble);

    const saveBtn = $('styleEditorSaveBtn');
    if (saveBtn) saveBtn.addEventListener('click', handleEditorSave);

    const exportBtn = $('styleEditorExportBtn');
    if (exportBtn) exportBtn.addEventListener('click', handleEditorExport);

    const deleteBtn = $('styleEditorDeleteBtn');
    if (deleteBtn) deleteBtn.addEventListener('click', handleEditorDelete);
}

// ========================================
// Extract-from-books modal
// ========================================

function _renderExtractFileList() {
    const container = $('styleExtractFileList');
    const runBtn = $('styleExtractRunBtn');
    if (runBtn) runBtn.disabled = _extractFiles.length === 0;
    if (!container) return;
    container.innerHTML = '';
    if (_extractFiles.length === 0) return;

    const summary = document.createElement('div');
    summary.className = 'ner-dropzone-hint';
    summary.textContent = t('style:extract_file_count', { count: _extractFiles.length });
    container.appendChild(summary);

    const list = document.createElement('ul');
    list.style.margin = '8px 0 0';
    list.style.paddingLeft = '0';
    list.style.listStyle = 'none';
    _extractFiles.forEach((file, index) => {
        const li = document.createElement('li');
        li.style.display = 'flex';
        li.style.alignItems = 'center';
        li.style.gap = '8px';
        li.style.padding = '4px 0';

        const nameSpan = document.createElement('span');
        nameSpan.textContent = file.name;
        li.appendChild(nameSpan);

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'file-action-btn delete';
        removeBtn.title = t('common:delete');
        removeBtn.setAttribute('aria-label', removeBtn.title);
        removeBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">close</span>';
        removeBtn.addEventListener('click', () => _removeExtractFile(index));
        li.appendChild(removeBtn);

        list.appendChild(li);
    });
    container.appendChild(list);
}

function _removeExtractFile(index) {
    _extractFiles.splice(index, 1);
    _renderExtractFileList();
}

function _addExtractFiles(fileList) {
    const incoming = Array.from(fileList || []);
    for (const file of incoming) {
        const ext = _fileExt(file.name);
        if (!EXTRACT_ACCEPTED_EXTS.includes(ext)) {
            toast.warn(t('style:unsupported_file', { ext: ext || '?', accepted: EXTRACT_ACCEPTED_EXTS.join(', ') }));
            continue;
        }
        if (_extractFiles.length >= EXTRACT_MAX_FILES) {
            toast.warn(t('style:too_many_files', { max: EXTRACT_MAX_FILES }));
            break;
        }
        _extractFiles.push(file);
    }
    _renderExtractFileList();
}

function _wireExtractDropzone() {
    const dropzone = $('styleExtractDropzone');
    const fileInput = $('styleExtractFileInput');
    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => {
        _addExtractFiles(fileInput.files);
        fileInput.value = '';
    });

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
        _addExtractFiles(e.dataTransfer.files);
    });
}

function _populateExtractLangSelects() {
    const srcSelect = $('styleExtractSourceLang');
    const tgtSelect = $('styleExtractTargetLang');
    const translateSrc = $('sourceLang');
    const translateTgt = $('targetLang');

    if (srcSelect && translateSrc) {
        srcSelect.innerHTML = translateSrc.innerHTML;
        // "Auto-detect" (empty value) isn't a valid extraction language —
        // the extract-style endpoint defaults to English server-side.
        srcSelect.value = translateSrc.value || 'English';
        if (srcSelect.selectedIndex < 0) srcSelect.selectedIndex = 0;
    }
    if (tgtSelect && translateTgt) {
        tgtSelect.innerHTML = translateTgt.innerHTML;
        tgtSelect.value = translateTgt.value || 'English';
        if (tgtSelect.selectedIndex < 0) tgtSelect.selectedIndex = 0;
    }
}

function _setExtractHint(showNoRulesHint) {
    const hint = document.querySelector('#styleExtractResults .ner-results-hint');
    if (!hint) return;
    hint.textContent = showNoRulesHint ? t('style:no_rules_selected') : t('style:candidates_hint');
}

function _setExtractPreview(assembled) {
    const transEl = $('stylePreviewTranslation');
    const refEl = $('stylePreviewRefinement');
    if (transEl) transEl.textContent = (assembled && assembled.translation) || '';
    if (refEl) refEl.textContent = (assembled && assembled.refinement) || '';
}

function _syncCreateEnablement() {
    const createBtn = $('styleExtractCreateBtn');
    const anyChecked = _extractCandidates.some((c) => c.checked && (c.instruction || '').trim());
    if (createBtn) createBtn.disabled = !anyChecked;
    _setExtractHint(!anyChecked);
}

function _buildCandidateCard(candidate, index) {
    const card = document.createElement('div');
    card.className = 'style-rule-card';
    card.dataset.candidateIndex = String(index);

    const head = document.createElement('div');
    head.className = 'style-rule-card-head';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = !!candidate.checked;

    const dimBadge = document.createElement('span');
    dimBadge.className = 'style-rule-dimension-badge';

    const flagIcon = document.createElement('span');
    flagIcon.className = 'material-symbols-outlined style-rule-flag hidden';
    flagIcon.textContent = 'warning';

    head.appendChild(cb);
    head.appendChild(dimBadge);
    head.appendChild(flagIcon);

    const instrInput = document.createElement('textarea');
    instrInput.className = 'form-control style-rule-instruction';
    instrInput.rows = 2;
    instrInput.value = candidate.instruction || '';

    const evidence = document.createElement('p');
    evidence.className = 'style-rule-evidence hidden';

    // Rebuilds only the label, the flag icon and the evidence line (never the
    // checkbox or the instruction textarea) so a locale switch or a server
    // round-trip never steals focus from whatever the user is editing.
    const refresh = () => {
        dimBadge.textContent = t(`style:dimension_${candidate.dimension || 'other'}`);

        const flagged = candidate.flags && candidate.flags.length > 0;
        flagIcon.classList.toggle('hidden', !flagged);
        if (flagged) {
            flagIcon.title = candidate.flags.map((code) => t(`style:flag_${code}`)).join(' · ');
            flagIcon.setAttribute('aria-label', flagIcon.title);
        }
        card.classList.toggle('is-muted', flagged && !candidate.checked);

        evidence.classList.toggle('hidden', !candidate.evidence);
        if (candidate.evidence) {
            evidence.textContent = candidate.evidence;
            evidence.title = t('style:evidence_tooltip');
        }
    };
    refresh();

    cb.addEventListener('change', () => {
        candidate.checked = cb.checked;
        refresh();
        _syncCreateEnablement();
        _scheduleExtractAssemble();
    });
    instrInput.addEventListener('input', () => {
        candidate.instruction = instrInput.value;
        _autoGrow(instrInput);
        _syncCreateEnablement();
        _scheduleExtractAssemble();
    });

    card.appendChild(head);
    card.appendChild(instrInput);
    card.appendChild(evidence);
    card._refresh = refresh;
    return card;
}

function _renderExtractCandidatesTable() {
    const body = $('styleRulesCandidatesBody');
    if (!body) return;
    body.innerHTML = '';
    _extractCandidates.forEach((candidate, index) => {
        body.appendChild(_buildCandidateCard(candidate, index));
    });
    // scrollHeight is only meaningful once the textarea is in the document.
    body.querySelectorAll('textarea.style-rule-instruction').forEach(_autoGrow);

    const selectAll = $('styleExtractSelectAll');
    if (selectAll) {
        selectAll.checked = _extractCandidates.length > 0 && _extractCandidates.every((c) => c.checked);
    }

    const notice = document.querySelector('#styleExtractResults [data-i18n="style:abstraction_notice"]');
    if (notice) notice.classList.toggle('hidden', !_extractCandidates.some((c) => c.flags && c.flags.length > 0));

    _syncCreateEnablement();
}

function _scheduleExtractAssemble() {
    clearTimeout(_extractAssembleTimer);
    _extractAssembleTimer = setTimeout(_runExtractAssemble, ASSEMBLE_DEBOUNCE_MS);
}

async function _runExtractAssemble() {
    const reqId = ++_extractAssembleReqId;
    const checked = _extractCandidates.filter((c) => c.checked && (c.instruction || '').trim());

    if (checked.length === 0) {
        _setExtractPreview({ translation: '', refinement: '' });
        return;
    }

    const modeModel = $('styleExtractModeModel');
    const mode = modeModel && modeModel.checked ? 'model' : 'source';
    const contextTextarea = $('styleExtractContext');
    const context = contextTextarea ? contextTextarea.value : '';

    let resp;
    try {
        resp = await ApiClient.assembleCustomInstruction(
            mode,
            checked.map((c) => ({ dimension: c.dimension, instruction: c.instruction })),
            context
        );
    } catch (err) {
        console.error('Style assemble failed:', err);
        return;
    }

    // Out-of-order guard (plan risk R7): a stale response must never touch
    // the DOM after a newer edit already superseded it.
    if (reqId !== _extractAssembleReqId) return;

    // The server re-lints every submitted instruction on each assemble call
    // (plan Phase 5: flags[i] = lint_instruction(rules[i].instruction)), so
    // an edited instruction's dimming always reflects the latest answer.
    checked.forEach((c, i) => {
        c.flags = (resp && resp.flags && resp.flags[i]) || [];
    });

    const body = $('styleRulesCandidatesBody');
    if (body) {
        _extractCandidates.forEach((c, idx) => {
            const card = body.querySelector(`[data-candidate-index="${idx}"]`);
            if (card && typeof card._refresh === 'function') card._refresh();
        });
    }

    _setExtractPreview(resp || { translation: '', refinement: '' });
}

function _wireExtractSelectAll() {
    const selectAll = $('styleExtractSelectAll');
    if (!selectAll) return;
    selectAll.addEventListener('change', () => {
        _extractCandidates.forEach((c) => { c.checked = selectAll.checked; });
        _renderExtractCandidatesTable();
        _scheduleExtractAssemble();
    });
}

function _renderExtractResponse(resp) {
    const warningsEl = $('styleExtractWarnings');
    const warnings = (resp && resp.warnings) || [];
    if (warningsEl) {
        warningsEl.innerHTML = '';
        if (warnings.length > 0) {
            warningsEl.classList.remove('hidden');
            const ul = document.createElement('ul');
            for (const w of warnings) {
                const li = document.createElement('li');
                li.textContent = String(w);
                ul.appendChild(li);
            }
            warningsEl.appendChild(ul);
        } else {
            warningsEl.classList.add('hidden');
        }
    }

    _extractSummary = (resp && resp.summary) || '';
    _extractCandidates = ((resp && resp.rules) || []).map((rule) => ({
        dimension: rule.dimension || 'other',
        instruction: rule.instruction || '',
        evidence: rule.evidence || '',
        flags: rule.flags || [],
        checked: !(rule.flags && rule.flags.length > 0),
    }));

    const nameInput = $('styleExtractName');
    if (nameInput) nameInput.value = (resp && resp.suggested_name) || '';

    const contextTextarea = $('styleExtractContext');
    // In 'model' mode the server deliberately returns "" — a reference
    // author's setting must not be imposed on an unrelated text, so the
    // field simply comes up empty for the user to fill in themselves.
    if (contextTextarea) contextTextarea.value = (resp && resp.context) || '';

    const results = $('styleExtractResults');
    if (results) results.classList.remove('hidden');

    _renderExtractCandidatesTable();
    _setExtractPreview((resp && resp.assembled) || { translation: '', refinement: '' });

    const runBtn = $('styleExtractRunBtn');
    const createBtn = $('styleExtractCreateBtn');
    if (runBtn) { runBtn.classList.add('hidden'); runBtn.disabled = false; }
    if (createBtn) createBtn.classList.remove('hidden');
}

async function handleExtractRun() {
    if (_extractFiles.length === 0) return;

    const runBtn = $('styleExtractRunBtn');
    const loading = $('styleExtractLoading');
    if (runBtn) runBtn.disabled = true;
    if (loading) loading.classList.remove('hidden');

    const modeModel = $('styleExtractModeModel');
    const mode = modeModel && modeModel.checked ? 'model' : 'source';
    const sourceLang = DomHelpers.getValue('styleExtractSourceLang') || 'English';
    const targetLang = DomHelpers.getValue('styleExtractTargetLang') || 'English';
    const maxChars = DomHelpers.getValue('styleExtractMaxChars');
    const sampleCount = DomHelpers.getValue('styleExtractSampleCount');

    // Reuse the provider/model/endpoint/key from the main translate form,
    // same read logic as the glossary NER modal (glossary-manager.js).
    const provider = (DomHelpers.getValue('llmProvider') || '').trim();
    const model = (DomHelpers.getValue('model') || '').trim();
    const apiEndpoint = (provider === 'openai'
        ? DomHelpers.getValue('openaiEndpoint')
        : DomHelpers.getValue('apiEndpoint')) || '';
    const apiKey = provider ? ApiKeyUtils.getValueForProvider(provider) : '';

    const formData = new FormData();
    for (const file of _extractFiles) formData.append('files', file);
    formData.append('mode', mode);
    formData.append('source_lang', sourceLang);
    formData.append('target_lang', targetLang);
    if (maxChars) formData.append('max_chars', String(maxChars));
    if (sampleCount) formData.append('sample_count', String(sampleCount));
    if (provider) formData.append('provider', provider);
    if (model) formData.append('model', model);
    if (apiEndpoint) formData.append('api_endpoint', apiEndpoint);
    if (apiKey) formData.append('api_key', apiKey);

    let resp;
    try {
        resp = await ApiClient.extractStyle(formData);
    } catch (err) {
        console.error('Style extraction failed:', err);
        toast.error(t('style:extract_failed', { error: err.message || t('glossary:unknown_error') }));
        if (runBtn) runBtn.disabled = false;
        if (loading) loading.classList.add('hidden');
        return;
    }

    if (loading) loading.classList.add('hidden');
    _renderExtractResponse(resp);
}

async function handleExtractCreate() {
    const nameInput = $('styleExtractName');
    const name = (nameInput ? nameInput.value : '').trim();
    if (!name) {
        toast.warn(t('errors:field_required', { field: t('style:preset_name_label') }));
        if (nameInput) nameInput.focus();
        return;
    }

    const checked = _extractCandidates.filter((c) => c.checked && (c.instruction || '').trim());
    if (checked.length === 0) return;

    const modeModel = $('styleExtractModeModel');
    const mode = modeModel && modeModel.checked ? 'model' : 'source';
    const contextTextarea = $('styleExtractContext');

    const payload = {
        name,
        description: _extractSummary || '',
        context: contextTextarea ? contextTextarea.value : '',
        mode,
        source_files: _extractFiles.map((f) => f.name),
        rules: checked.map((c) => ({ dimension: c.dimension, instruction: c.instruction })),
        overwrite: false,
    };

    const createBtn = $('styleExtractCreateBtn');
    if (createBtn) createBtn.disabled = true;

    try {
        await ApiClient.createCustomInstruction(payload);
    } catch (err) {
        if (_isConflictError(err) && confirm(t('style:confirm_overwrite', { name }))) {
            try {
                await ApiClient.createCustomInstruction({ ...payload, overwrite: true });
            } catch (err2) {
                console.error('Style create (overwrite) failed:', err2);
                toast.error(t('style:save_failed', { error: err2.message || t('glossary:unknown_error') }));
                if (createBtn) createBtn.disabled = false;
                return;
            }
        } else {
            if (!_isConflictError(err)) {
                console.error('Style create failed:', err);
                toast.error(t('style:save_failed', { error: err.message || t('glossary:unknown_error') }));
            }
            if (createBtn) createBtn.disabled = false;
            return;
        }
    }

    closeExtractModal();
    toast.success(t('style:created'));
    window.dispatchEvent(new CustomEvent('customInstructionsChanged'));
    await loadList();
    if (createBtn) createBtn.disabled = false;
}

function openExtractModal() {
    const modal = $('styleExtractModal');
    if (!modal) return;

    _extractFiles = [];
    _extractCandidates = [];
    _extractSummary = '';

    _populateExtractLangSelects();
    _renderExtractFileList();

    const sourceRadio = $('styleExtractModeSource');
    if (sourceRadio) sourceRadio.checked = true;

    const results = $('styleExtractResults');
    if (results) results.classList.add('hidden');

    const warnings = $('styleExtractWarnings');
    if (warnings) { warnings.classList.add('hidden'); warnings.innerHTML = ''; }

    const loading = $('styleExtractLoading');
    if (loading) loading.classList.add('hidden');

    const runBtn = $('styleExtractRunBtn');
    if (runBtn) runBtn.classList.remove('hidden');

    const createBtn = $('styleExtractCreateBtn');
    if (createBtn) { createBtn.classList.add('hidden'); createBtn.disabled = true; }

    const nameInput = $('styleExtractName');
    if (nameInput) nameInput.value = '';

    const contextTextarea = $('styleExtractContext');
    if (contextTextarea) contextTextarea.value = '';

    const body = $('styleRulesCandidatesBody');
    if (body) body.innerHTML = '';

    const fileInput = $('styleExtractFileInput');
    if (fileInput) fileInput.value = '';

    modal.classList.remove('hidden');
}

function closeExtractModal() {
    const modal = $('styleExtractModal');
    if (modal) modal.classList.add('hidden');
    clearTimeout(_extractAssembleTimer);
}

function wireExtractModal() {
    const openBtn = $('styleNewFromBooksBtn');
    if (openBtn) openBtn.addEventListener('click', openExtractModal);

    const closeBtn = $('styleExtractCloseBtn');
    if (closeBtn) closeBtn.addEventListener('click', closeExtractModal);

    _wireExtractDropzone();
    _wireExtractSelectAll();

    const contextTextarea = $('styleExtractContext');
    if (contextTextarea) {
        contextTextarea.addEventListener('input', () => {
            const results = $('styleExtractResults');
            if (results && !results.classList.contains('hidden')) _scheduleExtractAssemble();
        });
    }

    const modeSourceRadio = $('styleExtractModeSource');
    const modeModelRadio = $('styleExtractModeModel');
    [modeSourceRadio, modeModelRadio].forEach((radio) => {
        if (!radio) return;
        radio.addEventListener('change', () => {
            const results = $('styleExtractResults');
            if (results && !results.classList.contains('hidden')) _scheduleExtractAssemble();
        });
    });

    const runBtn = $('styleExtractRunBtn');
    if (runBtn) runBtn.addEventListener('click', handleExtractRun);

    const createBtn = $('styleExtractCreateBtn');
    if (createBtn) createBtn.addEventListener('click', handleExtractCreate);
}

function _rerenderExtractModal() {
    const modal = $('styleExtractModal');
    if (!modal || modal.classList.contains('hidden')) return;
    _renderExtractFileList();
    const body = $('styleRulesCandidatesBody');
    if (body) {
        body.querySelectorAll('.style-rule-card').forEach((card) => {
            if (typeof card._refresh === 'function') card._refresh();
        });
    }
    _syncCreateEnablement();
}

// ========================================
// Locale reactivity
// ========================================

function _rerenderAll() {
    const listView = $('styleListView');
    if (listView && !listView.classList.contains('hidden')) {
        loadList().catch(() => { /* swallow — UI text only */ });
    }
    const editorView = $('styleEditorView');
    if (editorView && !editorView.classList.contains('hidden')) {
        _refreshEditorRuleLabels();
        _updateManualBadge();
    }
    _rerenderExtractModal();
}

// ========================================
// Public API
// ========================================

export const StyleManager = {
    initialize() {
        wireListView();
        wireEditorView();
        wireExtractModal();

        // Re-render dynamic style content when the UI locale changes: the
        // list tbody, an open editor's rule-dimension labels, and an open
        // extract modal's candidate labels/flags/hint are all JS-rendered.
        // Each helper no-ops if its container isn't currently mounted.
        window.addEventListener('localeChanged', _rerenderAll);
    },
    refreshList() {
        return loadList();
    },
};
