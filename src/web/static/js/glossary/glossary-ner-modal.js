/**
 * Glossary NER auto-extract modal
 *
 * Lets the user pick a document, asks the backend to suggest glossary
 * terms from samples of it (using the provider configured on the
 * Translate tab), and adds the selected candidates in one bulk request.
 */

import { ApiClient } from '../core/api-client.js';
import { toast } from '../ui/toast.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import { t } from '../i18n/i18n.js';
import { $, glossaryState, buildCategorySelect, buildGenderSelect } from './glossary-shared.js';
import { openEditor, _resolveLangValue } from './glossary-editor.js';

let nerSelectedFile = null;
let nerLastCandidates = [];
const NER_ACCEPTED_EXTS = ['txt', 'srt', 'epub', 'docx', 'pdf'];

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

export function openNerModal() {
    if (glossaryState.currentGlossaryId == null) return;
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
    if (glossaryState.currentGlossaryId == null) return;
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
        resp = await ApiClient.suggestGlossaryTerms(glossaryState.currentGlossaryId, payload);
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
    if (glossaryState.currentGlossaryId == null) return;
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
    const gid = glossaryState.currentGlossaryId;
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

    if (glossaryState.currentGlossaryId === gid) {
        await openEditor(gid);
    }
}

export function wireNerModal() {
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
