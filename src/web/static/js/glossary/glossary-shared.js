/**
 * Glossary shared state and helpers
 *
 * Module state shared by every glossary view (the open glossary and its
 * cached terms), the storage key and option constants, and the small DOM
 * helpers used by more than one view.
 */

import { toast } from '../ui/toast.js';
import { t } from '../i18n/i18n.js';

// ========================================
// Module state
// ========================================

// Mutable state shared across the glossary views. Kept on one object so
// every module sees (and can reassign) the same values through the import.
export const glossaryState = {
    currentGlossaryId: null,

    // In-memory cache of the terms for the currently open glossary so we can
    // re-render after sort / filter / bulk operations without round-tripping.
    currentTerms: [],
};

export const STORAGE_KEY = 'selected_glossary_id';

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

export function $(id) {
    return document.getElementById(id);
}

export function isConflictError(err) {
    if (!err) return false;
    if (err.status === 409) return true;
    if (!err.message) return false;
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

export function buildCategorySelect(currentValue) {
    return buildOptionSelect('category', CATEGORY_OPTIONS, currentValue);
}

export function buildGenderSelect(currentValue) {
    return buildOptionSelect('gender', GENDER_OPTIONS, currentValue);
}

export function genderLabel(value) {
    const opt = GENDER_OPTIONS.find((o) => o.value === (value || ''));
    return opt ? t(opt.labelKey) : (value || '');
}

export function getRowFieldValues(tr) {
    const out = { source: '', target: '', category: '', gender: '' };
    const inputs = tr.querySelectorAll('.glossary-cell-input');
    inputs.forEach((el) => {
        const f = el.dataset.field;
        if (f) out[f] = el.value;
    });
    return out;
}

export function flashRow(tr, color) {
    const prev = tr.style.backgroundColor;
    tr.style.backgroundColor = color;
    setTimeout(() => { tr.style.backgroundColor = prev; }, 800);
}

// Show "Saving..." (intermediate) / "Saved" (success) inside an indicator
// span next to a meta-field label.
export function setSavedIndicator(indicatorEl, state) {
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
// Drag & drop import (editor + empty list)
// ========================================

export function _wireDragDropImport(targetEl, onFile) {
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
