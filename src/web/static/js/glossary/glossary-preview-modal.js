/**
 * Glossary prompt-block preview modal
 *
 * Renders the glossary block the backend would inject into the prompt
 * for a sample text typed by the user.
 */

import { ApiClient } from '../core/api-client.js';
import { toast } from '../ui/toast.js';
import { t } from '../i18n/i18n.js';
import { $, glossaryState } from './glossary-shared.js';

// ========================================
// Preview prompt block modal
// ========================================

export function openPreviewModal() {
    if (glossaryState.currentGlossaryId == null) return;
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
    if (glossaryState.currentGlossaryId == null) return;
    const input = $('glossary-preview-input');
    const out = $('glossary-preview-output');
    const meta = $('glossary-preview-meta');
    const text = input ? input.value : '';

    if (out) out.textContent = t('glossary:preview_loading');
    if (meta) meta.textContent = '';

    try {
        const resp = await ApiClient.previewGlossaryBlock(glossaryState.currentGlossaryId, text);
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

export function wirePreviewModal() {
    const closeBtn = $('glossary-preview-close');
    const cancelBtn = $('glossary-preview-cancel');
    const renderBtn = $('glossary-preview-render');
    if (closeBtn) closeBtn.addEventListener('click', closePreviewModal);
    if (cancelBtn) cancelBtn.addEventListener('click', closePreviewModal);
    if (renderBtn) renderBtn.addEventListener('click', handlePreviewRender);
}
