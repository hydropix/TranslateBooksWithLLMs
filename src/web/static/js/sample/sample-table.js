/**
 * Sample & Compare result renderer — stacked layout.
 *
 * Renders one "card" per sample (source extract on top), with each LLM block
 * stacked vertically inside. Replaces the original column-table layout, which
 * became unreadable at 3+ columns × 5+ samples.
 *
 * Data model unchanged: SampleManager still talks to SampleTable through
 * `render/updateCell/toMarkdown/hasResults` — only the DOM shape changed.
 */

import { t } from '../i18n/i18n.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { inlineDiff } from './sample-diff.js';

// Shared HTML escaper (was a local copy here and in sample-diff.js).
const escapeHtml = DomHelpers.escapeHtml;

function formatLatency(ms) {
    if (ms == null) return '–';
    if (ms < 1000) return `${ms} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
}

function formatCost(cost) {
    if (cost == null) return null;
    if (cost < 0.0001) return '$<0.0001';
    return `$${cost.toFixed(4)}`;
}

function metricsLine(metrics) {
    if (!metrics) return '';
    const parts = [];
    if (metrics.latency_ms != null) parts.push(`⏱ ${formatLatency(metrics.latency_ms)}`);
    if (metrics.prompt_tokens != null || metrics.completion_tokens != null) {
        const pin = metrics.prompt_tokens || 0;
        const pout = metrics.completion_tokens || 0;
        parts.push(`⇄ ${pin}/${pout}`);
    }
    if (metrics.cost_usd != null) {
        const c = formatCost(metrics.cost_usd);
        if (c) parts.push(c);
    }
    if (metrics.length_ratio != null) parts.push(`× ${metrics.length_ratio}`);
    const badges = [];
    if (metrics.was_fallback) badges.push(`<span class="sample-badge sample-badge-fallback">${t('sample:badge_fallback')}</span>`);
    if (metrics.was_truncated) badges.push(`<span class="sample-badge sample-badge-truncated">${t('sample:badge_truncated')}</span>`);
    const text = parts.map(escapeHtml).join(' · ');
    return `<div class="sample-cell-footer">${text}${badges.length ? ' ' + badges.join(' ') : ''}</div>`;
}

function llmHeader(col, colIdx) {
    const provider = escapeHtml(col.provider || '?');
    const model = escapeHtml(col.model || '?');
    return `
        <div class="sample-llm-header">
            <span class="sample-llm-index">#${colIdx + 1}</span>
            <span class="sample-llm-provider">${provider}</span>
            <span class="sample-llm-model">${model}</span>
        </div>
    `;
}

function sourceHeader(item, rowIdx) {
    const truncBadge = item.truncated
        ? `<span class="sample-badge sample-badge-truncated">${t('sample:badge_truncated')}</span>`
        : '';
    return `
        <div class="sample-card-source">
            <button type="button" class="sample-card-remove" data-row="${rowIdx}"
                    data-i18n-attr="title:sample:remove_sample;aria-label:sample:remove_sample"
                    title="${t('sample:remove_sample')}"
                    aria-label="${t('sample:remove_sample')}">
                <span class="material-symbols-outlined">close</span>
            </button>
            <div class="sample-card-source-meta">
                <span class="sample-card-source-index">#${escapeHtml(item.index)}</span>
                <span class="sample-card-source-label" data-i18n="sample:source_col_header">${t('sample:source_col_header')}</span>
                ${truncBadge}
            </div>
            <div class="sample-card-source-text">${escapeHtml(item.source_text)}</div>
        </div>
    `;
}

export const SampleTable = {
    _root: null,
    _mode: 'translate',
    _items: [],
    _columns: [],
    _cellState: new Map(), // key "row:col" -> {translate?, refine?}

    /**
     * Render the comparison grid.
     *
     * `options.prefilled` is a `Map<"row:col", {translate?, refine?}>` that
     * seeds individual cells with already-known results (typically pulled
     * from the cross-Run cache in sample-manager). Seeded cells render their
     * content immediately; the rest show a skeleton until a `sample_update`
     * event arrives over WebSocket.
     */
    render(rootEl, items, columns, mode, options = {}) {
        this._root = rootEl;
        this._items = items;
        this._columns = columns;
        this._mode = mode;
        this._cellState = new Map();

        const prefilled = options.prefilled instanceof Map ? options.prefilled : null;
        // Cells in `runningCells` get the animated shimmer (work actually in
        // flight). Cells absent from both prefilled and runningCells get a
        // static "Click Run" hint — no animation, since nothing is happening.
        const runningCells = options.runningCells instanceof Set ? options.runningCells : null;

        // `items === null` is "no file uploaded yet" — show the upload-first
        // message and stop. An empty array is "file uploaded, user removed
        // every sample" — fall through and just render the Add button.
        if (!items) {
            rootEl.innerHTML = `<p data-i18n="sample:no_results_yet">${t('sample:no_results_yet')}</p>`;
            return;
        }

        const pendingHtml = `<div class="sample-cell-pending" data-i18n="sample:cell_pending">${escapeHtml(t('sample:cell_pending'))}</div>`;
        const skeletonHtml = `<div class="sample-cell-skeleton" aria-live="polite"></div>`;

        const cards = items.map((item, rowIdx) => {
            const llmBlocks = columns.map((col, colIdx) => {
                let content = pendingHtml;
                const seed = prefilled ? prefilled.get(`${rowIdx}:${colIdx}`) : null;
                if (seed) {
                    this._cellState.set(`${rowIdx}:${colIdx}`, seed);
                    content = this._renderLlmContent(seed);
                } else if (runningCells && runningCells.has(`${rowIdx}:${colIdx}`)) {
                    content = skeletonHtml;
                }
                return `
                    <div class="sample-llm-block" data-row="${rowIdx}" data-col="${colIdx}">
                        ${llmHeader(col, colIdx)}
                        <div class="sample-llm-content">${content}</div>
                    </div>
                `;
            }).join('');
            return `
                <article class="sample-card" data-row="${rowIdx}">
                    ${sourceHeader(item, rowIdx)}
                    <div class="sample-card-results">${llmBlocks}</div>
                </article>
            `;
        }).join('');

        const emptyMsg = items.length === 0
            ? `<p class="sample-empty" data-i18n="sample:no_samples_left">${t('sample:no_samples_left')}</p>`
            : '';

        const addRow = `
            <div class="sample-add-row">
                <button type="button" class="btn btn-secondary" id="sampleAddSampleBtn">
                    <span class="material-symbols-outlined" style="font-size: 1rem; vertical-align: middle;">add</span>
                    <span data-i18n="sample:add_sample">Add a sample</span>
                </button>
            </div>
        `;

        rootEl.innerHTML = `${emptyMsg}<div class="sample-cards">${cards}</div>${addRow}`;
    },

    updateCell(payload) {
        if (!this._root) return;
        const { row, col, phase, output, metrics, error, type } = payload;
        const key = `${row}:${col}`;
        const prev = this._cellState.get(key) || {};
        prev[phase] = { status: type === 'cell_done' ? 'done' : 'error', output, metrics, error };
        this._cellState.set(key, prev);

        const block = this._root.querySelector(`.sample-llm-block[data-row="${row}"][data-col="${col}"]`);
        if (!block) return;
        block.querySelector('.sample-llm-content').innerHTML = this._renderLlmContent(prev);
    },

    _renderLlmContent(perPhase) {
        const tr = perPhase.translate;
        const rf = perPhase.refine;

        const renderPhase = (label, block) => {
            if (!block) return '';
            if (block.status === 'error') {
                return `
                    <div class="sample-phase sample-phase-error">
                        <div class="sample-phase-label">${escapeHtml(label)}</div>
                        <div class="sample-error-text">${escapeHtml(block.error || t('errors:unknown_error'))}</div>
                        ${metricsLine(block.metrics)}
                    </div>
                `;
            }
            return `
                <div class="sample-phase">
                    <div class="sample-phase-label">${escapeHtml(label)}</div>
                    <div class="sample-output">${escapeHtml(block.output || '')}</div>
                    ${metricsLine(block.metrics)}
                </div>
            `;
        };

        if (this._mode === 'translate_refine') {
            const blocks = [];
            blocks.push(renderPhase(t('sample:phase_translated'), tr));
            if (rf) {
                if (rf.status === 'error') {
                    blocks.push(renderPhase(t('sample:phase_refined'), rf));
                } else {
                    const draft = tr && tr.output ? tr.output : '';
                    const diffHtml = draft ? inlineDiff(draft, rf.output || '') : escapeHtml(rf.output || '');
                    blocks.push(`
                        <div class="sample-phase">
                            <div class="sample-phase-label">${escapeHtml(t('sample:phase_refined'))}</div>
                            <div class="sample-output sample-output-diff">${diffHtml}</div>
                            ${metricsLine(rf.metrics)}
                        </div>
                    `);
                }
            }
            return blocks.join('') || `<div class="sample-cell-pending" data-i18n="sample:cell_pending">${escapeHtml(t('sample:cell_pending'))}</div>`;
        }

        if (this._mode === 'refine') {
            return rf ? renderPhase(t('sample:phase_refined'), rf) : `<div class="sample-cell-pending" data-i18n="sample:cell_pending">${escapeHtml(t('sample:cell_pending'))}</div>`;
        }

        return tr ? renderPhase(t('sample:phase_translated'), tr) : `<div class="sample-cell-pending" data-i18n="sample:cell_pending">${escapeHtml(t('sample:cell_pending'))}</div>`;
    },

    /**
     * Serialize results as stacked Markdown — one section per sample, one
     * sub-section per LLM. The previous wide-table format became unreadable
     * past 2-3 columns; the stacked version mirrors what the UI shows.
     */
    toMarkdown() {
        if (!this._items || this._items.length === 0) return '';
        const lines = [];
        for (let r = 0; r < this._items.length; r += 1) {
            const item = this._items[r];
            lines.push(`## Sample #${item.index}${item.truncated ? ' (truncated)' : ''}`);
            lines.push('');
            lines.push('> ' + (item.source_text || '').replace(/\n+/g, '\n> '));
            lines.push('');
            for (let c = 0; c < this._columns.length; c += 1) {
                const col = this._columns[c];
                const state = this._cellState.get(`${r}:${c}`) || {};
                lines.push(`### ${col.provider} / ${col.model}`);
                lines.push('');

                const emitBlock = (label, block) => {
                    if (!block) {
                        lines.push(`*(pending)*`);
                        return;
                    }
                    lines.push(`**${label}:**`);
                    lines.push('');
                    if (block.status === 'error') {
                        lines.push(`> _Error: ${(block.error || '').replace(/\n/g, ' ')}_`);
                    } else {
                        lines.push(block.output || '');
                    }
                    lines.push('');
                    if (block.metrics) {
                        const parts = [];
                        if (block.metrics.latency_ms != null) parts.push(`latency: ${formatLatency(block.metrics.latency_ms)}`);
                        if (block.metrics.prompt_tokens != null) parts.push(`tokens in/out: ${block.metrics.prompt_tokens}/${block.metrics.completion_tokens || 0}`);
                        if (block.metrics.cost_usd != null) parts.push(`cost: ${formatCost(block.metrics.cost_usd)}`);
                        if (block.metrics.length_ratio != null) parts.push(`length ratio: ${block.metrics.length_ratio}`);
                        if (parts.length) {
                            lines.push(`_${parts.join(' · ')}_`);
                            lines.push('');
                        }
                    }
                };

                if (this._mode === 'translate_refine') {
                    emitBlock('Translated', state.translate);
                    emitBlock('Refined', state.refine);
                } else if (this._mode === 'refine') {
                    emitBlock('Refined', state.refine);
                } else {
                    emitBlock('Translated', state.translate);
                }
            }
            lines.push('---');
            lines.push('');
        }
        return lines.join('\n').trimEnd();
    },

    hasResults() {
        return this._items && this._items.length > 0;
    },
};
