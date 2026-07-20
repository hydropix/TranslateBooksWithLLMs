/**
 * Quick LLM test — a "light" sanity check before launching a big translation.
 *
 * Triggered by the flask icon on a Selected-Files row. Grabs 5 segments from
 * THAT file and translates them once, with the LLM + options currently chosen
 * in the Translate/Settings form (read-only — no editing here). Results stream
 * into a blurred overlay so the user can eyeball quality before committing.
 *
 * It is a single-column Sample run under the hood: it reuses the existing
 * /api/sample/run endpoint (no `items` → the server auto-samples; immediate
 * dispatch, no cache/skip machinery) and the `sample_update` WebSocket stream.
 * Its own `sample_id` keeps it isolated from the full Sample tab — each side
 * filters incoming events by the id it owns, so the two never collide.
 */

import { ApiClient } from '../core/api-client.js';
import { WebSocketManager } from '../core/websocket-manager.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { t, applyToDOM } from '../i18n/i18n.js';
import { navigateToSetting } from '../ui/settings-summary.js';
import { SampleManager } from '../sample/sample-manager.js';
import { SAMPLE_DEFAULT_N_SAMPLES, SAMPLE_DEFAULT_MAX_CHARS } from '../sample/sample-defaults.js';

// Samples exactly like the full Sample tool — same shared defaults.
const N_SAMPLES = SAMPLE_DEFAULT_N_SAMPLES;
const MAX_CHARS = SAMPLE_DEFAULT_MAX_CHARS;

const PROVIDER_LABELS = {
    ollama: 'Ollama',
    gemini: 'Gemini',
    openai: 'OpenAI',
    openrouter: 'OpenRouter',
    mistral: 'Mistral',
    deepseek: 'DeepSeek',
    atlascloud: 'Atlas Cloud',
    poe: 'Poe',
    nim: 'NVIDIA NIM',
};

const escapeHtml = DomHelpers.escapeHtml;

const state = {
    sampleId: null,
    running: false,
    rowCount: 0,
    currentFile: null,  // the Selected-Files entry under test (for the "Compare" hand-off)
};

function $(id) {
    return document.getElementById(id);
}

/**
 * Snapshot the LLM configuration currently selected in the Translate/Settings
 * form. Same DOM ids the Settings Summary reads — this IS the config the big
 * translation would use, which is the whole point of the test.
 */
function readGlobalLlmConfig() {
    const provider = (DomHelpers.getValue('llmProvider') || 'ollama').trim().toLowerCase();
    const model = (DomHelpers.getValue('model') || '').trim();

    // Endpoint applies to ollama + openai only; '' lets the server fall back to
    // its own config in _instantiate_provider.
    let apiEndpoint;
    if (provider === 'ollama') apiEndpoint = (DomHelpers.getValue('apiEndpoint') || '').trim() || undefined;
    else if (provider === 'openai') apiEndpoint = (DomHelpers.getValue('openaiEndpoint') || '').trim() || undefined;

    return {
        provider,
        model,
        api_key: '__USE_ENV__',
        api_endpoint: apiEndpoint,
        custom_instruction_file: (DomHelpers.getValue('customInstructionSelect') || '').trim(),
        glossary_id: (DomHelpers.getValue('glossarySelect') || '').trim() || null,
    };
}

function readPromptOptions() {
    return {
        preserve_technical_content: !!($('preserveTechnicalContent')?.checked),
        text_cleanup: !!($('textCleanup')?.checked),
    };
}

function formatLatency(ms) {
    if (ms == null) return '–';
    return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`;
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
        parts.push(`⇄ ${metrics.prompt_tokens || 0}/${metrics.completion_tokens || 0}`);
    }
    const c = formatCost(metrics.cost_usd);
    if (c) parts.push(c);
    if (metrics.length_ratio != null) parts.push(`× ${metrics.length_ratio}`);
    return `<div class="sample-cell-footer">${parts.map(escapeHtml).join(' · ')}</div>`;
}

/** Compact, read-only summary of the LLM + options being tested. */
function renderSummary(file, cfg, mode) {
    const provider = PROVIDER_LABELS[cfg.provider] || cfg.provider || '—';
    const model = cfg.model || '—';
    const langs = mode === 'refine'
        ? t('translation:summary_refining_in', { lang: file.targetLanguage || '—' })
        : `${file.sourceLanguage || t('translation:summary_lang_auto_detect')} → ${file.targetLanguage || '—'}`;

    const chips = [];
    const po = readPromptOptions();
    if (cfg.glossary_id) chips.push(t('translation:quicktest_chip_glossary'));
    if (cfg.custom_instruction_file) chips.push(t('translation:quicktest_chip_instructions'));
    if (po.text_cleanup) chips.push(t('translation:summary_ocr_cleanup'));

    const chipsHtml = chips.length
        ? `<div class="quick-test-chips">${chips.map((c) => `<span class="quick-test-chip">${escapeHtml(c)}</span>`).join('')}</div>`
        : '';

    return `
        <div class="quick-test-summary-llm">
            <strong>${escapeHtml(provider)}</strong>
            <span class="quick-test-summary-sep">·</span>
            <span>${escapeHtml(model)}</span>
            <span class="quick-test-summary-sep">·</span>
            <span>${escapeHtml(langs)}</span>
        </div>
        ${chipsHtml}
    `;
}

function setRunningUi(running) {
    state.running = running;
    const stopBtn = $('quickTestStopBtn');
    if (stopBtn) stopBtn.classList.toggle('hidden', !running);
    const spinner = $('quickTestSpinner');
    if (spinner) spinner.classList.toggle('hidden', !running);
}

function showError(message) {
    const body = $('quickTestBody');
    if (!body) return;
    body.innerHTML = `<div class="quick-test-error">${escapeHtml(message)}</div>`;
}

/** Render one skeleton card per sampled segment, ready to be filled by stream. */
function renderSkeletons(items) {
    const body = $('quickTestBody');
    if (!body) return;
    state.rowCount = items.length;
    body.innerHTML = items.map((item, rowIdx) => `
        <article class="quick-test-card" data-row="${rowIdx}">
            <div class="quick-test-source">${escapeHtml(item.source_text)}</div>
            <div class="quick-test-result" data-row="${rowIdx}">
                <div class="sample-cell-skeleton" aria-live="polite"></div>
            </div>
        </article>
    `).join('');
}

function fillResult(rowIdx, html) {
    const slot = $('quickTestBody')?.querySelector(`.quick-test-result[data-row="${rowIdx}"]`);
    if (slot) slot.innerHTML = html;
}

function handleSampleUpdate(payload) {
    // Only react to events for the run we own — the full Sample tab shares this
    // same WebSocket channel but carries a different sample_id.
    if (!payload || !state.sampleId || payload.sample_id !== state.sampleId) return;

    if (payload.type === 'cell_done') {
        fillResult(payload.row, `
            <div class="sample-output">${escapeHtml(payload.output || '')}</div>
            ${metricsLine(payload.metrics)}
        `);
        return;
    }
    if (payload.type === 'cell_error') {
        fillResult(payload.row, `
            <div class="sample-error-text">${escapeHtml(payload.error || t('translation:quicktest_cell_error'))}</div>
            ${metricsLine(payload.metrics)}
        `);
        return;
    }
    if (payload.type === 'sample_done' || payload.type === 'sample_stopped') {
        setRunningUi(false);
    }
}

function openOverlay() {
    const overlay = $('quickTestModal');
    if (overlay) overlay.classList.remove('hidden');
}

function closeOverlay() {
    if (state.running) stopRun();
    const overlay = $('quickTestModal');
    if (overlay) overlay.classList.add('hidden');
    state.sampleId = null;
}

async function stopRun() {
    if (!state.sampleId) return;
    try {
        await fetch(`/api/sample/${encodeURIComponent(state.sampleId)}/stop`, { method: 'POST' });
    } catch (err) {
        console.warn('[quick-test] stop failed', err);
    }
    setRunningUi(false);
}

/**
 * Launch a quick test for `file` (a Selected-Files queue entry). No-op while a
 * test is already running.
 */
async function open(file) {
    if (state.running) return;
    if (!file || !file.filePath) {
        openOverlay();
        showError(t('translation:quicktest_error_no_file'));
        return;
    }

    state.currentFile = file;
    const cfg = readGlobalLlmConfig();
    const mode = (file.operation === 'refine') ? 'refine' : 'translate';

    openOverlay();

    // Header summary reflects exactly what the big run would use.
    const summary = $('quickTestSummary');
    if (summary) {
        summary.innerHTML = renderSummary(file, cfg, mode);
        applyToDOM(summary);
    }

    if (!cfg.model) {
        showError(t('translation:quicktest_error_no_model'));
        const link = $('quickTestGoSettings');
        if (link) link.classList.remove('hidden');
        return;
    }
    const link = $('quickTestGoSettings');
    if (link) link.classList.add('hidden');

    const targetLanguage = file.targetLanguage || '';
    if (mode !== 'refine' && !targetLanguage) {
        showError(t('translation:quicktest_error_no_target'));
        return;
    }

    const body = $('quickTestBody');
    if (body) body.innerHTML = `<div class="quick-test-loading" data-i18n="translation:quicktest_sampling">${escapeHtml(t('translation:quicktest_sampling'))}</div>`;
    setRunningUi(true);

    const payload = {
        file_path: file.filePath,
        file_type: file.fileType,
        source_language: file.sourceLanguage || '',
        // For refine the server overrides target with the source; it just needs
        // a non-empty value to pass validation.
        target_language: targetLanguage || file.sourceLanguage || 'English',
        mode,
        n_samples: N_SAMPLES,
        max_chars: MAX_CHARS,
        columns: [cfg],
        prompt_options: readPromptOptions(),
    };

    try {
        const r = await fetch('/api/sample/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const resp = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(resp.error || `HTTP ${r.status}`);

        state.sampleId = resp.sample_id;
        renderSkeletons(resp.items || []);
        if (!resp.items || resp.items.length === 0) {
            showError(t('translation:quicktest_error_empty'));
            setRunningUi(false);
        }
    } catch (err) {
        console.error('[quick-test] run failed', err);
        showError(t('translation:quicktest_error_run', { error: err.message || String(err) }));
        setRunningUi(false);
    }
}

function wire() {
    $('quickTestCloseBtn')?.addEventListener('click', closeOverlay);
    $('quickTestDoneBtn')?.addEventListener('click', closeOverlay);
    $('quickTestStopBtn')?.addEventListener('click', stopRun);
    $('quickTestCompareBtn')?.addEventListener('click', () => {
        // Hand the file over to the full Sample & Compare tab so the user can
        // pit several LLMs against each other — without re-dropping the file.
        const file = state.currentFile;
        closeOverlay();
        if (typeof window.switchTopTab === 'function') window.switchTopTab('sample');
        if (file && file.filePath) {
            SampleManager.loadServerFile({
                name: file.name,
                size: file.size,
                filePath: file.filePath,
                fileType: file.fileType,
                thumbnail: file.thumbnail,
                sourceLanguage: file.sourceLanguage,
                targetLanguage: file.targetLanguage,
            });
        }
    });
    $('quickTestGoSettings')?.addEventListener('click', () => {
        closeOverlay();
        navigateToSetting('model');
    });
    // Click the dimmed backdrop (but not the panel) to dismiss.
    $('quickTestModal')?.addEventListener('click', (e) => {
        if (e.target === $('quickTestModal')) closeOverlay();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !$('quickTestModal')?.classList.contains('hidden')) closeOverlay();
    });
}

export const QuickTestManager = {
    init() {
        wire();
        WebSocketManager.on('sample_update', handleSampleUpdate);
    },
    open,
};
