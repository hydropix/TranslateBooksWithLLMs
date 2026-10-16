/**
 * Ask the Book tab.
 *
 * Upload a book, then ask questions about it. Answers come only from the book's
 * own text (POST /api/ask/query); questions the book does not cover are shown
 * as "out of scope". Uses the LLM currently selected in Settings, exactly like
 * Quick Test.
 *
 * i18n: static labels in rendered messages carry data-i18n attributes, so
 * applyToDOM() re-translates them on locale switch. LLM answers are content
 * and are never translated.
 */
import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { applyToDOM, t } from '../i18n/i18n.js';
import { readGlobalLlmConfig } from '../translation/quick-test.js';

const SUPPORTED_EXTENSIONS = ['txt', 'epub', 'docx', 'pdf'];
const MAX_HISTORY_TURNS = 6;

const escapeHtml = DomHelpers.escapeHtml;

const state = {
    filePath: null,
    fileName: '',
    book: null,          // {passages, language, chars, search}
    history: [],         // [{question, answer}] — in-scope turns only
    busy: false,
};

function $(id) {
    return document.getElementById(id);
}

function setStatus(key, params) {
    const el = $('askStatus');
    if (!el) return;
    if (!key) {
        el.classList.add('hidden');
        el.removeAttribute('data-i18n');
        el.removeAttribute('data-i18n-params');
        el.textContent = '';
        return;
    }
    el.setAttribute('data-i18n', key);
    if (params) el.setAttribute('data-i18n-params', JSON.stringify(params));
    else el.removeAttribute('data-i18n-params');
    el.textContent = t(key, params);
    el.classList.remove('hidden');
}

function updateControls() {
    const ready = !!state.book && !state.busy;
    const input = $('askQuestionInput');
    if (input) input.disabled = !ready;
    const send = $('askSendBtn');
    if (send) send.disabled = !ready;
    const clear = $('askClearBtn');
    if (clear) clear.disabled = state.busy || !state.history.length;
}

function renderBookInfo() {
    const info = $('askFileInfo');
    const upload = $('askFileUpload');
    if (!info || !upload) return;
    if (!state.filePath) {
        info.classList.add('hidden');
        upload.classList.remove('hidden');
        return;
    }
    upload.classList.add('hidden');
    info.classList.remove('hidden');
    $('askFileName').textContent = state.fileName;
    const details = $('askFileDetails');
    if (state.book) {
        details.setAttribute('data-i18n', state.book.search === 'hybrid' ? 'ask:book_details_hybrid' : 'ask:book_details_bm25');
        details.setAttribute('data-i18n-params', JSON.stringify({
            passages: state.book.passages,
            language: state.book.language,
        }));
    } else {
        details.setAttribute('data-i18n', 'ask:indexing');
        details.removeAttribute('data-i18n-params');
    }
    applyToDOM(info);
}

/** Escape the answer, keep line breaks, and turn [P12] markers into citation chips. */
function formatAnswer(text) {
    return escapeHtml(text)
        .replace(/\[P(\d+)\]/g, '<span class="ask-cite">P$1</span>')
        .replace(/\n/g, '<br>');
}

function appendMessage(html, cls) {
    const log = $('askConversation');
    if (!log) return null;
    $('askEmptyState')?.classList.add('hidden');
    const div = document.createElement('div');
    div.className = `ask-message ${cls}`;
    div.innerHTML = html;
    log.appendChild(div);
    applyToDOM(div);
    // Bring the start of the new message into view; long answers are then read top-down.
    div.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return div;
}

function renderResult(result) {
    if (!result.in_scope) {
        const reasonKey = `ask:reason_${result.reason || 'no_match'}`;
        appendMessage(
            `<div class="ask-oos-title"><span class="material-symbols-outlined">block</span>
                <span data-i18n="ask:out_of_scope"></span></div>
             <div class="ask-oos-reason" data-i18n="${escapeHtml(reasonKey)}"></div>`,
            'ask-answer ask-out-of-scope',
        );
        return;
    }
    const sources = (result.citations || []).map((c) => `
        <details class="ask-source">
            <summary><span class="ask-cite">P${Number(c.passage_id)}</span> ${escapeHtml(c.section || '')}</summary>
            <div class="ask-source-text">${escapeHtml(c.text)}</div>
        </details>`).join('');
    appendMessage(
        `<div class="ask-answer-text">${formatAnswer(result.answer)}</div>
         ${sources ? `<div class="ask-sources"><div class="ask-sources-label" data-i18n="ask:sources"></div>${sources}</div>` : ''}`,
        'ask-answer',
    );
}

async function indexBook() {
    state.book = null;
    renderBookInfo();
    updateControls();
    try {
        const resp = await fetch('/api/ask/index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: state.filePath }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            setStatus(data.error === 'unsupported_file_type' ? 'ask:error_unsupported' : 'ask:error_index',
                { details: data.error || resp.status });
            return;
        }
        state.book = data;
        setStatus(null);
    } catch (err) {
        setStatus('ask:error_index', { details: err.message });
    } finally {
        renderBookInfo();
        updateControls();
    }
}

async function handleFile(file) {
    if (!file) return;
    const ext = (file.name.split('.').pop() || '').toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
        setStatus('ask:error_unsupported', { details: ext });
        return;
    }
    setStatus('ask:uploading');
    try {
        const uploaded = await ApiClient.uploadFile(file);
        state.filePath = uploaded.file_path;
        state.fileName = uploaded.filename || file.name;
        clearConversation();
        await indexBook();
    } catch (err) {
        setStatus('ask:error_index', { details: err.message });
    }
}

function removeBook() {
    state.filePath = null;
    state.fileName = '';
    state.book = null;
    clearConversation();
    setStatus(null);
    const input = $('askFileInput');
    if (input) input.value = '';
    renderBookInfo();
    updateControls();
}

function clearConversation() {
    state.history = [];
    const log = $('askConversation');
    if (log) log.querySelectorAll('.ask-message').forEach((el) => el.remove());
    $('askEmptyState')?.classList.remove('hidden');
    updateControls();
}

async function sendQuestion() {
    const input = $('askQuestionInput');
    const question = (input?.value || '').trim();
    if (!question || !state.book || state.busy) return;

    const llm = readGlobalLlmConfig();
    if (!llm.model) {
        setStatus('ask:error_no_model');
        return;
    }
    setStatus(null);
    input.value = '';
    appendMessage(escapeHtml(question), 'ask-question');
    const pending = appendMessage('<span class="ask-thinking" data-i18n="ask:thinking"></span>', 'ask-answer ask-pending');

    state.busy = true;
    updateControls();
    try {
        const resp = await fetch('/api/ask/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: state.filePath,
                question,
                history: state.history.slice(-MAX_HISTORY_TURNS),
                answer_language: $('askAnswerLanguage')?.value || 'auto',
                provider: llm.provider,
                model: llm.model,
                api_key: llm.api_key,
                api_endpoint: llm.api_endpoint,
            }),
        });
        const data = await resp.json();
        pending?.remove();
        if (!resp.ok) {
            setStatus('ask:error_answer', { details: data.error || resp.status });
            return;
        }
        renderResult(data);
        if (data.in_scope) state.history.push({ question, answer: data.answer });
    } catch (err) {
        pending?.remove();
        setStatus('ask:error_answer', { details: err.message });
    } finally {
        state.busy = false;
        updateControls();
        input.focus();
    }
}

export const AskManager = {
    init() {
        const fileInput = $('askFileInput');
        if (!fileInput) return;
        fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

        const drop = $('askFileUpload');
        drop?.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('dragging'); });
        drop?.addEventListener('dragleave', () => drop.classList.remove('dragging'));
        drop?.addEventListener('drop', (e) => {
            e.preventDefault();
            drop.classList.remove('dragging');
            handleFile(e.dataTransfer.files[0]);
        });

        $('askFileRemoveBtn')?.addEventListener('click', removeBook);
        $('askClearBtn')?.addEventListener('click', clearConversation);
        $('askSendBtn')?.addEventListener('click', sendQuestion);
        $('askQuestionInput')?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuestion();
            }
        });
        renderBookInfo();
        updateControls();
    },
};
