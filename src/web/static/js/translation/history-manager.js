/**
 * History Manager - Completed translations history
 *
 * Lists the jobs that finished successfully (newest first) and offers the
 * download / open / reveal actions for their output files. The list is a
 * read-only view of `GET /api/history`; paused, interrupted and failed jobs
 * stay in the "Paused Translations" section instead.
 */

import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { FileActions } from '../files/file-actions.js';
import { getFileIcon } from './progress-title.js';
import { t, tHtml, getCurrentLocale, applyToDOM } from '../i18n/i18n.js';

/**
 * Fetch the translated-files listing without letting its failure sink the
 * history render: a missing or erroring file list only means no output file
 * can be confirmed, and the history itself is still worth showing.
 * @returns {Promise<Set<string>>} Filenames present in translated_files/
 */
async function loadAvailableFiles() {
    try {
        const data = await ApiClient.getFileList();
        const files = (data && data.files) || [];
        return new Set(files.map(file => file.filename));
    } catch (error) {
        console.warn('Could not load the file list for the history availability check:', error);
        return new Set();
    }
}

/** Above this many rows the list scrolls, so a filter box is shown. */
const HISTORY_FILTER_THRESHOLD = 6;

export const HistoryManager = {
    /** IDs of the listed entries whose output file no longer exists. */
    _missingIds: [],

    /**
     * Build the card for one completed job.
     *
     * Every label goes through t() at render time (the whole list is rebuilt
     * on locale switch) and every dynamic value is escaped before it reaches
     * the HTML string.
     * @param {Object} item - History item from GET /api/history
     * @param {Set<string>} availableFiles - Filenames still present in translated_files/
     * @returns {HTMLElement} Card element for the job
     */
    buildHistoryCard(item, availableFiles) {
        const unknownText = t('translation:job_card_unknown');
        const inputFilename = item.input_filename || unknownText;
        const outputFilename = item.output_filename || unknownText;

        // Same display name as the resumable cards: drop the 16-hex upload
        // prefix and the extension, then upper-case the first letter.
        const inputMatch = inputFilename.match(/^([a-f0-9]{16})_(.+)$/);
        const inputOriginalName = inputMatch ? inputMatch[2] : inputFilename;
        const displayName = inputOriginalName.replace(/\.[^.]+$/, '');
        const displayNameFormatted = displayName
            ? displayName.charAt(0).toUpperCase() + displayName.slice(1)
            : inputOriginalName;

        const fileType = (item.file_type || 'txt').toUpperCase();
        const completedChunks = item.completed_chunks || 0;
        const totalChunks = item.total_chunks || 0;

        // `completed_at` is set when the job reaches 'completed'; older rows
        // may only carry `updated_at`. The row shows a short date; the full
        // timestamp lives in the tooltip.
        const completedRaw = item.completed_at || item.updated_at;
        const completedAt = completedRaw ? new Date(completedRaw) : null;
        const locale = getCurrentLocale();
        const shortDate = completedAt
            ? completedAt.toLocaleDateString(locale, { day: 'numeric', month: 'short', year: 'numeric' })
            : t('translation:job_card_na');
        const fullDate = completedAt
            ? completedAt.toLocaleString(locale)
            : t('translation:job_card_na');

        const safeDisplayName = DomHelpers.escapeHtml(displayNameFormatted);
        const safeOutputFilename = DomHelpers.escapeHtml(outputFilename);

        const card = document.createElement('div');
        card.className = 'history-row';
        // Lower-cased haystack for the client-side filter.
        card.dataset.search = `${displayNameFormatted} ${outputFilename}`.toLowerCase();

        card.innerHTML = `
            <span class="history-row__icon">${getFileIcon(item.file_type)}</span>
            <div class="history-row__main">
                <div class="history-row__name" title="${safeDisplayName}">${safeDisplayName}</div>
                <div class="history-row__meta">
                    <span class="history-row__output" title="→ ${safeOutputFilename}">→ ${safeOutputFilename}</span>
                    <span class="history-row__tag">${DomHelpers.escapeHtml(fileType)}</span>
                    <span class="history-row__chunks">${tHtml('translation:history_chunks', { completed: completedChunks, total: totalChunks })}</span>
                </div>
            </div>
            <time class="history-row__date" title="${DomHelpers.escapeHtml(t('translation:history_completed_at', { date: fullDate }))}">${DomHelpers.escapeHtml(shortDate)}</time>
            <div class="history-row__actions"></div>
            <button type="button" class="history-row__remove material-symbols-outlined"
                    title="${DomHelpers.escapeHtml(t('translation:history_remove_title'))}"
                    aria-label="${DomHelpers.escapeHtml(t('translation:history_remove_title'))}">close</button>
        `;

        card.querySelector('.history-row__remove').addEventListener('click', () => {
            this.removeEntry(item.translation_id, card);
        });

        const actionsContainer = card.querySelector('.history-row__actions');
        if (item.output_filename && availableFiles.has(item.output_filename)) {
            actionsContainer.appendChild(FileActions.createActionGroup({
                actions: ['download', 'open', 'reveal'],
                filename: item.output_filename,
                variant: 'compact',
            }));
        } else {
            // The row outlives its output file (deleted from the Files tab, or
            // pruned): say so instead of offering actions that would 404.
            const missing = document.createElement('span');
            missing.className = 'history-row__missing material-symbols-outlined';
            missing.textContent = 'link_off';
            missing.title = t('translation:history_output_missing');
            missing.setAttribute('aria-label', t('translation:history_output_missing'));
            actionsContainer.appendChild(missing);
        }

        return card;
    },

    /**
     * Show only the rows whose name or output file contains the filter text.
     */
    applyFilter() {
        const input = DomHelpers.getElement('historyFilter');
        const listContainer = DomHelpers.getElement('historyList');
        const noMatch = DomHelpers.getElement('historyNoMatch');
        if (!listContainer) return;

        const query = (input ? input.value : '').trim().toLowerCase();
        let visible = 0;
        listContainer.querySelectorAll('.history-row').forEach((row) => {
            const match = !query || row.dataset.search.includes(query);
            row.hidden = !match;
            if (match) visible++;
        });
        if (noMatch) noMatch.hidden = visible > 0;
    },

    /**
     * Refresh the cleanup menu's "entries without a file" item: its count
     * goes through data-i18n-params so a locale switch re-renders it.
     */
    updateCleanupMenu() {
        const label = DomHelpers.getElement('historyCleanupMissingLabel');
        const item = DomHelpers.getElement('historyCleanupMissing');
        const count = this._missingIds.length;
        if (label) {
            label.setAttribute('data-i18n-params', JSON.stringify({ count }));
            applyToDOM(label.parentElement);
        }
        if (item) item.disabled = count === 0;
    },

    toggleCleanupMenu(open) {
        const menu = DomHelpers.getElement('historyCleanupMenu');
        const button = DomHelpers.getElement('historyCleanupBtn');
        if (!menu || !button) return;
        const show = open === undefined ? menu.hidden : open;
        menu.hidden = !show;
        button.setAttribute('aria-expanded', String(show));
    },

    /**
     * Remove one entry from the history. The output file is not touched, so
     * no confirmation is asked: the row only points at the Files tab.
     * @param {string} translationId - Job identifier
     * @param {HTMLElement} row - The row to drop on success
     */
    async removeEntry(translationId, row) {
        try {
            await ApiClient.deleteHistoryEntry(translationId);
            row.remove();
            this._missingIds = this._missingIds.filter(id => id !== translationId);
            this.afterRowsRemoved();
        } catch (error) {
            MessageLogger.showMessage(t('translation:history_clear_error', { error: error.message }), 'error');
        }
    },

    /**
     * Clear either the entries whose output file is gone or the whole history.
     * @param {'missing'|'all'} scope
     */
    async cleanup(scope) {
        this.toggleCleanupMenu(false);
        const ids = scope === 'missing' ? [...this._missingIds] : undefined;
        if (scope === 'missing' && ids.length === 0) return;

        const question = scope === 'missing'
            ? t('translation:history_confirm_clear_missing', { count: ids.length })
            : t('translation:history_confirm_clear_all');
        if (!confirm(question)) return;

        try {
            const result = await ApiClient.clearHistory(ids);
            const deleted = (result && result.deleted) || 0;
            MessageLogger.showMessage(t('translation:history_cleared', { count: deleted }), 'success');
        } catch (error) {
            MessageLogger.showMessage(t('translation:history_clear_error', { error: error.message }), 'error');
        }
        await this.loadHistory();
    },

    /** Keep the count, filter and empty state in step after a local removal. */
    afterRowsRemoved() {
        const listContainer = DomHelpers.getElement('historyList');
        const remaining = listContainer ? listContainer.querySelectorAll('.history-row').length : 0;
        if (remaining === 0) {
            // Same as an empty listing: the section disappears.
            this.loadHistory();
            return;
        }
        const count = DomHelpers.getElement('historyCount');
        if (count) count.textContent = String(remaining);
        const filterWrap = DomHelpers.getElement('historyFilterWrap');
        if (filterWrap) filterWrap.hidden = remaining <= HISTORY_FILTER_THRESHOLD;
        this.updateCleanupMenu();
        this.applyFilter();
    },

    /**
     * Load and display the completed translations history
     */
    async loadHistory() {
        const section = DomHelpers.getElement('historySection');
        const loading = DomHelpers.getElement('historyLoading');
        const listContainer = DomHelpers.getElement('historyList');
        const emptyMessage = DomHelpers.getElement('historyEmpty');

        // Show loading, hide list and empty message (use inline style to override)
        if (loading) loading.style.display = 'block';
        if (listContainer) listContainer.style.display = 'none';
        if (emptyMessage) emptyMessage.style.display = 'none';

        try {
            const [data, availableFiles] = await Promise.all([
                ApiClient.getJobHistory(50),
                loadAvailableFiles(),
            ]);
            const history = (data && data.history) || [];

            // Hide loading
            if (loading) loading.style.display = 'none';

            if (history.length === 0) {
                // Hide the section entirely when there is nothing to show
                if (section) section.style.display = 'none';
                if (emptyMessage) emptyMessage.style.display = 'block';
                return;
            }

            if (!listContainer) {
                console.error('Error: historyList element not found');
                return;
            }

            // Show section and populate the list (use inline style to override)
            if (section) section.style.display = 'block';
            listContainer.style.display = 'block';

            listContainer.innerHTML = '';
            history.forEach((item) => {
                listContainer.appendChild(this.buildHistoryCard(item, availableFiles));
            });

            // Translate the data-i18n markup carried by the injected cards
            applyToDOM(listContainer);

            const count = DomHelpers.getElement('historyCount');
            if (count) {
                count.textContent = String(history.length);
                count.hidden = false;
            }
            // The filter only earns its place once the list needs scrolling.
            const filterWrap = DomHelpers.getElement('historyFilterWrap');
            if (filterWrap) filterWrap.hidden = history.length <= HISTORY_FILTER_THRESHOLD;
            this.applyFilter();

            // Entries whose output file is gone are what "clean up" targets first.
            this._missingIds = history
                .filter(item => !(item.output_filename && availableFiles.has(item.output_filename)))
                .map(item => item.translation_id);
            this.updateCleanupMenu();

            MessageLogger.addLog(t('translation:history_count_log', { count: history.length }));

        } catch (error) {
            // Hide loading, show error message
            if (loading) loading.style.display = 'none';
            if (emptyMessage) {
                emptyMessage.style.display = 'block';
                emptyMessage.innerHTML = `<p style="color: var(--danger-light);">${t('translation:history_load_error', { error: DomHelpers.escapeHtml(error.message) })}</p>`;
            }
            // Hide the section on error
            if (section) section.style.display = 'none';
            console.error('Error loading translation history:', error);
        }
    },

    /**
     * Initialize history manager
     */
    initialize() {
        // Load the history on initialization
        this.loadHistory();

        const filter = DomHelpers.getElement('historyFilter');
        if (filter) filter.addEventListener('input', () => this.applyFilter());

        const cleanupBtn = DomHelpers.getElement('historyCleanupBtn');
        if (cleanupBtn) {
            cleanupBtn.addEventListener('click', (event) => {
                event.stopPropagation();
                this.toggleCleanupMenu();
            });
        }
        DomHelpers.getElement('historyCleanupMissing')
            ?.addEventListener('click', () => this.cleanup('missing'));
        DomHelpers.getElement('historyCleanupAll')
            ?.addEventListener('click', () => this.cleanup('all'));

        // Close the menu on any outside click or on Escape.
        document.addEventListener('click', (event) => {
            const menu = DomHelpers.getElement('historyCleanupMenu');
            if (menu && !menu.hidden && !menu.contains(event.target)) this.toggleCleanupMenu(false);
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') this.toggleCleanupMenu(false);
        });

        // Card copy (status badge, type / chunks / date lines) is baked in via
        // t() at render time rather than through reactive data-i18n markup, and
        // the dates are formatted with the current locale. Re-render the whole
        // list on locale switch so it follows the rest of the UI.
        window.addEventListener('localeChanged', () => {
            this.loadHistory();
        });
    }
};
