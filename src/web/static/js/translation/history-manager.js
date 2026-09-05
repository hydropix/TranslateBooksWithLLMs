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
import { t, getCurrentLocale, applyToDOM } from '../i18n/i18n.js';

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

export const HistoryManager = {
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
        // may only carry `updated_at`.
        const completedRaw = item.completed_at || item.updated_at;
        const completedDate = completedRaw
            ? new Date(completedRaw).toLocaleString(getCurrentLocale())
            : t('translation:job_card_na');

        const safeDisplayName = DomHelpers.escapeHtml(displayNameFormatted);
        const safeOutputFilename = DomHelpers.escapeHtml(outputFilename);

        const card = document.createElement('div');
        card.className = 'history-job-card';
        card.style.cssText = 'border: 1px solid #e5e7eb; padding: 20px; margin-bottom: 15px; border-radius: 8px; background: #f9fafb;';

        card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: start; gap: 15px;">
                <div style="flex: 1; min-width: 0;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; min-width: 0;">
                        <span style="flex-shrink: 0; display: inline-flex; align-items: center;">${getFileIcon(item.file_type)}</span>
                        <span style="font-size: 18px; font-weight: 600; color: #1f2937; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${safeDisplayName}">${safeDisplayName}</span>
                        <span style="flex-shrink: 0; padding: 2px 8px; font-size: 11px; font-weight: 600; color: #166534; background: #dcfce7; border: 1px solid #22c55e; border-radius: 4px;">${t('translation:history_status_completed')}</span>
                    </div>
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="→ ${safeOutputFilename}">
                        → ${safeOutputFilename}
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 20px; font-size: 12px; color: #9ca3af; margin-top: 8px;">
                        <span>${t('translation:history_type_line', { type: DomHelpers.escapeHtml(fileType) })}</span>
                        <span>${t('translation:history_chunks', { completed: completedChunks, total: totalChunks })}</span>
                        <span>${t('translation:history_completed_at', { date: DomHelpers.escapeHtml(completedDate) })}</span>
                    </div>
                </div>
                <div class="history-job-actions" style="flex-shrink: 0;"></div>
            </div>
        `;

        const actionsContainer = card.querySelector('.history-job-actions');
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
            missing.style.cssText = 'font-size: 12px; color: #9ca3af; font-style: italic;';
            missing.textContent = t('translation:history_output_missing');
            actionsContainer.appendChild(missing);
        }

        return card;
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

            MessageLogger.addLog(t('translation:history_count_log', { count: history.length }));

        } catch (error) {
            // Hide loading, show error message
            if (loading) loading.style.display = 'none';
            if (emptyMessage) {
                emptyMessage.style.display = 'block';
                emptyMessage.innerHTML = `<p style="color: #ef4444;">${t('translation:history_load_error', { error: DomHelpers.escapeHtml(error.message) })}</p>`;
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

        // Card copy (status badge, type / chunks / date lines) is baked in via
        // t() at render time rather than through reactive data-i18n markup, and
        // the dates are formatted with the current locale. Re-render the whole
        // list on locale switch so it follows the rest of the UI.
        window.addEventListener('localeChanged', () => {
            this.loadHistory();
        });
    }
};
