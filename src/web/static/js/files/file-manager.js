/**
 * File Manager - File list management and operations
 *
 * Handles file list display, selection management, batch operations
 * (download/delete), and individual file actions.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { FileActions } from './file-actions.js';
import { t } from '../i18n/i18n.js';

export const FileManager = {
    /**
     * Initialize file manager
     */
    initialize() {
        this.setupEventListeners();
        this.refreshFileList();
    },

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Listen for file list changes
        window.addEventListener('fileListChanged', () => {
            this.refreshFileList();
        });

        // Re-render the list when the UI locale changes: rows are built with
        // t() at render time, so a full refresh applies the new translations
        // (including the "Download N" / "Delete N" selection buttons).
        window.addEventListener('localeChanged', () => {
            this.refreshFileList();
        });

        // Select all checkbox
        const selectAllCheckbox = DomHelpers.getElement('selectAllFiles');
        if (selectAllCheckbox) {
            selectAllCheckbox.addEventListener('change', () => {
                this.toggleSelectAll();
            });
        }

        // Batch download button
        const batchDownloadBtn = DomHelpers.getElement('batchDownloadBtn');
        if (batchDownloadBtn) {
            batchDownloadBtn.addEventListener('click', () => {
                this.downloadSelectedFiles();
            });
        }

        // Batch delete button
        const batchDeleteBtn = DomHelpers.getElement('batchDeleteBtn');
        if (batchDeleteBtn) {
            batchDeleteBtn.addEventListener('click', () => {
                this.deleteSelectedFiles();
            });
        }
    },

    /**
     * Refresh file list from server
     */
    async refreshFileList() {
        const loadingDiv = DomHelpers.getElement('fileListLoading');
        const containerDiv = DomHelpers.getElement('fileManagementContainer');
        const tableBody = DomHelpers.getElement('fileTableBody');
        const emptyDiv = DomHelpers.getElement('fileListEmpty');

        if (!tableBody) return;

        // Show loading, hide container (use inline style to override)
        if (loadingDiv) loadingDiv.style.display = 'block';
        if (containerDiv) containerDiv.style.display = 'none';

        try {
            const data = await ApiClient.getFileList();

            // Hide loading, show container (use inline style to override)
            if (loadingDiv) loadingDiv.style.display = 'none';
            if (containerDiv) containerDiv.style.display = 'block';

            // Clear existing table rows
            tableBody.innerHTML = '';

            // Clear selected files
            StateManager.setState('files.selected', new Set());

            // Reset "Select All" checkbox
            const selectAllCheckbox = DomHelpers.getElement('selectAllFiles');
            if (selectAllCheckbox) {
                selectAllCheckbox.checked = false;
            }

            this.updateFileSelectionButtons();

            if (data.files.length === 0) {
                if (emptyDiv) emptyDiv.style.display = 'block';
                const fileTable = containerDiv.querySelector('.file-table');
                if (fileTable) {
                    fileTable.style.display = 'none';
                }
            } else {
                if (emptyDiv) emptyDiv.style.display = 'none';
                const fileTable = containerDiv.querySelector('.file-table');
                if (fileTable) {
                    fileTable.style.display = 'table';
                }

                // Populate table with files
                data.files.forEach(file => {
                    const row = this.createFileRow(file);
                    tableBody.appendChild(row);
                });
            }

            // Update totals
            DomHelpers.setText('totalFileCount', data.total_files);
            DomHelpers.setText('totalFileSize', `${data.total_size_mb} MB`);

            // Store in state
            StateManager.setState('files.managed', data.files);

        } catch (error) {
            if (loadingDiv) loadingDiv.style.display = 'none';
            MessageLogger.showMessage(t('files:load_failed', { error: error.message }), 'error');
        }
    },

    /**
     * Create file row element
     * @param {Object} file - File data object
     * @returns {HTMLElement} Table row element
     */
    createFileRow(file) {
        const row = document.createElement('tr');

        const modifiedDate = new Date(file.modified_date);
        const formattedDate = modifiedDate.toLocaleString();

        const isAudioFile = file.file_type === 'opus' || file.file_type === 'mp3';
        const fileIconClass = file.file_type === 'epub' ? 'book' :
                        file.file_type === 'srt' ? 'movie' :
                        file.file_type === 'txt' ? 'description' :
                        file.file_type === 'pdf' ? 'picture_as_pdf' :
                        isAudioFile ? 'headphones' : 'attach_file';

        const supportsTTS = ['epub', 'txt', 'srt'].includes(file.file_type);
        const safeFilename = DomHelpers.escapeHtml(file.filename);
        const tooltipInfo = `${file.file_type.toUpperCase()} • ${file.size_mb} MB • ${formattedDate}`;

        row.innerHTML = `
            <td style="width: 36px; padding: 0.5rem;">
                <input type="checkbox" class="file-checkbox" data-filename="${safeFilename}">
            </td>
            <td style="max-width: 0;">
                <span class="clickable-filename" data-filename="${safeFilename}" data-action="open" title="${tooltipInfo}">
                    <span class="material-symbols-outlined file-icon-cell">${fileIconClass}</span>
                    <span class="filename-text">${safeFilename}</span>
                </span>
            </td>
            <td class="file-row-actions">
                <div class="file-action-group file-action-group--compact"></div>
            </td>
        `;

        const checkbox = row.querySelector('.file-checkbox');
        if (checkbox) {
            checkbox.addEventListener('change', () => this.toggleFileSelection(file.filename));
        }

        const openLink = row.querySelector('.clickable-filename');
        if (openLink) {
            openLink.addEventListener('click', () => FileActions.open(file.filename));
        }

        const actionsHost = row.querySelector('.file-action-group');

        if (supportsTTS) {
            const audiobookBtn = document.createElement('button');
            audiobookBtn.type = 'button';
            audiobookBtn.className = 'file-action-btn audiobook';
            audiobookBtn.title = t('translation:audiobook_btn_title');
            audiobookBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 0.875rem;">headphones</span>';
            audiobookBtn.addEventListener('click', () => window.createAudiobook(file.filename, file.file_path));
            actionsHost.appendChild(audiobookBtn);
        }

        const refreshAfterDelete = () => this.refreshFileList();
        ['open', 'reveal', 'download', 'delete'].forEach(action => {
            actionsHost.appendChild(FileActions.createActionButton({
                action,
                filename: file.filename,
                variant: 'compact',
                onAfter: action === 'delete' ? refreshAfterDelete : undefined
            }));
        });

        return row;
    },

    /**
     * Toggle file selection
     * @param {string} filename - Filename to toggle
     */
    toggleFileSelection(filename) {
        const selectedFiles = StateManager.getState('files.selected');

        if (selectedFiles.has(filename)) {
            selectedFiles.delete(filename);
        } else {
            selectedFiles.add(filename);
        }

        StateManager.setState('files.selected', selectedFiles);
        this.updateFileSelectionButtons();
    },

    /**
     * Select all files
     */
    selectAllFiles() {
        const checkboxes = DomHelpers.getElements('.file-checkbox');
        const selectedFiles = new Set();

        checkboxes.forEach(checkbox => {
            checkbox.checked = true;
            const filename = checkbox.getAttribute('data-filename');
            selectedFiles.add(filename);
        });

        StateManager.setState('files.selected', selectedFiles);
        this.updateFileSelectionButtons();
    },

    /**
     * Deselect all files
     */
    deselectAllFiles() {
        const checkboxes = DomHelpers.getElements('.file-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.checked = false;
        });

        StateManager.setState('files.selected', new Set());
        this.updateFileSelectionButtons();
    },

    /**
     * Toggle select all
     */
    toggleSelectAll() {
        const checkboxes = DomHelpers.getElements('.file-checkbox');
        const selectAllFiles = DomHelpers.getElement('selectAllFiles');

        // Use the Select All checkbox state
        const isChecked = selectAllFiles.checked;

        if (isChecked) {
            this.selectAllFiles();
        } else {
            this.deselectAllFiles();
        }
    },

    /**
     * Update file selection button states
     */
    updateFileSelectionButtons() {
        const selectedFiles = StateManager.getState('files.selected');
        const hasSelection = selectedFiles.size > 0;

        // Update button states
        DomHelpers.setDisabled('batchDownloadBtn', !hasSelection);
        DomHelpers.setDisabled('batchDeleteBtn', !hasSelection);

        // Update "Select All" checkbox state based on actual selection
        const checkboxes = DomHelpers.getElements('.file-checkbox');
        const selectAllCheckbox = DomHelpers.getElement('selectAllFiles');
        if (selectAllCheckbox && checkboxes.length > 0) {
            const allChecked = Array.from(checkboxes).every(cb => cb.checked);
            selectAllCheckbox.checked = allChecked;
        }

        // Update button text with count
        const downloadBtn = DomHelpers.getElement('batchDownloadBtn');
        const deleteBtn = DomHelpers.getElement('batchDeleteBtn');
        if (hasSelection) {
            if (downloadBtn) downloadBtn.innerHTML = `<span class="material-symbols-outlined">download</span> ${t('files:download_selected_with_count', { count: selectedFiles.size })}`;
            if (deleteBtn) deleteBtn.innerHTML = `<span class="material-symbols-outlined">delete</span> ${t('files:delete_selected_with_count', { count: selectedFiles.size })}`;
        } else {
            if (downloadBtn) downloadBtn.innerHTML = `<span class="material-symbols-outlined">download</span> ${t('files:download_selected')}`;
            if (deleteBtn) deleteBtn.innerHTML = `<span class="material-symbols-outlined">delete</span> ${t('files:delete_selected')}`;
        }
    },

    async deleteSingleFile(filename) {
        await FileActions.delete(filename, { onDeleted: () => this.refreshFileList() });
    },

    /**
     * Download selected files as ZIP
     */
    async downloadSelectedFiles() {
        const selectedFiles = StateManager.getState('files.selected');

        if (selectedFiles.size === 0) {
            MessageLogger.showMessage(t('files:no_selection_download'), 'error');
            return;
        }

        try {
            const blob = await ApiClient.batchDownloadFiles(Array.from(selectedFiles));
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `translated_files_${new Date().getTime()}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            MessageLogger.showMessage(t('files:downloaded_as_zip', { count: selectedFiles.size }), 'success');
        } catch (error) {
            if (error.status) {
                // The server answered with an error status: show its label only.
                MessageLogger.showMessage(error.body?.error || t('files:download_failed_default'), 'error');
            } else {
                MessageLogger.showMessage(t('files:download_error', { error: error.message }), 'error');
            }
        }
    },

    /**
     * Delete selected files
     */
    async deleteSelectedFiles() {
        const selectedFiles = StateManager.getState('files.selected');

        if (selectedFiles.size === 0) {
            MessageLogger.showMessage(t('files:no_selection_delete'), 'error');
            return;
        }

        if (!confirm(t('files:confirm_delete_selected', { count: selectedFiles.size }))) {
            return;
        }

        try {
            const data = await ApiClient.batchDeleteFiles(Array.from(selectedFiles));
            const message = data.failed.length > 0
                ? t('files:deleted_summary_with_failed', { count: data.total_deleted, failed: data.failed.length })
                : t('files:deleted_summary', { count: data.total_deleted });
            MessageLogger.showMessage(message, data.failed.length > 0 ? 'info' : 'success');
            this.refreshFileList();
        } catch (error) {
            if (error.status) {
                // The server answered with an error status: show its label only.
                MessageLogger.showMessage(error.body?.error || t('files:delete_failed_default'), 'error');
            } else {
                MessageLogger.showMessage(t('files:delete_error', { error: error.message }), 'error');
            }
        }
    },

};

// Selection toggle stays here (state lives in FileManager); the per-file
// actions (open/reveal/download) are exposed globally by FileActions itself.
window.toggleFileSelection = (filename) => FileManager.toggleFileSelection(filename);
window.deleteSingleFile = (filename) => FileManager.deleteSingleFile(filename);
