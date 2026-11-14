/**
 * File Actions - Shared file operations and action button factory
 *
 * Centralizes the open / reveal / download / delete / goToFilesTab actions
 * used by both the Files tab table and the post-translation completion card.
 * Also provides a button factory so both surfaces render consistent buttons.
 */

import { describeApiError } from '../core/api-errors.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { toast } from '../ui/toast.js';
import { t } from '../i18n/i18n.js';

const ACTION_DEFS = {
    open: {
        icon: 'open_in_new',
        labelKey: 'files:open_action',
        compactClass: 'open',
        labeledClass: 'btn-secondary'
    },
    reveal: {
        icon: 'folder_open',
        labelKey: 'files:reveal_action',
        compactClass: 'reveal',
        labeledClass: 'btn-secondary'
    },
    download: {
        icon: 'download',
        labelKey: 'files:download_action',
        compactClass: 'download',
        labeledClass: 'btn-primary'
    },
    delete: {
        icon: 'delete',
        labelKey: 'files:delete_action',
        compactClass: 'delete',
        labeledClass: 'btn-danger'
    },
    'files-tab': {
        icon: 'folder',
        labelKey: 'files:go_to_files_tab',
        compactClass: 'files-tab',
        labeledClass: 'btn-secondary'
    }
};

export const FileActions = {
    async open(filename) {
        if (!filename) return;
        try {
            await ApiClient.openLocalFile(filename);
            MessageLogger.addLog(t('files:open_log', { name: filename }));
        } catch (error) {
            // A toast is visible from every tab; #messages lives in the
            // Translate tab and is hidden while the Files tab is shown.
            toast.error(t('files:open_error', { error: describeApiError(error, { context: 'open' }) }));
        }
    },

    async reveal(filename) {
        if (!filename) return;
        try {
            await ApiClient.revealLocalFile(filename);
            MessageLogger.addLog(t('files:reveal_log', { name: filename }));
        } catch (error) {
            toast.error(t('files:reveal_error', { error: describeApiError(error) }));
        }
    },

    async openOutputFolder() {
        try {
            const data = await ApiClient.openOutputFolder();
            MessageLogger.addLog(
                data.folder_path
                    ? t('files:folder_opened_log_with_path', { path: data.folder_path })
                    : t('files:folder_opened_log')
            );
        } catch (error) {
            toast.error(t('files:folder_open_error', { error: describeApiError(error) }));
        }
    },

    download(filename) {
        if (!filename) return;
        window.location.href = ApiClient.getFileDownloadUrl(filename);
    },

    async delete(filename, { confirm: needConfirm = true, onDeleted } = {}) {
        if (!filename) return false;
        if (needConfirm && !window.confirm(t('files:confirm_delete_named', { name: filename }))) {
            return false;
        }
        try {
            await ApiClient.deleteFile(filename);
            // The server message is English-only; always show the localized one.
            toast.success(t('files:delete_default_msg', { name: filename }));
            if (typeof onDeleted === 'function') onDeleted(filename);
            return true;
        } catch (error) {
            toast.error(t('files:delete_failed', { error: describeApiError(error) }));
            return false;
        }
    },

    goToFilesTab() {
        if (typeof window.switchTopTab === 'function') {
            window.switchTopTab('files');
        }
    },

    invoke(action, filename, options) {
        switch (action) {
            case 'open':       return this.open(filename);
            case 'reveal':     return this.reveal(filename);
            case 'download':   return this.download(filename);
            case 'delete':     return this.delete(filename, options);
            case 'files-tab':  return this.goToFilesTab();
            default:           return undefined;
        }
    },

    /**
     * Build a single action button.
     * @param {Object} opts
     * @param {string} opts.action - One of: open, reveal, download, delete, files-tab
     * @param {string} [opts.filename] - File to operate on (not needed for files-tab)
     * @param {'compact'|'labeled'} [opts.variant='compact']
     * @param {Function} [opts.onAfter] - Callback after action resolves
     * @returns {HTMLElement}
     */
    createActionButton({ action, filename, variant = 'compact', onAfter } = {}) {
        const def = ACTION_DEFS[action];
        if (!def) throw new Error(`Unknown file action: ${action}`);

        const label = t(def.labelKey);

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.dataset.action = action;
        if (filename) btn.dataset.filename = filename;
        btn.title = label;

        if (variant === 'compact') {
            btn.className = `file-action-btn ${def.compactClass}`;
            btn.innerHTML = `<span class="material-symbols-outlined" style="font-size: 0.875rem;">${def.icon}</span>`;
        } else {
            btn.className = `btn ${def.labeledClass}`;
            btn.innerHTML = `<span class="material-symbols-outlined">${def.icon}</span> ${label}`;
        }

        btn.addEventListener('click', async () => {
            const result = this.invoke(action, filename, { onDeleted: onAfter });
            if (result && typeof result.then === 'function') await result;
            if (typeof onAfter === 'function' && action !== 'delete') onAfter(action, filename);
        });

        return btn;
    },

    /**
     * Build a container of action buttons.
     * @param {Object} opts
     * @param {string[]} opts.actions - Ordered list of action names
     * @param {string} [opts.filename]
     * @param {'compact'|'labeled'} [opts.variant='compact']
     * @param {Function} [opts.onAfter]
     * @returns {HTMLElement}
     */
    createActionGroup({ actions, filename, variant = 'compact', onAfter } = {}) {
        const wrap = document.createElement('div');
        wrap.className = variant === 'compact' ? 'file-action-group file-action-group--compact' : 'file-action-group file-action-group--labeled';
        actions.forEach(action => {
            wrap.appendChild(this.createActionButton({ action, filename, variant, onAfter }));
        });
        return wrap;
    }
};

// Expose action handlers globally (used by completion-card and any inline onclicks)
if (typeof window !== 'undefined') {
    window.openLocalFile = (filename) => FileActions.open(filename);
    window.revealLocalFile = (filename) => FileActions.reveal(filename);
    window.downloadSingleFile = (filename) => FileActions.download(filename);
}
