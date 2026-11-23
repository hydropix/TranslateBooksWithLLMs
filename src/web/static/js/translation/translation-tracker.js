/**
 * Translation Tracker - Track active translations and handle WebSocket updates
 *
 * Manages active translation state, WebSocket event handling,
 * translation completion, error handling, and batch queue progression.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { StatusManager } from '../utils/status-manager.js';
import { FileUpload, getFileStatusLabel } from '../files/file-upload.js';
import { ProgressManager } from './progress-manager.js';
import { renderTranslationTitle } from './progress-title.js';
import { LifecycleManager } from '../utils/lifecycle-manager.js';
import { t } from '../i18n/i18n.js';
import { renderCompletionCard as renderCompletionCardView } from './completion-card.js';

// Storage configuration with versioning
const STORAGE_VERSION = 1;
const STORAGE_KEY_PREFIX = 'tbl_translation_state';
const TRANSLATION_STATE_STORAGE_KEY = `${STORAGE_KEY_PREFIX}_v${STORAGE_VERSION}`;

// Statuses that end a job server-side. Only these can reset the UI to idle.
const TERMINAL_STATUSES = new Set(['completed', 'partial', 'error', 'interrupted', 'rate_limited']);
// Statuses of a job that is still being worked on server-side.
const ACTIVE_STATUSES = new Set(['running', 'queued']);
const MAX_OWNED_JOB_IDS = 50;

// Debounce timer for discovering a job started on another device (issue #271).
let foreignDiscoveryTimer = null;
const FOREIGN_DISCOVERY_DELAY_MS = 1000;

/**
 * Last segment of a path, split on both `/` and `\` so a Windows path
 * matches as well as a POSIX one.
 * @param {string|null|undefined} path
 * @returns {string} Basename, or '' when there is no path
 */
function basenameOf(path) {
    if (typeof path !== 'string' || !path) return '';
    const parts = path.split(/[\\/]/);
    return parts[parts.length - 1] || '';
}

/**
 * Validate translation state structure
 * @param {any} data - Data to validate
 * @returns {boolean} True if valid
 */
function validateTranslationState(data) {
    if (!data || typeof data !== 'object') return false;

    // Check required fields
    if (!('version' in data)) return false;
    if (!('currentJob' in data)) return false;
    if (!('isBatchActive' in data)) return false;
    if (!('activeJobs' in data)) return false;
    if (!('hasActive' in data)) return false;

    // Validate types
    if (typeof data.isBatchActive !== 'boolean') return false;
    if (typeof data.hasActive !== 'boolean') return false;
    if (!Array.isArray(data.activeJobs)) return false;

    // Validate currentJob if present
    if (data.currentJob !== null) {
        if (typeof data.currentJob !== 'object') return false;
        if (!('translationId' in data.currentJob)) return false;
        if (!('fileRef' in data.currentJob)) return false;
    }

    return true;
}

export const TranslationTracker = {
    // Debounce timer for saving state
    _saveStateTimer: null,
    _saveStateDebounceMs: 100,

    /**
     * Initialize translation tracker
     */
    async initialize() {
        // Clean up old storage versions
        this.cleanupOldStorageVersions();

        // Setup event listeners FIRST (they need to be ready before any state changes)
        this.setupEventListeners();

        // CRITICAL: Check server session BEFORE restoring state
        // This prevents restoring state from a previous server session
        try {
            const serverWasRestarted = await LifecycleManager.getServerSessionCheck();

            if (serverWasRestarted) {
                this.initializeDefaultTranslationState();
            } else {
                this.restoreTranslationStateSync();

                await Promise.all([
                    this.updateActiveTranslationsState(),
                    this.reconcileStateWithServer()
                ]);
            }
        } catch (error) {
            console.error('Failed to initialize translation state:', error);
            MessageLogger.addLog(t('translation:session_init_failed'));

            // Fallback: restore from localStorage anyway
            this.restoreTranslationStateSync();
        }

        // Mark initialization as complete
        this._initializationComplete = true;
    },

    /**
     * Check if initialization is complete
     * @returns {boolean} True if initialization is complete
     */
    isInitialized() {
        return this._initializationComplete === true;
    },

    /**
     * Clean up old localStorage versions
     */
    cleanupOldStorageVersions() {
        try {
            // Remove old non-versioned key
            const oldKey = 'tbl_translation_state';
            if (localStorage.getItem(oldKey)) {
                localStorage.removeItem(oldKey);
            }

            // Remove any other versions (future-proofing)
            for (let i = 0; i < STORAGE_VERSION; i++) {
                const oldVersionKey = `${STORAGE_KEY_PREFIX}_v${i}`;
                if (localStorage.getItem(oldVersionKey)) {
                    localStorage.removeItem(oldVersionKey);
                }
            }
        } catch (error) {
            console.warn('Failed to cleanup old storage versions:', error);
        }
    },

    /**
     * Restore translation state from localStorage synchronously
     * This ensures the UI shows the translation state immediately on page load
     */
    restoreTranslationStateSync() {
        try {
            const stored = localStorage.getItem(TRANSLATION_STATE_STORAGE_KEY);

            if (!stored) {
                this.initializeDefaultTranslationState();
                return;
            }

            const savedState = JSON.parse(stored);

            if (!validateTranslationState(savedState)) {
                MessageLogger.addLog(t('translation:session_corrupted_log'));
                this.initializeDefaultTranslationState();
                this.clearTranslationState();
                return;
            }

            if (savedState.version !== STORAGE_VERSION) {
                this.initializeDefaultTranslationState();
                this.clearTranslationState();
                return;
            }

            if (savedState.isBatchActive && savedState.currentJob) {
                StateManager.setState('translation.currentJob', savedState.currentJob);
                StateManager.setState('translation.isBatchActive', savedState.isBatchActive);
                StateManager.setState('translation.activeJobs', savedState.activeJobs || []);
                StateManager.setState('translation.hasActive', savedState.hasActive || false);
                StateManager.setState('translation.ownedJobIds', savedState.ownedJobIds || []);

                DomHelpers.show('progressSection');
                DomHelpers.show('interruptBtn');

                const translateBtn = DomHelpers.getElement('translateBtn');
                if (translateBtn) {
                    translateBtn.disabled = true;
                    translateBtn.innerHTML = t('translation:batch_in_progress');
                }

                MessageLogger.addLog(t('translation:session_restored_log'));
            } else {
                this.initializeDefaultTranslationState();
            }
        } catch (error) {
            console.error('Failed to restore translation state from localStorage:', error);
            MessageLogger.addLog(t('translation:session_could_not_restore'));
            this.initializeDefaultTranslationState();
        }
    },

    /**
     * Reconcile local state with server state
     * Checks if localStorage state matches server reality
     */
    async reconcileStateWithServer() {
        try {
            const currentJob = StateManager.getState('translation.currentJob');

            // If we have a local job, verify it exists on server
            if (currentJob && currentJob.translationId) {
                try {
                    const serverState = await ApiClient.getTranslationStatus(currentJob.translationId);

                    if (TERMINAL_STATUSES.has(serverState.status)) {

                        MessageLogger.addLog(t('translation:session_sync_log', { status: serverState.status }));
                        this.resetUIToIdle();
                    } else if (ACTIVE_STATUSES.has(serverState.status)) {
                        // Calculate progress from stats if available
                        if (serverState.stats) {
                            this.updateStats(currentJob.fileRef.fileType, serverState.stats);
                        }
                    }
                } catch (error) {
                    if (error.status === 404) {
                        MessageLogger.addLog(t('translation:session_job_missing_log'));
                        this.resetUIToIdle();
                    }
                }
            }

            await this.restoreActiveTranslation();

        } catch (error) {
            console.warn('Failed to reconcile state with server:', error);
        }
    },

    /**
     * Initialize default translation state (when no saved state exists)
     */
    initializeDefaultTranslationState() {
        StateManager.setState('translation.currentJob', null);
        StateManager.setState('translation.isBatchActive', false);
        StateManager.setState('translation.activeJobs', []);
        StateManager.setState('translation.hasActive', false);
        StateManager.setState('translation.ownedJobIds', []);
    },

    /**
     * Save translation state to localStorage (debounced)
     */
    saveTranslationState() {
        // Clear existing timer
        if (this._saveStateTimer) {
            clearTimeout(this._saveStateTimer);
        }

        // Debounce to avoid multiple rapid saves
        this._saveStateTimer = setTimeout(() => {
            this._performSaveTranslationState();
        }, this._saveStateDebounceMs);
    },

    /**
     * Perform the actual save to localStorage
     * @private
     */
    _performSaveTranslationState() {
        try {
            const state = {
                version: STORAGE_VERSION,
                currentJob: StateManager.getState('translation.currentJob'),
                isBatchActive: StateManager.getState('translation.isBatchActive'),
                activeJobs: StateManager.getState('translation.activeJobs'),
                hasActive: StateManager.getState('translation.hasActive'),
                ownedJobIds: StateManager.getState('translation.ownedJobIds') || [],
                timestamp: Date.now()
            };

            localStorage.setItem(TRANSLATION_STATE_STORAGE_KEY, JSON.stringify(state));
        } catch (error) {
            console.error('Failed to save translation state to localStorage:', error);

            // Check if it's a quota exceeded error
            if (error.name === 'QuotaExceededError') {
                MessageLogger.addLog(t('translation:session_state_save_quota'));
            } else {
                MessageLogger.addLog(t('translation:session_state_save_failed'));
            }
        }
    },

    /**
     * Clear translation state from localStorage
     */
    clearTranslationState() {
        try {
            // Clear any pending save
            if (this._saveStateTimer) {
                clearTimeout(this._saveStateTimer);
                this._saveStateTimer = null;
            }

            localStorage.removeItem(TRANSLATION_STATE_STORAGE_KEY);
        } catch (error) {
            console.error('Failed to clear translation state from localStorage:', error);
        }
    },

    /**
     * Record a translation id as owned by this tab. Idempotent. FIFO-capped at
     * MAX_OWNED_JOB_IDS so a long-lived tab cannot grow the persisted state
     * without bound.
     * @param {string} translationId
     */
    registerOwnedJob(translationId) {
        if (!translationId) return;

        const owned = StateManager.getState('translation.ownedJobIds') || [];
        if (owned.includes(translationId)) return;

        const next = [...owned, translationId];
        while (next.length > MAX_OWNED_JOB_IDS) {
            next.shift();
        }

        StateManager.setState('translation.ownedJobIds', next);
    },

    /**
     * @param {string} translationId
     * @returns {boolean} true only if registerOwnedJob was called for this id
     *   (in this tab, or in a previous session restored from localStorage).
     */
    ownsJob(translationId) {
        if (!translationId) return false;
        const owned = StateManager.getState('translation.ownedJobIds') || [];
        return owned.includes(translationId);
    },

    /**
     * Build the stats payload ProgressManager expects from a record of
     * `/api/translations` (a job summary, not a full stats object).
     * @param {Object} job - Active-job record from ApiClient.getActiveTranslations()
     * @returns {Object|null} Stats object, or null when the job has not
     *   reported a chunk total yet (nothing meaningful to display).
     */
    _buildStatsFromActiveJob(job) {
        if (!job || !(job.total_chunks > 0)) return null;
        return {
            total_chunks: job.total_chunks,
            completed_chunks: job.completed_chunks || 0,
            failed_chunks: job.failed_chunks || 0,
            elapsed_time: job.elapsed_time,
            progress_percent: job.progress_percent,
            current_phase: job.current_phase,
            enable_refinement: job.enable_refinement || false
        };
    },

    /**
     * Push an active-job record into the progress panel: stats (when the job
     * has a chunk total) and the last translated segment preview.
     * @param {string} fileType - File type of the job ('txt', 'epub', ...)
     * @param {Object} job - Active-job record from ApiClient.getActiveTranslations()
     */
    _applyActiveJobSnapshot(fileType, job) {
        const stats = this._buildStatsFromActiveJob(job);
        if (stats) {
            this.updateStats(fileType, stats);
        }
        if (job.last_translation) {
            MessageLogger.updateTranslationPreview(job.last_translation);
        }
    },

    /**
     * Restore active translation state if there's one running on the server.
     *
     * Only ONE job is restored, on purpose. BatchController runs a batch
     * strictly one job at a time through the single `translation.currentJob`
     * slot, so at most one active job belongs to this tab's batch; any other
     * active job was started elsewhere (another tab or device). The rest of
     * the batch needs no restoring here: the files still waiting are persisted
     * by FileUpload as 'Queued' entries of `files.toProcess`, and when the
     * restored job reports a terminal status, finishCurrentFileTranslation
     * dispatches `processNextFile`, which makes BatchController pick the next
     * 'Queued' file.
     *
     * Closing or reloading the page does not interrupt the running job: it
     * keeps going on the server and is restored here on the next load.
     *
     * Limitation: a job that finished while the page was closed is not
     * followed by the next file (reconcileStateWithServer resets to idle), so
     * a new job is never started on page load without a user action; the
     * remaining files stay queued until the user restarts the batch.
     */
    async restoreActiveTranslation() {
        try {
            const response = await ApiClient.getActiveTranslations();
            const activeJobs = (response.translations || []).filter(
                job => ACTIVE_STATUSES.has(job.status)
            );

            if (activeJobs.length === 0) return;

            const filesToProcess = StateManager.getState('files.toProcess') || [];
            const persistedJob = StateManager.getState('translation.currentJob');

            // Prefer the job this tab persisted, then any job it owns, and only
            // then the server's order (most recent first). Without this, a job
            // started on another device could be restored in place of ours.
            const rank = (job) => {
                if (persistedJob && job.translation_id === persistedJob.translationId) return 0;
                if (this.ownsJob(job.translation_id)) return 1;
                return 2;
            };
            const candidates = [...activeJobs].sort((a, b) => rank(a) - rank(b));

            for (const job of candidates) {
                // `input_filename` is what /api/translations returns (a display
                // basename); `input_file` is kept for older payloads. Either
                // may carry a Windows path, hence basenameOf.
                const jobFileName = basenameOf(job.input_filename || job.input_file);

                let matchingFile = filesToProcess.find(f => f.translationId === job.translation_id);
                let inQueue = !!matchingFile;

                // The submitted file is removed from files.toProcess, so the
                // persisted fileRef (thumbnail, languages, operation) is usually
                // the best description of the job this tab was running.
                if (!matchingFile && persistedJob?.fileRef
                    && persistedJob.translationId === job.translation_id) {
                    matchingFile = persistedJob.fileRef;
                }

                if (!matchingFile && jobFileName) {
                    matchingFile = filesToProcess.find(f =>
                        (job.input_file && f.filePath === job.input_file) ||
                        f.name === jobFileName
                    );
                    inQueue = !!matchingFile;
                }

                // If no matching file found, create a virtual file reference from server data
                // This allows restoration after browser refresh even if filesToProcess is empty
                if (!matchingFile && job.input_filename) {
                    const fileType = job.file_type || 'txt';
                    matchingFile = {
                        name: jobFileName || job.input_filename,
                        translationId: job.translation_id,
                        status: 'Processing',
                        // `fileType` is what every renderer reads; `type` is
                        // kept for backward compatibility.
                        fileType,
                        type: fileType,
                        isVirtual: true
                    };
                }

                if (!matchingFile) continue;

                StateManager.setState('translation.currentJob', {
                    fileRef: matchingFile,
                    translationId: job.translation_id
                });
                this.registerOwnedJob(job.translation_id);
                StateManager.setState('translation.isBatchActive', true);

                DomHelpers.show('progressSection');
                this.updateTranslationTitle(matchingFile);

                this._applyActiveJobSnapshot(matchingFile.fileType, job);

                const translateBtn = DomHelpers.getElement('translateBtn');
                if (translateBtn) {
                    translateBtn.disabled = true;
                    translateBtn.innerHTML = t('translation:batch_in_progress');
                }
                DomHelpers.show('interruptBtn');

                if (inQueue && !matchingFile.isVirtual) {
                    this.updateFileStatusInList(matchingFile.name, 'Processing', job.translation_id);
                }

                // One job only: see the method comment.
                return;
            }
        } catch (error) {
            console.warn('Failed to restore active translation:', error);
        }
    },

    /**
     * Re-synchronize the current job with the server after a WebSocket
     * reconnect, since any update emitted while the socket was down is lost.
     *
     * - Still running/queued: refresh the stats and preview from the active
     *   list, exactly as restoreActiveTranslation does.
     * - Gone from the active list: read its final status and, if terminal,
     *   replay it through handleTranslationUpdate (completion card, queue
     *   advance, ...).
     * - Unknown to the server (404): reset the UI to idle.
     *
     * Never throws.
     */
    async resyncCurrentJob() {
        try {
            const currentJob = StateManager.getState('translation.currentJob');
            const translationId = currentJob?.translationId;
            if (!translationId) return;

            // The awaits below can race a live update (or the user) changing
            // the current job; only act if it is still the same one.
            const isStillCurrent = () =>
                StateManager.getState('translation.currentJob')?.translationId === translationId;

            let activeRecord = null;
            try {
                const response = await ApiClient.getActiveTranslations();
                activeRecord = (response.translations || [])
                    .find(job => job.translation_id === translationId) || null;
            } catch (error) {
                console.warn('Resync: failed to list active translations:', error);
            }

            if (!isStillCurrent()) return;

            if (activeRecord && ACTIVE_STATUSES.has(activeRecord.status)) {
                this._applyActiveJobSnapshot(currentJob.fileRef?.fileType, activeRecord);
                return;
            }

            let data;
            try {
                data = await ApiClient.getTranslationStatus(translationId);
            } catch (error) {
                if (error?.status === 404) {
                    if (isStillCurrent()) this.resetUIToIdle();
                } else {
                    console.warn('Resync: failed to read translation status:', error);
                }
                return;
            }

            if (!data || !isStillCurrent()) return;

            const status = data.status;
            if (TERMINAL_STATUSES.has(status)) {
                this.handleTranslationUpdate({
                    translation_id: translationId,
                    status,
                    result: data.result_preview || `[${status}]`,
                    error: data.error,
                    stats: data.stats
                });
            } else if (ACTIVE_STATUSES.has(status) && data.stats) {
                // Listed as inactive a moment ago but running now (or the
                // list call failed): still refresh the counters we have.
                this.updateStats(currentJob.fileRef?.fileType, data.stats);
            }
        } catch (error) {
            console.warn('Failed to resync current job:', error);
        }
    },

    /**
     * Schedule a debounced discovery of a job started on another device.
     * Only fires when this tab is genuinely idle, so it can never hijack
     * a local batch (issue #271).
     * @param {string} translationId
     */
    _scheduleForeignJobDiscovery(translationId) {
        if (!this.isInitialized()) return;
        if (StateManager.getState('translation.currentJob')) return;
        if (StateManager.getState('translation.isBatchActive')) return;
        if (this.ownsJob(translationId)) return;

        if (foreignDiscoveryTimer) return;

        foreignDiscoveryTimer = setTimeout(async () => {
            foreignDiscoveryTimer = null;
            if (StateManager.getState('translation.currentJob')) return;
            await this.restoreActiveTranslation();
        }, FOREIGN_DISCOVERY_DELAY_MS);
    },

    setupEventListeners() {
        StateManager.subscribe('translation.currentJob', () => {
            this.saveTranslationState();
        });

        StateManager.subscribe('translation.isBatchActive', () => {
            this.saveTranslationState();
        });

        StateManager.subscribe('translation.hasActive', () => {
            this.updateResumeButtonsState();
            this.saveTranslationState();
        });

        StateManager.subscribe('translation.activeJobs', () => {
            this.saveTranslationState();
        });

        StateManager.subscribe('translation.ownedJobIds', () => {
            this.saveTranslationState();
        });
    },

    /**
     * Handle translation update from WebSocket
     * @param {Object} data - Translation update data
     */
    handleTranslationUpdate(data) {
        const currentJob = StateManager.getState('translation.currentJob');

        if (!currentJob || data.translation_id !== currentJob.translationId) {
            // A terminal event with no current job is only ours to act on if this tab
            // started the job. Without this check, a straggler from a previous file
            // (currentJob is nulled before the queue advances) or an event for another
            // tab's job wipes the batch UI mid-run. Issue #225.
            if (!currentJob && data.translation_id
                && TERMINAL_STATUSES.has(data.status)
                && this.ownsJob(data.translation_id)) {
                this.resetUIToIdle();
            }

            // A non-terminal event for a job we don't know about and don't own is a
            // candidate for a job started on another device. Debounce a discovery
            // attempt rather than acting immediately (issue #271).
            if (!currentJob && data.translation_id && !TERMINAL_STATUSES.has(data.status)) {
                this._scheduleForeignJobDiscovery(data.translation_id);
            }
            return;
        }

        const currentFile = currentJob.fileRef;

        if (data.log) {
            MessageLogger.addLog(`[${currentFile.name}] ${data.log}`);
        }

        // Progress is now calculated from stats in ProgressManager.update()
        // No need to call updateProgress() separately
        if (data.stats) {
            this.updateStats(currentFile.fileType, data.stats);
        }

        if (data.log_entry
            && (data.log_entry.type === 'llm_response' || data.log_entry.type === 'refinement_response')
            && data.log_entry.data && data.log_entry.data.response) {
            MessageLogger.updateTranslationPreview(data.log_entry.data.response);
        }

        if (data.status === 'completed') {
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_completed_msg', { name: currentFile.name }),
                'success',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'partial') {
            // Finished, but some units stayed failed after the automatic
            // retries. The output file exists (best effort) and the job is
            // resumable; the completion card explains and gives advice.
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_partial_msg', { name: currentFile.name }),
                'info',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'interrupted') {
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_interrupted_msg', { name: currentFile.name }),
                'info',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'rate_limited') {
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_rate_limited_msg', { name: currentFile.name }),
                'info',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'error') {
            MessageLogger.resetProgressTracking();
            // `nothing_translated`: the run finished but no unit got a
            // translation. The job keeps its checkpoint and shows up in the
            // resumable list, which the localized message points to.
            const errorMessage = data.error_code === 'nothing_translated'
                ? t('translation:translation_nothing_translated_msg', { name: currentFile.name })
                : t('translation:translation_error_msg', { name: currentFile.name, error: data.error || t('translation:translation_unknown_error') });
            this.finishCurrentFileTranslation(
                errorMessage,
                'error',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'running') {
            MessageLogger.resetProgressTracking();
            DomHelpers.show('progressSection');
            DomHelpers.show('statsGrid');
            this.updateTranslationTitle(currentFile);
            this.resetOpenRouterCostDisplay();

            MessageLogger.showMessage(t('translation:translation_in_progress', { name: currentFile.name }), 'info');
            this.updateFileStatusInList(currentFile.name, 'Processing');
        }
    },

    /**
     * Update translation title with file icon/thumbnail and name
     * @param {Object} file - File object
     */
    updateTranslationTitle(file) {
        renderTranslationTitle(file);
    },

    /**
     * Update statistics display
     * @param {string} fileType - File type (txt, epub, srt)
     * @param {Object} stats - Statistics object
     */
    updateStats(fileType, stats) {
        ProgressManager.update({ stats: stats }, fileType);
        this.updateOpenRouterCost(stats);
    },

    /**
     * Update OpenRouter cost display
     * @param {Object} stats - Statistics object containing cost data
     */
    updateOpenRouterCost(stats) {
        const costGrid = DomHelpers.getElement('openrouterCostGrid');
        if (!costGrid) return;

        const cost = stats.openrouter_cost || 0;
        const promptTokens = stats.openrouter_prompt_tokens || 0;
        const completionTokens = stats.openrouter_completion_tokens || 0;
        const totalTokens = promptTokens + completionTokens;

        // Show cost grid if there's any cost or token data
        if (cost > 0 || totalTokens > 0) {
            DomHelpers.show('openrouterCostGrid');
            // '~' marks an estimate: usage.cost was missing for some requests
            DomHelpers.setText('openrouterCost', (stats.openrouter_cost_estimated ? '~$' : '$') + cost.toFixed(4));
            DomHelpers.setText('openrouterTokens', totalTokens.toLocaleString());
        }
    },

    /**
     * Reset OpenRouter cost display for a new translation
     */
    resetOpenRouterCostDisplay() {
        DomHelpers.hide('openrouterCostGrid');
        DomHelpers.setText('openrouterCost', '$0.0000');
        DomHelpers.setText('openrouterTokens', '0');
    },

    /**
     * Update file status in UI list
     * @param {string} fileName - File name
     * @param {string} newStatus - New status text
     * @param {string} [translationId] - Translation ID
     */
    updateFileStatusInList(fileName, newStatus, translationId = null) {
        const fileListItem = DomHelpers.getOne(`#fileListContainer li[data-filename="${CSS.escape(fileName)}"] .file-status`);
        if (fileListItem) {
            DomHelpers.setText(fileListItem, `(${getFileStatusLabel(newStatus)})`);
        }

        // Update in state
        const filesToProcess = StateManager.getState('files.toProcess') || [];
        const fileObj = filesToProcess.find(f => f.name === fileName);
        if (fileObj) {
            fileObj.status = newStatus;
            if (translationId) {
                fileObj.translationId = translationId;
            }
            StateManager.setState('files.toProcess', filesToProcess);
            // Persist to localStorage
            FileUpload.notifyFileListChanged();
        }
    },

    /**
     * Finish current file translation and update UI
     * @param {string} statusMessage - Status message to display
     * @param {string} messageType - Message type (success, error, info)
     * @param {Object} resultData - Translation result data
     */
    finishCurrentFileTranslation(statusMessage, messageType, resultData) {
        const currentJob = StateManager.getState('translation.currentJob');
        if (!currentJob) return;

        const currentFile = currentJob.fileRef;
        currentFile.status = resultData.status || 'unknown_error';
        currentFile.result = resultData.result;

        MessageLogger.showMessage(statusMessage, messageType);
        this.updateFileStatusInList(
            currentFile.name,
            resultData.status === 'completed' ? 'Completed' :
            resultData.status === 'partial' ? 'Partial' :
            resultData.status === 'interrupted' ? 'Interrupted' :
            resultData.status === 'rate_limited' ? 'Rate Limited' : 'Error'
        );

        if (resultData.status === 'completed' || resultData.status === 'partial') {
            // Partial jobs still produced a best-effort output file; the card
            // surfaces it together with the warning block and its advice.
            this.renderCompletionCard(currentFile, resultData);
        }

        // Retire the id before nulling currentJob: this file's terminal event has
        // been handled, so a duplicate copy of it arriving while the queue advances
        // is no longer owned and is ignored by handleTranslationUpdate. Issue #225.
        if (currentJob.translationId) {
            const owned = StateManager.getState('translation.ownedJobIds') || [];
            if (owned.includes(currentJob.translationId)) {
                StateManager.setState(
                    'translation.ownedJobIds',
                    owned.filter(id => id !== currentJob.translationId)
                );
            }
        }

        StateManager.setState('translation.currentJob', null);

        if (resultData.status === 'completed') {
            this.processNextFileInQueue();
        } else if (resultData.status === 'interrupted') {
            MessageLogger.addLog(t('translation:batch_stopped_user_log'));
            this.resetUIToIdle();
        } else if (resultData.status === 'rate_limited') {
            MessageLogger.addLog(t('translation:batch_paused_log'));
            this.resetUIToIdle();
        } else {
            this.processNextFileInQueue();
        }
    },

    /**
     * Render a persistent success card for a completed file. Thin delegate to
     * completion-card.js, which owns the card DOM, its warning block and its
     * locale / active-state refresh.
     * @param {Object} file - The file that just finished
     * @param {Object} resultData - Final payload from the server (output_filename, output_dir)
     */
    renderCompletionCard(file, resultData) {
        renderCompletionCardView(file, resultData);
    },

    /**
     * Process next file in queue (delegates to batch-controller when available)
     */
    processNextFileInQueue() {
        // Trigger event for batch controller to handle
        window.dispatchEvent(new CustomEvent('processNextFile'));
    },

    /**
     * Check and update active translations state
     */
    async updateActiveTranslationsState() {
        try {
            const response = await ApiClient.getActiveTranslations();
            const activeJobs = (response.translations || []).filter(
                t => t.status === 'running' || t.status === 'queued'
            );

            const wasActive = StateManager.getState('translation.hasActive');
            const hasActive = activeJobs.length > 0;

            StateManager.setState('translation.hasActive', hasActive);
            StateManager.setState('translation.activeJobs', activeJobs);

            // If state changed, update UI
            if (wasActive !== hasActive) {
                this.updateResumeButtonsState();
            }

            return { hasActive, activeJobs };
        } catch {
            return {
                hasActive: StateManager.getState('translation.hasActive'),
                activeJobs: StateManager.getState('translation.activeJobs')
            };
        }
    },

    /**
     * Update the state of all resume buttons based on active translations
     */
    updateResumeButtonsState() {
        const resumeButtons = DomHelpers.getElements('button[onclick^="resumeJob"]');
        const hasActive = StateManager.getState('translation.hasActive');

        resumeButtons.forEach(button => {
            if (hasActive) {
                button.disabled = true;
                button.style.opacity = '0.5';
                button.style.cursor = 'not-allowed';
                button.title = t('translation:cannot_resume_in_progress_title');
            } else {
                button.disabled = false;
                button.style.opacity = '1';
                button.style.cursor = 'pointer';
                button.title = t('translation:resume_btn_title');
            }
        });

        // Update warning banner
        this.updateResumableJobsWarningBanner();
    },

    /**
     * Update or create the warning banner in resumable jobs section
     */
    updateResumableJobsWarningBanner() {
        const listContainer = DomHelpers.getElement('resumableJobsList');
        if (!listContainer) return;

        const existingBanner = listContainer.querySelector('.active-translation-warning');
        const hasActive = StateManager.getState('translation.hasActive');
        const activeJobs = StateManager.getState('translation.activeJobs');

        if (hasActive) {
            const activeNames = activeJobs.map(job => job.output_filename || t('translation:job_card_unknown')).join(', ');
            const bannerHtml = `
                <div class="active-translation-warning" style="background: var(--warning-light-bg); border: 1px solid var(--warning-light-border); padding: 12px; margin-bottom: 15px; border-radius: 6px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 20px;">⚠️</span>
                        <div style="flex: 1;">
                            <strong style="color: var(--warning-light-text);">${t('translation:active_translation_warning_title')}</strong>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: var(--warning-light-text);">
                                ${t('translation:active_translation_warning_desc', { names: DomHelpers.escapeHtml(activeNames) })}
                            </p>
                        </div>
                    </div>
                </div>
            `;

            if (existingBanner) {
                existingBanner.outerHTML = bannerHtml;
            } else {
                // Insert at the beginning of the container
                listContainer.insertAdjacentHTML('afterbegin', bannerHtml);
            }
        } else if (existingBanner) {
            // Remove banner if no active translations
            existingBanner.remove();
        }
    },

    resetUIToIdle() {
        StateManager.setState('translation.isBatchActive', false);
        StateManager.setState('translation.currentJob', null);

        this.clearTranslationState();

        DomHelpers.hide('interruptBtn');
        DomHelpers.setDisabled('interruptBtn', false);
        DomHelpers.setText('interruptBtn', t('translation:interrupt_batch_with_icon'));

        const filesToProcess = StateManager.getState('files.toProcess');
        DomHelpers.setDisabled('translateBtn', filesToProcess.length === 0 || !StatusManager.isConnected());
        DomHelpers.setText('translateBtn', t('translation:start_batch_with_icon'));

        if (filesToProcess.length === 0) {
            DomHelpers.hide('progressSection');
        }

        this.updateActiveTranslationsState();

        if (window.loadResumableJobs) {
            window.loadResumableJobs();
        }
    }
};
