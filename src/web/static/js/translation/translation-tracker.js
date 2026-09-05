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
import { FileUpload } from '../files/file-upload.js';
import { FileActions } from '../files/file-actions.js';
import { ProgressManager, formatElapsedTime, deriveRateContext, buildRecommendationContent } from './progress-manager.js';
import { renderTranslationTitle, getFileIcon, createGenericEPUBIcon } from './progress-title.js';
import { LifecycleManager } from '../utils/lifecycle-manager.js';
import { t } from '../i18n/i18n.js';
import {
    overridePanelHtml,
    openOverridePanel,
    toggleOverridePanel,
    readOverrideConfig,
    destroyOverridePickers,
} from './model-override-panel.js';

// Storage configuration with versioning
const STORAGE_VERSION = 1;
const STORAGE_KEY_PREFIX = 'tbl_translation_state';
const TRANSLATION_STATE_STORAGE_KEY = `${STORAGE_KEY_PREFIX}_v${STORAGE_VERSION}`;

// Statuses that end a job server-side. Only these can reset the UI to idle.
const TERMINAL_STATUSES = new Set(['completed', 'partial', 'error', 'interrupted', 'rate_limited']);
const MAX_OWNED_JOB_IDS = 50;

// Debounce timer for discovering a job started on another device (issue #271).
let foreignDiscoveryTimer = null;
const FOREIGN_DISCOVERY_DELAY_MS = 1000;

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

/**
 * Read a per-run counter from a finished job's stats, falling back to its
 * accumulated twin only when the `run_*` key is absent.
 *
 * Same presence-based precedence as `_unfinished()` in
 * src/api/completion_status.py and `deriveRateContext` in progress-manager.js:
 * a `run_*` key that is present and 0 is trusted (that pass really did zero),
 * and only a missing key falls back to the accumulated field (legacy payloads,
 * or a format that never learned to emit the twin).
 *
 * @param {Object} stats - Job stats payload
 * @param {string} runKey - Per-run key (e.g. 'run_placeholder_errors')
 * @param {string} accumulatedKey - Cross-pass twin (e.g. 'placeholder_errors')
 * @returns {number}
 */
function runCounter(stats, runKey, accumulatedKey) {
    const s = stats || {};
    return (s[runKey] !== undefined && s[runKey] !== null)
        ? (s[runKey] || 0)
        : (s[accumulatedKey] || 0);
}

/**
 * Number of chunks the delivered file still owes (issue #261, design decision
 * D9). Trust `unfinished_chunks` whenever the key is present — including when
 * it is 0 while `fallback_used` is not, which is exactly what a fully healed
 * retry pass looks like — and fall back to the historical `fallback_used`
 * counter only for legacy payloads that never emitted it.
 *
 * @param {Object} stats - Job stats payload
 * @returns {number}
 */
function unfinishedChunkCount(stats) {
    const s = stats || {};
    return typeof s.unfinished_chunks === 'number'
        ? s.unfinished_chunks
        : (s.fallback_used || 0);
}

/**
 * Number of chunks the delivered file carries with only approximate tag
 * positions — translated by the token-alignment repair (Phase 2), tags
 * reinserted proportionally.
 *
 * Read from the live map the EPUB pipeline emits (`degraded_files`, total
 * `degraded_chunks`), NOT from `token_alignment_used`: that counter is an
 * accumulated, cross-pass tally of Phase 2 *events*, so it stays positive after
 * those chunks have been retranslated successfully and would make a clean book
 * claim defects — the same class of bug as reading `fallback_used` for
 * "still untranslated".
 *
 * Same presence-based precedence as `unfinishedChunkCount` above and
 * `_unfinished()` in src/api/completion_status.py: a key that is present and 0
 * is trusted, and only an absent pair falls back to the accumulated counter
 * (legacy payloads, and formats that never emit the map).
 *
 * @param {Object} stats - Job stats payload
 * @returns {number}
 */
function degradedChunkCount(stats) {
    const s = stats || {};
    if (typeof s.degraded_chunks === 'number') return s.degraded_chunks;
    const files = s.degraded_files;
    if (files && typeof files === 'object' && !Array.isArray(files)) {
        return Object.values(files).reduce(
            (sum, indices) => sum + (Array.isArray(indices) ? indices.length : 0), 0);
    }
    return s.token_alignment_used || 0;
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

                    if (serverState.status === 'completed' ||
                        serverState.status === 'error' ||
                        serverState.status === 'interrupted' ||
                        serverState.status === 'rate_limited') {

                        MessageLogger.addLog(t('translation:session_sync_log', { status: serverState.status }));
                        this.resetUIToIdle();
                    } else if (serverState.status === 'running' || serverState.status === 'queued') {
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
     * Restore active translation state if there's one running on the server
     */
    async restoreActiveTranslation() {
        try {
            const response = await ApiClient.getActiveTranslations();
            const activeJobs = (response.translations || []).filter(
                t => t.status === 'running' || t.status === 'queued'
            );

            if (activeJobs.length === 0) return;

            // Find matching file in our queue
            const filesToProcess = StateManager.getState('files.toProcess') || [];

            for (const job of activeJobs) {
                let matchingFile = filesToProcess.find(f =>
                    f.translationId === job.translation_id ||
                    f.filePath === job.input_file ||
                    f.name === job.input_file?.split('/').pop()
                );

                // If no matching file found, create a virtual file reference from server data
                // This allows restoration after browser refresh even if filesToProcess is empty
                if (!matchingFile && job.input_filename) {
                    matchingFile = {
                        name: job.input_filename,
                        translationId: job.translation_id,
                        status: 'Processing',
                        type: job.file_type || 'txt',
                        isVirtual: true
                    };
                }

                if (matchingFile) {
                    StateManager.setState('translation.currentJob', {
                        fileRef: matchingFile,
                        translationId: job.translation_id
                    });
                    this.registerOwnedJob(job.translation_id);
                    StateManager.setState('translation.isBatchActive', true);

                    DomHelpers.show('progressSection');
                    this.updateTranslationTitle(matchingFile);

                    // Calculate progress from stats (job contains total_chunks, completed_chunks, etc.)
                    if (job.total_chunks > 0) {
                        const stats = {
                            total_chunks: job.total_chunks,
                            completed_chunks: job.completed_chunks || 0,
                            failed_chunks: job.failed_chunks || 0,
                            elapsed_time: job.elapsed_time,
                            progress_percent: job.progress_percent,
                            current_phase: job.current_phase,
                            enable_refinement: job.enable_refinement || false
                        };
                        this.updateStats(matchingFile.fileType, stats);
                    }

                    if (job.last_translation) {
                        MessageLogger.updateTranslationPreview(job.last_translation);
                    }

                    const translateBtn = DomHelpers.getElement('translateBtn');
                    if (translateBtn) {
                        translateBtn.disabled = true;
                        translateBtn.innerHTML = t('translation:batch_in_progress');
                    }
                    DomHelpers.show('interruptBtn');

                    if (!matchingFile.isVirtual) {
                        this.updateFileStatusInList(matchingFile.name, 'Processing', job.translation_id);
                    }

                    break;
                }
            }
        } catch (error) {
            console.warn('Failed to restore active translation:', error);
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
            this.finishCurrentFileTranslation(
                t('translation:translation_error_msg', { name: currentFile.name, error: data.error || t('translation:translation_unknown_error') }),
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
            DomHelpers.setText('openrouterCost', '$' + cost.toFixed(4));
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
        const fileListItem = DomHelpers.getOne(`#fileListContainer li[data-filename="${fileName}"] .file-status`);
        if (fileListItem) {
            DomHelpers.setText(fileListItem, `(${newStatus})`);
        }

        // Update in state
        const filesToProcess = StateManager.getState('files.toProcess');
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
     * Render a persistent success card for a completed file, with quick actions
     * to locate it on disk.
     * @param {Object} file - The file that just finished
     * @param {Object} resultData - Final payload from the server (output_filename, output_dir)
     */
    renderCompletionCard(file, resultData) {
        const container = DomHelpers.getElement('completionCardsContainer');
        if (!container) return;

        const card = document.createElement('div');
        card.className = 'completion-card';
        this._populateCompletionCard(card, file, resultData);
        container.appendChild(card);
        this._ensureCompletionCardsLocaleListener();

        DomHelpers.hide('progressSection');
    },

    /**
     * Fill (or rebuild) an existing completion card with localized content.
     * Pulled out of `renderCompletionCard` so the same DOM tree can be
     * re-rendered on `localeChanged` without dropping the card from the page.
     *
     * Stashes the source payload on the element itself so the locale listener
     * can rebuild without coordinating extra storage.
     */
    _populateCompletionCard(card, file, resultData) {
        card._tblPayload = { file, resultData };

        // The card is rebuilt in place (locale switch, active-state change), so
        // any override picker created by a previous render is about to lose its
        // DOM. Release it first, otherwise its SearchableSelect registrations
        // leak and a second picker gets registered for the same job.
        destroyOverridePickers(card);

        const outputFilename = resultData.output_filename || file.outputFilename || file.name;
        const safeFilename = DomHelpers.escapeHtml(outputFilename);
        const statsHtml = this._buildCompletionStatsHtml(file, resultData);
        const dismissLabel = t('translation:completion_card_dismiss');
        const isPartial = resultData.status === 'partial';
        const titleIcon = isPartial ? 'warning' : 'check_circle';
        const titleText = t(isPartial
            ? 'translation:translation_partial_card_title'
            : 'translation:translation_completed_card_title');

        card.innerHTML = '';

        const topRow = document.createElement('div');
        topRow.className = 'completion-card__top';
        topRow.appendChild(this._buildCompletionThumb(file));

        const main = document.createElement('div');
        main.className = 'completion-card__main';
        main.innerHTML = `
            <div class="completion-card__header">
                <h3 class="completion-card__title">
                    <span class="material-symbols-outlined">${titleIcon}</span>
                    <span>${titleText}${statsHtml}</span>
                </h3>
                <button type="button" class="completion-card__close" title="${dismissLabel}" aria-label="${dismissLabel}">
                    <span class="material-symbols-outlined">close</span>
                </button>
            </div>
            <div class="completion-card__filename" title="${safeFilename}">${safeFilename}</div>
        `;
        topRow.appendChild(main);
        card.appendChild(topRow);

        const warningBlock = this._buildCompletionWarningBlock(file, resultData);
        if (warningBlock) {
            card.appendChild(warningBlock);
        }

        const actionsGroup = FileActions.createActionGroup({
            actions: ['download', 'open', 'reveal', 'files-tab'],
            filename: outputFilename,
            variant: 'labeled'
        });
        actionsGroup.classList.add('completion-card__actions');
        card.appendChild(actionsGroup);

        card.querySelector('.completion-card__close').addEventListener('click', () => {
            destroyOverridePickers(card);
            card.remove();
        });
    },

    /**
     * Re-render every visible completion card whenever the user switches
     * locale, so the dynamically interpolated strings (title, stat badges,
     * warning block, action labels) stay in sync with the rest of the UI.
     * Bound once, lazily, the first time a card is rendered.
     */
    _ensureCompletionCardsLocaleListener() {
        if (this._completionLocaleListenerBound) return;
        this._completionLocaleListenerBound = true;
        window.addEventListener('localeChanged', () => this._refreshCompletionCards());

        // The "fix these chunks" affordance mirrors the resumable-job card's
        // active-translation guard, and that guard is only correct at render
        // time. The resumable list solves it by reloading on this same state
        // key; do the equivalent here so a card rendered while the batch was
        // still running does not keep a permanently disabled button (or offer
        // a resume while another job is running).
        // `translation.hasActive` is re-set on every poll of the active-jobs
        // endpoint, so only a real transition triggers a rebuild — otherwise an
        // open panel would be torn down and re-created for nothing.
        StateManager.subscribe('translation.hasActive', (hasActive, wasActive) => {
            if (hasActive === wasActive) return;
            this._refreshCompletionCards();
        });
    },

    /**
     * Rebuild every visible completion card from its stashed payload.
     *
     * A rebuild replaces the model-override panel, so an open one is reopened
     * afterwards (with a fresh picker — the old one is destroyed by
     * `_populateCompletionCard`, never left orphaned). The picker is only
     * created once the card is back in the document, which is why this lives
     * here rather than in the builder.
     * @private
     */
    _refreshCompletionCards() {
        const container = DomHelpers.getElement('completionCardsContainer');
        if (!container) return;
        container.querySelectorAll('.completion-card').forEach((card) => {
            if (!card._tblPayload) return;
            const previousPanel = card.querySelector('.completion-override');
            const wasOpen = !!previousPanel && previousPanel.style.display !== 'none';

            this._populateCompletionCard(card, card._tblPayload.file, card._tblPayload.resultData);

            if (!wasOpen) return;
            const panel = card.querySelector('.completion-override');
            if (!panel) return;
            panel.style.display = 'block';
            openOverridePanel(panel);
        });
    },

    /**
     * Build the thumbnail element for the completion card.
     * Uses the book cover for EPUBs (with SVG fallback), generic icon otherwise.
     * @param {Object} file - File object (fileType, thumbnail)
     * @returns {HTMLElement} Thumb wrapper element
     */
    _buildCompletionThumb(file) {
        const wrap = document.createElement('div');
        wrap.className = 'completion-card__thumb';

        if (file.fileType === 'epub' && file.thumbnail) {
            const img = document.createElement('img');
            img.src = `/api/thumbnails/${encodeURIComponent(file.thumbnail)}`;
            img.alt = 'Cover';
            img.onerror = () => {
                wrap.innerHTML = createGenericEPUBIcon();
            };
            wrap.appendChild(img);
        } else {
            wrap.innerHTML = getFileIcon(file.fileType);
        }

        return wrap;
    },

    /**
     * Build the stats block HTML for the completion card.
     *
     * The chips describe the state of the file that was just delivered, not
     * the accumulated history of the job (issue #261). Concretely:
     *   - the approximate-tag chip counts the chunks that are STILL only
     *     approximately tagged (`degraded_files` / `degraded_chunks`), not the
     *     accumulated `token_alignment_used` tally of Phase 2 events: those
     *     chunks are never retried automatically (design decision D3), but the
     *     card now offers to retry them explicitly, and a book whose chunks
     *     were all repaired that way must stop reporting them.
     *   - Phase 3 fallbacks are no longer merged into that number. A chunk left
     *     in the source language is not "approximately tagged", and once it has
     *     been retried successfully it is neither — it gets its own chip driven
     *     by the current unfinished count (D9).
     *   - `placeholder_errors` is read per-run: retry-attempt noise from an
     *     earlier pass is not a property of the output.
     *
     * @param {Object} file - File object (for fileType)
     * @param {Object} resultData - Final payload (contains stats)
     * @returns {string} HTML for the stats block (empty string if no stats)
     */
    _buildCompletionStatsHtml(file, resultData) {
        const stats = resultData.stats || {};

        const failed = stats.failed_chunks || 0;
        const elapsed = stats.elapsed_time;
        const isSrt = !!(file && file.fileType === 'srt');
        const fallbacks = isSrt ? 0 : degradedChunkCount(stats);
        const placeholderErrors = isSrt
            ? 0
            : runCounter(stats, 'run_placeholder_errors', 'placeholder_errors');

        // Content still in the source language in the delivered file. On the
        // TXT/SRT/DOCX path every unfinished unit is also counted in
        // `failed_chunks` (which has its own chip), so subtracting avoids
        // reporting the same chunks twice in the header.
        const untranslated = Math.max(0, unfinishedChunkCount(stats) - failed);

        // Plain Text Mode paragraph alignment (issue #253). Zero on every other
        // path, so no gating by file type is needed: a segment the model
        // returned with the wrong paragraph count was re-translated until the
        // count matched, and whatever is left over stayed misaligned.
        const paragraphMismatches = stats.paragraph_count_mismatches || 0;
        const paragraphRepairFailed = stats.paragraph_repair_failed || 0;
        const paragraphRealigned = Math.max(0, paragraphMismatches - paragraphRepairFailed);

        const cost = stats.openrouter_cost || 0;
        const promptTokens = stats.openrouter_prompt_tokens || 0;
        const completionTokens = stats.openrouter_completion_tokens || 0;
        const totalTokens = promptTokens + completionTokens;

        const items = [];

        if (typeof elapsed === 'number' && elapsed > 0) {
            items.push(formatElapsedTime(elapsed));
        }

        if (failed > 0) {
            items.push(`<span class="completion-card__stat--error">${t('translation:completion_failed_chunks', { count: failed })}</span>`);
        }

        // Missing content, not a formatting blemish: same severity styling as
        // the failed-chunks chip rather than the `--warn` formatting ones.
        if (untranslated > 0) {
            items.push(`<span class="completion-card__stat--error">${t('translation:completion_warning_untranslated', { count: untranslated })}</span>`);
        }

        if (fallbacks > 0) {
            items.push(`<span class="completion-card__stat--warn">${t('translation:completion_fallback_chunks', { count: fallbacks })}</span>`);
        }

        if (placeholderErrors > 0) {
            items.push(`<span class="completion-card__stat--warn">${t('translation:completion_placeholder_errors', { count: placeholderErrors })}</span>`);
        }

        if (paragraphRealigned > 0) {
            items.push(t('translation:completion_paragraph_realigned', { count: paragraphRealigned }));
        }

        if (paragraphRepairFailed > 0) {
            items.push(`<span class="completion-card__stat--warn">${t('translation:completion_paragraph_repair_failed', { count: paragraphRepairFailed })}</span>`);
        }

        if (cost > 0 || totalTokens > 0) {
            items.push(`$${cost.toFixed(4)} · ${totalTokens.toLocaleString()} tokens`);
        }

        if (items.length === 0) return '';

        return `<span class="completion-card__stats"> - ${items.join(' · ')}</span>`;
    },

    /**
     * Build the warning block surfaced beneath the title when something is
     * still worth saying about the delivered file: work the job still owes,
     * chunks that failed, chunks whose tags are only approximate, or
     * placeholder trouble in this run. Mirrors the live recommendation panel
     * from progress-manager so the post-translation advice stays in sync with
     * what was shown during the run.
     *
     * Issue #261: the block claims only what is currently true of the output.
     * A healed retry pass must not be told to "use a more capable LLM" for
     * chunks it just fixed, so the rate-based advice is gated on this pass's
     * own counters — the same `run_*` values `deriveRateContext` divides by.
     *
     * @param {Object} file - File object (used to gate by file type)
     * @param {Object} resultData - Final payload (contains stats)
     * @returns {HTMLElement|null} Warning block element, or null when there is
     *   nothing worth surfacing.
     */
    _buildCompletionWarningBlock(file, resultData) {
        const stats = resultData.stats || {};
        if (file && file.fileType === 'srt') {
            return this._buildSrtCompletionWarningBlock(stats);
        }

        // What is still true of the delivered file. The approximate-tag count
        // comes from the live degraded map, never from the accumulated
        // `token_alignment_used` tally (see `degradedChunkCount`): a book whose
        // approximately-tagged chunks were all retranslated must stop claiming
        // them. `placeholder_errors` becomes per-run — an earlier pass's retry
        // noise says nothing about the output.
        const placeholderErrors = runCounter(stats, 'run_placeholder_errors', 'placeholder_errors');
        const failed = stats.failed_chunks || 0;
        const tokenAlignment = degradedChunkCount(stats);
        // The map itself: {file_href: [chunk_index, ...]}. Non-empty is the one
        // gate for the explicit retry affordance below — it is exactly what
        // handlers.py reads to decide to KEEP the checkpoint that retry needs,
        // so the button can never be offered without something behind it.
        const degradedFiles = stats.degraded_files;
        const hasDegradedMap = !!degradedFiles
            && typeof degradedFiles === 'object'
            && !Array.isArray(degradedFiles)
            && Object.keys(degradedFiles).length > 0;

        // What THIS pass struggled with. Only these may gate the rate-based
        // recommendation, whose percentages `deriveRateContext` already derives
        // from the same `run_*` values: advising "use a more capable LLM" after
        // a flawless retry pass — on chunks that pass just healed — is exactly
        // the defect being fixed here.
        const runFallbacks = runCounter(stats, 'run_token_alignment_used', 'token_alignment_used')
            + runCounter(stats, 'run_fallback_used', 'fallback_used');

        // Issue #261: `stats.unfinished_chunks` / `stats.unfinished_files` are
        // the authoritative "work still pending" signals. They cover both
        // outright failures and Phase 3 fallbacks, and — unlike the counters
        // above — they go back to 0 once those chunks have been retried
        // successfully. `fallback_used` is a historical tally of what happened
        // during the run: reading it here would claim a fully healed job still
        // holds source-language content. Same precedence as `_unfinished()` in
        // src/api/completion_status.py (D9): trust `unfinished_chunks` whenever
        // the key is present — including when it is 0 — and fall back to
        // `fallback_used` only when it is absent (legacy payloads).
        const unfinishedFiles = stats.unfinished_files;
        const hasUnfinishedFilesMap = !!unfinishedFiles
            && typeof unfinishedFiles === 'object'
            && !Array.isArray(unfinishedFiles)
            && Object.keys(unfinishedFiles).length > 0;
        const unfinishedNow = typeof stats.unfinished_chunks === 'number'
            ? stats.unfinished_chunks
            : (stats.fallback_used || 0);

        // The part of the remaining work to attribute to "kept in source
        // language". On the TXT/SRT/DOCX path every unfinished unit is also
        // counted in `failed_chunks`, so subtracting avoids reporting the same
        // chunks twice in the breakdown below.
        const untranslated = Math.max(0, unfinishedNow - failed);

        // The rate-based advice describes the pass that just ran, so it is
        // gated on the per-run signals only.
        const showRecommendations = runFallbacks > 0 || placeholderErrors > 0;
        // The block itself renders when there is something currently true to
        // say: work still owed, chunks that failed, approximate tags still in
        // the book, or placeholder trouble in this run. `showRecommendations` is
        // ORed in so the advice can never be computed without a home.
        const hasCurrentIssue = unfinishedNow > 0 || hasUnfinishedFilesMap
            || failed > 0 || tokenAlignment > 0 || placeholderErrors > 0;
        if (!hasCurrentIssue && !showRecommendations) {
            return null;
        }

        const block = document.createElement('div');
        block.className = 'completion-card__warning';

        const heading = document.createElement('div');
        heading.className = 'completion-card__warning-heading';
        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined';
        icon.textContent = 'warning';
        heading.appendChild(icon);
        const headingText = document.createElement('span');
        // When chunks are still in the source language or outright failed, the
        // optimistic "translations are correct" heading is misleading — surface
        // the missing-content message instead. A run whose fallbacks were all
        // retried successfully has no remaining work, so it keeps the
        // optimistic heading (issue #261).
        const hasUntranslatedContent = unfinishedNow > 0 || failed > 0
            || hasUnfinishedFilesMap;
        headingText.textContent = t(hasUntranslatedContent
            ? 'translation:completion_warning_heading_untranslated'
            : 'translation:completion_warning_heading');
        heading.appendChild(headingText);
        block.appendChild(heading);

        const breakdownItems = [];
        if (tokenAlignment > 0) {
            breakdownItems.push(t('translation:completion_warning_token_alignment', { count: tokenAlignment }));
        }
        if (untranslated > 0) {
            breakdownItems.push(t('translation:completion_warning_untranslated', { count: untranslated }));
        }
        if (placeholderErrors > 0) {
            breakdownItems.push(t('translation:completion_warning_placeholder_errors', { count: placeholderErrors }));
        }
        if (failed > 0) {
            breakdownItems.push(t('translation:completion_warning_failed', { count: failed }));
        }
        if (breakdownItems.length > 0) {
            const breakdown = document.createElement('div');
            breakdown.className = 'completion-card__warning-breakdown';
            breakdown.textContent = breakdownItems.join(' · ');
            block.appendChild(breakdown);
        }

        // Name the files still holding untranslated content (issue #261), so
        // the user knows where to look instead of just a count. Reuses the
        // breakdown line's style rather than introducing a new class.
        if (hasUnfinishedFilesMap) {
            const fileHrefs = Object.keys(unfinishedFiles);
            const shown = fileHrefs.slice(0, 5);
            const extra = fileHrefs.length - shown.length;
            const filesLine = document.createElement('div');
            filesLine.className = 'completion-card__warning-breakdown';
            filesLine.textContent = t('translation:completion_warning_untranslated_files', {
                files: shown.join(', '),
                // A neutral numeric suffix ("(+N)") rather than a worded
                // "N more" — it needs no translation of its own and is simply
                // '' when nothing was truncated (5 or fewer files).
                more: extra > 0 ? ` (+${extra})` : '',
            });
            block.appendChild(filesLine);
        }

        // Issue #261: those chunks are kept and retryable, so the card can offer
        // the fix instead of only advising "use a more capable LLM".
        //
        // CRITICAL: gated on `partial`. A `completed` job whose chunks are all
        // finished has had its chunks and upload directory pruned by
        // checkpoint_manager.prune_job_data (see src/api/handlers.py),
        // so there is nothing left to resume and the button would only produce
        // an error. Without a translation id there is nothing to act on either
        // — in both cases the block stays exactly as it was before this feature
        // (breakdown, file list, recommendations). The degraded affordance
        // below has its own, narrower gate for the `completed` case.
        const unfinishedFromMap = hasUnfinishedFilesMap
            ? Object.values(unfinishedFiles)
                .reduce((sum, indices) => sum + (Array.isArray(indices) ? indices.length : 0), 0)
            : 0;
        const retryableChunks = unfinishedNow > 0 ? unfinishedNow : unfinishedFromMap;
        const translationId = resultData.translation_id || (file && file.translationId) || '';
        if (resultData.status === 'partial' && translationId && retryableChunks > 0) {
            this._appendCompletionFixAffordance(block, {
                translationId,
                count: retryableChunks,
                resultData,
            });
        }

        // Second, separate affordance: retranslate the chunks that ARE
        // translated but whose inline tags were only approximately
        // repositioned. Opt-in and reachable ONLY from this card while it is on
        // screen — the checkpoint behind it is kept for a `completed` job
        // precisely because these chunks exist (src/api/handlers.py), and the
        // job deliberately never appears in the resumable-jobs list: a finished
        // book is not unfinished work.
        //
        // Gated on the degraded MAP, not on `token_alignment_used`: that
        // counter never goes back down, so it would keep offering a repair for
        // chunks that have already been repaired. Unlike the affordance above
        // this one is not restricted to `partial` — its whole point is the
        // `completed` job whose only imperfection is tag placement.
        if (translationId && hasDegradedMap && tokenAlignment > 0) {
            this._appendCompletionFixAffordance(block, {
                translationId,
                count: tokenAlignment,
                resultData,
                adviceKey: 'translation:completion_degraded_advice',
                toggleKey: 'translation:completion_degraded_toggle',
                toggleTitleKey: 'translation:completion_degraded_toggle_title',
                // The apply button says "Retranslate these chunks" in both
                // affordances, so its key is shared rather than duplicated.
                extraOverrides: { retry_token_aligned: true },
            });
        }

        // Expert-level note about the Phase 2 trade-off
        // (EPUB_TOKEN_ALIGNMENT_ENABLED, src/config.py): these chunks were
        // salvaged with approximate tag placement. Turning the setting off
        // would leave them untranslated instead — a defensible choice, since
        // untranslated chunks are retryable on any resume (issue #261).
        // Only shown when it is actionable: EPUB, and at least one chunk that
        // is STILL approximately tagged (the degraded map, not the accumulated
        // Phase 2 counter — a repaired book has nothing left to explain).
        // Built here rather than inside the recommendations sub-block so it
        // survives a pass that earns no advice: it describes chunks still in
        // the book, not this run.
        let tokenAlignmentNote = null;
        if (file && file.fileType === 'epub' && tokenAlignment > 0) {
            tokenAlignmentNote = document.createElement('p');
            tokenAlignmentNote.className = 'completion-card__warning-note';
            tokenAlignmentNote.textContent = t('translation:completion_token_alignment_note', { count: tokenAlignment });
        }

        // Only renew the rate-based recommendations when this pass actually
        // produced fallbacks or placeholder issues — a run with only
        // `failed_chunks` (e.g. provider errors) is not really a "tune the LLM"
        // situation, and neither is a retry pass that healed everything.
        if (showRecommendations) {
            const recommendations = document.createElement('div');
            recommendations.className = 'completion-card__warning-recommendations';
            buildRecommendationContent(
                recommendations,
                deriveRateContext(stats),
                'translation:completion_warning_intro',
            );
            if (tokenAlignmentNote) {
                recommendations.appendChild(tokenAlignmentNote);
            }
            block.appendChild(recommendations);
        } else if (tokenAlignmentNote) {
            block.appendChild(tokenAlignmentNote);
        }

        return block;
    },

    /**
     * Append an inline "retranslate these chunks with a better model"
     * affordance to a completion warning block: one advice sentence, a
     * disclosure button, and the shared model-override panel.
     *
     * Two callers, one wiring: the `partial` job's unfinished chunks (default
     * keys) and the `completed` job's approximately-tagged chunks, which add
     * `retry_token_aligned: true` to the overrides via `extraOverrides`. Only
     * the copy and that extra field differ, so the panel, the active-job guard
     * and the card teardown are shared rather than duplicated.
     *
     * Applying calls `window.resumeJob` (bound in index.js) rather than
     * importing ResumeManager: resume-manager.js already imports this module,
     * so an import here would close the cycle.
     *
     * @param {HTMLElement} block - Warning block to append to
     * @param {Object} opts
     * @param {string} opts.translationId - Job to resume
     * @param {number} opts.count - Number of retryable chunks
     * @param {Object} opts.resultData - Final payload (model/provider seed)
     * @param {string} [opts.adviceKey] - i18n key for the advice sentence
     * @param {string} [opts.toggleKey] - i18n key for the disclosure label
     * @param {string} [opts.toggleTitleKey] - i18n key for its tooltip
     * @param {string} [opts.applyLabelKey] - i18n key for the apply button
     * @param {Object} [opts.extraOverrides] - Extra fields merged into the
     *   resume body (the model overrides always win nothing here: these are
     *   pass-scoped flags the endpoint validates on its own).
     * @private
     */
    _appendCompletionFixAffordance(block, {
        translationId,
        count,
        resultData,
        adviceKey = 'translation:completion_fix_advice',
        toggleKey = 'translation:completion_fix_toggle',
        toggleTitleKey = 'translation:completion_fix_toggle_title',
        applyLabelKey = 'translation:completion_fix_apply_btn',
        extraOverrides = null,
    }) {
        // Same source of truth as the resumable-job card: a resume is refused
        // server-side while another job runs, so the affordance is disabled
        // rather than failing on click. The card is re-rendered when this state
        // flips (see _ensureCompletionCardsLocaleListener).
        const hasActiveTranslation = StateManager.getState('translation.hasActive') || false;

        const advice = document.createElement('div');
        advice.className = 'completion-card__warning-breakdown';
        advice.textContent = t(adviceKey, { count });
        block.appendChild(advice);

        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'completion-card__fix-toggle';
        toggle.title = hasActiveTranslation
            ? t('translation:cannot_resume_in_progress_title')
            : t(toggleTitleKey);
        const toggleIcon = document.createElement('span');
        toggleIcon.className = 'material-symbols-outlined';
        toggleIcon.textContent = 'tune';
        toggle.appendChild(toggleIcon);
        const toggleLabel = document.createElement('span');
        toggleLabel.textContent = t(toggleKey, { count });
        toggle.appendChild(toggleLabel);
        toggle.disabled = hasActiveTranslation;
        block.appendChild(toggle);

        const holder = document.createElement('div');
        holder.innerHTML = overridePanelHtml({
            tid: translationId,
            provider: resultData.llm_provider,
            model: resultData.model,
            endpoint: resultData.llm_api_endpoint,
            applyLabelKey,
            panelClass: 'completion-override',
        });
        const panel = holder.firstElementChild;
        if (!panel) return;
        block.appendChild(panel);

        toggle.addEventListener('click', () => {
            if (toggle.disabled) return;
            toggleOverridePanel(panel);
        });

        const applyBtn = panel.querySelector('.resume-apply');
        if (!applyBtn) return;
        applyBtn.disabled = hasActiveTranslation;
        if (hasActiveTranslation) {
            applyBtn.title = t('translation:cannot_resume_in_progress_title');
        }
        applyBtn.addEventListener('click', () => {
            if (applyBtn.disabled) return;
            const picked = readOverrideConfig(panel);
            // `undefined` means the panel refused the input (no model picked)
            // and already told the user; `null` means "resume as configured".
            if (picked === undefined) return;
            // A pass-scoped flag (retry_token_aligned) travels in the same body
            // as the model overrides, so "resume as configured" still has to
            // send one. ApiClient.resumeJob omits the body only for an empty
            // object, which is exactly the no-flag no-override case.
            const overrides = extraOverrides
                ? Object.assign({}, picked || {}, extraOverrides)
                : picked;
            if (typeof window.resumeJob !== 'function') {
                console.error('window.resumeJob is not available; cannot retry unfinished chunks');
                return;
            }

            // This card is superseded the moment the retry starts: the resumed
            // run overwrites the very file its Download button points at, and a
            // fresh completion card is rendered when the retry finishes. Two
            // cards for one book — the stale one first — is exactly what we do
            // not want, so drop it once the resume is really under way.
            //
            // "Really" matters: ResumeManager.resumeJob returns undefined in
            // every case (it is async and reports nothing), and it can bail out
            // before doing anything — another job is active, or the user
            // dismisses its confirm() dialog — in which case this card must
            // survive. Its only success signal is the `translationResumed`
            // event, dispatched after the server accepted the resume, so that
            // is what we listen for.
            const card = block.closest('.completion-card');
            const onResumed = (event) => {
                if (!event.detail || event.detail.translationId !== translationId) return;
                window.removeEventListener('translationResumed', onResumed);
                if (!card) return;
                // Same discipline as _populateCompletionCard and the dismiss
                // handler: release the override picker before its DOM goes
                // away, or its SearchableSelect registration leaks.
                destroyOverridePickers(card);
                card.remove();
            };
            window.addEventListener('translationResumed', onResumed);
            Promise.resolve(window.resumeJob(translationId, overrides))
                .catch(() => { /* resumeJob reports its own errors */ })
                .finally(() => {
                    // Cancelled or rejected: the event never fired, so tear the
                    // listener down instead of leaving it on the window. On the
                    // success path onResumed already removed it.
                    window.removeEventListener('translationResumed', onResumed);
                });
        });
    },

    /**
     * SRT variant of the completion warning block. Shown when subtitle
     * blocks still failed after the automatic marker-validation retries:
     * the affected cues kept the source-language text. Mirrors the EPUB
     * fallback panel structure (heading + breakdown + advice list) with
     * SRT-specific recommendations.
     *
     * @param {Object} stats - Final stats payload
     * @returns {HTMLElement|null} Warning block, or null when nothing failed
     */
    _buildSrtCompletionWarningBlock(stats) {
        const failed = stats.failed_chunks || 0;
        if (failed === 0) {
            return null;
        }

        const block = document.createElement('div');
        block.className = 'completion-card__warning';

        const heading = document.createElement('div');
        heading.className = 'completion-card__warning-heading';
        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined';
        icon.textContent = 'warning';
        heading.appendChild(icon);
        const headingText = document.createElement('span');
        headingText.textContent = t('translation:srt_completion_warning_heading');
        heading.appendChild(headingText);
        block.appendChild(heading);

        const breakdown = document.createElement('div');
        breakdown.className = 'completion-card__warning-breakdown';
        breakdown.textContent = t('translation:srt_completion_warning_blocks', { count: failed });
        block.appendChild(breakdown);

        const recommendations = document.createElement('div');
        recommendations.className = 'completion-card__warning-recommendations';
        const intro = document.createElement('strong');
        intro.textContent = t('translation:srt_completion_warning_intro');
        recommendations.appendChild(intro);

        const list = document.createElement('ul');
        list.className = 'recommendation-list';
        const llmTip = document.createElement('li');
        llmTip.textContent = t('translation:fallback_panel_tip_llm');
        list.appendChild(llmTip);
        const blockSizeTip = document.createElement('li');
        blockSizeTip.textContent = t('translation:srt_completion_tip_block_size');
        list.appendChild(blockSizeTip);
        recommendations.appendChild(list);
        block.appendChild(recommendations);

        return block;
    },

    /**
     * Remove all completion cards. Currently unused — cards are dismissed
     * individually by the user via the card's close button.
     */
    clearCompletionCards() {
        const container = DomHelpers.getElement('completionCardsContainer');
        if (container) container.innerHTML = '';
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
                <div class="active-translation-warning" style="background: #fef3c7; border: 1px solid #f59e0b; padding: 12px; margin-bottom: 15px; border-radius: 6px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 20px;">⚠️</span>
                        <div style="flex: 1;">
                            <strong style="color: #92400e;">${t('translation:active_translation_warning_title')}</strong>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: #78350f;">
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
