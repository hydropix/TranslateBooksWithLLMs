/**
 * Lifecycle Manager - Page lifecycle and connection management
 *
 * Handles page initialization, cleanup, visibility changes,
 * and connection state management.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { t } from '../i18n/i18n.js';

// Storage configuration with versioning
const STORAGE_VERSION = 1;
const SERVER_SESSION_KEY = `tbl_server_session_id_v${STORAGE_VERSION}`;
const TRANSLATION_STATE_STORAGE_KEY = `tbl_translation_state_v${STORAGE_VERSION}`;

export const LifecycleManager = {
    /**
     * Handlers invoked when a visibility/interval consistency check finds the
     * server and the UI disagree. Injected from index.js so this module does not
     * import the tracker (translation-tracker.js already imports this module).
     * @type {{onTerminalStatus: (data: object) => void, onJobMissing: () => void} | null}
     */
    _desyncHandlers: null,

    /**
     * @param {{onTerminalStatus: (data: object) => void, onJobMissing: () => void}} handlers
     */
    setDesyncHandlers(handlers) {
        this._desyncHandlers = handlers;
    },

    /**
     * Initialize lifecycle manager
     */
    initialize() {
        // Clean up old storage versions
        this.cleanupOldStorageVersions();

        this.setupPageLoadHandler();
        // No pagehide/unload handler on purpose: a job belongs to the server,
        // not to the tab. Closing, reloading or backgrounding the page must
        // leave it running (issue #349); restoreActiveTranslation picks it up
        // again on the next load.
        this.setupVisibilityChangeHandler();
        this.startStateConsistencyCheck();
    },

    /**
     * Clean up old localStorage versions
     */
    cleanupOldStorageVersions() {
        try {
            // Remove old non-versioned keys
            const oldKeys = [
                'tbl_server_session_id',
                'tbl_translation_state'
            ];

            oldKeys.forEach(oldKey => {
                if (localStorage.getItem(oldKey)) {
                    localStorage.removeItem(oldKey);
                }
            });

            // Remove any other versions (future-proofing)
            for (let i = 0; i < STORAGE_VERSION; i++) {
                const oldSessionKey = `tbl_server_session_id_v${i}`;
                const oldTranslationKey = `tbl_translation_state_v${i}`;

                if (localStorage.getItem(oldSessionKey)) {
                    localStorage.removeItem(oldSessionKey);
                }
                if (localStorage.getItem(oldTranslationKey)) {
                    localStorage.removeItem(oldTranslationKey);
                }
            }
        } catch (error) {
            console.warn('Failed to cleanup old storage versions:', error);
        }
    },

    /**
     * Set up page load handler
     */
    setupPageLoadHandler() {
        window.addEventListener('load', async () => {
            try {
                const healthData = await ApiClient.healthCheck();
                MessageLogger.addLog(t('common:server_health_ok'));

                if (healthData.supported_formats) {
                    MessageLogger.addLog(t('common:supported_formats', { formats: healthData.supported_formats.join(', ') }));
                }
                // The WebSocket is opened once by index.js, right after its
                // handlers are wired; connecting here as well was redundant.

            } catch (error) {
                MessageLogger.showMessage(
                    t('common:server_unavailable', { url: ApiClient.getBaseUrl(), error: error.message }),
                    'error'
                );
                MessageLogger.addLog(t('common:server_connect_failed_log', { error: error.message }));
            }
        });
    },

    /**
     * Get server session check promise (to be called early in initialization)
     * This ensures server restart detection happens BEFORE state restoration
     * @returns {Promise<boolean>} True if server was restarted
     */
    async getServerSessionCheck() {
        try {
            const healthData = await ApiClient.healthCheck();
            return await this.checkServerRestart(healthData);
        } catch (error) {
            console.warn('Could not check server session:', error);
            return false;
        }
    },

    /**
     * Check if server was restarted and clean up stale state
     * @param {Object} healthData - Health check response data
     * @returns {boolean} True if server was restarted
     */
    async checkServerRestart(healthData) {
        try {
            const serverSessionId = healthData.session_id || healthData.startup_time;

            if (!serverSessionId) {
                return false;
            }

            const lastSessionId = localStorage.getItem(SERVER_SESSION_KEY);

            if (lastSessionId && lastSessionId !== String(serverSessionId)) {
                MessageLogger.addLog(t('common:server_restart_detected'));

                localStorage.removeItem(TRANSLATION_STATE_STORAGE_KEY);

                StateManager.setState('translation.currentJob', null);
                StateManager.setState('translation.isBatchActive', false);
                StateManager.setState('translation.activeJobs', []);
                StateManager.setState('translation.hasActive', false);

                try {
                    const progressSection = document.getElementById('progressSection');
                    const interruptBtn = document.getElementById('interruptBtn');
                    const translateBtn = document.getElementById('translateBtn');

                    if (progressSection) progressSection.style.display = 'none';
                    if (interruptBtn) interruptBtn.style.display = 'none';
                    if (translateBtn) {
                        translateBtn.disabled = false;
                        translateBtn.innerHTML = t('translation:start_batch_with_icon');
                    }
                } catch (uiError) {
                    console.warn('Could not reset UI elements:', uiError);
                }

                localStorage.setItem(SERVER_SESSION_KEY, String(serverSessionId));

                return true;
            }

            localStorage.setItem(SERVER_SESSION_KEY, String(serverSessionId));

            return false;

        } catch (error) {
            console.error('Error checking server restart:', error);
            MessageLogger.addLog(t('common:server_session_check_failed'));
            return false;
        }
    },


    setupVisibilityChangeHandler() {
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                this.checkStateConsistency();
            }
        });
    },

    async checkStateConsistency() {
        const isBatchActive = StateManager.getState('translation.isBatchActive');
        const currentJob = StateManager.getState('translation.currentJob');

        if (!isBatchActive || !currentJob) {
            return;
        }

        const tidToCheck = currentJob.translationId;

        try {
            const data = await ApiClient.getTranslationStatus(tidToCheck);
            const serverStatus = data.status;

            if (serverStatus === 'completed' || serverStatus === 'error' || serverStatus === 'interrupted' || serverStatus === 'rate_limited' || serverStatus === 'partial') {
                MessageLogger.addLog(t('common:state_desync_syncing', { status: serverStatus }));

                this._desyncHandlers?.onTerminalStatus({
                    translation_id: tidToCheck,
                    status: serverStatus,
                    result: data.result_preview || `[${serverStatus}]`,
                    error: data.error,
                });
            }
        } catch (error) {
            if (error.status === 404) {
                MessageLogger.addLog(t('common:translation_job_missing'));
                this._desyncHandlers?.onJobMissing();
            } else {
                console.error('Error checking state consistency:', error);
            }
        }
    },

    /**
     * Start periodic state consistency checks (every 10 seconds)
     */
    startStateConsistencyCheck() {
        setInterval(() => {
            this.checkStateConsistency();
        }, 10000);
    }
};
