/**
 * Status Manager - LLM connection status indicator
 *
 * Manages the visual status indicator in the header showing LLM connection state
 */

import { DomHelpers } from '../ui/dom-helpers.js';
import { StateManager } from '../core/state-manager.js';
import { t } from '../i18n/i18n.js';

/**
 * Status types and their visual representations
 */
const STATUS_TYPES = {
    checking: {
        textKey: 'common:llm_checking',
        dotClass: 'checking'
    },
    connected: {
        textKey: 'common:llm_connected',
        dotClass: 'connected'
    },
    disconnected: {
        textKey: 'settings:llm_disconnected',
        dotClass: 'disconnected'
    },
    error: {
        textKey: 'settings:llm_error',
        dotClass: 'error'
    },
    waiting: {
        textKey: 'settings:llm_waiting',
        dotClass: 'waiting'
    }
};

/**
 * Current status
 */
let currentStatus = 'checking';

export const StatusManager = {
    /**
     * Initialize status manager
     */
    initialize() {
        // Set initial checking state
        this.setStatus('checking');
    },

    /**
     * Set connection status
     * @param {string} status - Status type ('checking', 'connected', 'disconnected', 'error', 'waiting')
     * @param {string} customText - Optional custom text to override default
     */
    setStatus(status, customText = null) {
        const statusInfo = STATUS_TYPES[status];
        if (!statusInfo) {
            return;
        }

        currentStatus = status;

        // Update text
        const statusText = DomHelpers.getElement('providerStatusText');
        if (statusText) {
            const text = customText || t(statusInfo.textKey);
            statusText.textContent = text;
            // Colour is driven by style.css from the dot's state class: muted
            // while healthy, escalated only on disconnected / error.
            // The pill truncates long text with an ellipsis; expose the full
            // message on hover so a verbose connection error stays readable.
            statusText.title = text;
        }

        // Update dot
        const statusDot = document.querySelector('.header .status-indicator .status-dot');
        if (statusDot) {
            // Remove all status classes
            statusDot.classList.remove('checking', 'connected', 'disconnected', 'error', 'waiting');
            // Add new status class
            statusDot.classList.add(statusInfo.dotClass);
        }

        // Open Settings foldout on connection problems
        if (status === 'disconnected' || status === 'error' || status === 'waiting') {
            this.openSettingsFoldout();
        }

        // Update translate button based on connection status
        this.updateTranslateButton(status);
    },

    /**
     * Update translate button based on connection status
     * @param {string} status - Current status
     */
    updateTranslateButton(status) {
        const translateBtn = document.getElementById('translateBtn');
        if (!translateBtn) return;

        const isConnected = status === 'connected';
        const filesToProcess = StateManager.getState('files.toProcess') || [];
        const isBatchActive = StateManager.getState('translation.isBatchActive') || false;

        if (!isConnected) {
            translateBtn.disabled = true;
            translateBtn.title = t('settings:llm_not_connected_title');
        } else {
            // Enable if there are files and no batch is active
            translateBtn.disabled = filesToProcess.length === 0 || isBatchActive;
            translateBtn.title = '';
        }
    },

    /**
     * Check if LLM is connected
     * @returns {boolean} True if connected
     */
    isConnected() {
        return currentStatus === 'connected';
    },

    /**
     * Open the Settings foldout if it's closed
     */
    openSettingsFoldout() {
        // Use requestAnimationFrame to ensure DOM is ready
        requestAnimationFrame(() => {
            const section = document.getElementById('settingsOptionsSection');
            const icon = document.getElementById('settingsOptionsIcon');

            if (section && section.classList.contains('hidden')) {
                section.classList.remove('hidden');
                if (icon) {
                    icon.style.transform = 'rotate(180deg)';
                }
            }
        });
    },

    /**
     * Set checking status
     */
    setChecking() {
        this.setStatus('checking');
    },

    /**
     * Set connected status
     * @param {string} provider - Provider name (optional)
     * @param {number} modelCount - Number of models available (optional)
     */
    setConnected(provider = null, modelCount = null) {
        let text = t('common:llm_connected');
        if (provider) {
            const providerName = provider.charAt(0).toUpperCase() + provider.slice(1);
            if (modelCount) {
                text = modelCount === 1
                    ? t('settings:llm_with_provider_count_one', { provider: providerName, count: modelCount })
                    : t('settings:llm_with_provider_count_other', { provider: providerName, count: modelCount });
            } else {
                text = t('settings:llm_with_provider', { provider: providerName });
            }
        }
        this.setStatus('connected', text);
    },

    /**
     * Set disconnected status
     * @param {string} reason - Optional reason for disconnection
     */
    setDisconnected(reason = null) {
        const text = reason
            ? t('settings:llm_disconnected_with_reason', { reason })
            : t('settings:llm_disconnected');
        this.setStatus('disconnected', text);
    },

    /**
     * Set error status
     * @param {string} message - Optional error message
     */
    setError(message = null) {
        const text = message
            ? t('settings:llm_error_with_msg', { message })
            : t('settings:llm_error');
        this.setStatus('error', text);
    },

    /**
     * Set waiting status
     * @param {string} message - Optional waiting message
     */
    setWaiting(message = null) {
        const text = message || t('settings:llm_waiting_default');
        this.setStatus('waiting', text);
    },

    /**
     * Get current status
     * @returns {string} Current status type
     */
    getCurrentStatus() {
        return currentStatus;
    }
};
