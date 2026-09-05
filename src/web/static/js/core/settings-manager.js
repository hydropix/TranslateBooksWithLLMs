/**
 * Settings Manager - User preferences persistence
 *
 * Handles saving/loading user preferences via:
 * 1. localStorage for quick preferences (last model, provider, languages)
 * 2. Server API for sensitive data (API keys saved to .env)
 * 3. Server API (`/api/preferences`) for the small subset of purely local
 *    preferences that must follow the user across devices (SYNCED_PREF_KEYS).
 *    The server wins at page load, later changes are pushed after the auto-save
 *    debounce, and localStorage keeps acting as an offline cache. Provider,
 *    model, endpoints, API keys and the output filename pattern are NOT synced:
 *    they stay governed by .env so a single owner remains.
 */

import { ApiClient } from './api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { MessageLogger } from '../ui/message-logger.js';
import { t } from '../i18n/i18n.js';

// Storage configuration with versioning
const STORAGE_VERSION = 1;
const STORAGE_KEY_PREFIX = 'tbl_user_preferences';
const STORAGE_KEY = `${STORAGE_KEY_PREFIX}_v${STORAGE_VERSION}`;

/**
 * Preference keys mirrored to the server so they follow the user across devices.
 *
 * This list is also a security whitelist: it must never be widened to provider,
 * model, endpoint or API key fields — those belong to .env and are never sent
 * to `/api/preferences`.
 */
const SYNCED_PREF_KEYS = [
    'lastSourceLanguage',
    'lastTargetLanguage',
    'ttsEnabled',
    'textCleanup',
    'bilingualMode',
    'plainTextMode',
    'customInstructionFile'
];

/**
 * Validate user preferences structure
 * @param {any} data - Data to validate
 * @returns {boolean} True if valid
 */
function validatePreferences(data) {
    if (!data || typeof data !== 'object') return false;

    // Check version
    if (!('version' in data)) return false;

    // Validate types for known fields (non-exhaustive, just critical ones)
    if ('ttsEnabled' in data && typeof data.ttsEnabled !== 'boolean') return false;
    if ('textCleanup' in data && typeof data.textCleanup !== 'boolean') return false;

    return true;
}

/**
 * Flag to prevent localStorage from overriding .env default model
 * Set to true once the .env model has been applied
 */
let envModelApplied = false;

/**
 * Debounce timer for auto-save
 */
let autoSaveTimer = null;
const AUTO_SAVE_DELAY = 1000; // 1 second debounce

/**
 * Flag to prevent auto-save during initial load
 */
let isInitializing = true;

export const SettingsManager = {
    /**
     * Initialize settings manager - load saved preferences and setup auto-save
     */
    initialize() {
        // Clean up old storage versions
        this.cleanupOldStorageVersions();

        this.loadLocalPreferences();

        // Pull the server-synced subset (fire and forget: a slow or unreachable
        // server must never block UI initialization).
        this._syncPreferencesFromServer();

        // Listen for custom instructions loaded event
        window.addEventListener('customInstructionsLoaded', () => {
            this.applyPendingCustomInstructionSelection();
        });

        // Setup auto-save listeners after a short delay to avoid triggering during initial load
        setTimeout(() => {
            this._setupAutoSaveListeners();
            isInitializing = false;
        }, 500);
    },

    /**
     * Clean up old localStorage versions
     */
    cleanupOldStorageVersions() {
        try {
            // Remove old non-versioned key
            const oldKey = 'tbl_user_preferences';
            if (localStorage.getItem(oldKey)) {
                // Migrate data from old key before removing
                const oldData = localStorage.getItem(oldKey);
                if (oldData) {
                    try {
                        const parsed = JSON.parse(oldData);
                        // Add version and save to new key
                        parsed.version = STORAGE_VERSION;
                        localStorage.setItem(STORAGE_KEY, JSON.stringify(parsed));
                    } catch (e) {
                        console.warn('Could not migrate old preferences:', e);
                    }
                }
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
     * Setup event listeners for auto-save on all settings elements
     * @private
     */
    _setupAutoSaveListeners() {
        // Auto-save targets localStorage only. Fields that persist to .env
        // (provider, model, endpoints, API keys, naming convention) are saved
        // exclusively via the explicit "Save Settings to .env" button.
        const localAutoSaveElements = [
            { id: 'sourceLang', event: 'change' },
            { id: 'targetLang', event: 'change' },
            { id: 'customSourceLang', event: 'change' },
            { id: 'customTargetLang', event: 'change' },
            { id: 'ttsEnabled', event: 'change' },
            { id: 'textCleanup', event: 'change' },
            { id: 'bilingualMode', event: 'change' },
            { id: 'plainTextMode', event: 'change' },
            { id: 'customInstructionSelect', event: 'change' }
        ];

        localAutoSaveElements.forEach(({ id, event }) => {
            const element = DomHelpers.getElement(id);
            if (element) {
                element.addEventListener(event, () => this._triggerAutoSave());
            }
        });

        // Dirty-tracking for .env fields: any change marks the Save button as
        // having pending changes; saveAllSettings(true) clears it on success.
        const envDirtyElements = [
            { id: 'llmProvider', event: 'change' },
            { id: 'model', event: 'change' },
            { id: 'apiEndpoint', event: 'input' },
            { id: 'openaiEndpoint', event: 'input' },
            { id: 'outputFilenamePattern', event: 'input' },
            { id: 'geminiApiKey', event: 'input' },
            { id: 'openaiApiKey', event: 'input' },
            { id: 'openrouterApiKey', event: 'input' },
            { id: 'mistralApiKey', event: 'input' },
            { id: 'deepseekApiKey', event: 'input' },
            { id: 'poeApiKey', event: 'input' },
            { id: 'nimApiKey', event: 'input' },
            { id: 'disableAutoPause', event: 'change' },
            { id: 'parallelWorkers', event: 'input' },
            { id: 'maxTokensPerChunk', event: 'input' }
        ];

        envDirtyElements.forEach(({ id, event }) => {
            const element = DomHelpers.getElement(id);
            if (element) {
                element.addEventListener(event, () => this._markEnvDirty());
            }
        });
    },

    /**
     * Enable the .env Save button — there are pending unsaved changes
     */
    _markEnvDirty() {
        if (isInitializing) return;
        const btn = DomHelpers.getElement('saveSettingsBtn');
        if (btn) btn.disabled = false;
    },

    /**
     * Disable the .env Save button — nothing to save
     */
    _clearEnvDirty() {
        const btn = DomHelpers.getElement('saveSettingsBtn');
        if (btn) btn.disabled = true;
    },

    /**
     * Trigger auto-save with debounce
     * @private
     */
    _triggerAutoSave() {
        if (isInitializing) return;

        // Clear existing timer
        if (autoSaveTimer) {
            clearTimeout(autoSaveTimer);
        }

        // Set new timer
        autoSaveTimer = setTimeout(async () => {
            await this._performAutoSave();
        }, AUTO_SAVE_DELAY);
    },

    /**
     * Perform the actual auto-save
     * @private
     */
    async _performAutoSave() {
        try {
            await this.saveAllSettings(false);
        } catch {
            // Auto-save failed silently
        }
    },

    /**
     * Get all local preferences from localStorage
     * @returns {Object} Saved preferences
     */
    getLocalPreferences() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);

            if (!stored) return {};

            const parsed = JSON.parse(stored);

            // Validate structure
            if (!validatePreferences(parsed)) {
                console.warn('Invalid preferences structure, resetting to defaults');
                localStorage.removeItem(STORAGE_KEY);
                return {};
            }

            // Check version compatibility
            if (parsed.version !== STORAGE_VERSION) {
                console.warn(`Preferences version mismatch (found ${parsed.version}, expected ${STORAGE_VERSION})`);
                // Could implement migration here in the future
                localStorage.removeItem(STORAGE_KEY);
                return {};
            }

            return parsed;
        } catch (error) {
            console.error('Failed to load preferences from localStorage:', error);
            MessageLogger.addLog(t('translation:preferences_load_failed_log'));
            return {};
        }
    },

    /**
     * Save preferences to localStorage
     * @param {Object} prefs - Preferences to save
     */
    saveLocalPreferences(prefs) {
        try {
            const current = this.getLocalPreferences();
            const updated = {
                ...current,
                ...prefs,
                version: STORAGE_VERSION,
                timestamp: Date.now()
            };

            localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
        } catch (error) {
            console.error('Failed to save preferences to localStorage:', error);

            // Check if it's a quota exceeded error
            if (error.name === 'QuotaExceededError') {
                MessageLogger.addLog(t('translation:preferences_save_quota'));
            } else {
                MessageLogger.addLog(t('translation:preferences_save_failed_log'));
            }
        }
    },

    /**
     * Load and apply saved local preferences to the form
     */
    loadLocalPreferences() {
        const prefs = this.getLocalPreferences();

        // Apply last model (after models are loaded)
        if (prefs.lastModel) {
            // Store for later application after models load
            window.__pendingModelSelection = prefs.lastModel;
        }

        // Apply last languages
        if (prefs.lastSourceLanguage) {
            this._setLanguage('sourceLang', 'customSourceLang', prefs.lastSourceLanguage);
        }
        if (prefs.lastTargetLanguage) {
            this._setLanguage('targetLang', 'customTargetLang', prefs.lastTargetLanguage);
        }

        // Apply API endpoints BEFORE setting provider (so models load with correct endpoint)
        if (prefs.lastApiEndpoint) {
            DomHelpers.setValue('apiEndpoint', prefs.lastApiEndpoint);
        }
        if (prefs.lastOpenaiEndpoint) {
            DomHelpers.setValue('openaiEndpoint', prefs.lastOpenaiEndpoint);
        }

        // Apply output filename pattern (naming convention)
        if (prefs.outputFilenamePattern) {
            DomHelpers.setValue('outputFilenamePattern', prefs.outputFilenamePattern);
        }

        // Apply last provider AFTER endpoints are set
        // NOTE: We set the provider value but DON'T trigger the change event here.
        // The change event would trigger model loading, but we need to wait for
        // FormManager.loadDefaultConfig() to complete and update the endpoint
        // from the server configuration first (fixes GitHub issue #108 part 2).
        if (prefs.lastProvider) {
            const providerSelect = DomHelpers.getElement('llmProvider');
            if (providerSelect) {
                providerSelect.value = prefs.lastProvider;
                // Don't trigger change event - ProviderManager will handle model loading
                // after the 'defaultConfigLoaded' event is dispatched
            }
        }

        // Apply TTS Enabled setting
        if (prefs.ttsEnabled !== undefined) {
            const ttsEnabledCheckbox = DomHelpers.getElement('ttsEnabled');
            if (ttsEnabledCheckbox) {
                ttsEnabledCheckbox.checked = prefs.ttsEnabled;
                // Show/hide the TTS options panel based on checkbox state
                const ttsOptions = DomHelpers.getElement('ttsOptions');
                if (ttsOptions) {
                    ttsOptions.style.display = prefs.ttsEnabled ? 'block' : 'none';
                }
            }
        }

        // Apply Prompt Options settings
        if (prefs.textCleanup !== undefined) {
            const cleanupCheckbox = DomHelpers.getElement('textCleanup');
            if (cleanupCheckbox) {
                cleanupCheckbox.checked = prefs.textCleanup;
            }
        }
        if (prefs.bilingualMode !== undefined) {
            const bilingualCheckbox = DomHelpers.getElement('bilingualMode');
            if (bilingualCheckbox) {
                bilingualCheckbox.checked = prefs.bilingualMode;
            }
        }
        if (prefs.plainTextMode !== undefined) {
            const plainTextCheckbox = DomHelpers.getElement('plainTextMode');
            if (plainTextCheckbox) {
                plainTextCheckbox.checked = prefs.plainTextMode;
            }
        }
        // Note: disableAutoPause is now loaded from .env via /api/config in FormManager,
        // not from localStorage.

        // Store custom instruction file for later application (after loadCustomInstructions completes)
        if (prefs.customInstructionFile) {
            window.__pendingCustomInstructionSelection = prefs.customInstructionFile;
        }
    },

    /**
     * Keep only the preference keys that are mirrored to the server
     * @param {Object} prefs - Any preferences object
     * @returns {Object} Subset limited to SYNCED_PREF_KEYS with defined values
     * @private
     */
    _pickSynced(prefs) {
        const picked = {};
        if (!prefs || typeof prefs !== 'object') return picked;

        SYNCED_PREF_KEYS.forEach((key) => {
            if (prefs[key] !== undefined) {
                picked[key] = prefs[key];
            }
        });

        return picked;
    },

    /**
     * Pull the shared preferences from the server and apply them (server wins
     * at page load). When the server has nothing yet, the local subset is
     * pushed once so an existing install migrates silently.
     * @private
     */
    async _syncPreferencesFromServer() {
        let server;
        try {
            server = await ApiClient.getPreferences();
        } catch {
            // Offline or server error: localStorage remains the source of truth
            return;
        }

        const synced = this._pickSynced(server);

        if (Object.keys(synced).length === 0) {
            const local = this._pickSynced(this.getLocalPreferences());
            if (Object.keys(local).length > 0) {
                this._pushSyncedPreferences(local);
            }
            return;
        }

        // Refresh the offline cache (saveLocalPreferences merges, so the
        // non-synced local keys survive) then apply to the form.
        this.saveLocalPreferences(synced);
        this._applySyncedPreferences(synced);
    },

    /**
     * Apply the server-synced subset to the form without side effects
     *
     * Auto-save is suspended for the duration and no `change` event is
     * dispatched, so applying server values cannot echo back a PUT nor trigger
     * a model reload.
     * @param {Object} prefs - Synced preferences subset
     * @private
     */
    _applySyncedPreferences(prefs) {
        const wasInit = isInitializing;
        isInitializing = true;

        try {
            if (prefs.lastSourceLanguage) {
                this._setLanguage('sourceLang', 'customSourceLang', prefs.lastSourceLanguage);
            }
            if (prefs.lastTargetLanguage) {
                this._setLanguage('targetLang', 'customTargetLang', prefs.lastTargetLanguage);
            }

            if (prefs.ttsEnabled !== undefined) {
                const ttsEnabledCheckbox = DomHelpers.getElement('ttsEnabled');
                if (ttsEnabledCheckbox) {
                    ttsEnabledCheckbox.checked = prefs.ttsEnabled;
                    // Show/hide the TTS options panel based on checkbox state
                    const ttsOptions = DomHelpers.getElement('ttsOptions');
                    if (ttsOptions) {
                        ttsOptions.style.display = prefs.ttsEnabled ? 'block' : 'none';
                    }
                }
            }
            if (prefs.textCleanup !== undefined) {
                const cleanupCheckbox = DomHelpers.getElement('textCleanup');
                if (cleanupCheckbox) {
                    cleanupCheckbox.checked = prefs.textCleanup;
                }
            }
            if (prefs.bilingualMode !== undefined) {
                const bilingualCheckbox = DomHelpers.getElement('bilingualMode');
                if (bilingualCheckbox) {
                    bilingualCheckbox.checked = prefs.bilingualMode;
                }
            }
            if (prefs.plainTextMode !== undefined) {
                const plainTextCheckbox = DomHelpers.getElement('plainTextMode');
                if (plainTextCheckbox) {
                    plainTextCheckbox.checked = prefs.plainTextMode;
                }
            }

            if (prefs.customInstructionFile) {
                // No-op while the select is still empty; the
                // 'customInstructionsLoaded' listener applies it later.
                window.__pendingCustomInstructionSelection = prefs.customInstructionFile;
                this.applyPendingCustomInstructionSelection();
            }
        } finally {
            isInitializing = wasInit;
        }
    },

    /**
     * Push the synced subset to the server (fire and forget)
     *
     * PUT is a full replacement, so the complete subset is always sent.
     * @param {Object} prefs - Preferences to mirror (filtered by _pickSynced)
     * @private
     */
    _pushSyncedPreferences(prefs) {
        ApiClient.savePreferences(this._pickSynced(prefs)).catch(() => {
            // Best effort: the local cache already holds the value
        });
    },

    /**
     * Set language in select/custom input
     * @private
     */
    _setLanguage(selectId, customInputId, value) {
        const select = DomHelpers.getElement(selectId);
        const customInput = DomHelpers.getElement(customInputId);

        if (!select) return;

        // Check if value exists in options (excluding "Other" which is just a placeholder)
        let found = false;
        for (let option of select.options) {
            // Skip "Other" option - we only want to match actual language values
            if (option.value === 'Other') continue;

            if (option.value.toLowerCase() === value.toLowerCase()) {
                select.value = option.value;
                found = true;
                break;
            }
        }

        // If language is not in the predefined list, use "Other" and fill custom input
        if (!found && customInput) {
            select.value = 'Other';
            customInput.value = value;
            // Show the custom input - need both class removal AND style change
            // because HTML has inline style="display: none"
            customInput.classList.remove('hidden');
            customInput.style.display = 'block';
        }
    },

    /**
     * Save current form state to local preferences
     * @param {Object} [options] - Options
     * @param {boolean} [options.remote=true] - Also mirror the synced subset to
     *   the server. Disabled on `beforeunload`, where a fetch is unreliable.
     */
    saveCurrentState({ remote = true } = {}) {
        const ttsEnabledCheckbox = DomHelpers.getElement('ttsEnabled');
        const textCleanupCheckbox = DomHelpers.getElement('textCleanup');
        const bilingualModeCheckbox = DomHelpers.getElement('bilingualMode');
        const plainTextModeCheckbox = DomHelpers.getElement('plainTextMode');

        const prefs = {
            lastProvider: DomHelpers.getValue('llmProvider'),
            lastModel: DomHelpers.getValue('model'),
            lastSourceLanguage: this._getLanguageValue('sourceLang', 'customSourceLang'),
            lastTargetLanguage: this._getLanguageValue('targetLang', 'customTargetLang'),
            lastApiEndpoint: DomHelpers.getValue('apiEndpoint'),
            lastOpenaiEndpoint: DomHelpers.getValue('openaiEndpoint'),
            outputFilenamePattern: DomHelpers.getValue('outputFilenamePattern'),
            ttsEnabled: ttsEnabledCheckbox ? ttsEnabledCheckbox.checked : false,
            textCleanup: textCleanupCheckbox ? textCleanupCheckbox.checked : false,
            bilingualMode: bilingualModeCheckbox ? bilingualModeCheckbox.checked : false,
            plainTextMode: plainTextModeCheckbox ? plainTextModeCheckbox.checked : false,
            customInstructionFile: DomHelpers.getValue('customInstructionSelect') || ''
        };

        this.saveLocalPreferences(prefs);

        if (remote) {
            this._pushSyncedPreferences(prefs);
        }
    },

    /**
     * Get language value from select or custom input
     * @private
     */
    _getLanguageValue(selectId, customInputId) {
        const selectVal = DomHelpers.getValue(selectId);
        if (selectVal === 'Other') {
            return DomHelpers.getValue(customInputId) || selectVal;
        }
        return selectVal;
    },

    /**
     * Save all current settings (both local and to .env)
     * @param {boolean} includeApiKeys - Whether to save API keys to .env
     * @returns {Promise<Object>} Result with success status
     */
    async saveAllSettings(includeApiKeys = false) {
        // Save local preferences
        this.saveCurrentState();

        if (includeApiKeys) {
            // Collect API keys to save
            const envSettings = {};
            const provider = DomHelpers.getValue('llmProvider');

            if (provider === 'gemini') {
                const key = DomHelpers.getValue('geminiApiKey');
                if (key) envSettings['GEMINI_API_KEY'] = key;
            } else if (provider === 'openai') {
                const key = DomHelpers.getValue('openaiApiKey');
                if (key) envSettings['OPENAI_API_KEY'] = key;
            } else if (provider === 'openrouter') {
                const key = DomHelpers.getValue('openrouterApiKey');
                if (key) envSettings['OPENROUTER_API_KEY'] = key;
            } else if (provider === 'mistral') {
                const key = DomHelpers.getValue('mistralApiKey');
                if (key) envSettings['MISTRAL_API_KEY'] = key;
            } else if (provider === 'deepseek') {
                const key = DomHelpers.getValue('deepseekApiKey');
                if (key) envSettings['DEEPSEEK_API_KEY'] = key;
            } else if (provider === 'poe') {
                const key = DomHelpers.getValue('poeApiKey');
                if (key) envSettings['POE_API_KEY'] = key;
            } else if (provider === 'nim') {
                const key = DomHelpers.getValue('nimApiKey');
                if (key) envSettings['NIM_API_KEY'] = key;
            }

            // Save endpoints to .env
            const ollamaEndpoint = DomHelpers.getValue('apiEndpoint');
            const openaiEndpoint = DomHelpers.getValue('openaiEndpoint');
            if (ollamaEndpoint) {
                envSettings['OLLAMA_API_ENDPOINT'] = ollamaEndpoint;
            }
            if (openaiEndpoint) {
                envSettings['OPENAI_API_ENDPOINT'] = openaiEndpoint;
            }

            // Save output filename pattern (naming convention)
            const filenamePattern = DomHelpers.getValue('outputFilenamePattern');
            if (filenamePattern) {
                envSettings['OUTPUT_FILENAME_PATTERN'] = filenamePattern;
            }

            // Save disable auto-pause flag (runtime behavior default)
            const disableAutoPauseCheckbox = DomHelpers.getElement('disableAutoPause');
            envSettings['DISABLE_AUTO_PAUSE'] = (disableAutoPauseCheckbox && disableAutoPauseCheckbox.checked) ? 'true' : 'false';

            // Save parallel-requests default (the per-job request still overrides
            // this; it only seeds the input on next load). Backend clamps it.
            const parallelWorkersInput = DomHelpers.getElement('parallelWorkers');
            if (parallelWorkersInput) {
                const pw = parseInt(parallelWorkersInput.value, 10);
                envSettings['PARALLEL_TRANSLATIONS'] = String(Number.isFinite(pw) && pw > 0 ? pw : 1);
            }

            // Save chunker budget. Backend floors it at MIN_CHUNK_SIZE_TOKENS
            // and applies it to new jobs without a restart (reloadable setting).
            const maxTokensInput = DomHelpers.getElement('maxTokensPerChunk');
            if (maxTokensInput) {
                const mt = parseInt(maxTokensInput.value, 10);
                if (Number.isFinite(mt) && mt > 0) {
                    envSettings['MAX_TOKENS_PER_CHUNK'] = String(mt);
                }
            }

            // Webhook notifications — always serialized (even empty) so the user
            // can disable notifications by clearing the URL and clicking Save.
            const notifyUrl = DomHelpers.getElement('notifyWebhookUrl');
            if (notifyUrl) {
                envSettings['NOTIFY_WEBHOOK_URL'] = notifyUrl.value.trim();
                envSettings['NOTIFY_WEBHOOK_METHOD'] = DomHelpers.getValue('notifyWebhookMethod') || 'POST';
                envSettings['NOTIFY_WEBHOOK_HEADERS'] = (DomHelpers.getValue('notifyWebhookHeaders') || '').trim();
                envSettings['NOTIFY_WEBHOOK_PAYLOAD'] = (DomHelpers.getValue('notifyWebhookPayload') || '').trim();
                const onSuccess = DomHelpers.getElement('notifyOnSuccess');
                const onFailure = DomHelpers.getElement('notifyOnFailure');
                const onInterruption = DomHelpers.getElement('notifyOnInterruption');
                envSettings['NOTIFY_ON_SUCCESS'] = (onSuccess && onSuccess.checked) ? 'true' : 'false';
                envSettings['NOTIFY_ON_FAILURE'] = (onFailure && onFailure.checked) ? 'true' : 'false';
                envSettings['NOTIFY_ON_INTERRUPTION'] = (onInterruption && onInterruption.checked) ? 'true' : 'false';
                const timeoutRaw = DomHelpers.getValue('notifyTimeoutSeconds');
                const timeoutNum = parseInt(timeoutRaw, 10);
                envSettings['NOTIFY_TIMEOUT_SECONDS'] = Number.isFinite(timeoutNum) && timeoutNum > 0 ? String(timeoutNum) : '5';
            }

            // Also save provider and model as defaults
            envSettings['LLM_PROVIDER'] = provider;
            const model = DomHelpers.getValue('model');
            if (model) {
                // Save to provider-specific model variable
                if (provider === 'openrouter') {
                    envSettings['OPENROUTER_MODEL'] = model;
                } else if (provider === 'gemini') {
                    envSettings['GEMINI_MODEL'] = model;
                } else if (provider === 'mistral') {
                    envSettings['MISTRAL_MODEL'] = model;
                } else if (provider === 'deepseek') {
                    envSettings['DEEPSEEK_MODEL'] = model;
                } else if (provider === 'poe') {
                    envSettings['POE_MODEL'] = model;
                } else if (provider === 'nim') {
                    envSettings['NIM_MODEL'] = model;
                } else {
                    // Ollama and OpenAI use DEFAULT_MODEL
                    envSettings['DEFAULT_MODEL'] = model;
                }
            }

            // Languages are no longer saved to .env - they are:
            // - Source: auto-detected from file content
            // - Target: auto-detected from browser language per session

            if (Object.keys(envSettings).length > 0) {
                try {
                    const result = await ApiClient.saveSettings(envSettings);
                    // Reset the lock since user explicitly saved their choice
                    this.resetEnvModelApplied();
                    this._clearEnvDirty();
                    return { success: true, savedToEnv: result.saved_keys };
                } catch (e) {
                    return { success: false, error: e.message };
                }
            }
        }

        return { success: true, savedToEnv: [] };
    },

    /**
     * Apply pending model selection after models are loaded
     * Called by provider-manager after loading models
     */
    applyPendingModelSelection() {
        // Don't apply localStorage preference if .env model was already applied
        if (envModelApplied) {
            delete window.__pendingModelSelection;
            return;
        }

        if (window.__pendingModelSelection) {
            const modelSelect = DomHelpers.getElement('model');
            if (modelSelect && modelSelect.options.length > 0) {
                // Check if the model exists in options
                let found = false;
                for (let option of modelSelect.options) {
                    if (option.value === window.__pendingModelSelection) {
                        modelSelect.value = window.__pendingModelSelection;
                        found = true;
                        break;
                    }
                }
                if (found) {
                    delete window.__pendingModelSelection;
                }
            }
        }
    },

    /**
     * Apply pending custom instruction selection after custom instructions are loaded
     * Called when 'customInstructionsLoaded' event is fired
     */
    applyPendingCustomInstructionSelection() {
        if (window.__pendingCustomInstructionSelection) {
            const select = DomHelpers.getElement('customInstructionSelect');
            if (select && select.options.length > 0) {
                // Check if the value exists in options
                let found = false;
                for (let option of select.options) {
                    if (option.value === window.__pendingCustomInstructionSelection) {
                        select.value = window.__pendingCustomInstructionSelection;
                        found = true;
                        console.log('[SettingsManager] Restored custom instruction:', window.__pendingCustomInstructionSelection);
                        break;
                    }
                }
                if (!found) {
                    console.warn('[SettingsManager] Custom instruction not found:', window.__pendingCustomInstructionSelection);
                }
                delete window.__pendingCustomInstructionSelection;
            }
        }
    },

    /**
     * Mark that the .env default model has been applied
     * This prevents localStorage from overriding it
     */
    markEnvModelApplied() {
        envModelApplied = true;
    },

    /**
     * Reset the envModelApplied flag
     * Called after user explicitly saves settings to .env
     */
    resetEnvModelApplied() {
        envModelApplied = false;
    },

    /**
     * Check if .env model was already applied
     * @returns {boolean}
     */
    isEnvModelApplied() {
        return envModelApplied;
    },

    /**
     * Mark an endpoint as customized by the user
     * @param {string} endpointType - 'ollama' or 'openai'
     */
    markEndpointCustomized(endpointType) {
        const key = endpointType === 'openai' ? 'openaiEndpointCustomized' : 'apiEndpointCustomized';
        this.saveLocalPreferences({ [key]: true });
        this.updateEndpointBadge(endpointType, true);
    },

    /**
     * Check if an endpoint was customized by the user
     * @param {string} endpointType - 'ollama' or 'openai'
     * @returns {boolean}
     */
    isEndpointCustomized(endpointType) {
        const prefs = this.getLocalPreferences();
        return endpointType === 'openai' 
            ? prefs.openaiEndpointCustomized 
            : prefs.apiEndpointCustomized;
    },

    /**
     * Reset endpoint to server default (.env value)
     * @param {string} endpointType - 'ollama' or 'openai'
     * @param {string} serverValue - The value from server config
     */
    resetEndpointToServerDefault(endpointType, serverValue) {
        const inputId = endpointType === 'openai' ? 'openaiEndpoint' : 'apiEndpoint';
        const key = endpointType === 'openai' ? 'openaiEndpointCustomized' : 'apiEndpointCustomized';
        const storageKey = endpointType === 'openai' ? 'lastOpenaiEndpoint' : 'lastApiEndpoint';
        
        // Update input field
        DomHelpers.setValue(inputId, serverValue);
        
        // Clear customized flag
        const prefs = this.getLocalPreferences();
        delete prefs[key];
        delete prefs[storageKey];
        this.saveLocalPreferences(prefs);
        
        // Update badge
        this.updateEndpointBadge(endpointType, false);
        
        // Reload models with new endpoint
        const currentProvider = DomHelpers.getValue('llmProvider');
        if (currentProvider === endpointType || (endpointType === 'ollama' && currentProvider === 'ollama')) {
            window.dispatchEvent(new Event('endpointReset'));
        }
        
        MessageLogger.addLog(`↺ ${t('common:endpoint_reset_log')}`);
    },

    /**
     * Update the visual badge for endpoint customization
     * @param {string} endpointType - 'ollama' or 'openai'
     * @param {boolean} isCustomized - Whether the endpoint is customized
     */
    updateEndpointBadge(endpointType, isCustomized) {
        const badgeId = endpointType === 'openai' ? 'openaiEndpointBadge' : 'apiEndpointBadge';
        const badge = DomHelpers.getElement(badgeId);
        if (badge) {
            badge.style.display = isCustomized ? 'inline-block' : 'none';
        }
        
        // Also show/hide the reset button
        const resetBtnId = endpointType === 'openai' ? 'resetOpenaiEndpointBtn' : 'resetApiEndpointBtn';
        const resetBtn = DomHelpers.getElement(resetBtnId);
        if (resetBtn) {
            resetBtn.style.display = isCustomized ? 'inline-flex' : 'none';
        }
    },

    /**
     * Initialize endpoint badges on page load
     * Call this after server config is loaded
     * @param {Object} serverConfig - The config from /api/config
     */
    initializeEndpointBadges(serverConfig) {
        const prefs = this.getLocalPreferences();
        
        // Check Ollama endpoint
        if (prefs.apiEndpointCustomized && prefs.lastApiEndpoint) {
            const serverEndpoint = serverConfig.ollama_api_endpoint || serverConfig.api_endpoint;
            if (prefs.lastApiEndpoint !== serverEndpoint) {
                this.updateEndpointBadge('ollama', true);
            }
        }
        
        // Check OpenAI endpoint
        if (prefs.openaiEndpointCustomized && prefs.lastOpenaiEndpoint) {
            const serverEndpoint = serverConfig.openai_api_endpoint;
            if (prefs.lastOpenaiEndpoint !== serverEndpoint) {
                this.updateEndpointBadge('openai', true);
            }
        }
    }
};

// Auto-save preferences when leaving page (local only: a fetch fired during
// unload is unreliable, and the debounced auto-save already pushed the change)
if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', () => {
        SettingsManager.saveCurrentState({ remote: false });
    });
}
