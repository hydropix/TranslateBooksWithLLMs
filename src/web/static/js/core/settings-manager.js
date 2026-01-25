/**
 * Settings Manager - User preferences persistence
 *
 * Handles saving/loading user preferences via:
 * 1. localStorage for quick preferences (last model, provider, languages)
 * 2. Server API for sensitive data (API keys saved to .env)
 */

import { ApiClient } from './api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { MessageLogger } from '../ui/message-logger.js';

// Storage configuration with versioning
const STORAGE_VERSION = 1;
const STORAGE_KEY_PREFIX = 'tbl_user_preferences';
const STORAGE_KEY = `${STORAGE_KEY_PREFIX}_v${STORAGE_VERSION}`;

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
    if ('refineTranslation' in data && typeof data.refineTranslation !== 'boolean') return false;

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

/**
 * Settings that are saved to localStorage (non-sensitive)
 */
const LOCAL_SETTINGS = [
    'lastProvider',
    'lastModel',
    'lastSourceLanguage',
    'lastTargetLanguage',
    'lastApiEndpoint',
    'lastOpenaiEndpoint',
    'ttsEnabled',
    'textCleanup',
    'refineTranslation'
];

/**
 * Settings that are saved to .env via API (sensitive)
 */
const ENV_SETTINGS_MAP = {
    'geminiApiKey': 'GEMINI_API_KEY',
    'openaiApiKey': 'OPENAI_API_KEY',
    'openrouterApiKey': 'OPENROUTER_API_KEY',
    'mistralApiKey': 'MISTRAL_API_KEY',
    'deepseekApiKey': 'DEEPSEEK_API_KEY'
};

export const SettingsManager = {
    /**
     * Initialize settings manager - load saved preferences and setup auto-save
     */
    initialize() {
        // Clean up old storage versions
        this.cleanupOldStorageVersions();

        this.loadLocalPreferences();
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
        // Elements that trigger auto-save
        const autoSaveElements = [
            // Provider and model
            { id: 'llmProvider', event: 'change' },
            { id: 'model', event: 'change' },
            // API endpoints
            { id: 'apiEndpoint', event: 'change' },
            { id: 'openaiEndpoint', event: 'change' },
            // API keys (save to .env)
            { id: 'geminiApiKey', event: 'change' },
            { id: 'openaiApiKey', event: 'change' },
            { id: 'openrouterApiKey', event: 'change' },
            { id: 'mistralApiKey', event: 'change' },
            { id: 'deepseekApiKey', event: 'change' },
            // Languages
            { id: 'sourceLang', event: 'change' },
            { id: 'targetLang', event: 'change' },
            { id: 'customSourceLang', event: 'change' },
            { id: 'customTargetLang', event: 'change' },
            // Checkboxes
            { id: 'ttsEnabled', event: 'change' },
            { id: 'textCleanup', event: 'change' },
            { id: 'refineTranslation', event: 'change' }
        ];

        autoSaveElements.forEach(({ id, event }) => {
            const element = DomHelpers.getElement(id);
            if (element) {
                element.addEventListener(event, () => this._triggerAutoSave());
            }
        });

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
            await this.saveAllSettings(true);
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
            MessageLogger.addLog('⚠️ Could not load saved preferences');
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
                MessageLogger.addLog('⚠️ Browser storage full, could not save preferences');
            } else {
                MessageLogger.addLog('⚠️ Failed to save preferences');
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

        // Apply last provider AFTER endpoints are set
        // This triggers model loading with the correct endpoint
        if (prefs.lastProvider) {
            const providerSelect = DomHelpers.getElement('llmProvider');
            if (providerSelect) {
                providerSelect.value = prefs.lastProvider;
                // Trigger change event to show correct settings panel and load models
                providerSelect.dispatchEvent(new Event('change'));
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
        if (prefs.refineTranslation !== undefined) {
            const refineCheckbox = DomHelpers.getElement('refineTranslation');
            if (refineCheckbox) {
                refineCheckbox.checked = prefs.refineTranslation;
            }
        }

        // Keep Prompt Options section open if any option is checked
        const hasAnyPromptOption = prefs.textCleanup || prefs.refineTranslation;
        if (hasAnyPromptOption) {
            const promptOptionsSection = DomHelpers.getElement('promptOptionsSection');
            const promptOptionsIcon = DomHelpers.getElement('promptOptionsIcon');
            if (promptOptionsSection) {
                promptOptionsSection.classList.remove('hidden');
            }
            if (promptOptionsIcon) {
                promptOptionsIcon.style.transform = 'rotate(180deg)';
            }
        }
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
     */
    saveCurrentState() {
        const ttsEnabledCheckbox = DomHelpers.getElement('ttsEnabled');
        const textCleanupCheckbox = DomHelpers.getElement('textCleanup');
        const refineTranslationCheckbox = DomHelpers.getElement('refineTranslation');

        const prefs = {
            lastProvider: DomHelpers.getValue('llmProvider'),
            lastModel: DomHelpers.getValue('model'),
            lastSourceLanguage: this._getLanguageValue('sourceLang', 'customSourceLang'),
            lastTargetLanguage: this._getLanguageValue('targetLang', 'customTargetLang'),
            lastApiEndpoint: DomHelpers.getValue('apiEndpoint'),
            lastOpenaiEndpoint: DomHelpers.getValue('openaiEndpoint'),
            ttsEnabled: ttsEnabledCheckbox ? ttsEnabledCheckbox.checked : false,
            textCleanup: textCleanupCheckbox ? textCleanupCheckbox.checked : false,
            refineTranslation: refineTranslationCheckbox ? refineTranslationCheckbox.checked : false
        };

        this.saveLocalPreferences(prefs);
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
     * Save API key to .env file via server
     * @param {string} provider - Provider name (gemini, openai, openrouter)
     * @param {string} apiKey - The API key to save
     * @returns {Promise<boolean>} Success status
     */
    async saveApiKey(provider, apiKey) {
        const keyMap = {
            'gemini': 'GEMINI_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY',
            'mistral': 'MISTRAL_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY'
        };

        const envKey = keyMap[provider];
        if (!envKey) {
            console.error('Unknown provider:', provider);
            return false;
        }

        try {
            const result = await ApiClient.saveSettings({ [envKey]: apiKey });
            if (result.success) {
                MessageLogger.addLog(`API key saved for ${provider}`);
                return true;
            }
            return false;
        } catch {
            return false;
        }
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
    }
};

// Auto-save preferences when leaving page
if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', () => {
        SettingsManager.saveCurrentState();
    });
}
