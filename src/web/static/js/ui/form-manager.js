/**
 * Form Manager - Form configuration and settings management
 *
 * Handles form state, custom language toggles, advanced settings,
 * default configuration loading, and form reset functionality.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from './dom-helpers.js';
import { MessageLogger } from './message-logger.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import { TranslationTracker } from '../translation/translation-tracker.js';

/**
 * Set default language in select/input
 * @param {string} selectId - Select element ID
 * @param {string} customInputId - Custom input element ID
 * @param {string} defaultLanguage - Default language value
 * @param {boolean} [forceOverwrite=false] - If true, overwrite even if "Other" is selected with a value
 */
function setDefaultLanguage(selectId, customInputId, defaultLanguage, forceOverwrite = false) {
    const select = DomHelpers.getElement(selectId);
    const customInput = DomHelpers.getElement(customInputId);
    const containerId = customInputId + 'Container';
    const container = DomHelpers.getElement(containerId);

    if (!select || !customInput) return;

    // Don't overwrite if "Other" is already selected with a custom value (restored from file)
    // This preserves custom languages across page reloads
    if (!forceOverwrite && select.value === 'Other' && customInput.value.trim()) {
        // Keep the existing "Other" selection and show the container
        if (container) container.style.display = 'block';
        return;
    }

    // Check if the default language is in the dropdown options (excluding "Other")
    let languageFound = false;
    for (let option of select.options) {
        // Skip "Other" option - we only want to match actual language values
        if (option.value === 'Other') continue;

        if (option.value.toLowerCase() === defaultLanguage.toLowerCase()) {
            select.value = option.value;
            languageFound = true;
            if (container) container.style.display = 'none';
            break;
        }
    }

    // If language not found in dropdown, use "Other" and set custom input
    if (!languageFound) {
        select.value = 'Other';
        customInput.value = defaultLanguage;
        // Show the container (not just the input)
        if (container) container.style.display = 'block';
    }
}

export const FormManager = {
    /**
     * Initialize form manager
     */
    initialize() {
        this.setupEventListeners();
        this.loadDefaultConfig();
    },

    /**
     * Set up event listeners for form elements
     */
    setupEventListeners() {
        // Source language change
        const sourceLang = DomHelpers.getElement('sourceLang');
        if (sourceLang) {
            sourceLang.addEventListener('change', (e) => {
                this.checkCustomSourceLanguage(e.target);
            });
        }

        // Target language change
        const targetLang = DomHelpers.getElement('targetLang');
        if (targetLang) {
            targetLang.addEventListener('change', (e) => {
                this.checkCustomTargetLanguage(e.target);
            });
        }

        // TTS enabled checkbox
        const ttsEnabled = DomHelpers.getElement('ttsEnabled');
        if (ttsEnabled) {
            ttsEnabled.addEventListener('change', (e) => {
                this.handleTtsToggle(e.target.checked);
            });
        }

        // Prompt options checkboxes - keep section open if any is checked
        const textCleanup = DomHelpers.getElement('textCleanup');
        const refineTranslation = DomHelpers.getElement('refineTranslation');

        [textCleanup, refineTranslation].forEach(checkbox => {
            if (checkbox) {
                checkbox.addEventListener('change', () => {
                    this.handlePromptOptionChange();
                });
            }
        });

        // Reset button
        const resetBtn = DomHelpers.getElement('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.resetForm();
            });
        }
    },

    /**
     * Check if custom source language input should be shown
     * @param {HTMLSelectElement} selectElement - Source language select element
     */
    checkCustomSourceLanguage(selectElement) {
        const container = DomHelpers.getElement('customSourceLangContainer');
        const customLangInput = DomHelpers.getElement('customSourceLang');
        if (!container || !customLangInput) return;

        if (selectElement.value === 'Other') {
            container.style.display = 'block';
            customLangInput.focus();
        } else {
            container.style.display = 'none';
        }
    },

    /**
     * Check if custom target language input should be shown
     * @param {HTMLSelectElement} selectElement - Target language select element
     */
    checkCustomTargetLanguage(selectElement) {
        const container = DomHelpers.getElement('customTargetLangContainer');
        const customLangInput = DomHelpers.getElement('customTargetLang');
        if (!container || !customLangInput) return;

        if (selectElement.value === 'Other') {
            container.style.display = 'block';
            customLangInput.focus();
        } else {
            container.style.display = 'none';
        }
    },


    /**
     * Toggle settings options panel
     */
    toggleSettingsOptions() {
        const section = DomHelpers.getElement('settingsOptionsSection');
        const icon = DomHelpers.getElement('settingsOptionsIcon');

        if (!section || !icon) return;

        const isHidden = section.classList.toggle('hidden');
        icon.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';

        // Update state
        StateManager.setState('ui.isSettingsOptionsOpen', !isHidden);
    },

    /**
     * Toggle prompt options panel
     */
    togglePromptOptions() {
        const section = DomHelpers.getElement('promptOptionsSection');
        const icon = DomHelpers.getElement('promptOptionsIcon');

        if (!section || !icon) return;

        const isHidden = section.classList.toggle('hidden');
        icon.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';

        // Update state
        StateManager.setState('ui.isPromptOptionsOpen', !isHidden);
    },

    /**
     * Toggle activity log panel
     */
    toggleActivityLog() {
        const section = DomHelpers.getElement('activityLogSection');
        const icon = DomHelpers.getElement('activityLogIcon');

        if (!section || !icon) return;

        const isHidden = section.classList.toggle('hidden');
        icon.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';

        // Update state
        StateManager.setState('ui.isActivityLogOpen', !isHidden);
    },

    /**
     * Handle prompt option checkbox change - keep section open if any option is checked
     */
    handlePromptOptionChange() {
        const textCleanup = DomHelpers.getElement('textCleanup');
        const refineTranslation = DomHelpers.getElement('refineTranslation');

        const anyChecked = (textCleanup?.checked || refineTranslation?.checked);

        if (anyChecked) {
            const section = DomHelpers.getElement('promptOptionsSection');
            const icon = DomHelpers.getElement('promptOptionsIcon');

            if (section && section.classList.contains('hidden')) {
                section.classList.remove('hidden');
                if (icon) {
                    icon.style.transform = 'rotate(180deg)';
                }
                StateManager.setState('ui.isPromptOptionsOpen', true);
            }
        }
    },

    /**
     * Handle TTS toggle
     * @param {boolean} isChecked - Whether TTS is enabled
     */
    handleTtsToggle(isChecked) {
        const ttsOptions = DomHelpers.getElement('ttsOptions');

        if (ttsOptions) {
            if (isChecked) {
                ttsOptions.style.display = 'block';
            } else {
                ttsOptions.style.display = 'none';
            }
        }

        // Dispatch event for other components
        window.dispatchEvent(new CustomEvent('ttsChanged', { detail: { enabled: isChecked } }));
    },

    /**
     * Detect browser language and map to full language name
     * @returns {string} Full language name (e.g., "French", "English")
     */
    detectBrowserLanguage() {
        // Get browser language (e.g., "fr-FR", "en-US", "zh-CN")
        const browserLang = navigator.language || navigator.userLanguage || 'en';
        const langCode = browserLang.split('-')[0].toLowerCase();

        // Map language codes to full names used in the UI
        const languageMap = {
            'en': 'English',
            'zh': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ja': 'Japanese',
            'ko': 'Korean',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'it': 'Italian',
            'nl': 'Dutch',
            'pl': 'Polish',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'fi': 'Finnish',
            'el': 'Greek',
            'hu': 'Hungarian',
            'cs': 'Czech',
            'sk': 'Slovak',
            'ro': 'Romanian',
            'bg': 'Bulgarian',
            'hr': 'Croatian',
            'sr': 'Serbian',
            'uk': 'Ukrainian',
            'ca': 'Catalan',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ur': 'Urdu',
            'pa': 'Punjabi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'vi': 'Vietnamese',
            'th': 'Thai',
            'id': 'Indonesian',
            'ms': 'Malay',
            'tl': 'Tagalog',
            'my': 'Burmese',
            'fa': 'Persian',
            'tr': 'Turkish',
            'he': 'Hebrew',
            'sw': 'Swahili',
            'am': 'Amharic'
        };

        return languageMap[langCode] || 'English'; // Default to English if not found
    },

    /**
     * Load default configuration from server
     */
    async loadDefaultConfig() {
        try {
            const config = await ApiClient.getConfig();

            // Detect browser language for target language (no default from .env)
            const browserLanguage = this.detectBrowserLanguage();

            // Set target language from browser detection
            setDefaultLanguage('targetLang', 'customTargetLang', browserLanguage);

            // Source language: preserve "Other" + custom value if already set (restored from file)
            // Pass empty string as default - setDefaultLanguage will keep existing "Other" selection
            setDefaultLanguage('sourceLang', 'customSourceLang', '')

            // Set other configuration values
            if (config.api_endpoint) {
                DomHelpers.setValue('apiEndpoint', config.api_endpoint);
            }
            // Handle API keys - show indicator if configured in .env, otherwise keep placeholder
            ApiKeyUtils.setupField('geminiApiKey', config.gemini_api_key_configured, config.gemini_api_key);
            ApiKeyUtils.setupField('openaiApiKey', config.openai_api_key_configured, config.openai_api_key);
            ApiKeyUtils.setupField('openrouterApiKey', config.openrouter_api_key_configured, config.openrouter_api_key);
            ApiKeyUtils.setupField('mistralApiKey', config.mistral_api_key_configured, config.mistral_api_key);
            ApiKeyUtils.setupField('deepseekApiKey', config.deepseek_api_key_configured, config.deepseek_api_key);

            // Store in state
            StateManager.setState('ui.defaultConfig', config);

            // After loading defaults, dispatch event to sync any pending file languages
            // This ensures restored file languages override browser-detected defaults
            window.dispatchEvent(new CustomEvent('defaultConfigLoaded'));

        } catch {
            MessageLogger.showMessage('Failed to load default configuration', 'warning');
            // Still dispatch event even on error so file languages can be synced
            window.dispatchEvent(new CustomEvent('defaultConfigLoaded'));
        }
    },

    /**
     * Reset form to default state
     */
    async resetForm() {
        // Get current files to process
        const filesToProcess = StateManager.getState('files.toProcess');
        const currentJob = StateManager.getState('translation.currentJob');
        const isBatchActive = StateManager.getState('translation.isBatchActive');

        // First, interrupt current translation if active
        if (currentJob && currentJob.translationId && isBatchActive) {
            MessageLogger.addLog("🛑 Interrupting current translation before clearing files...");
            try {
                await ApiClient.interruptTranslation(currentJob.translationId);
            } catch {
                // Interrupt failed
            }
        }

        // Collect file paths to delete from server
        const uploadedFilePaths = filesToProcess
            .filter(file => file.filePath)
            .map(file => file.filePath);

        // Clear client-side state
        StateManager.setState('files.toProcess', []);
        StateManager.setState('translation.currentJob', null);
        StateManager.setState('translation.isBatchActive', false);

        // Clear saved translation state from localStorage
        if (TranslationTracker && TranslationTracker.clearTranslationState) {
            TranslationTracker.clearTranslationState();
        }

        // Reset file input
        DomHelpers.setValue('fileInput', '');

        // Hide progress section
        DomHelpers.hide('progressSection');

        // Reset buttons
        DomHelpers.setText('translateBtn', '▶️ Start Translation Batch');
        DomHelpers.setDisabled('translateBtn', true);
        DomHelpers.hide('interruptBtn');
        DomHelpers.setDisabled('interruptBtn', false);

        // Reset language selectors
        const sourceContainer = DomHelpers.getElement('customSourceLangContainer');
        const targetContainer = DomHelpers.getElement('customTargetLangContainer');
        if (sourceContainer) sourceContainer.style.display = 'none';
        if (targetContainer) targetContainer.style.display = 'none';
        DomHelpers.getElement('sourceLang').selectedIndex = 0;
        DomHelpers.getElement('targetLang').selectedIndex = 0;

        // Reset stats and progress
        DomHelpers.show('statsGrid');
        this.updateProgress(0);
        MessageLogger.showMessage('', '');

        // Delete uploaded files from server
        if (uploadedFilePaths.length > 0) {
            MessageLogger.addLog(`🗑️ Deleting ${uploadedFilePaths.length} uploaded file(s) from server...`);
            try {
                const result = await ApiClient.clearUploads(uploadedFilePaths);

                MessageLogger.addLog(`✅ Successfully deleted ${result.total_deleted} uploaded file(s).`);
                if (result.failed && result.failed.length > 0) {
                    MessageLogger.addLog(`⚠️ Failed to delete ${result.failed.length} file(s).`);
                }
            } catch {
                MessageLogger.addLog("⚠️ Error occurred while deleting uploaded files.");
            }
        }

        MessageLogger.addLog("Form and file list reset.");

        // Trigger UI update
        window.dispatchEvent(new CustomEvent('formReset'));
    },

    /**
     * Update progress bar
     * @param {number} percent - Progress percentage (0-100)
     */
    updateProgress(percent) {
        const progressBar = DomHelpers.getElement('progressBar');
        if (!progressBar) return;

        progressBar.style.width = percent + '%';
        DomHelpers.setText(progressBar, Math.round(percent) + '%');
    },

    /**
     * Get form configuration for translation
     * @returns {Object} Translation configuration object
     */
    getTranslationConfig() {
        // Get source language
        let sourceLanguageVal = DomHelpers.getValue('sourceLang');
        if (sourceLanguageVal === 'Other') {
            sourceLanguageVal = DomHelpers.getValue('customSourceLang').trim();
        }

        // Get target language
        let targetLanguageVal = DomHelpers.getValue('targetLang');
        if (targetLanguageVal === 'Other') {
            targetLanguageVal = DomHelpers.getValue('customTargetLang').trim();
        }

        // Get provider and model
        const provider = DomHelpers.getValue('llmProvider');
        const model = DomHelpers.getValue('model');

        // Get API endpoint based on provider
        let apiEndpoint;
        if (provider === 'openai') {
            apiEndpoint = DomHelpers.getValue('openaiEndpoint');
        } else {
            apiEndpoint = DomHelpers.getValue('apiEndpoint');
        }

        // Get API keys - use helper to handle .env configured keys
        const geminiApiKey = provider === 'gemini' ? ApiKeyUtils.getValue('geminiApiKey') : '';
        const openaiApiKey = provider === 'openai' ? ApiKeyUtils.getValue('openaiApiKey') : '';
        const openrouterApiKey = provider === 'openrouter' ? ApiKeyUtils.getValue('openrouterApiKey') : '';

        // Get TTS configuration
        const ttsEnabled = DomHelpers.getElement('ttsEnabled')?.checked || false;

        return {
            source_language: sourceLanguageVal,
            target_language: targetLanguageVal,
            model: model,
            llm_api_endpoint: apiEndpoint,
            llm_provider: provider,
            gemini_api_key: geminiApiKey,
            openai_api_key: openaiApiKey,
            openrouter_api_key: openrouterApiKey,
            // Prompt options (optional system prompt instructions)
            // Technical content protection is always enabled
            prompt_options: {
                preserve_technical_content: true,
                text_cleanup: DomHelpers.getElement('textCleanup')?.checked || false,
                refine: DomHelpers.getElement('refineTranslation')?.checked || false
            },
            // Bilingual output (original + translation interleaved)
            bilingual_output: DomHelpers.getElement('bilingualMode')?.checked || false,
            // TTS configuration
            tts_enabled: ttsEnabled,
            tts_voice: ttsEnabled ? (DomHelpers.getValue('ttsVoice') || '') : '',
            tts_rate: ttsEnabled ? (DomHelpers.getValue('ttsRate') || '+0%') : '+0%',
            tts_format: ttsEnabled ? (DomHelpers.getValue('ttsFormat') || 'opus') : 'opus',
            tts_bitrate: ttsEnabled ? (DomHelpers.getValue('ttsBitrate') || '64k') : '64k'
        };
    },

    /**
     * Validate form configuration
     * @returns {Object} { valid: boolean, message: string }
     */
    validateConfig() {
        const config = this.getTranslationConfig();

        if (!config.source_language) {
            return { valid: false, message: 'Please specify the source language.' };
        }

        if (!config.target_language) {
            return { valid: false, message: 'Please specify the target language.' };
        }

        if (!config.model) {
            return { valid: false, message: 'Please select an LLM model.' };
        }

        if (!config.llm_api_endpoint) {
            return { valid: false, message: 'API endpoint cannot be empty.' };
        }

        // Validate API keys for cloud providers using shared utility
        const apiKeyValidation = ApiKeyUtils.validateForProvider(config.llm_provider, config.llm_api_endpoint);
        if (!apiKeyValidation.valid) {
            return apiKeyValidation;
        }

        return { valid: true, message: '' };
    }
};
