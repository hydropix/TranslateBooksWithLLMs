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
import { SettingsManager } from '../core/settings-manager.js';
import { t } from '../i18n/i18n.js';

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
        this.loadCustomInstructions();

        // The Styles tab (style-manager.js) owns the Custom_Instructions
        // CRUD flows; it never writes to #customInstructionSelect directly,
        // it only signals that the on-disk list changed so this dropdown
        // stays in sync without a page reload.
        window.addEventListener('customInstructionsChanged', () => this.loadCustomInstructions());
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

        // Custom instructions refresh button
        const refreshCustomInstructionsBtn = DomHelpers.getElement('refreshCustomInstructionsBtn');
        if (refreshCustomInstructionsBtn) {
            refreshCustomInstructionsBtn.addEventListener('click', () => {
                this.loadCustomInstructions();
            });
        }

        // Custom instructions manage button — jumps to the Styles tab, which owns
        // the Custom_Instructions CRUD flows. Mirrors #glossaryManageBtn, which
        // does the same for the Glossaries tab (glossary-manager.js).
        const customInstructionsManageBtn = DomHelpers.getElement('customInstructionsManageBtn');
        if (customInstructionsManageBtn) {
            customInstructionsManageBtn.addEventListener('click', () => {
                // Global rather than imported: switchTopTab lives in
                // glossary-manager.js and importing it here would close a cycle.
                if (typeof window.switchTopTab === 'function') {
                    window.switchTopTab('styles');
                }
            });
        }

        // Reset button
        const resetBtn = DomHelpers.getElement('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.resetForm();
            });
        }

        // Endpoint change listeners - detect manual modifications
        const apiEndpoint = DomHelpers.getElement('apiEndpoint');
        if (apiEndpoint) {
            apiEndpoint.addEventListener('change', () => {
                SettingsManager.markEndpointCustomized('ollama');
                console.log('[FormManager] Ollama endpoint customized by user');
            });
        }

        const openaiEndpoint = DomHelpers.getElement('openaiEndpoint');
        if (openaiEndpoint) {
            openaiEndpoint.addEventListener('change', () => {
                SettingsManager.markEndpointCustomized('openai');
                console.log('[FormManager] OpenAI endpoint customized by user');
            });
        }

        // Reset endpoint to server default buttons
        const resetApiEndpointBtn = DomHelpers.getElement('resetApiEndpointBtn');
        if (resetApiEndpointBtn) {
            resetApiEndpointBtn.addEventListener('click', () => {
                const config = StateManager.getState('ui.defaultConfig');
                const serverEndpoint = config?.ollama_api_endpoint || config?.api_endpoint;
                if (serverEndpoint) {
                    SettingsManager.resetEndpointToServerDefault('ollama', serverEndpoint);
                }
            });
        }

        const resetOpenaiEndpointBtn = DomHelpers.getElement('resetOpenaiEndpointBtn');
        if (resetOpenaiEndpointBtn) {
            resetOpenaiEndpointBtn.addEventListener('click', () => {
                const config = StateManager.getState('ui.defaultConfig');
                const serverEndpoint = config?.openai_api_endpoint;
                if (serverEndpoint) {
                    SettingsManager.resetEndpointToServerDefault('openai', serverEndpoint);
                }
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
        const fullLangCode = browserLang.toLowerCase();
        const langCode = fullLangCode.split('-')[0];

        // Map language codes to full names used in the UI
        const languageMap = {
            'en': 'English',
            'zh': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ja': 'Japanese',
            'ko': 'Korean',
            'pt': 'Portuguese (Brazil)',
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

        // Regional variant overrides: check full locale before base code.
        // Any pt-* locale other than pt-BR is closer to European Portuguese,
        // and a bare "pt" keeps the Brazilian default set in languageMap.
        if (langCode === 'pt' && fullLangCode !== 'pt' && !fullLangCode.startsWith('pt-br')) {
            return 'Portuguese (Portugal)';
        }

        return languageMap[langCode] || 'English'; // Default to English if not found
    },

    /**
     * Load default configuration from server
     */
    async loadDefaultConfig() {
        try {
            const config = await ApiClient.getConfig();

            // Store config in state first so other modules can access it
            StateManager.setState('ui.defaultConfig', config);

            // Set target language from server config if available
            // This fixes GitHub issue #108: DEFAULT_TARGET_LANGUAGE was ignored
            // Only override if server has a default target language configured
            if (config.default_target_language && config.default_target_language.trim()) {
                console.log('[FormManager] Applying DEFAULT_TARGET_LANGUAGE from server:', config.default_target_language);
                setDefaultLanguage('targetLang', 'customTargetLang', config.default_target_language);
            } else {
                console.log('[FormManager] No DEFAULT_TARGET_LANGUAGE from server, keeping current value');
            }

            // Set source language from server config if available
            const sourceLanguage = config.default_source_language && config.default_source_language.trim()
                ? config.default_source_language
                : '';  // Empty = auto-detect from file
            setDefaultLanguage('sourceLang', 'customSourceLang', sourceLanguage)

            // Set provider-specific API endpoints with smart merge:
            // - If user has customized endpoint, keep it and show badge
            // - Otherwise use server default (.env)
            // Ollama endpoint (for Ollama provider)
            const ollamaEndpoint = config.ollama_api_endpoint || config.api_endpoint;
            if (ollamaEndpoint) {
                const prefs = SettingsManager.getLocalPreferences();
                // Check if user has a customized endpoint
                if (prefs.apiEndpointCustomized && prefs.lastApiEndpoint) {
                    // User customized - keep their value but show badge
                    SettingsManager.updateEndpointBadge('ollama', true);
                    console.log('[FormManager] Using customized Ollama endpoint:', prefs.lastApiEndpoint);
                } else {
                    // Use server default
                    DomHelpers.setValue('apiEndpoint', ollamaEndpoint);
                }
            }
            
            // OpenAI endpoint (for OpenAI-compatible providers like OpenAI, LM Studio)
            if (config.openai_api_endpoint) {
                const prefs = SettingsManager.getLocalPreferences();
                // Check if user has a customized endpoint
                if (prefs.openaiEndpointCustomized && prefs.lastOpenaiEndpoint) {
                    // User customized - keep their value but show badge
                    SettingsManager.updateEndpointBadge('openai', true);
                    console.log('[FormManager] Using customized OpenAI endpoint:', prefs.lastOpenaiEndpoint);
                } else {
                    // Use server default
                    DomHelpers.setValue('openaiEndpoint', config.openai_api_endpoint);
                }
            }
            
            // Output filename pattern (naming convention)
            if (config.output_filename_pattern) {
                DomHelpers.setValue('outputFilenamePattern', config.output_filename_pattern);
            }

            // Disable auto-pause on rate limit (runtime behavior default)
            if (typeof config.disable_auto_pause === 'boolean') {
                const disableAutoPauseCheckbox = DomHelpers.getElement('disableAutoPause');
                if (disableAutoPauseCheckbox) {
                    disableAutoPauseCheckbox.checked = config.disable_auto_pause;
                }
            }

            // Provider + model saved in .env win over localStorage / HTML defaults.
            if (config.llm_provider) {
                const providerSelect = DomHelpers.getElement('llmProvider');
                if (providerSelect) {
                    providerSelect.value = config.llm_provider;
                }
            }

            // Parallel requests default (seeds the input; per-job request overrides it)
            if (config.parallel_translations) {
                const parallelWorkersInput = DomHelpers.getElement('parallelWorkers');
                if (parallelWorkersInput) {
                    parallelWorkersInput.value = String(config.parallel_translations);
                }
                if (config.max_parallel_translations) {
                    const pwEl = DomHelpers.getElement('parallelWorkers');
                    if (pwEl) pwEl.max = String(config.max_parallel_translations);
                }
            }

            // Chunker budget (seeds the input; saved back to .env on Save)
            if (config.max_tokens_per_chunk) {
                const maxTokensInput = DomHelpers.getElement('maxTokensPerChunk');
                if (maxTokensInput) {
                    maxTokensInput.value = String(config.max_tokens_per_chunk);
                }
            }

            // Webhook notifications — populate fields from .env
            if ('notify_webhook_url' in config) {
                DomHelpers.setValue('notifyWebhookUrl', config.notify_webhook_url || '');
            }
            if (config.notify_webhook_method) {
                DomHelpers.setValue('notifyWebhookMethod', config.notify_webhook_method);
            }
            if ('notify_webhook_headers' in config) {
                DomHelpers.setValue('notifyWebhookHeaders', config.notify_webhook_headers || '');
            }
            if ('notify_webhook_payload' in config) {
                DomHelpers.setValue('notifyWebhookPayload', config.notify_webhook_payload || '');
            }
            const setCheckbox = (id, value) => {
                if (typeof value === 'boolean') {
                    const cb = DomHelpers.getElement(id);
                    if (cb) cb.checked = value;
                }
            };
            setCheckbox('notifyOnSuccess', config.notify_on_success);
            setCheckbox('notifyOnFailure', config.notify_on_failure);
            setCheckbox('notifyOnInterruption', config.notify_on_interruption);
            if (typeof config.notify_timeout_seconds === 'number') {
                DomHelpers.setValue('notifyTimeoutSeconds', String(config.notify_timeout_seconds));
            }
            const notifyBadge = DomHelpers.getElement('notifyStatusBadge');
            if (notifyBadge) {
                notifyBadge.style.display = config.notify_configured ? 'inline-block' : 'none';
            }

            // Handle API keys - show indicator if configured in .env, otherwise keep placeholder
            ApiKeyUtils.setupField('geminiApiKey', config.gemini_api_key_configured, config.gemini_api_key, config.gemini_api_key_count);
            ApiKeyUtils.setupField('openaiApiKey', config.openai_api_key_configured, config.openai_api_key, config.openai_api_key_count);
            ApiKeyUtils.setupField('openrouterApiKey', config.openrouter_api_key_configured, config.openrouter_api_key, config.openrouter_api_key_count);
            ApiKeyUtils.setupField('mistralApiKey', config.mistral_api_key_configured, config.mistral_api_key, config.mistral_api_key_count);
            ApiKeyUtils.setupField('deepseekApiKey', config.deepseek_api_key_configured, config.deepseek_api_key, config.deepseek_api_key_count);
            ApiKeyUtils.setupField('poeApiKey', config.poe_api_key_configured, config.poe_api_key, config.poe_api_key_count);
            ApiKeyUtils.setupField('nimApiKey', config.nim_api_key_configured, config.nim_api_key, config.nim_api_key_count);
            ApiKeyUtils.setupField('anthropicApiKey', config.anthropic_api_key_configured, config.anthropic_api_key, config.anthropic_api_key_count);
            ApiKeyUtils.setupField('xaiApiKey', config.xai_api_key_configured, config.xai_api_key, config.xai_api_key_count);
            ApiKeyUtils.setupField('opencodeApiKey', config.opencode_api_key_configured, config.opencode_api_key, config.opencode_api_key_count);
            ApiKeyUtils.setupField('opencodegoApiKey', config.opencodego_api_key_configured, config.opencodego_api_key, config.opencodego_api_key_count);
            ApiKeyUtils.setupField('ollamacloudApiKey', config.ollamacloud_api_key_configured, config.ollamacloud_api_key, config.ollamacloud_api_key_count);

            // After loading defaults, dispatch event to notify other modules
            console.log('[FormManager] Default config loaded, dispatching event');
            window.dispatchEvent(new CustomEvent('defaultConfigLoaded'));

        } catch (error) {
            console.error('[FormManager] Failed to load default configuration:', error);
            MessageLogger.showMessage(t('settings:default_config_load_failed'), 'warning');
            // Still dispatch event even on error so other modules aren't blocked
            console.log('[FormManager] Dispatching defaultConfigLoaded event despite error');
            window.dispatchEvent(new CustomEvent('defaultConfigLoaded'));
        }
    },

    /**
     * Load available custom instruction files
     */
    async loadCustomInstructions() {
        try {
            console.log('[CustomInstructions] Loading custom instructions...');
            const data = await ApiClient.getCustomInstructions();
            console.log('[CustomInstructions] Data received:', data);

            const select = DomHelpers.getElement('customInstructionSelect');
            if (!select) {
                console.warn('[CustomInstructions] Select element not found!');
                return;
            }

            // Save current value before resetting dropdown
            const currentValue = select.value;

            // Reset dropdown to default
            select.innerHTML = `<option value="" data-i18n="settings:select_none">${t('settings:select_none')}</option>` +
                `<option value="__auto__" data-i18n="settings:style_auto_option">${t('settings:style_auto_option')}</option>`;

            // Populate dropdown with available files
            if (data.files && data.files.length > 0) {
                console.log('[CustomInstructions] Adding', data.files.length, 'files to dropdown');
                data.files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file.filename;
                    option.textContent = file.display_name;
                    select.appendChild(option);
                    console.log('[CustomInstructions] Added option:', file.display_name);
                });
            } else {
                console.warn('[CustomInstructions] No files found in response');
            }

            // Restore previously selected value if it still exists in the list
            if (currentValue) {
                // Check if the value exists in options
                let found = false;
                for (let option of select.options) {
                    if (option.value === currentValue) {
                        select.value = currentValue;
                        found = true;
                        console.log('[CustomInstructions] Restored selection:', currentValue);
                        break;
                    }
                }
                if (!found) {
                    console.warn('[CustomInstructions] Previously selected file not found:', currentValue);
                }
            }

            // Dispatch event to notify that custom instructions are loaded
            // This allows SettingsManager to restore the saved value
            window.dispatchEvent(new CustomEvent('customInstructionsLoaded'));
        } catch (error) {
            console.error('[CustomInstructions] Error loading custom instructions:', error);
            // Graceful degradation - dropdown will only show "None" option
            // Still dispatch event even on error
            window.dispatchEvent(new CustomEvent('customInstructionsLoaded'));
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
            MessageLogger.addLog(t('translation:interrupt_before_clear_log'));
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
        DomHelpers.setText('translateBtn', t('translation:start_batch_with_icon'));
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
            MessageLogger.addLog(t('translation:delete_uploaded_log', { count: uploadedFilePaths.length }));
            try {
                const result = await ApiClient.clearUploads(uploadedFilePaths);

                MessageLogger.addLog(t('translation:delete_uploaded_success_log', { count: result.total_deleted }));
                if (result.failed && result.failed.length > 0) {
                    MessageLogger.addLog(t('translation:delete_uploaded_failed_log', { count: result.failed.length }));
                }
            } catch {
                MessageLogger.addLog(t('translation:delete_uploaded_error_log'));
            }
        }

        MessageLogger.addLog(t('translation:form_reset_log'));

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

        // Get API endpoint based on provider. Only providers with a visible
        // endpoint field (ollama, openai) send one from the form. Cloud
        // providers (anthropic, xai, opencode, ...) resolve their endpoint
        // server-side from the .env/config defaults; forwarding the Ollama
        // endpoint here used to redirect every request to the local server,
        // which answered 404 "model not found" and silently kept the source
        // language in the output file.
        let apiEndpoint = '';
        if (provider === 'openai') {
            apiEndpoint = DomHelpers.getValue('openaiEndpoint');
        } else if (provider === 'ollama') {
            apiEndpoint = DomHelpers.getValue('apiEndpoint');
        }

        // Get API keys - use helper to handle .env configured keys
        const geminiApiKey = provider === 'gemini' ? ApiKeyUtils.getValue('geminiApiKey') : '';
        const openaiApiKey = provider === 'openai' ? ApiKeyUtils.getValue('openaiApiKey') : '';
        const openrouterApiKey = provider === 'openrouter' ? ApiKeyUtils.getValue('openrouterApiKey') : '';

        // Get TTS configuration
        const ttsEnabled = DomHelpers.getElement('ttsEnabled')?.checked || false;

        const ciValue = DomHelpers.getValue('customInstructionSelect') || '';
        const promptOptions = {
            preserve_technical_content: true,
            text_cleanup: DomHelpers.getElement('textCleanup')?.checked || false,
            refine: false,
            custom_instruction_file: ciValue === '__auto__' ? '' : ciValue
        };
        if (ciValue === '__auto__') promptOptions.style_auto = true;

        return {
            source_language: sourceLanguageVal,
            target_language: targetLanguageVal,
            model: model,
            llm_api_endpoint: apiEndpoint,
            llm_provider: provider,
            gemini_api_key: geminiApiKey,
            openai_api_key: openaiApiKey,
            openrouter_api_key: openrouterApiKey,
            anthropic_api_key: provider === 'anthropic' ? ApiKeyUtils.getValue('anthropicApiKey') : '',
            xai_api_key: provider === 'xai' ? ApiKeyUtils.getValue('xaiApiKey') : '',
            opencode_api_key: provider === 'opencode' ? ApiKeyUtils.getValue('opencodeApiKey') : '',
            opencodego_api_key: provider === 'opencodego' ? ApiKeyUtils.getValue('opencodegoApiKey') : '',
            ollamacloud_api_key: provider === 'ollamacloud' ? ApiKeyUtils.getValue('ollamacloudApiKey') : '',
            // Prompt options (optional system prompt instructions)
            // Technical content protection is always enabled
            prompt_options: promptOptions,
            // Bilingual output (original + translation interleaved)
            bilingual_output: DomHelpers.getElement('bilingualMode')?.checked || false,
            // Disable auto-pause on rate limit (auto-resume after Retry-After)
            auto_pause_on_rate_limit: !(DomHelpers.getElement('disableAutoPause')?.checked || false),
            // Parallel chunk translation (cloud only; backend gates local to 1)
            parallel_workers: provider === 'ollama'
                ? 1
                : (parseInt(DomHelpers.getValue('parallelWorkers'), 10) || 1),
            max_tokens_per_chunk: parseInt(DomHelpers.getValue('maxTokensPerChunk'), 10) || undefined,
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
            return { valid: false, message: t('translation:validation_source_required') };
        }

        if (!config.target_language) {
            return { valid: false, message: t('translation:validation_target_required') };
        }

        if (!config.model) {
            return { valid: false, message: t('translation:validation_model_required') };
        }

        // The endpoint field is only meaningful for providers that expose one
        // (ollama, openai). Cloud providers resolve their endpoint from the
        // server defaults, so an empty llm_api_endpoint is valid for them.
        const provider = config.llm_provider;
        const needsEndpoint = provider === 'ollama' || provider === 'openai';
        if (needsEndpoint && !config.llm_api_endpoint) {
            return { valid: false, message: t('translation:validation_endpoint_empty') };
        }

        // Validate API keys for cloud providers using shared utility
        const apiKeyValidation = ApiKeyUtils.validateForProvider(config.llm_provider, config.llm_api_endpoint);
        if (!apiKeyValidation.valid) {
            return apiKeyValidation;
        }

        return { valid: true, message: '' };
    }
};
