/**
 * TTS Manager - Manages Text-to-Speech provider selection and configuration
 *
 * Handles provider switching between Edge-TTS and Chatterbox TTS,
 * voice prompt uploads for voice cloning, and GPU status display.
 */

import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { MessageLogger } from '../ui/message-logger.js';
import { t } from '../i18n/i18n.js';

/**
 * TTS Provider state and configuration
 */
const TTSState = {
    currentProvider: 'edge-tts',
    providers: {},
    gpuStatus: null,
    voicePrompts: [],
    isInitialized: false
};

/**
 * TTS Manager module
 */
export const TTSManager = {
    /**
     * Initialize TTS Manager
     */
    async initialize() {
        if (TTSState.isInitialized) {
            return;
        }

        console.log('Initializing TTS Manager...');

        // Set up event listeners
        this.setupEventListeners();

        // Load providers info
        await this.loadProvidersInfo();

        // Load GPU status
        await this.loadGPUStatus();

        // Load available voice prompts
        await this.loadVoicePrompts();

        TTSState.isInitialized = true;
        console.log('TTS Manager initialized');
    },

    /**
     * Set up event listeners for TTS controls
     */
    setupEventListeners() {
        // TTS enabled checkbox
        const ttsEnabled = DomHelpers.getElement('ttsEnabled');
        if (ttsEnabled) {
            ttsEnabled.addEventListener('change', () => this.onTTSEnabledChange());
        }

        // TTS provider selector
        const ttsProvider = DomHelpers.getElement('ttsProvider');
        if (ttsProvider) {
            ttsProvider.addEventListener('change', () => this.onProviderChange());
        }

        // Voice prompt file input
        const voicePromptInput = DomHelpers.getElement('voicePromptInput');
        if (voicePromptInput) {
            voicePromptInput.addEventListener('change', (e) => this.onVoicePromptUpload(e));
        }

        // Exaggeration slider
        const exaggerationSlider = DomHelpers.getElement('ttsExaggeration');
        if (exaggerationSlider) {
            exaggerationSlider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                const valueDisplay = DomHelpers.getElement('exaggerationValue');
                if (valueDisplay) {
                    valueDisplay.textContent = value.toFixed(2);
                }
            });
        }

        // CFG weight slider
        const cfgSlider = DomHelpers.getElement('ttsCfgWeight');
        if (cfgSlider) {
            cfgSlider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                const valueDisplay = DomHelpers.getElement('cfgWeightValue');
                if (valueDisplay) {
                    valueDisplay.textContent = value.toFixed(2);
                }
            });
        }
    },

    /**
     * Handle TTS enabled checkbox change
     */
    onTTSEnabledChange() {
        const ttsEnabled = DomHelpers.getElement('ttsEnabled');
        const ttsOptions = DomHelpers.getElement('ttsOptions');

        if (ttsEnabled && ttsOptions) {
            if (ttsEnabled.checked) {
                ttsOptions.style.display = 'block';
                // Refresh GPU status when TTS is enabled
                this.loadGPUStatus();
            } else {
                ttsOptions.style.display = 'none';
            }
        }
    },

    /**
     * Handle provider selection change
     */
    onProviderChange() {
        const providerSelect = DomHelpers.getElement('ttsProvider');
        if (!providerSelect) return;

        const selectedProvider = providerSelect.value;
        TTSState.currentProvider = selectedProvider;

        // Show/hide provider-specific options
        this.updateProviderOptions(selectedProvider);

        // Update voice selector
        this.updateVoiceSelector(selectedProvider);

        MessageLogger.addLog(t('tts:tts_provider_changed_log', { provider: selectedProvider }));
    },

    /**
     * Update UI based on selected provider
     * @param {string} provider - Selected provider name
     */
    updateProviderOptions(provider) {
        // Edge-TTS options
        const edgeTTSOptions = DomHelpers.getElement('edgeTTSOptions');
        // Chatterbox options
        const chatterboxOptions = DomHelpers.getElement('chatterboxOptions');
        // GPU status
        const gpuStatusSection = DomHelpers.getElement('gpuStatusSection');

        if (provider === 'edge-tts') {
            if (edgeTTSOptions) edgeTTSOptions.style.display = 'block';
            if (chatterboxOptions) chatterboxOptions.style.display = 'none';
            if (gpuStatusSection) gpuStatusSection.style.display = 'none';
        } else if (provider === 'chatterbox') {
            if (edgeTTSOptions) edgeTTSOptions.style.display = 'none';
            if (chatterboxOptions) chatterboxOptions.style.display = 'block';
            if (gpuStatusSection) gpuStatusSection.style.display = 'block';

            // Check if Chatterbox is available
            const providerInfo = TTSState.providers['chatterbox'];
            if (providerInfo && !providerInfo.available) {
                MessageLogger.showMessage(
                    t('tts:chatterbox_not_available'),
                    'error'
                );
            }
        }
    },

    /**
     * Update voice selector based on provider
     * @param {string} provider - Selected provider name
     */
    updateVoiceSelector(provider) {
        const voiceInput = DomHelpers.getElement('ttsVoice');
        const voiceLabel = document.querySelector('label[for="ttsVoice"]');
        const voiceHelp = DomHelpers.getElement('ttsVoiceHelp');

        if (provider === 'edge-tts') {
            if (voiceInput) {
                voiceInput.placeholder = t('tts:voice_placeholder');
                voiceInput.disabled = false;
            }
            if (voiceLabel) {
                voiceLabel.textContent = t('tts:voice_auto_select_label');
            }
            if (voiceHelp) {
                voiceHelp.textContent = t('tts:voice_auto_hint_target');
            }
        } else if (provider === 'chatterbox') {
            if (voiceInput) {
                voiceInput.placeholder = t('tts:voice_via_prompt_placeholder');
                voiceInput.disabled = true;
                voiceInput.value = '';
            }
            if (voiceLabel) {
                voiceLabel.textContent = t('tts:voice_via_prompt_label');
            }
            if (voiceHelp) {
                voiceHelp.textContent = t('tts:voice_via_prompt_hint');
            }
        }
    },

    /**
     * Load TTS providers information from server
     */
    async loadProvidersInfo() {
        try {
            const response = await ApiClient.getTTSProviders();
            TTSState.providers = response.providers || {};

            // Update provider selector availability
            this.updateProviderAvailability();

            return TTSState.providers;
        } catch (error) {
            console.error('Failed to load TTS providers:', error);
            MessageLogger.addLog(t('tts:tts_providers_load_failed', { error: error.message }));
        }
    },

    /**
     * Update provider selector based on availability
     */
    updateProviderAvailability() {
        const providerSelect = DomHelpers.getElement('ttsProvider');
        if (!providerSelect) return;

        // Update Chatterbox option based on availability
        const chatterboxOption = providerSelect.querySelector('option[value="chatterbox"]');
        if (chatterboxOption) {
            const isAvailable = TTSState.providers['chatterbox']?.available;
            if (!isAvailable) {
                chatterboxOption.textContent = t('tts:provider_chatterbox_full_unavailable');
                chatterboxOption.disabled = true;
            } else {
                chatterboxOption.textContent = t('tts:provider_chatterbox_full_local');
                chatterboxOption.disabled = false;
            }
        }
    },

    /**
     * Load GPU status from server
     */
    async loadGPUStatus() {
        try {
            const status = await ApiClient.getTTSGPUStatus();
            TTSState.gpuStatus = status;

            // Update GPU status display
            this.updateGPUStatusDisplay(status);

            return status;
        } catch (error) {
            console.error('Failed to load GPU status:', error);
            this.updateGPUStatusDisplay({ cuda_available: false, error: error.message });
        }
    },

    /**
     * Update GPU status display
     * @param {Object} status - GPU status information
     */
    updateGPUStatusDisplay(status) {
        const gpuStatusElement = DomHelpers.getElement('gpuStatusIndicator');
        const gpuNameElement = DomHelpers.getElement('gpuName');
        const gpuVramElement = DomHelpers.getElement('gpuVram');
        const gpuStatusDot = DomHelpers.getElement('gpuStatusDot');

        if (status.cuda_available) {
            if (gpuStatusElement) {
                gpuStatusElement.className = 'gpu-status gpu-available';
            }
            if (gpuStatusDot) {
                gpuStatusDot.className = 'status-dot available';
            }
            if (gpuNameElement) {
                gpuNameElement.textContent = status.gpu_name || t('tts:gpu_cuda');
            }
            if (gpuVramElement && status.vram_total) {
                const usedGB = (status.vram_used / 1024).toFixed(1);
                const totalGB = (status.vram_total / 1024).toFixed(1);
                gpuVramElement.textContent = `${usedGB} / ${totalGB} GB`;
            }
        } else {
            if (gpuStatusElement) {
                gpuStatusElement.className = 'gpu-status gpu-unavailable';
            }
            if (gpuStatusDot) {
                gpuStatusDot.className = 'status-dot unavailable';
            }
            if (gpuNameElement) {
                gpuNameElement.textContent = t('tts:gpu_cpu_no_cuda');
            }
            if (gpuVramElement) {
                gpuVramElement.textContent = t('tts:gpu_na');
            }
        }
    },

    /**
     * Load available voice prompts from server
     */
    async loadVoicePrompts() {
        try {
            const response = await ApiClient.getTTSVoicePrompts();
            TTSState.voicePrompts = response.voice_prompts || [];

            // Update voice prompts dropdown
            this.updateVoicePromptsDropdown();

            return TTSState.voicePrompts;
        } catch (error) {
            console.error('Failed to load voice prompts:', error);
        }
    },

    /**
     * Update voice prompts dropdown
     */
    updateVoicePromptsDropdown() {
        const dropdown = DomHelpers.getElement('voicePromptSelect');
        if (!dropdown) return;

        // Clear existing options except first
        while (dropdown.options.length > 1) {
            dropdown.remove(1);
        }

        // Add voice prompts
        TTSState.voicePrompts.forEach(prompt => {
            const option = document.createElement('option');
            option.value = prompt.path;
            option.textContent = prompt.filename;
            dropdown.appendChild(option);
        });
    },

    /**
     * Handle voice prompt upload
     * @param {Event} event - Change event from file input
     */
    async onVoicePromptUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const uploadBtn = DomHelpers.getElement('uploadVoicePromptBtn');
        const statusSpan = DomHelpers.getElement('voicePromptUploadStatus');

        try {
            if (uploadBtn) {
                uploadBtn.disabled = true;
                uploadBtn.textContent = t('tts:voice_prompt_uploading');
            }
            if (statusSpan) {
                statusSpan.textContent = '';
            }

            const result = await ApiClient.uploadTTSVoicePrompt(file);

            if (result.success) {
                MessageLogger.addLog(t('tts:voice_prompt_uploaded_log', { filename: result.filename }));
                MessageLogger.showMessage(t('tts:voice_prompt_uploaded'), 'success');

                // Reload voice prompts list
                await this.loadVoicePrompts();

                // Auto-select the new prompt
                const dropdown = DomHelpers.getElement('voicePromptSelect');
                if (dropdown) {
                    dropdown.value = result.path;
                }

                if (statusSpan) {
                    statusSpan.textContent = t('tts:voice_prompt_uploaded_short');
                    statusSpan.style.color = 'var(--success-light)';
                }
            }
        } catch (error) {
            MessageLogger.showMessage(t('tts:voice_prompt_upload_failed', { error: error.message }), 'error');
            if (statusSpan) {
                statusSpan.textContent = t('tts:voice_prompt_upload_failed_short');
                statusSpan.style.color = '#ef4444';
            }
        } finally {
            if (uploadBtn) {
                uploadBtn.disabled = false;
                uploadBtn.textContent = t('tts:voice_prompt_upload_btn');
            }
            // Reset file input
            event.target.value = '';
        }
    },

    /**
     * Delete a voice prompt
     * @param {string} filename - Filename to delete
     */
    async deleteVoicePrompt(filename) {
        if (!confirm(t('tts:voice_prompt_delete_confirm', { filename }))) {
            return;
        }

        try {
            const result = await ApiClient.deleteTTSVoicePrompt(filename);

            if (result.success) {
                MessageLogger.addLog(t('tts:voice_prompt_deleted_log', { filename }));
                MessageLogger.showMessage(t('tts:voice_prompt_deleted'), 'success');

                // Reload voice prompts
                await this.loadVoicePrompts();
            }
        } catch (error) {
            MessageLogger.showMessage(t('tts:voice_prompt_delete_failed', { error: error.message }), 'error');
        }
    },



    /**
     * Get current provider
     * @returns {string} Current provider name
     */
    getCurrentProvider() {
        return TTSState.currentProvider;
    },



    /**
     * Refresh GPU status (call periodically or on demand)
     */
    async refreshGPUStatus() {
        return await this.loadGPUStatus();
    }
};

// Export for global access
if (typeof window !== 'undefined') {
    window.__TTS_MANAGER__ = TTSManager;
}
