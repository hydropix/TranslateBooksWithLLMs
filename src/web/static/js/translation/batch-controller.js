/**
 * Batch Controller - Batch translation orchestration
 *
 * Manages batch translation queue processing, configuration validation,
 * and sequential file translation.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { Validators } from '../utils/validators.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import { StatusManager } from '../utils/status-manager.js';
import { ProgressManager } from './progress-manager.js';
import { FileUpload } from '../files/file-upload.js';
import { TranslationTracker } from './translation-tracker.js';

/**
 * Validation helper for early failures
 * @param {string} message - Error message
 */
function earlyValidationFail(message) {
    MessageLogger.showMessage(message, 'error');
    MessageLogger.addLog(`❌ Validation failed: ${message}`);
    return false;
}


/**
 * Get translation configuration from form
 * @param {Object} file - File to translate
 * @returns {Object} Translation configuration
 */
function getTranslationConfig(file) {
    // Use languages stored in the file object (captured when added to queue)
    // This ensures each file can have different source/target languages in batch
    const sourceLanguageVal = file.sourceLanguage;
    const targetLanguageVal = file.targetLanguage;

    const provider = DomHelpers.getValue('llmProvider');

    const promptOptions = {
        preserve_technical_content: true,
        text_cleanup: DomHelpers.getElement('textCleanup')?.checked || false,
        refine: DomHelpers.getElement('refineTranslation')?.checked || false,
        custom_instruction_file: DomHelpers.getValue('customInstructionSelect') || ''
    };

    // Get TTS configuration
    const ttsEnabled = DomHelpers.getElement('ttsEnabled')?.checked || false;

    const config = {
        source_language: sourceLanguageVal,
        target_language: targetLanguageVal,
        model: DomHelpers.getValue('model'),
        llm_api_endpoint: provider === 'openai' ?
                         DomHelpers.getValue('openaiEndpoint') :
                         DomHelpers.getValue('apiEndpoint'),
        llm_provider: provider,
        gemini_api_key: provider === 'gemini' ? ApiKeyUtils.getValue('geminiApiKey') : '',
        openai_api_key: provider === 'openai' ? ApiKeyUtils.getValue('openaiApiKey') : '',
        openrouter_api_key: provider === 'openrouter' ? ApiKeyUtils.getValue('openrouterApiKey') : '',
        mistral_api_key: provider === 'mistral' ? ApiKeyUtils.getValue('mistralApiKey') : '',
        deepseek_api_key: provider === 'deepseek' ? ApiKeyUtils.getValue('deepseekApiKey') : '',
        poe_api_key: provider === 'poe' ? ApiKeyUtils.getValue('poeApiKey') : '',
        input_filename: file.name,
        output_filename: file.outputFilename,
        file_type: file.fileType,
        prompt_options: promptOptions,
        bilingual_output: DomHelpers.getElement('bilingualMode')?.checked || false,
        tts_enabled: ttsEnabled,
        tts_voice: ttsEnabled ? (DomHelpers.getValue('ttsVoice') || '') : '',
        tts_rate: ttsEnabled ? (DomHelpers.getValue('ttsRate') || '+0%') : '+0%',
        tts_format: ttsEnabled ? (DomHelpers.getValue('ttsFormat') || 'opus') : 'opus',
        tts_bitrate: ttsEnabled ? (DomHelpers.getValue('ttsBitrate') || '64k') : '64k'
    };

    if (file.fileType === 'epub' || file.fileType === 'srt') {
        config.file_path = file.filePath;
    } else {
        if (file.content) {
            config.text = file.content;
        } else {
            config.file_path = file.filePath;
        }
    }

    return config;
}

/**
 * Update file status in the display
 * @param {string} filename - File name
 * @param {string} status - New status
 * @param {string} translationId - Optional translation ID
 */
function updateFileStatusInList(filename, status, translationId = null) {
    const filesToProcess = StateManager.getState('files.toProcess') || [];
    const fileIndex = filesToProcess.findIndex(f => f.name === filename);

    if (fileIndex !== -1) {
        filesToProcess[fileIndex].status = status;
        if (translationId) {
            filesToProcess[fileIndex].translationId = translationId;
        }
        StateManager.setState('files.toProcess', filesToProcess);
        // Persist to localStorage
        FileUpload.notifyFileListChanged();
    }

    // Emit event for UI update
    const event = new CustomEvent('fileStatusChanged', { detail: { filename, status, translationId } });
    window.dispatchEvent(event);
}

export const BatchController = {
    /**
     * Start batch translation
     */
    async startBatchTranslation() {
        if (!TranslationTracker.isInitialized || !TranslationTracker.isInitialized()) {
            await new Promise(resolve => setTimeout(resolve, 100));
            if (!TranslationTracker.isInitialized || !TranslationTracker.isInitialized()) {
                MessageLogger.showMessage('⚠️ System still initializing, please wait...', 'warning');
                return;
            }
        }

        const isBatchActive = StateManager.getState('translation.isBatchActive') || false;
        const filesToProcess = StateManager.getState('files.toProcess') || [];

        if (isBatchActive || filesToProcess.length === 0) return;

        // Validate configuration
        let sourceLanguageVal = DomHelpers.getValue('sourceLang');
        if (sourceLanguageVal === 'Other') {
            sourceLanguageVal = DomHelpers.getValue('customSourceLang').trim();
            if (!sourceLanguageVal) {
                return earlyValidationFail('Please specify the custom source language for the batch.');
            }
        }

        let targetLanguageVal = DomHelpers.getValue('targetLang');
        if (targetLanguageVal === 'Other') {
            targetLanguageVal = DomHelpers.getValue('customTargetLang').trim();
            if (!targetLanguageVal) {
                return earlyValidationFail('Please specify the custom target language for the batch.');
            }
        }

        const selectedModel = DomHelpers.getValue('model');
        if (!selectedModel) {
            return earlyValidationFail('Please select an LLM model for the batch.');
        }

        const provider = DomHelpers.getValue('llmProvider');
        if (provider === 'ollama') {
            const ollamaApiEndpoint = DomHelpers.getValue('apiEndpoint').trim();
            if (!ollamaApiEndpoint) {
                return earlyValidationFail('Ollama API Endpoint cannot be empty for the batch.');
            }
        }

        let filesUpdated = false;
        for (const file of filesToProcess) {
            if (file.status !== 'Queued') continue;

            if (!file.sourceLanguage || file.sourceLanguage === 'Other') {
                file.sourceLanguage = sourceLanguageVal;
                filesUpdated = true;
            }
            if (!file.targetLanguage || file.targetLanguage === 'Other') {
                file.targetLanguage = targetLanguageVal;
                filesUpdated = true;
            }
        }

        if (filesUpdated) {
            StateManager.setState('files.toProcess', filesToProcess);
        }

        StateManager.setState('translation.isBatchActive', true);

        const queuedFilesCount = filesToProcess.filter(f => f.status === 'Queued').length;

        // Update UI
        const translateBtn = DomHelpers.getElement('translateBtn');
        if (translateBtn) {
            translateBtn.disabled = true;
            translateBtn.innerHTML = '⏳ Batch in Progress...';
        }

        const interruptBtn = DomHelpers.getElement('interruptBtn');
        if (interruptBtn) {
            DomHelpers.show('interruptBtn');
            interruptBtn.disabled = false;
        }

        MessageLogger.addLog(`🚀 Batch translation started for ${queuedFilesCount} file(s).`);
        MessageLogger.showMessage(`Batch of ${queuedFilesCount} file(s) initiated.`, 'info');

        // Start processing queue
        this.processNextFileInQueue();
    },

    /**
     * Process next file in queue
     */
    async processNextFileInQueue() {
        const currentJob = StateManager.getState('translation.currentJob');
        if (currentJob) return;

        const filesToProcess = StateManager.getState('files.toProcess') || [];
        const fileToTranslate = filesToProcess.find(f => f.status === 'Queued');

        if (!fileToTranslate) {
            StateManager.setState('translation.isBatchActive', false);
            StateManager.setState('translation.currentJob', null);

            const translateBtn = DomHelpers.getElement('translateBtn');
            if (translateBtn) {
                translateBtn.disabled = filesToProcess.length === 0 || !StatusManager.isConnected();
                translateBtn.innerHTML = '▶️ Start Translation Batch';
            }

            DomHelpers.hide('interruptBtn');

            MessageLogger.showMessage('✅ Batch translation completed for all files!', 'success');
            MessageLogger.addLog('🏁 All files in the batch have been processed.');
            DomHelpers.setText('currentFileProgressTitle', `📊 Batch Completed`);
            return;
        }

        ProgressManager.reset();

        const lastTranslationPreview = DomHelpers.getElement('lastTranslationPreview');
        if (lastTranslationPreview) {
            lastTranslationPreview.innerHTML = '<div style="color: #6b7280; font-style: italic; padding: 10px;">No translation yet...</div>';
        }

        if (fileToTranslate.fileType === 'epub') {
            DomHelpers.hide('statsGrid');
        } else {
            DomHelpers.show('statsGrid');
        }

        this.updateTranslationTitle(fileToTranslate);
        ProgressManager.show();
        MessageLogger.addLog(`▶️ Starting translation for: ${fileToTranslate.name} (${fileToTranslate.fileType.toUpperCase()})`);
        updateFileStatusInList(fileToTranslate.name, 'Preparing...');

        const provider = DomHelpers.getValue('llmProvider');
        const endpoint = provider === 'openai' ? DomHelpers.getValue('openaiEndpoint') : '';
        const apiKeyValidation = ApiKeyUtils.validateForProvider(provider, endpoint);

        if (!apiKeyValidation.valid) {
            MessageLogger.addLog(`❌ Error: ${apiKeyValidation.message}`);
            MessageLogger.showMessage(apiKeyValidation.message, 'error');
            updateFileStatusInList(fileToTranslate.name, 'Error: Missing API key');
            StateManager.setState('translation.currentJob', null);
            this.processNextFileInQueue();
            return;
        }

        // Validate file path
        if (!fileToTranslate.filePath && !fileToTranslate.content) {
            MessageLogger.addLog(`❌ Critical Error: File ${fileToTranslate.name} has no server path or content`);
            MessageLogger.showMessage(`Cannot process ${fileToTranslate.name}: file path missing.`, 'error');
            updateFileStatusInList(fileToTranslate.name, 'Path Error');
            StateManager.setState('translation.currentJob', null);
            this.processNextFileInQueue();
            return;
        }

        const config = getTranslationConfig(fileToTranslate);

        try {
            const data = await ApiClient.startTranslation(config);

            StateManager.setState('translation.currentJob', {
                fileRef: fileToTranslate,
                translationId: data.translation_id
            });

            fileToTranslate.translationId = data.translation_id;
            updateFileStatusInList(fileToTranslate.name, 'Submitted', data.translation_id);

            DomHelpers.show('progressSection');
            DomHelpers.show('interruptBtn');

            requestAnimationFrame(() => {
                const progressSection = DomHelpers.getElement('progressSection');
                if (progressSection) {
                    progressSection.style.display = 'block';
                }
            });

            this.updateTranslationTitle(fileToTranslate);
            MessageLogger.addLog(`⏳ Translation submitted for ${fileToTranslate.name}...`);
            this.removeFileFromProcessingList(fileToTranslate.name);

            const event = new CustomEvent('translationStarted', { detail: { file: fileToTranslate, translationId: data.translation_id } });
            window.dispatchEvent(event);

        } catch (error) {
            MessageLogger.addLog(`❌ Error initiating translation for ${fileToTranslate.name}: ${error.message}`);
            MessageLogger.showMessage(`Error starting ${fileToTranslate.name}: ${error.message}`, 'error');
            updateFileStatusInList(fileToTranslate.name, 'Initiation Error');
            StateManager.setState('translation.currentJob', null);
            this.processNextFileInQueue();
        }
    },

    /**
     * Update translation title with file icon/thumbnail and name
     * @param {Object} file - File object
     */
    updateTranslationTitle(file) {
        const titleElement = DomHelpers.getElement('currentFileProgressTitle');
        if (!titleElement) return;

        // Clear existing content
        titleElement.innerHTML = '';

        // Create main container with vertical layout
        const mainContainer = document.createElement('div');
        mainContainer.style.display = 'flex';
        mainContainer.style.flexDirection = 'column';
        mainContainer.style.gap = '8px';

        // Add "Translating" text
        const translatingText = document.createElement('div');
        translatingText.textContent = 'Translating';
        translatingText.style.fontWeight = 'bold';
        mainContainer.appendChild(translatingText);

        // Create file info container (icon + filename)
        const fileInfoContainer = document.createElement('div');
        fileInfoContainer.style.display = 'flex';
        fileInfoContainer.style.alignItems = 'center';
        fileInfoContainer.style.gap = '8px';

        // Icon/thumbnail container
        const iconContainer = document.createElement('span');
        iconContainer.style.display = 'inline-flex';
        iconContainer.style.alignItems = 'center';
        iconContainer.style.fontSize = '24px';

        if (file.fileType === 'epub' && file.thumbnail) {
            // Show thumbnail
            const img = document.createElement('img');
            img.src = `/api/thumbnails/${encodeURIComponent(file.thumbnail)}`;
            img.alt = 'Cover';
            img.style.width = '48px';
            img.style.height = '72px';
            img.style.objectFit = 'cover';
            img.style.borderRadius = '3px';
            img.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';

            // Fallback to generic SVG on error
            img.onerror = () => {
                iconContainer.innerHTML = this._createGenericEPUBIcon();
            };

            iconContainer.appendChild(img);
        } else {
            // Generic icons
            iconContainer.innerHTML = this._getFileIcon(file.fileType);
        }

        fileInfoContainer.appendChild(iconContainer);

        // File name (split name and extension)
        const fileNameContainer = document.createElement('div');
        fileNameContainer.style.display = 'flex';
        fileNameContainer.style.flexDirection = 'column';
        fileNameContainer.style.gap = '4px';

        // Split filename and extension
        const lastDotIndex = file.name.lastIndexOf('.');
        const fileNameWithoutExt = lastDotIndex > 0 ? file.name.substring(0, lastDotIndex) : file.name;
        const fileExt = lastDotIndex > 0 ? file.name.substring(lastDotIndex) : '';

        // Create container for name + extension
        const nameRow = document.createElement('div');
        nameRow.style.display = 'flex';
        nameRow.style.alignItems = 'baseline';
        nameRow.style.gap = '2px';

        // File name (bold and larger)
        const fileNameSpan = document.createElement('span');
        fileNameSpan.textContent = fileNameWithoutExt;
        fileNameSpan.style.fontSize = '18px';
        fileNameSpan.style.fontWeight = 'bold';
        nameRow.appendChild(fileNameSpan);

        // Extension (normal size)
        if (fileExt) {
            const extSpan = document.createElement('span');
            extSpan.textContent = fileExt;
            extSpan.style.fontSize = '14px';
            extSpan.style.color = 'var(--text-muted-light)';
            nameRow.appendChild(extSpan);
        }

        fileNameContainer.appendChild(nameRow);

        // Language info (source → target)
        if (file.sourceLanguage && file.targetLanguage) {
            const langSpan = document.createElement('div');
            langSpan.textContent = `${file.sourceLanguage} → ${file.targetLanguage}`;
            langSpan.style.fontSize = '12px';
            langSpan.style.color = 'var(--text-muted-light)';
            langSpan.style.fontWeight = 'normal';
            fileNameContainer.appendChild(langSpan);
        }

        fileInfoContainer.appendChild(fileNameContainer);

        // Add file info to main container
        mainContainer.appendChild(fileInfoContainer);

        // Add main container to title element
        titleElement.appendChild(mainContainer);
    },

    /**
     * Get file icon based on file type
     * @param {string} fileType - File type ('txt', 'epub', 'srt')
     * @returns {string} HTML string for icon
     */
    _getFileIcon(fileType) {
        if (fileType === 'epub') {
            return this._createGenericEPUBIcon();
        } else if (fileType === 'srt') {
            return '🎬';
        }
        return '📄';
    },

    /**
     * Create generic EPUB icon as SVG
     * @returns {string} SVG HTML string
     */
    _createGenericEPUBIcon() {
        return `
            <svg style="width: 48px; height: 72px;" viewBox="0 0 48 72" xmlns="http://www.w3.org/2000/svg">
                <!-- Book cover -->
                <rect x="6" y="3" width="36" height="66" rx="2.5"
                      fill="#5a8ee8" stroke="#3676d8" stroke-width="2"/>
                <!-- Book spine line -->
                <path d="M6 13 L42 13" stroke="#3676d8" stroke-width="1.8"/>
                <!-- Text lines -->
                <path d="M10 22 L38 22 M10 32 L38 32 M10 42 L32 42"
                      stroke="white" stroke-width="2.2" stroke-linecap="round" opacity="0.8"/>
                <!-- EPUB badge -->
                <circle cx="24" cy="56" r="5" fill="white" opacity="0.9"/>
                <text x="24" y="60" text-anchor="middle" font-size="6"
                      fill="#3676d8" font-weight="bold">E</text>
            </svg>
        `;
    },

    /**
     * Stop batch translation
     */
    stopBatch() {
        StateManager.setState('translation.isBatchActive', false);
        StateManager.setState('translation.currentJob', null);

        // Clear saved translation state from localStorage
        if (TranslationTracker && TranslationTracker.clearTranslationState) {
            TranslationTracker.clearTranslationState();
        }

        const translateBtn = DomHelpers.getElement('translateBtn');
        const filesToProcess = StateManager.getState('files.toProcess') || [];
        if (translateBtn) {
            translateBtn.disabled = filesToProcess.length === 0 || !StatusManager.isConnected();
            translateBtn.innerHTML = '▶️ Start Translation Batch';
        }

        DomHelpers.hide('interruptBtn');

        MessageLogger.addLog('⏹️ Batch translation stopped');
        MessageLogger.showMessage('Batch translation stopped', 'info');
    },

    /**
     * Remove file from processing list
     * @param {string} filename - Filename to remove
     */
    removeFileFromProcessingList(filename) {
        const filesToProcess = StateManager.getState('files.toProcess');
        const fileIndex = filesToProcess.findIndex(f => f.name === filename);

        if (fileIndex !== -1) {
            filesToProcess.splice(fileIndex, 1);
            StateManager.setState('files.toProcess', filesToProcess);
            MessageLogger.addLog(`🗑️ Removed ${filename} from file list`);
            // Notify file list change to update UI and persist to localStorage
            FileUpload.notifyFileListChanged();
        }
    }
};
