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
import { renderTranslationTitle } from './progress-title.js';
import { FileUpload, generateOutputFilename } from '../files/file-upload.js';
import { TranslationTracker } from './translation-tracker.js';
import { t } from '../i18n/i18n.js';

/**
 * Validation helper for early failures
 * @param {string} message - Error message
 */
function earlyValidationFail(message) {
    MessageLogger.showMessage(message, 'error');
    MessageLogger.addLog(t('translation:validation_failed_log', { message }));
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
    const currentModel = DomHelpers.getValue('model') || '';

    // Regenerate output filename at translation time so placeholders like
    // {model}, {date}, {datetime} reflect the current run. The value stored
    // on the file object was computed at upload time and may be stale,
    // especially when the same file is re-translated with a different model.
    const outputPattern = DomHelpers.getValue('outputFilenamePattern')
        || '{originalName} ({targetLang}).{ext}';
    const resolvedOutputFilename = generateOutputFilename(
        { name: file.name },
        outputPattern,
        {
            sourceLang: sourceLanguageVal,
            targetLang: targetLanguageVal,
            model: currentModel
        }
    );

    const operation = file.operation || 'translate';
    const refineAfter = operation === 'translate' && !!file.refineAfter;

    const promptOptions = {
        preserve_technical_content: true,
        text_cleanup: DomHelpers.getElement('textCleanup')?.checked || false,
        refine: refineAfter,
        plain_text_mode: DomHelpers.getElement('plainTextMode')?.checked || false,
        custom_instruction_file: DomHelpers.getValue('customInstructionSelect') || ''
    };

    const glossaryId = DomHelpers.getValue('glossarySelect');
    if (glossaryId) {
        promptOptions.glossary_id = parseInt(glossaryId, 10);
    }

    // Get TTS configuration
    const ttsEnabled = DomHelpers.getElement('ttsEnabled')?.checked || false;

    const config = {
        source_language: sourceLanguageVal,
        target_language: targetLanguageVal,
        model: currentModel,
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
        nim_api_key: provider === 'nim' ? ApiKeyUtils.getValue('nimApiKey') : '',
        input_filename: file.name,
        output_filename: resolvedOutputFilename,
        file_type: file.fileType,
        prompt_options: promptOptions,
        bilingual_output: DomHelpers.getElement('bilingualMode')?.checked || false,
        refine_only: operation === 'refine',
        refine_after: refineAfter,
        auto_pause_on_rate_limit: !(DomHelpers.getElement('disableAutoPause')?.checked || false),
        // Parallel chunk translation; only honored for cloud providers (the
        // backend forces local providers back to 1).
        parallel_workers: provider === 'ollama'
            ? 1
            : (parseInt(DomHelpers.getValue('parallelWorkers'), 10) || 1),
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
                MessageLogger.showMessage(t('translation:system_initializing'), 'warning');
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
                return earlyValidationFail(t('translation:validation_custom_source'));
            }
        }

        let targetLanguageVal = DomHelpers.getValue('targetLang');
        if (targetLanguageVal === 'Other') {
            targetLanguageVal = DomHelpers.getValue('customTargetLang').trim();
            if (!targetLanguageVal) {
                return earlyValidationFail(t('translation:validation_custom_target'));
            }
        }

        const selectedModel = DomHelpers.getValue('model');
        if (!selectedModel) {
            return earlyValidationFail(t('translation:validation_model'));
        }

        const provider = DomHelpers.getValue('llmProvider');
        if (provider === 'ollama') {
            const ollamaApiEndpoint = DomHelpers.getValue('apiEndpoint').trim();
            if (!ollamaApiEndpoint) {
                return earlyValidationFail(t('translation:validation_ollama_endpoint'));
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
            translateBtn.innerHTML = t('translation:batch_in_progress');
        }

        const interruptBtn = DomHelpers.getElement('interruptBtn');
        if (interruptBtn) {
            DomHelpers.show('interruptBtn');
            interruptBtn.disabled = false;
        }

        MessageLogger.clearAlerts();
        MessageLogger.addLog(t('translation:batch_started_log', { count: queuedFilesCount }));
        MessageLogger.showMessage(t('translation:batch_initiated', { count: queuedFilesCount }), 'info');

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
                translateBtn.innerHTML = t('translation:start_batch_with_icon');
            }

            DomHelpers.hide('interruptBtn');

            MessageLogger.showMessage(t('translation:batch_completed'), 'success');
            MessageLogger.addLog(t('translation:batch_completed_log'));
            DomHelpers.setText('currentFileProgressTitle', t('translation:batch_completed_title'));
            return;
        }

        ProgressManager.reset();

        const lastTranslationPreview = DomHelpers.getElement('lastTranslationPreview');
        if (lastTranslationPreview) {
            lastTranslationPreview.innerHTML = `<div style="color: #6b7280; font-style: italic; padding: 10px;">${t('translation:no_translation_yet')}</div>`;
        }

        if (fileToTranslate.fileType === 'epub') {
            DomHelpers.hide('statsGrid');
        } else {
            DomHelpers.show('statsGrid');
        }

        this.updateTranslationTitle(fileToTranslate);
        ProgressManager.show();
        MessageLogger.addLog(t('translation:starting_translation_log', { name: fileToTranslate.name, type: fileToTranslate.fileType.toUpperCase() }));
        updateFileStatusInList(fileToTranslate.name, 'Preparing...');

        const provider = DomHelpers.getValue('llmProvider');
        const endpoint = provider === 'openai' ? DomHelpers.getValue('openaiEndpoint') : '';
        const apiKeyValidation = ApiKeyUtils.validateForProvider(provider, endpoint);

        if (!apiKeyValidation.valid) {
            MessageLogger.addLog(t('translation:api_key_error_log', { message: apiKeyValidation.message }));
            MessageLogger.showMessage(apiKeyValidation.message, 'error');
            updateFileStatusInList(fileToTranslate.name, 'Error: Missing API key');
            StateManager.setState('translation.currentJob', null);
            this.processNextFileInQueue();
            return;
        }

        // Validate file path
        if (!fileToTranslate.filePath && !fileToTranslate.content) {
            MessageLogger.addLog(t('translation:critical_no_path_log', { name: fileToTranslate.name }));
            MessageLogger.showMessage(t('translation:critical_no_path_msg', { name: fileToTranslate.name }), 'error');
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
            TranslationTracker.registerOwnedJob(data.translation_id);

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
            MessageLogger.addLog(t('translation:submitted_log', { name: fileToTranslate.name }));
            this.removeFileFromProcessingList(fileToTranslate.name);

            const event = new CustomEvent('translationStarted', { detail: { file: fileToTranslate, translationId: data.translation_id } });
            window.dispatchEvent(event);

        } catch (error) {
            MessageLogger.addLog(t('translation:init_error_log', { name: fileToTranslate.name, error: error.message }));
            MessageLogger.showMessage(t('translation:init_error_msg', { name: fileToTranslate.name, error: error.message }), 'error');
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
        renderTranslationTitle(file);
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
            translateBtn.innerHTML = t('translation:start_batch_with_icon');
        }

        DomHelpers.hide('interruptBtn');

        MessageLogger.addLog(t('translation:batch_stopped_log'));
        MessageLogger.showMessage(t('translation:batch_stopped'), 'info');
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
            MessageLogger.addLog(t('translation:file_removed_log', { name: filename }));
            // Notify file list change to update UI and persist to localStorage
            FileUpload.notifyFileListChanged();
        }
    }
};
