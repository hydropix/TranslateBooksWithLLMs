/**
 * Validators - Input validation utilities
 *
 * Provides validation functions for forms and user inputs
 */

import { MessageLogger } from '../ui/message-logger.js';
import { t } from '../i18n/i18n.js';

export const Validators = {
    /**
     * Show validation error message
     * @param {string} message - Error message
     * @returns {boolean} Always returns false for chaining
     */
    showError(message) {
        MessageLogger.showMessage(message, 'error');
        return false;
    },

    /**
     * Validate that a value is not empty
     */
    required(value, fieldName) {
        if (!value || value.trim() === '') {
            return this.showError(t('errors:field_required', { field: fieldName }));
        }
        return true;
    },

    /**
     * Validate language selection
     */
    validateLanguages(sourceLanguage, targetLanguage) {
        if (!sourceLanguage || sourceLanguage.trim() === '') {
            return this.showError(t('errors:no_source_language'));
        }
        if (!targetLanguage || targetLanguage.trim() === '') {
            return this.showError(t('errors:no_target_language'));
        }
        if (sourceLanguage.toLowerCase() === targetLanguage.toLowerCase()) {
            return this.showError(t('errors:same_source_target_language'));
        }
        return true;
    },

    /**
     * Validate model selection
     */
    validateModel(model) {
        if (!model || model.trim() === '') {
            return this.showError(t('errors:no_model_selected'));
        }
        return true;
    },

    /**
     * Validate API endpoint
     */
    validateApiEndpoint(endpoint) {
        if (!endpoint || endpoint.trim() === '') {
            return this.showError(t('errors:api_endpoint_empty'));
        }
        try {
            new URL(endpoint);
            return true;
        } catch {
            return this.showError(t('errors:api_endpoint_invalid'));
        }
    },

    /**
     * Validate provider API key
     */
    validateProviderApiKey(provider, apiKey, endpoint = '') {
        if (provider === 'gemini') {
            if (!apiKey || apiKey.trim() === '') {
                return this.showError(t('errors:api_key_required_gemini'));
            }
        }

        if (provider === 'openai') {
            const isLocalEndpoint = endpoint.includes('localhost') || endpoint.includes('127.0.0.1');
            if (!isLocalEndpoint && (!apiKey || apiKey.trim() === '')) {
                return this.showError(t('errors:api_key_required_openai'));
            }
        }

        return true;
    },


    /**
     * Validate file selection
     */
    validateFileSelection(files) {
        if (!files || files.length === 0) {
            return this.showError(t('errors:no_file_selected'));
        }
        return true;
    },

};
