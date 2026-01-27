/**
 * API Key Utilities - Centralized API key handling
 *
 * Provides shared functionality for API key value retrieval,
 * availability checking, and provider validation.
 */

import { DomHelpers } from '../ui/dom-helpers.js';

/**
 * Map of field IDs to their status span IDs
 */
const STATUS_ID_MAP = {
    'geminiApiKey': 'geminiKeyStatus',
    'openaiApiKey': 'openaiKeyStatus',
    'openrouterApiKey': 'openrouterKeyStatus',
    'mistralApiKey': 'mistralKeyStatus',
    'deepseekApiKey': 'deepseekKeyStatus',
    'poeApiKey': 'poeKeyStatus'
};

/**
 * Map of providers to their API key field IDs
 */
const PROVIDER_FIELD_MAP = {
    'gemini': 'geminiApiKey',
    'openai': 'openaiApiKey',
    'openrouter': 'openrouterApiKey',
    'mistral': 'mistralApiKey',
    'deepseek': 'deepseekApiKey',
    'poe': 'poeApiKey'
};

export const ApiKeyUtils = {
    /**
     * Get API key value from field, handling .env configured keys
     * If field is empty but configured in .env, returns special marker for backend
     * @param {string} fieldId - Field ID
     * @returns {string} API key value or '__USE_ENV__' marker
     */
    getValue(fieldId) {
        const field = DomHelpers.getElement(fieldId);
        if (!field) return '';

        const value = field.value.trim();

        // If user entered a value, use it
        if (value) {
            return value;
        }

        // If field is empty but .env has a key configured, tell backend to use .env key
        if (field.dataset.envConfigured === 'true') {
            return '__USE_ENV__';
        }

        return '';
    },

    /**
     * Check if API key is available (either user entered or configured in .env)
     * @param {string} fieldId - Field ID
     * @returns {boolean} True if key is available
     */
    isAvailable(fieldId) {
        const field = DomHelpers.getElement(fieldId);
        if (!field) return false;

        // Key is available if: user entered a value OR .env has it configured
        return field.value.trim() !== '' || field.dataset.envConfigured === 'true';
    },

    /**
     * Get the field ID for a given provider
     * @param {string} provider - Provider name (gemini, openai, openrouter)
     * @returns {string|null} Field ID or null if not found
     */
    getFieldIdForProvider(provider) {
        return PROVIDER_FIELD_MAP[provider] || null;
    },

    /**
     * Get the status span ID for a given field
     * @param {string} fieldId - Field ID
     * @returns {string|null} Status span ID or null if not found
     */
    getStatusIdForField(fieldId) {
        return STATUS_ID_MAP[fieldId] || null;
    },

    /**
     * Get API key value for a specific provider
     * @param {string} provider - Provider name
     * @returns {string} API key value or empty string
     */
    getValueForProvider(provider) {
        const fieldId = this.getFieldIdForProvider(provider);
        if (!fieldId) return '';
        return this.getValue(fieldId);
    },

    /**
     * Check if API key is available for a specific provider
     * @param {string} provider - Provider name
     * @returns {boolean} True if key is available
     */
    isAvailableForProvider(provider) {
        const fieldId = this.getFieldIdForProvider(provider);
        if (!fieldId) return false;
        return this.isAvailable(fieldId);
    },

    /**
     * Setup API key field with proper placeholder/indicator and status badge
     * @param {string} fieldId - Input field ID
     * @param {boolean} isConfigured - Whether key is configured in .env
     * @param {string} maskedValue - Masked value (e.g., "***1234") if configured
     */
    setupField(fieldId, isConfigured, maskedValue) {
        const field = DomHelpers.getElement(fieldId);
        if (!field) return;

        const statusSpan = DomHelpers.getElement(this.getStatusIdForField(fieldId));

        if (isConfigured) {
            // Key is configured in .env - show masked indicator as placeholder
            field.value = '';
            field.placeholder = maskedValue
                ? `Using .env key (${maskedValue})`
                : 'Using .env key';
            field.dataset.envConfigured = 'true';

            // Update status badge
            if (statusSpan) {
                statusSpan.textContent = '✓ Configured';
                statusSpan.className = 'key-status configured';
            }
        } else {
            // Key is NOT configured - show instruction placeholder
            field.value = '';
            field.dataset.envConfigured = 'false';
            // Keep original placeholder from HTML

            // Update status badge
            if (statusSpan) {
                statusSpan.textContent = '⚠ Not set';
                statusSpan.className = 'key-status not-configured';
            }
        }
    },

    /**
     * Validate API key for a provider, with special handling for OpenAI local endpoints
     * @param {string} provider - Provider name
     * @param {string} endpoint - API endpoint (used for OpenAI local endpoint detection)
     * @returns {{valid: boolean, message: string}} Validation result
     */
    validateForProvider(provider, endpoint = '') {
        const fieldId = this.getFieldIdForProvider(provider);

        // Provider doesn't require API key (e.g., ollama)
        if (!fieldId) {
            return { valid: true, message: '' };
        }

        const isAvailable = this.isAvailable(fieldId);

        if (provider === 'gemini' && !isAvailable) {
            return { valid: false, message: 'Gemini API key is required when using Gemini provider.' };
        }

        if (provider === 'openai' && !isAvailable) {
            // OpenAI API key is only required for official OpenAI endpoint
            // Local servers (llama.cpp, LM Studio, vLLM, etc.) don't need an API key
            const isLocalEndpoint = endpoint.includes('localhost') || endpoint.includes('127.0.0.1');
            const isOfficialEndpoint = endpoint.includes('api.openai.com');

            if (isOfficialEndpoint || !isLocalEndpoint) {
                return { valid: false, message: 'API key is required when using OpenAI cloud API.' };
            }
        }

        if (provider === 'openrouter' && !isAvailable) {
            return { valid: false, message: 'OpenRouter API key is required when using OpenRouter provider.' };
        }

        if (provider === 'mistral' && !isAvailable) {
            return { valid: false, message: 'Mistral API key is required when using Mistral provider.' };
        }

        if (provider === 'deepseek' && !isAvailable) {
            return { valid: false, message: 'DeepSeek API key is required when using DeepSeek provider.' };
        }

        if (provider === 'poe' && !isAvailable) {
            return { valid: false, message: 'Poe API key is required. Get your key at poe.com/api_key' };
        }

        return { valid: true, message: '' };
    }
};
