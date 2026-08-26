/**
 * API Key Utilities - Centralized API key handling
 *
 * Provides shared functionality for API key value retrieval,
 * availability checking, and provider validation.
 */

import { DomHelpers } from '../ui/dom-helpers.js';
import { t } from '../i18n/i18n.js';

/**
 * Map of field IDs to their status span IDs
 */
const STATUS_ID_MAP = {
    'geminiApiKey': 'geminiKeyStatus',
    'openaiApiKey': 'openaiKeyStatus',
    'openrouterApiKey': 'openrouterKeyStatus',
    'mistralApiKey': 'mistralKeyStatus',
    'deepseekApiKey': 'deepseekKeyStatus',
    'poeApiKey': 'poeKeyStatus',
    'nimApiKey': 'nimKeyStatus',
    'anthropicApiKey': 'anthropicKeyStatus',
    'xaiApiKey': 'xaiKeyStatus',
    'opencodeApiKey': 'opencodeKeyStatus',
    'opencodegoApiKey': 'opencodegoKeyStatus',
    'ollamacloudApiKey': 'ollamacloudKeyStatus'
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
    'poe': 'poeApiKey',
    'nim': 'nimApiKey',
    'anthropic': 'anthropicApiKey',
    'xai': 'xaiApiKey',
    'opencode': 'opencodeApiKey',
    'opencodego': 'opencodegoApiKey',
    'ollamacloud': 'ollamacloudApiKey'
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
     * Setup API key field with proper placeholder/indicator and status badge.
     * When `keyCount > 1`, signals the active rotation pool size so users can
     * verify their multi-key .env config actually parsed.
     *
     * @param {string} fieldId - Input field ID
     * @param {boolean} isConfigured - Whether key is configured in .env
     * @param {string} maskedValue - Masked indicator (e.g. "***1234") for the last key
     * @param {number} [keyCount=1] - Number of keys in the rotation pool
     */
    setupField(fieldId, isConfigured, maskedValue, keyCount = 1) {
        const field = DomHelpers.getElement(fieldId);
        if (!field) return;

        const statusSpan = DomHelpers.getElement(this.getStatusIdForField(fieldId));

        if (isConfigured) {
            field.value = '';
            const count = Math.max(1, keyCount | 0);
            if (count > 1) {
                field.placeholder = maskedValue
                    ? t('settings:key_using_env_multi_with_masked', { count, masked: maskedValue })
                    : t('settings:key_using_env_multi', { count });
            } else {
                field.placeholder = maskedValue
                    ? t('settings:key_using_env_single_with_masked', { masked: maskedValue })
                    : t('settings:key_using_env_single');
            }
            field.dataset.envConfigured = 'true';
            field.dataset.envKeyCount = String(count);

            if (statusSpan) {
                statusSpan.textContent = count > 1
                    ? t('settings:key_status_configured_rotation', { count })
                    : t('settings:key_status_configured');
                statusSpan.className = 'key-status configured';
            }
        } else {
            field.value = '';
            field.dataset.envConfigured = 'false';
            field.dataset.envKeyCount = '0';

            if (statusSpan) {
                statusSpan.textContent = t('settings:key_status_not_configured_badge');
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
            return { valid: false, message: t('errors:api_key_required_gemini') };
        }

        if (provider === 'openai' && !isAvailable) {
            // OpenAI API key is only required for official OpenAI endpoint
            // Local servers (llama.cpp, LM Studio, vLLM, etc.) don't need an API key
            const isLocalEndpoint = endpoint.includes('localhost') || endpoint.includes('127.0.0.1');
            const isOfficialEndpoint = endpoint.includes('api.openai.com');

            if (isOfficialEndpoint || !isLocalEndpoint) {
                return { valid: false, message: t('errors:api_key_required_openai') };
            }
        }

        if (provider === 'openrouter' && !isAvailable) {
            return { valid: false, message: t('errors:api_key_required_openrouter') };
        }

        if (provider === 'mistral' && !isAvailable) {
            return { valid: false, message: t('errors:api_key_required_mistral') };
        }

        if (provider === 'deepseek' && !isAvailable) {
            return { valid: false, message: t('errors:api_key_required_deepseek') };
        }

        if (provider === 'poe' && !isAvailable) {
            return { valid: false, message: t('errors:api_key_required_poe') };
        }

        if (provider === 'nim' && !isAvailable) {
            return { valid: false, message: t('errors:api_key_required_nim') };
        }

        if (['anthropic', 'xai', 'opencode', 'opencodego', 'ollamacloud'].includes(provider) && !isAvailable) {
            return { valid: false, message: t('errors:api_key_required') };
        }

        if (provider === 'chatgpt') {
            const signedIn = document.getElementById('chatgptOauthStatus')
                ?.getAttribute('data-i18n') === 'settings:chatgpt_signed_in';
            if (!signedIn) {
                return { valid: false, message: t('settings:chatgpt_sign_in_to_load') };
            }
        }

        return { valid: true, message: '' };
    }
};
