/**
 * Provider Manager - LLM provider switching and model loading
 *
 * Manages switching between different LLM providers (Ollama, Gemini, OpenAI)
 * and loading available models for each provider.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { ModelDetector } from './model-detector.js';
import { SettingsManager } from '../core/settings-manager.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import { StatusManager } from '../utils/status-manager.js';

/**
 * Common OpenAI models list
 */
const OPENAI_MODELS = [
    { value: 'gpt-4o', label: 'GPT-4o (Latest)' },
    { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
    { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
    { value: 'gpt-4', label: 'GPT-4' },
    { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' }
];

/**
 * Fallback OpenRouter models list (used when API fetch fails)
 * Sorted by cost: cheap first
 */
const OPENROUTER_FALLBACK_MODELS = [
    // Cheap models
    { value: 'google/gemini-2.0-flash-001', label: 'Gemini 2.0 Flash' },
    { value: 'meta-llama/llama-3.3-70b-instruct', label: 'Llama 3.3 70B' },
    { value: 'qwen/qwen-2.5-72b-instruct', label: 'Qwen 2.5 72B' },
    { value: 'mistralai/mistral-small-24b-instruct-2501', label: 'Mistral Small 24B' },
    // Mid-tier models
    { value: 'anthropic/claude-3-5-haiku-20241022', label: 'Claude 3.5 Haiku' },
    { value: 'openai/gpt-4o-mini', label: 'GPT-4o Mini' },
    { value: 'google/gemini-1.5-pro', label: 'Gemini 1.5 Pro' },
    { value: 'deepseek/deepseek-chat', label: 'DeepSeek Chat' },
    // Premium models
    { value: 'anthropic/claude-sonnet-4', label: 'Claude Sonnet 4' },
    { value: 'openai/gpt-4o', label: 'GPT-4o' },
    { value: 'anthropic/claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet' }
];

/**
 * Auto-retry configuration for Ollama
 */
const OLLAMA_RETRY_INTERVAL = 3000; // 3 seconds
const OLLAMA_MAX_SILENT_RETRIES = 5; // Show message after 5 failed attempts
let ollamaRetryTimer = null;
let ollamaRetryCount = 0;

/**
 * Format price for display (per 1M tokens)
 * @param {number} price - Price per 1M tokens
 * @returns {string} Formatted price string
 */
function formatPrice(price) {
    if (price === 0) return 'Free';
    if (price < 0.01) return '<$0.01';
    if (price < 1) return `$${price.toFixed(2)}`;
    return `$${price.toFixed(2)}`;
}

/**
 * Populate model select with options
 * @param {Array} models - Array of model objects or strings
 * @param {string} defaultModel - Default model to select (from .env)
 * @param {string} provider - Provider type ('ollama', 'gemini', 'openai', 'openrouter')
 * @returns {boolean} True if defaultModel was found and selected
 */
function populateModelSelect(models, defaultModel = null, provider = 'ollama') {
    const modelSelect = DomHelpers.getElement('model');
    if (!modelSelect) return false;

    modelSelect.innerHTML = '';
    let defaultModelFound = false;

    if (provider === 'gemini') {
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = `${model.displayName || model.name} - ${model.description || ''}`;
            option.title = `Input: ${model.inputTokenLimit || 'N/A'} tokens, Output: ${model.outputTokenLimit || 'N/A'} tokens`;
            if (model.name === defaultModel) {
                option.selected = true;
                defaultModelFound = true;
            }
            modelSelect.appendChild(option);
        });
    } else if (provider === 'openai') {
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.label;
            if (model.value === defaultModel) {
                option.selected = true;
                defaultModelFound = true;
            }
            modelSelect.appendChild(option);
        });
    } else if (provider === 'openrouter') {
        models.forEach(model => {
            const option = document.createElement('option');
            // Handle both API response format (id) and fallback format (value)
            const modelId = model.id || model.value;
            option.value = modelId;

            // Format label with pricing info if available
            if (model.pricing && model.pricing.prompt_per_million !== undefined) {
                const inputPrice = formatPrice(model.pricing.prompt_per_million);
                const outputPrice = formatPrice(model.pricing.completion_per_million);
                option.textContent = `${model.name || modelId} (In: ${inputPrice}/M, Out: ${outputPrice}/M)`;
                option.title = `Context: ${model.context_length || 'N/A'} tokens`;
            } else {
                // Fallback format
                option.textContent = model.label || model.name || modelId;
            }

            if (modelId === defaultModel) {
                option.selected = true;
                defaultModelFound = true;
            }
            modelSelect.appendChild(option);
        });
    } else if (provider === 'mistral') {
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.label;
            if (model.context_length) {
                option.title = `Context: ${model.context_length} tokens`;
            }
            if (model.value === defaultModel) {
                option.selected = true;
                defaultModelFound = true;
            }
            modelSelect.appendChild(option);
        });
    } else {
        // Ollama - models are strings
        models.forEach(modelName => {
            const option = document.createElement('option');
            option.value = modelName;
            option.textContent = modelName;
            if (modelName === defaultModel) {
                option.selected = true;
                defaultModelFound = true;
            }
            modelSelect.appendChild(option);
        });
    }

    return defaultModelFound;
}

export const ProviderManager = {
    /**
     * Initialize provider manager
     */
    initialize() {
        const providerSelect = DomHelpers.getElement('llmProvider');

        if (providerSelect) {
            providerSelect.addEventListener('change', () => {
                // Stop any ongoing Ollama retries when switching providers
                this.stopOllamaAutoRetry();
                this.toggleProviderSettings();
            });
        }

        // Add listener for OpenAI endpoint changes (for local server support)
        const openaiEndpoint = DomHelpers.getElement('openaiEndpoint');
        if (openaiEndpoint) {
            // Use debounce to avoid too many requests while typing
            let endpointTimeout = null;
            openaiEndpoint.addEventListener('input', () => {
                clearTimeout(endpointTimeout);
                endpointTimeout = setTimeout(() => {
                    const currentProvider = DomHelpers.getValue('llmProvider');
                    if (currentProvider === 'openai') {
                        this.loadOpenAIModels();
                    }
                }, 500); // Wait 500ms after user stops typing
            });
        }

        // Show initial provider settings and load models immediately
        this.toggleProviderSettings(true);
    },

    /**
     * Toggle provider-specific settings visibility
     * @param {boolean} loadModels - Whether to load models (default: true)
     */
    toggleProviderSettings(loadModels = true) {
        const provider = DomHelpers.getValue('llmProvider');

        // Update state
        StateManager.setState('ui.currentProvider', provider);

        // Get provider settings elements
        const ollamaSettings = DomHelpers.getElement('ollamaSettings');
        const geminiSettings = DomHelpers.getElement('geminiSettings');
        const openaiApiKeyGroup = DomHelpers.getElement('openaiApiKeyGroup');
        const openaiEndpointRow = DomHelpers.getElement('openaiEndpointRow');
        const openrouterSettings = DomHelpers.getElement('openrouterSettings');

        // Get mistral settings element once
        const mistralSettings = DomHelpers.getElement('mistralSettings');

        // Show/hide provider-specific settings (use inline style for elements with inline display:none)
        if (provider === 'ollama') {
            DomHelpers.show('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (loadModels) this.loadOllamaModels();
        } else if (provider === 'gemini') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'block';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (loadModels) this.loadGeminiModels();
        } else if (provider === 'openai') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'block';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'block';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (loadModels) this.loadOpenAIModels();
        } else if (provider === 'openrouter') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'block';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (loadModels) this.loadOpenRouterModels();
        } else if (provider === 'mistral') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'block';
            if (loadModels) this.loadMistralModels();
        }
    },

    /**
     * Refresh models for current provider
     */
    refreshModels() {
        const provider = DomHelpers.getValue('llmProvider');

        if (provider === 'ollama') {
            this.loadOllamaModels();
        } else if (provider === 'gemini') {
            this.loadGeminiModels();
        } else if (provider === 'openai') {
            this.loadOpenAIModels();
        } else if (provider === 'openrouter') {
            this.loadOpenRouterModels();
        } else if (provider === 'mistral') {
            this.loadMistralModels();
        }
    },

    /**
     * Load Ollama models with auto-retry on failure
     */
    async loadOllamaModels() {
        const modelSelect = DomHelpers.getElement('model');
        if (!modelSelect) return;

        // Cancel any pending request
        const currentRequest = StateManager.getState('models.currentLoadRequest');
        if (currentRequest) {
            currentRequest.cancelled = true;
        }

        // Create new request tracker
        const thisRequest = { cancelled: false };
        StateManager.setState('models.currentLoadRequest', thisRequest);

        modelSelect.innerHTML = '<option value="">Loading models...</option>';
        StatusManager.setChecking();

        try {
            const apiEndpoint = DomHelpers.getValue('apiEndpoint');
            const data = await ApiClient.getModels('ollama', { apiEndpoint });

            // Check if request was cancelled
            if (thisRequest.cancelled) {
                console.log('Model load request was cancelled');
                return;
            }

            // Verify provider hasn't changed
            const currentProvider = DomHelpers.getValue('llmProvider');
            if (currentProvider !== 'ollama') {
                console.log('Provider changed during model load, ignoring Ollama response');
                return;
            }

            if (data.models && data.models.length > 0) {
                // Success - stop auto-retry
                this.stopOllamaAutoRetry();

                MessageLogger.showMessage('', '');
                const envModelApplied = populateModelSelect(data.models, data.default, 'ollama');
                MessageLogger.addLog(`✅ ${data.count} LLM model(s) loaded. Default: ${data.default}`);

                // If .env model was found and applied, lock it in
                if (envModelApplied && data.default) {
                    SettingsManager.markEnvModelApplied();
                }

                // Apply saved model preference if any (will be skipped if .env model was applied)
                SettingsManager.applyPendingModelSelection();

                ModelDetector.checkAndShowRecommendation();

                // Update available models in state
                StateManager.setState('models.availableModels', data.models);

                // Update status to connected
                StatusManager.setConnected('ollama', data.count);
            } else {
                // No models available - start auto-retry
                const errorMessage = data.error || 'No LLM models available. Ensure Ollama is running and accessible.';

                // Show message only after several retries
                if (ollamaRetryCount >= OLLAMA_MAX_SILENT_RETRIES) {
                    MessageLogger.showMessage(`⚠️ ${errorMessage}`, 'error');
                    MessageLogger.addLog(`⚠️ No models available from Ollama at ${apiEndpoint} (auto-retrying every ${OLLAMA_RETRY_INTERVAL/1000}s...)`);
                }

                modelSelect.innerHTML = '<option value="">Waiting for Ollama...</option>';
                StatusManager.setWaiting('Waiting for Ollama...');
                this.startOllamaAutoRetry();
            }

        } catch (error) {
            if (!thisRequest.cancelled) {
                // Connection error - start auto-retry
                if (ollamaRetryCount >= OLLAMA_MAX_SILENT_RETRIES) {
                    MessageLogger.showMessage(`⚠️ Waiting for Ollama to start...`, 'warning');
                    MessageLogger.addLog(`⚠️ Ollama not accessible, auto-retrying every ${OLLAMA_RETRY_INTERVAL/1000}s...`);
                }

                modelSelect.innerHTML = '<option value="">Waiting for Ollama...</option>';
                StatusManager.setDisconnected('Not accessible');
                this.startOllamaAutoRetry();
            }
        } finally {
            // Clear request tracker if it's still ours
            if (StateManager.getState('models.currentLoadRequest') === thisRequest) {
                StateManager.setState('models.currentLoadRequest', null);
            }
        }
    },

    /**
     * Start auto-retry mechanism for Ollama
     */
    startOllamaAutoRetry() {
        // Don't start if already running
        if (ollamaRetryTimer) {
            return;
        }

        ollamaRetryCount++;

        ollamaRetryTimer = setTimeout(() => {
            ollamaRetryTimer = null;

            // Only retry if still on Ollama provider
            const currentProvider = DomHelpers.getValue('llmProvider');
            if (currentProvider === 'ollama') {
                console.log(`Auto-retrying Ollama connection (attempt ${ollamaRetryCount})...`);
                this.loadOllamaModels();
            }
        }, OLLAMA_RETRY_INTERVAL);
    },

    /**
     * Stop auto-retry mechanism for Ollama
     */
    stopOllamaAutoRetry() {
        if (ollamaRetryTimer) {
            clearTimeout(ollamaRetryTimer);
            ollamaRetryTimer = null;
        }
        ollamaRetryCount = 0;
    },

    /**
     * Load Gemini models
     */
    async loadGeminiModels() {
        const modelSelect = DomHelpers.getElement('model');
        if (!modelSelect) return;

        modelSelect.innerHTML = '<option value="">Loading Gemini models...</option>';
        StatusManager.setChecking();

        try {
            // Use ApiKeyUtils to get API key (returns '__USE_ENV__' if configured in .env)
            const apiKey = ApiKeyUtils.getValue('geminiApiKey');
            const data = await ApiClient.getModels('gemini', { apiKey });

            if (data.models && data.models.length > 0) {
                MessageLogger.showMessage('', '');
                const envModelApplied = populateModelSelect(data.models, data.default, 'gemini');
                MessageLogger.addLog(`✅ ${data.count} Gemini model(s) loaded (excluding thinking models)`);

                // If .env model was found and applied, lock it in
                if (envModelApplied && data.default) {
                    SettingsManager.markEnvModelApplied();
                }

                // Apply saved model preference if any (will be skipped if .env model was applied)
                SettingsManager.applyPendingModelSelection();

                ModelDetector.checkAndShowRecommendation();

                // Update available models in state
                StateManager.setState('models.availableModels', data.models);

                // Update status to connected
                StatusManager.setConnected('gemini', data.count);
            } else {
                const errorMessage = data.error || 'No Gemini models available.';
                MessageLogger.showMessage(`⚠️ ${errorMessage}`, 'error');
                modelSelect.innerHTML = '<option value="">No models available</option>';
                MessageLogger.addLog(`⚠️ No Gemini models available`);
                StatusManager.setError('No models');
            }

        } catch (error) {
            MessageLogger.showMessage(`❌ Error fetching Gemini models: ${error.message}`, 'error');
            MessageLogger.addLog(`❌ Failed to retrieve Gemini model list: ${error.message}`);
            modelSelect.innerHTML = '<option value="">Error loading models</option>';
            StatusManager.setError(error.message);
        }
    },

    /**
     * Load OpenAI-compatible models dynamically
     * Always tries to fetch models dynamically from any OpenAI-compatible endpoint.
     * Falls back to static list if dynamic fetch fails.
     */
    async loadOpenAIModels() {
        const modelSelect = DomHelpers.getElement('model');
        if (!modelSelect) return;

        const apiEndpoint = DomHelpers.getValue('openaiEndpoint') || 'https://api.openai.com/v1/chat/completions';

        modelSelect.innerHTML = '<option value="">Loading models...</option>';
        StatusManager.setChecking();

        try {
            const apiKey = ApiKeyUtils.getValue('openaiApiKey');
            const data = await ApiClient.getModels('openai', { apiKey, apiEndpoint });

            if (data.models && data.models.length > 0) {
                MessageLogger.showMessage('', '');

                // Format models for the dropdown
                const formattedModels = data.models.map(m => ({
                    value: m.id,
                    label: m.name || m.id
                }));

                const envModelApplied = populateModelSelect(formattedModels, data.default, 'openai');
                MessageLogger.addLog(`✅ ${data.count} model(s) loaded from OpenAI-compatible endpoint`);

                if (envModelApplied && data.default) {
                    SettingsManager.markEnvModelApplied();
                }

                SettingsManager.applyPendingModelSelection();
                ModelDetector.checkAndShowRecommendation();

                StateManager.setState('models.availableModels', formattedModels.map(m => m.value));
                StatusManager.setConnected('openai', data.count);
                return;
            } else {
                // No models returned from endpoint
                const errorMsg = data.error || 'No models available from endpoint';
                MessageLogger.showMessage(`⚠️ ${errorMsg}. Using fallback OpenAI models.`, 'warning');
                MessageLogger.addLog(`⚠️ ${errorMsg}. Using fallback list.`);
            }
        } catch (error) {
            MessageLogger.showMessage(`⚠️ Could not connect to endpoint. Using fallback OpenAI models.`, 'warning');
            MessageLogger.addLog(`⚠️ Connection error: ${error.message}. Using fallback list.`);
        }

        // Fallback: use static OpenAI models list
        populateModelSelect(OPENAI_MODELS, null, 'openai');
        MessageLogger.addLog(`✅ OpenAI models loaded (common models)`);

        SettingsManager.applyPendingModelSelection();
        ModelDetector.checkAndShowRecommendation();

        StateManager.setState('models.availableModels', OPENAI_MODELS.map(m => m.value));
        StatusManager.setConnected('openai', OPENAI_MODELS.length);
    },

    /**
     * Load OpenRouter models dynamically from API (text-only models, sorted by price)
     */
    async loadOpenRouterModels() {
        const modelSelect = DomHelpers.getElement('model');
        if (!modelSelect) return;

        modelSelect.innerHTML = '<option value="">Loading OpenRouter models...</option>';
        StatusManager.setChecking();

        try {
            // Use ApiKeyUtils to get API key (returns '__USE_ENV__' if configured in .env)
            const apiKey = ApiKeyUtils.getValue('openrouterApiKey');
            const data = await ApiClient.getModels('openrouter', { apiKey });

            if (data.models && data.models.length > 0) {
                MessageLogger.showMessage('', '');
                const envModelApplied = populateModelSelect(data.models, data.default, 'openrouter');
                MessageLogger.addLog(`✅ ${data.count} OpenRouter text models loaded (sorted by price, cheapest first)`);

                // If .env model was found and applied, lock it in
                if (envModelApplied && data.default) {
                    SettingsManager.markEnvModelApplied();
                }

                // Apply saved model preference if any (will be skipped if .env model was applied)
                SettingsManager.applyPendingModelSelection();

                ModelDetector.checkAndShowRecommendation();

                // Update available models in state
                StateManager.setState('models.availableModels', data.models.map(m => m.id));

                // Update status to connected
                StatusManager.setConnected('openrouter', data.count);
            } else {
                // Use fallback list
                const errorMessage = data.error || 'Could not load models from OpenRouter API';
                MessageLogger.showMessage(`⚠️ ${errorMessage}. Using fallback list.`, 'warning');
                populateModelSelect(OPENROUTER_FALLBACK_MODELS, 'anthropic/claude-sonnet-4', 'openrouter');
                MessageLogger.addLog(`⚠️ Using fallback OpenRouter models list`);

                // Update available models in state
                StateManager.setState('models.availableModels', OPENROUTER_FALLBACK_MODELS.map(m => m.value));

                // Still mark as connected since we have fallback models
                StatusManager.setConnected('openrouter', OPENROUTER_FALLBACK_MODELS.length);
            }

        } catch (error) {
            // Use fallback list on error
            MessageLogger.showMessage(`⚠️ Error fetching OpenRouter models. Using fallback list.`, 'warning');
            MessageLogger.addLog(`⚠️ OpenRouter API error: ${error.message}. Using fallback list.`);
            populateModelSelect(OPENROUTER_FALLBACK_MODELS, 'anthropic/claude-sonnet-4', 'openrouter');

            // Update available models in state
            StateManager.setState('models.availableModels', OPENROUTER_FALLBACK_MODELS.map(m => m.value));

            // Still mark as connected since we have fallback models
            StatusManager.setConnected('openrouter', OPENROUTER_FALLBACK_MODELS.length);
        }
    },

    /**
     * Load Mistral models dynamically from API
     */
    async loadMistralModels() {
        const modelSelect = DomHelpers.getElement('model');
        if (!modelSelect) return;

        modelSelect.innerHTML = '<option value="">Loading Mistral models...</option>';
        StatusManager.setChecking();

        try {
            // Use ApiKeyUtils to get API key (returns '__USE_ENV__' if configured in .env)
            const apiKey = ApiKeyUtils.getValue('mistralApiKey');
            if (!apiKey) {
                MessageLogger.showMessage('⚠️ Mistral API key required', 'warning');
                modelSelect.innerHTML = '<option value="">Enter API key first</option>';
                StatusManager.setError('No API key');
                return;
            }

            const data = await ApiClient.getModels('mistral', { apiKey });

            if (data.models && data.models.length > 0) {
                MessageLogger.showMessage('', '');

                // Format models for the dropdown
                const formattedModels = data.models.map(m => ({
                    value: m.id,
                    label: m.name || m.id,
                    context_length: m.context_length
                }));

                populateModelSelect(formattedModels, data.default, 'mistral');
                MessageLogger.addLog(`✅ ${data.count} Mistral model(s) loaded`);

                SettingsManager.applyPendingModelSelection();
                ModelDetector.checkAndShowRecommendation();

                StateManager.setState('models.availableModels', formattedModels.map(m => m.value));
                StatusManager.setConnected('mistral', data.count);
            } else {
                const errorMessage = data.error || 'No Mistral models available';
                MessageLogger.showMessage(`⚠️ ${errorMessage}`, 'error');
                modelSelect.innerHTML = '<option value="">No models available</option>';
                StatusManager.setError('No models');
            }
        } catch (error) {
            MessageLogger.showMessage(`❌ Error: ${error.message}`, 'error');
            modelSelect.innerHTML = '<option value="">Error loading models</option>';
            StatusManager.setError(error.message);
        }
    },

    /**
     * Get current provider
     * @returns {string} Current provider ('ollama', 'gemini', 'openai', 'openrouter')
     */
    getCurrentProvider() {
        return StateManager.getState('ui.currentProvider') || DomHelpers.getValue('llmProvider');
    },

    /**
     * Get current model
     * @returns {string} Current model name
     */
    getCurrentModel() {
        return StateManager.getState('ui.currentModel') || DomHelpers.getValue('model');
    },

    /**
     * Set current model
     * @param {string} modelName - Model name to set
     */
    setCurrentModel(modelName) {
        DomHelpers.setValue('model', modelName);
        StateManager.setState('ui.currentModel', modelName);
        ModelDetector.checkAndShowRecommendation();
    }
};
