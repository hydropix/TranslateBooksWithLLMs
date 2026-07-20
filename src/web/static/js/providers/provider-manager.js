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
import { SearchableSelectFactory } from '../ui/searchable-select.js';
import { t } from '../i18n/i18n.js';
import {
    PROVIDER_LOGOS,
    PROVIDER_META,
    populateModelSelectInto,
    setPlaceholderOption as setPlaceholderOptionShared,
    renderProviderOption,
    providerDisplayHtml,
} from './provider-select-helpers.js';

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
 * Fallback DeepSeek models list (used when API fetch fails)
 */
const DEEPSEEK_FALLBACK_MODELS = [
    { value: 'deepseek-v4-pro', label: 'DeepSeek V4 Pro' },
    { value: 'deepseek-v4-flash', label: 'DeepSeek V4 Flash' },
    { value: 'deepseek-reasoner', label: 'DeepSeek Reasoner (Thinking)' }
];

const ATLASCLOUD_FALLBACK_MODELS = [
    { value: 'qwen/qwen3.5-flash', label: 'Qwen 3.5 Flash' },
    { value: 'deepseek-ai/deepseek-v4-pro', label: 'DeepSeek V4 Pro' }
];

/**
 * Comprehensive Poe models list - organized by provider
 * Poe has a /v1/models API endpoint, but this fallback list is used when API fails
 * Updated based on https://api.poe.com/v1/models response
 * Get full list at: https://poe.com/explore
 */
const POE_FALLBACK_MODELS = [
    // === Anthropic Claude ===
    { value: 'claude-opus-4.5', label: 'Claude Opus 4.5 (196k ctx)', group: 'Anthropic' },
    { value: 'claude-opus-4.1', label: 'Claude Opus 4.1 (196k ctx)', group: 'Anthropic' },
    { value: 'claude-sonnet-4.5', label: 'Claude Sonnet 4.5 (983k ctx)', group: 'Anthropic' },
    { value: 'claude-haiku-4.5', label: 'Claude Haiku 4.5 (192k ctx)', group: 'Anthropic' },
    { value: 'Claude-Sonnet-4', label: 'Claude Sonnet 4', group: 'Anthropic' },
    { value: 'Claude-3.5-Sonnet', label: 'Claude 3.5 Sonnet', group: 'Anthropic' },
    { value: 'Claude-3.5-Haiku', label: 'Claude 3.5 Haiku', group: 'Anthropic' },

    // === OpenAI GPT ===
    { value: 'gpt-5', label: 'GPT-5 (400k ctx)', group: 'OpenAI' },
    { value: 'gpt-5-mini', label: 'GPT-5 Mini (400k ctx)', group: 'OpenAI' },
    { value: 'gpt-5-nano', label: 'GPT-5 Nano (400k ctx)', group: 'OpenAI' },
    { value: 'gpt-5.2', label: 'GPT-5.2 (400k ctx)', group: 'OpenAI' },
    { value: 'gpt-5.1', label: 'GPT-5.1 (400k ctx)', group: 'OpenAI' },
    { value: 'o3-pro', label: 'o3 Pro (200k ctx, reasoning)', group: 'OpenAI' },
    { value: 'GPT-4o', label: 'GPT-4o (128k ctx)', group: 'OpenAI' },
    { value: 'GPT-4o-Mini', label: 'GPT-4o Mini', group: 'OpenAI' },

    // === Google Gemini ===
    { value: 'gemini-3-pro', label: 'Gemini 3 Pro (1M ctx)', group: 'Google' },
    { value: 'gemini-3-flash', label: 'Gemini 3 Flash (1M ctx)', group: 'Google' },
    { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro (1M ctx)', group: 'Google' },
    { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash (1M ctx)', group: 'Google' },
    { value: 'gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash Lite (1M ctx)', group: 'Google' },

    // === xAI Grok ===
    { value: 'grok-4', label: 'Grok 4 (256k ctx)', group: 'xAI' },
    { value: 'grok-4.1-fast-reasoning', label: 'Grok 4.1 Fast Reasoning (2M ctx)', group: 'xAI' },
    { value: 'grok-4-fast-reasoning', label: 'Grok 4 Fast Reasoning (2M ctx)', group: 'xAI' },
    { value: 'grok-4-fast-non-reasoning', label: 'Grok 4 Fast Non-Reasoning (2M ctx)', group: 'xAI' },

    // === DeepSeek ===
    { value: 'deepseek-r1', label: 'DeepSeek R1 (160k ctx, reasoning)', group: 'DeepSeek' },
    { value: 'deepseek-v3.2', label: 'DeepSeek V3.2 (164k ctx)', group: 'DeepSeek' },
    { value: 'deepseek-v3.2-exp', label: 'DeepSeek V3.2 Exp (160k ctx)', group: 'DeepSeek' },

    // === Qwen (Alibaba) ===
    { value: 'qwen3-max-thinking', label: 'Qwen3 Max Thinking (256k ctx)', group: 'Qwen' },
    { value: 'qwen3-next-80b', label: 'Qwen3 Next 80B (65k ctx)', group: 'Qwen' },
    { value: 'qwen-3-next-80b-think', label: 'Qwen3 Next 80B Think (65k ctx)', group: 'Qwen' },

    // === GLM (Zhipu) ===
    { value: 'glm-4.7', label: 'GLM 4.7 (131k ctx)', group: 'GLM' },
    { value: 'glm-4.7-n', label: 'GLM 4.7-N (205k ctx)', group: 'GLM' },
    { value: 'glm-4.7-flash', label: 'GLM 4.7 Flash (200k ctx)', group: 'GLM' },
    { value: 'glm-4.6', label: 'GLM 4.6 (205k ctx)', group: 'GLM' },

    // === Mistral ===
    { value: 'mistral-medium-3.1', label: 'Mistral Medium 3.1 (131k ctx)', group: 'Mistral' },
    { value: 'Mistral-Large', label: 'Mistral Large', group: 'Mistral' },
    { value: 'Codestral', label: 'Codestral', group: 'Mistral' },

    // === MiniMax ===
    { value: 'minimax-m2.1', label: 'MiniMax M2.1 (205k ctx)', group: 'MiniMax' },
    { value: 'minimax-m2', label: 'MiniMax M2 (200k ctx)', group: 'MiniMax' },

    // === Amazon Nova ===
    { value: 'nova-premier-1.0', label: 'Nova Premier 1.0 (1M ctx)', group: 'Amazon' },
    { value: 'nova-pro-1.0', label: 'Nova Pro 1.0 (300k ctx)', group: 'Amazon' },
    { value: 'nova-lite-1.0', label: 'Nova Lite 1.0 (300k ctx)', group: 'Amazon' },
    { value: 'nova-micro-1.0', label: 'Nova Micro 1.0 (128k ctx)', group: 'Amazon' },

    // === Other ===
    { value: 'kimi-k2-thinking', label: 'Kimi K2 Thinking (256k ctx)', group: 'Other' },
    { value: 'manus', label: 'Manus (Autonomous Agent)', group: 'Other' },

    // === Poe Assistant Bots ===
    { value: 'assistant', label: 'Assistant (Router)', group: 'Poe Bots' },
    { value: 'exa-answer', label: 'Exa Answer (Web Search)', group: 'Poe Bots' },
    { value: 'exa-search', label: 'Exa Search', group: 'Poe Bots' }
];

/**
 * Fallback NVIDIA NIM models list (used when API fetch fails)
 * See all models at: https://build.nvidia.com/explore/discover
 */
const NIM_FALLBACK_MODELS = [
    { value: 'meta/llama-3.1-8b-instruct', label: 'Llama 3.1 8B Instruct (128k ctx)' },
    { value: 'meta/llama-3.1-70b-instruct', label: 'Llama 3.1 70B Instruct (128k ctx)' },
    { value: 'meta/llama-3.1-405b-instruct', label: 'Llama 3.1 405B Instruct (128k ctx)' },
    { value: 'meta/llama-3.2-1b-instruct', label: 'Llama 3.2 1B Instruct (128k ctx)' },
    { value: 'meta/llama-3.2-3b-instruct', label: 'Llama 3.2 3B Instruct (128k ctx)' },
    { value: 'mistralai/mistral-nemo-12b-instruct', label: 'Mistral Nemo 12B Instruct (128k ctx)' },
    { value: 'mistralai/mixtral-8x7b-instruct-v0.1', label: 'Mixtral 8x7B Instruct v0.1 (32k ctx)' },
    { value: 'nvidia/llama-3.1-nemotron-70b-instruct', label: 'Llama 3.1 Nemotron 70B Instruct (128k ctx)' },
    { value: 'deepseek-ai/deepseek-v3', label: 'DeepSeek V3 (128k ctx)' },
    { value: 'deepseek-ai/deepseek-r1', label: 'DeepSeek R1 (128k ctx)' }
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
 * Replace the model dropdown content with a single placeholder option whose
 * text comes from i18n key `i18nKey`. The data-i18n attribute ensures
 * applyToDOM re-translates it on language switch — so a dropdown stuck in
 * "Loading...", "Waiting for Ollama", "Enter API key first", etc. follows
 * the UI locale without requiring us to re-run the original load logic.
 */
function setPlaceholderOption(modelSelect, i18nKey) {
    // Delegates to the shared helper so Settings + Sample dropdowns share the
    // same placeholder semantics (data-i18n keeps the text reactive on locale
    // switch).
    setPlaceholderOptionShared(modelSelect, i18nKey);
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

    // All per-provider rendering (Gemini token labels, OpenRouter / Poe
    // pricing labels, Poe optgroups, Mistral / DeepSeek / NIM tooltips,
    // Ollama plain strings) lives in the shared helper; this wrapper just
    // targets the Settings dropdown and fires the legacy `modelChanged`
    // CustomEvent the cost estimator listens to.
    const defaultFound = populateModelSelectInto(modelSelect, models, defaultModel, provider);

    window.dispatchEvent(new CustomEvent('modelChanged', {
        detail: { value: modelSelect.value }
    }));

    return defaultFound;
}

export const ProviderManager = {
    /**
     * Initialize provider manager
     */
    initialize() {
        const providerSelect = DomHelpers.getElement('llmProvider');

        if (providerSelect) {
            // Initialize SearchableSelect for provider dropdown with logos
            this.initSearchableProviderSelect();

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

        // Initialize SearchableSelect for model dropdown
        this.initSearchableModelSelect();

        // Show initial provider settings UI but DON'T load models yet.
        // We must wait for FormManager.loadDefaultConfig() to complete
        // and update the API endpoints from server configuration.
        // This fixes GitHub issue #108 part 2: Ollama endpoint was using
        // localhost instead of the configured remote server.
        this.toggleProviderSettings(false);

        // Check if config is already loaded (race condition fix)
        const serverConfig = StateManager.getState('ui.defaultConfig');
        if (serverConfig) {
            console.log('[ProviderManager] Config already loaded, loading models immediately');
            this.toggleProviderSettings(true);
        } else {
            // Listen for server config to be loaded, THEN load models with correct endpoint
            console.log('[ProviderManager] Waiting for defaultConfigLoaded event');
            window.addEventListener('defaultConfigLoaded', () => {
                console.log('[ProviderManager] Server config loaded, now loading models with correct endpoint');
                this.toggleProviderSettings(true);
            }, { once: true });
        }
    },

    /**
     * Initialize searchable select for provider dropdown with logos
     */
    initSearchableProviderSelect() {
        const providerSelect = DomHelpers.getElement('llmProvider');
        if (providerSelect) {
            // Logo+name+description row and selected-chip rendering are
            // factored into provider-select-helpers.js so the Sample tab
            // column dropdowns produce visually identical UI.
            SearchableSelectFactory.create('llmProvider', {
                placeholder: t('settings:search_providers_placeholder'),
                showBadge: false,
                renderOption: renderProviderOption,
                onSelect: (option) => {
                    this.updateProviderDisplay(option.value);
                }
            });
            const currentValue = providerSelect.value;
            if (currentValue) {
                this.updateProviderDisplay(currentValue);
            }
        }
    },

    /**
     * Update provider display with logo
     * @param {string} providerValue - Provider value
     */
    updateProviderDisplay(providerValue) {
        const instance = SearchableSelectFactory.get('llmProvider');
        if (instance && instance.displayText) {
            instance.displayText.innerHTML = providerDisplayHtml(providerValue);
        }
    },

    /**
     * Initialize searchable select for model dropdown
     */
    initSearchableModelSelect() {
        const modelSelect = DomHelpers.getElement('model');
        if (modelSelect) {
            SearchableSelectFactory.create('model', {
                placeholder: t('settings:search_models_placeholder'),
                allowCustomValue: true, // Allow custom bot names for Poe
                onSelect: (option) => {
                    // Trigger model detection check
                    ModelDetector.checkAndShowRecommendation();
                    StateManager.setState('ui.currentModel', option.value);
                    window.dispatchEvent(new CustomEvent('modelChanged', {
                        detail: { value: option.value }
                    }));
                }
            });
        }
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

        // Get mistral, deepseek, poe and nim settings elements once
        const mistralSettings = DomHelpers.getElement('mistralSettings');
        const deepseekSettings = DomHelpers.getElement('deepseekSettings');
        const atlascloudSettings = DomHelpers.getElement('atlascloudSettings');
        const poeSettings = DomHelpers.getElement('poeSettings');
        const nimSettings = DomHelpers.getElement('nimSettings');

        // Show/hide provider-specific settings (use inline style for elements with inline display:none)
        if (provider === 'ollama') {
            DomHelpers.show('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (atlascloudSettings) atlascloudSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadOllamaModels();
        } else if (provider === 'poe') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (atlascloudSettings) atlascloudSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'block';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadPoeModels();
        } else if (provider === 'gemini') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'block';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (atlascloudSettings) atlascloudSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadGeminiModels();
        } else if (provider === 'openai') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'block';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'block';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (atlascloudSettings) atlascloudSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadOpenAIModels();
        } else if (provider === 'openrouter') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'block';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (atlascloudSettings) atlascloudSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadOpenRouterModels();
        } else if (provider === 'mistral') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'block';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (atlascloudSettings) atlascloudSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadMistralModels();
        } else if (provider === 'deepseek') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'block';
            if (atlascloudSettings) atlascloudSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadDeepSeekModels();
        } else if (provider === 'atlascloud') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (atlascloudSettings) atlascloudSettings.style.display = 'block';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadAtlasCloudModels();
        } else if (provider === 'nim') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (atlascloudSettings) atlascloudSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'block';
            if (loadModels) this.loadNimModels();
        }

        // Parallel translation is only useful for cloud providers; a single
        // local Ollama instance serializes requests anyway (mirrors the backend
        // LOCAL_PROVIDERS gate). Hide the control for local providers.
        const parallelGroup = DomHelpers.getElement('parallelWorkersGroup');
        if (parallelGroup) {
            const isLocal = provider === 'ollama';
            parallelGroup.style.display = isLocal ? 'none' : 'block';
        }
    },

    /**
     * Refresh models for current provider
     */
    refreshModels() {
        const provider = DomHelpers.getValue('llmProvider');

        if (provider === 'ollama') {
            this.loadOllamaModels();
        } else if (provider === 'poe') {
            this.loadPoeModels();
        } else if (provider === 'gemini') {
            this.loadGeminiModels();
        } else if (provider === 'openai') {
            this.loadOpenAIModels();
        } else if (provider === 'openrouter') {
            this.loadOpenRouterModels();
        } else if (provider === 'mistral') {
            this.loadMistralModels();
        } else if (provider === 'deepseek') {
            this.loadDeepSeekModels();
        } else if (provider === 'atlascloud') {
            this.loadAtlasCloudModels();
        } else if (provider === 'nim') {
            this.loadNimModels();
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

        setPlaceholderOption(modelSelect, 'settings:search_models_loading');
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
                MessageLogger.addLog(t('settings:models_loaded_ollama_log', { count: data.count, default: data.default }));

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
                const errorMessage = data.error || t('settings:no_models_default');

                // Show message only after several retries
                if (ollamaRetryCount >= OLLAMA_MAX_SILENT_RETRIES) {
                    MessageLogger.showMessage(t('settings:no_models_warning', { message: errorMessage }), 'error');
                    MessageLogger.addLog(t('settings:waiting_for_ollama_log', { endpoint: apiEndpoint, interval: OLLAMA_RETRY_INTERVAL / 1000 }));
                }

                setPlaceholderOption(modelSelect, 'settings:search_models_waiting_ollama');
                StatusManager.setWaiting(t('settings:search_models_waiting_ollama'));
                this.startOllamaAutoRetry();
            }

        } catch (error) {
            if (!thisRequest.cancelled) {
                // Connection error - start auto-retry
                if (ollamaRetryCount >= OLLAMA_MAX_SILENT_RETRIES) {
                    MessageLogger.showMessage(t('settings:waiting_for_ollama_msg'), 'warning');
                    MessageLogger.addLog(t('settings:ollama_not_accessible_log', { interval: OLLAMA_RETRY_INTERVAL / 1000 }));
                }

                setPlaceholderOption(modelSelect, 'settings:search_models_waiting_ollama');
                StatusManager.setDisconnected(t('settings:status_not_accessible'));
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

        setPlaceholderOption(modelSelect, 'settings:search_models_loading_gemini');
        StatusManager.setChecking();

        try {
            // Use ApiKeyUtils to get API key (returns '__USE_ENV__' if configured in .env)
            const apiKey = ApiKeyUtils.getValue('geminiApiKey');
            const data = await ApiClient.getModels('gemini', { apiKey });

            if (data.models && data.models.length > 0) {
                MessageLogger.showMessage('', '');
                const envModelApplied = populateModelSelect(data.models, data.default, 'gemini');
                MessageLogger.addLog(t('settings:models_loaded_gemini_log', { count: data.count }));

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
                const errorMessage = data.error || t('settings:no_models_gemini');
                MessageLogger.showMessage(t('settings:no_models_warning', { message: errorMessage }), 'error');
                setPlaceholderOption(modelSelect, 'settings:search_models_no_models_available');
                MessageLogger.addLog(t('settings:models_no_gemini_log'));
                StatusManager.setError(t('settings:status_no_models'));
            }

        } catch (error) {
            MessageLogger.showMessage(t('settings:gemini_fetch_error', { error: error.message }), 'error');
            MessageLogger.addLog(t('settings:gemini_fetch_error_log', { error: error.message }));
            setPlaceholderOption(modelSelect, 'settings:search_models_error');
            StatusManager.setError(error.message);
        }
    },

    /**
     * Load OpenAI-compatible models dynamically.
     * For api.openai.com, falls back to a static cloud list if the live fetch fails.
     * For any custom endpoint, surfaces the error instead of falling back, since
     * the cloud model names would not exist on a local server.
     */
    async loadOpenAIModels() {
        const modelSelect = DomHelpers.getElement('model');
        if (!modelSelect) return;

        const apiEndpoint = DomHelpers.getValue('openaiEndpoint') || 'https://api.openai.com/v1/chat/completions';
        const isOfficialOpenAI = /^https?:\/\/api\.openai\.com(\/|$)/i.test(apiEndpoint);
        const isLocalHttps = /^https:\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])/i.test(apiEndpoint);

        setPlaceholderOption(modelSelect, 'settings:search_models_loading');
        StatusManager.setChecking();

        if (isLocalHttps) {
            MessageLogger.addLog(t('settings:openai_local_https_warning_log', { host: apiEndpoint.slice(8) }));
        }

        try {
            const apiKey = ApiKeyUtils.getValue('openaiApiKey');
            const data = await ApiClient.getModels('openai', { apiKey, apiEndpoint });

            if (data.status === 'openai_connected' && data.models && data.models.length > 0) {
                MessageLogger.showMessage('', '');

                const formattedModels = data.models.map(m => ({
                    value: m.id,
                    label: m.name || m.id
                }));

                const envModelApplied = populateModelSelect(formattedModels, data.default, 'openai');
                MessageLogger.addLog(t('settings:openai_models_loaded_log', { count: data.count }));

                if (envModelApplied && data.default) {
                    SettingsManager.markEnvModelApplied();
                }

                SettingsManager.applyPendingModelSelection();
                ModelDetector.checkAndShowRecommendation();

                StateManager.setState('models.availableModels', formattedModels.map(m => m.value));
                StatusManager.setConnected('openai', data.count);
                return;
            }

            if (data.status === 'openai_error') {
                const errorMsg = data.error || t('settings:openai_endpoint_error_default');
                MessageLogger.showMessage(t('settings:openai_endpoint_error_warning', { error: errorMsg }), 'warning');
                MessageLogger.addLog(t('settings:openai_endpoint_error_log', { error: errorMsg }));
                setPlaceholderOption(modelSelect, 'settings:search_models_no_check_endpoint');
                StateManager.setState('models.availableModels', []);
                StatusManager.setError(errorMsg);
                return;
            }

            if (data.status === 'openai_static' && data.models && data.models.length > 0) {
                const reason = data.error ? ` (${data.error})` : '';
                MessageLogger.showMessage(t('settings:openai_static_warning', { reason }), 'warning');
                MessageLogger.addLog(t('settings:openai_static_log', { reason }));

                const formattedModels = data.models.map(m => ({
                    value: m.id,
                    label: m.name || m.id
                }));
                populateModelSelect(formattedModels, data.default, 'openai');
                SettingsManager.applyPendingModelSelection();
                ModelDetector.checkAndShowRecommendation();

                StateManager.setState('models.availableModels', formattedModels.map(m => m.value));
                StatusManager.setConnected('openai', data.count);
                return;
            }
        } catch (error) {
            MessageLogger.showMessage(t('settings:openai_cannot_reach_endpoint', { error: error.message }), 'warning');
            MessageLogger.addLog(t('settings:openai_connection_error_log', { error: error.message }));

            if (!isOfficialOpenAI) {
                setPlaceholderOption(modelSelect, 'settings:search_models_no_check_endpoint');
                StateManager.setState('models.availableModels', []);
                StatusManager.setError(error.message);
                return;
            }
        }

        // Last-resort static fallback — only reached for the official OpenAI host
        populateModelSelect(OPENAI_MODELS, null, 'openai');
        MessageLogger.addLog(t('settings:openai_models_common_log'));

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

        setPlaceholderOption(modelSelect, 'settings:search_models_loading_openrouter');
        StatusManager.setChecking();

        try {
            // Use ApiKeyUtils to get API key (returns '__USE_ENV__' if configured in .env)
            const apiKey = ApiKeyUtils.getValue('openrouterApiKey');
            const data = await ApiClient.getModels('openrouter', { apiKey });

            if (data.models && data.models.length > 0) {
                MessageLogger.showMessage('', '');
                const envModelApplied = populateModelSelect(data.models, data.default, 'openrouter');
                MessageLogger.addLog(t('settings:openrouter_models_loaded_log', { count: data.count }));

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
                const errorMessage = data.error || t('settings:openrouter_default_error');
                MessageLogger.showMessage(t('settings:openrouter_fallback_warn', { message: errorMessage }), 'warning');
                populateModelSelect(OPENROUTER_FALLBACK_MODELS, 'anthropic/claude-sonnet-4', 'openrouter');
                MessageLogger.addLog(t('settings:openrouter_fallback_log'));

                // Update available models in state
                StateManager.setState('models.availableModels', OPENROUTER_FALLBACK_MODELS.map(m => m.value));

                // Still mark as connected since we have fallback models
                StatusManager.setConnected('openrouter', OPENROUTER_FALLBACK_MODELS.length);
            }

        } catch (error) {
            // Use fallback list on error
            MessageLogger.showMessage(t('settings:openrouter_fetch_error_msg'), 'warning');
            MessageLogger.addLog(t('settings:openrouter_fetch_error_log', { error: error.message }));
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

        setPlaceholderOption(modelSelect, 'settings:search_models_loading_mistral');
        StatusManager.setChecking();

        try {
            // Use ApiKeyUtils to get API key (returns '__USE_ENV__' if configured in .env)
            const apiKey = ApiKeyUtils.getValue('mistralApiKey');
            if (!apiKey) {
                MessageLogger.showMessage(t('settings:mistral_key_required'), 'warning');
                setPlaceholderOption(modelSelect, 'settings:search_models_enter_key_first');
                StatusManager.setError(t('settings:status_no_api_key'));
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
                MessageLogger.addLog(t('settings:mistral_models_loaded_log', { count: data.count }));

                SettingsManager.applyPendingModelSelection();
                ModelDetector.checkAndShowRecommendation();

                StateManager.setState('models.availableModels', formattedModels.map(m => m.value));
                StatusManager.setConnected('mistral', data.count);
            } else {
                const errorMessage = data.error || t('settings:mistral_no_models_default');
                MessageLogger.showMessage(t('settings:no_models_warning', { message: errorMessage }), 'error');
                setPlaceholderOption(modelSelect, 'settings:search_models_no_models_available');
                StatusManager.setError(t('settings:status_no_models'));
            }
        } catch (error) {
            MessageLogger.showMessage(t('settings:mistral_error', { error: error.message }), 'error');
            setPlaceholderOption(modelSelect, 'settings:search_models_error');
            StatusManager.setError(error.message);
        }
    },

    /**
     * Load DeepSeek models dynamically from API
     */
    async loadDeepSeekModels() {
        const modelSelect = DomHelpers.getElement('model');
        if (!modelSelect) return;

        setPlaceholderOption(modelSelect, 'settings:search_models_loading_deepseek');
        StatusManager.setChecking();

        try {
            // Use ApiKeyUtils to get API key (returns '__USE_ENV__' if configured in .env)
            const apiKey = ApiKeyUtils.getValue('deepseekApiKey');
            if (!apiKey) {
                MessageLogger.showMessage(t('settings:deepseek_key_required'), 'warning');
                setPlaceholderOption(modelSelect, 'settings:search_models_enter_key_first');
                StatusManager.setError(t('settings:status_no_api_key'));
                return;
            }

            const data = await ApiClient.getModels('deepseek', { apiKey });

            if (data.models && data.models.length > 0) {
                MessageLogger.showMessage('', '');

                // Format models for the dropdown
                const formattedModels = data.models.map(m => ({
                    value: m.id,
                    label: m.name || m.id,
                    context_length: m.context_length
                }));

                populateModelSelect(formattedModels, data.default, 'deepseek');
                MessageLogger.addLog(t('settings:deepseek_models_loaded_log', { count: data.count }));

                SettingsManager.applyPendingModelSelection();
                ModelDetector.checkAndShowRecommendation();

                StateManager.setState('models.availableModels', formattedModels.map(m => m.value));
                StatusManager.setConnected('deepseek', data.count);
            } else {
                // Use fallback list
                const errorMessage = data.error || t('settings:deepseek_default_error');
                MessageLogger.showMessage(t('settings:deepseek_fallback_msg', { message: errorMessage }), 'warning');
                populateModelSelect(DEEPSEEK_FALLBACK_MODELS, 'deepseek-v4-pro', 'deepseek');
                MessageLogger.addLog(t('settings:deepseek_fallback_log'));

                StateManager.setState('models.availableModels', DEEPSEEK_FALLBACK_MODELS.map(m => m.value));
                StatusManager.setConnected('deepseek', DEEPSEEK_FALLBACK_MODELS.length);
            }
        } catch (error) {
            // Use fallback list on error
            MessageLogger.showMessage(t('settings:deepseek_error_fallback_msg', { error: error.message }), 'warning');
            MessageLogger.addLog(t('settings:deepseek_error_fallback_log', { error: error.message }));
            populateModelSelect(DEEPSEEK_FALLBACK_MODELS, 'deepseek-v4-pro', 'deepseek');

            StateManager.setState('models.availableModels', DEEPSEEK_FALLBACK_MODELS.map(m => m.value));
            StatusManager.setConnected('deepseek', DEEPSEEK_FALLBACK_MODELS.length);
        }
    },

    async loadAtlasCloudModels() {
        const modelSelect = DomHelpers.getElement('model');
        if (!modelSelect) return;

        setPlaceholderOption(modelSelect, 'settings:search_models_loading_atlascloud');
        StatusManager.setChecking();

        try {
            const apiKey = ApiKeyUtils.getValue('atlascloudApiKey');
            if (!apiKey) {
                MessageLogger.showMessage(t('settings:atlascloud_key_required'), 'warning');
                setPlaceholderOption(modelSelect, 'settings:search_models_enter_key_first');
                StatusManager.setError(t('settings:status_no_api_key'));
                return;
            }

            const data = await ApiClient.getModels('atlascloud', { apiKey });

            if (data.models && data.models.length > 0) {
                MessageLogger.showMessage('', '');

                const formattedModels = data.models.map(m => ({
                    value: m.id,
                    label: m.name || m.id,
                    context_length: m.context_length
                }));

                populateModelSelect(formattedModels, data.default, 'atlascloud');
                MessageLogger.addLog(t('settings:atlascloud_models_loaded_log', { count: data.count }));

                SettingsManager.applyPendingModelSelection();
                ModelDetector.checkAndShowRecommendation();

                StateManager.setState('models.availableModels', formattedModels.map(m => m.value));
                StatusManager.setConnected('atlascloud', data.count);
            } else {
                const errorMessage = data.error || t('settings:atlascloud_default_error');
                MessageLogger.showMessage(t('settings:deepseek_fallback_msg', { message: errorMessage }), 'warning');
                populateModelSelect(ATLASCLOUD_FALLBACK_MODELS, 'qwen/qwen3.5-flash', 'atlascloud');
                MessageLogger.addLog(t('settings:atlascloud_fallback_log'));

                StateManager.setState('models.availableModels', ATLASCLOUD_FALLBACK_MODELS.map(m => m.value));
                StatusManager.setConnected('atlascloud', ATLASCLOUD_FALLBACK_MODELS.length);
            }
        } catch (error) {
            MessageLogger.showMessage(t('settings:deepseek_error_fallback_msg', { error: error.message }), 'warning');
            MessageLogger.addLog(t('settings:atlascloud_error_fallback_log', { error: error.message }));
            populateModelSelect(ATLASCLOUD_FALLBACK_MODELS, 'qwen/qwen3.5-flash', 'atlascloud');

            StateManager.setState('models.availableModels', ATLASCLOUD_FALLBACK_MODELS.map(m => m.value));
            StatusManager.setConnected('atlascloud', ATLASCLOUD_FALLBACK_MODELS.length);
        }
    },

    /**
     * Load NVIDIA NIM models dynamically from API
     */
    async loadNimModels() {
        const modelSelect = DomHelpers.getElement('model');
        if (!modelSelect) return;

        setPlaceholderOption(modelSelect, 'settings:search_models_loading_nim');
        StatusManager.setChecking();

        try {
            const apiKey = ApiKeyUtils.getValue('nimApiKey');
            if (!apiKey) {
                MessageLogger.showMessage(t('settings:nim_key_required'), 'warning');
                setPlaceholderOption(modelSelect, 'settings:search_models_enter_key_first');
                StatusManager.setError(t('settings:status_no_api_key'));
                return;
            }

            const data = await ApiClient.getModels('nim', { apiKey });

            if (data.models && data.models.length > 0) {
                MessageLogger.showMessage('', '');

                const formattedModels = data.models.map(m => ({
                    value: m.id,
                    label: m.name || m.id,
                    context_length: m.context_length
                }));

                populateModelSelect(formattedModels, data.default, 'nim');
                MessageLogger.addLog(t('settings:nim_models_loaded_log', { count: data.count }));

                SettingsManager.applyPendingModelSelection();
                ModelDetector.checkAndShowRecommendation();

                StateManager.setState('models.availableModels', formattedModels.map(m => m.value));
                StatusManager.setConnected('nim', data.count);
            } else {
                const errorMessage = data.error || t('settings:nim_default_error');
                MessageLogger.showMessage(t('settings:deepseek_fallback_msg', { message: errorMessage }), 'warning');
                populateModelSelect(NIM_FALLBACK_MODELS, 'meta/llama-3.1-70b-instruct', 'nim');
                MessageLogger.addLog(t('settings:nim_fallback_log'));

                StateManager.setState('models.availableModels', NIM_FALLBACK_MODELS.map(m => m.value));
                StatusManager.setConnected('nim', NIM_FALLBACK_MODELS.length);
            }
        } catch (error) {
            MessageLogger.showMessage(t('settings:deepseek_error_fallback_msg', { error: error.message }), 'warning');
            MessageLogger.addLog(t('settings:nim_error_fallback_log', { error: error.message }));
            populateModelSelect(NIM_FALLBACK_MODELS, 'meta/llama-3.1-70b-instruct', 'nim');

            StateManager.setState('models.availableModels', NIM_FALLBACK_MODELS.map(m => m.value));
            StatusManager.setConnected('nim', NIM_FALLBACK_MODELS.length);
        }
    },

    /**
     * Load Poe models dynamically from API
     */
    async loadPoeModels() {
        const modelSelect = DomHelpers.getElement('model');
        if (!modelSelect) return;

        setPlaceholderOption(modelSelect, 'settings:search_models_loading_poe');
        StatusManager.setChecking();

        try {
            // Use ApiKeyUtils to get API key (returns '__USE_ENV__' if configured in .env)
            const apiKey = ApiKeyUtils.getValue('poeApiKey');
            if (!apiKey) {
                MessageLogger.showMessage(t('settings:poe_key_required'), 'warning');
                setPlaceholderOption(modelSelect, 'settings:search_models_enter_key_first');
                StatusManager.setError(t('settings:status_no_api_key'));
                return;
            }

            const data = await ApiClient.getModels('poe', { apiKey });

            if (data.models && data.models.length > 0) {
                MessageLogger.showMessage('', '');

                // Pass models directly (same format as OpenRouter)
                populateModelSelect(data.models, data.default, 'poe');
                MessageLogger.addLog(t('settings:poe_models_loaded_log', { count: data.count }));

                SettingsManager.applyPendingModelSelection();
                ModelDetector.checkAndShowRecommendation();

                StateManager.setState('models.availableModels', data.models.map(m => m.id));
                StatusManager.setConnected('poe', data.count);
            } else {
                // Use fallback list
                const errorMessage = data.error || t('settings:poe_default_error');
                MessageLogger.showMessage(t('settings:deepseek_fallback_msg', { message: errorMessage }), 'warning');
                populateModelSelect(POE_FALLBACK_MODELS, 'Claude-Sonnet-4', 'poe');
                MessageLogger.addLog(t('settings:poe_fallback_log'));

                StateManager.setState('models.availableModels', POE_FALLBACK_MODELS.map(m => m.value));
                StatusManager.setConnected('poe', POE_FALLBACK_MODELS.length);
            }
        } catch (error) {
            // Use fallback list on error
            MessageLogger.showMessage(t('settings:deepseek_error_fallback_msg', { error: error.message }), 'warning');
            MessageLogger.addLog(t('settings:poe_error_fallback_log', { error: error.message }));
            populateModelSelect(POE_FALLBACK_MODELS, 'Claude-Sonnet-4', 'poe');

            StateManager.setState('models.availableModels', POE_FALLBACK_MODELS.map(m => m.value));
            StatusManager.setConnected('poe', POE_FALLBACK_MODELS.length);
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
