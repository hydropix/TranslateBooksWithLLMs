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
 * Monotonically increasing token of the latest model-list request. Every
 * loadModels() call takes a new token; a response (or error) whose token is
 * no longer the latest is discarded without touching the DOM or the state,
 * so a slow answer for provider X cannot overwrite the list after the user
 * switched to provider Y (or re-triggered a load for the same provider).
 */
let latestModelLoadToken = 0;

/** Map backend `{id, name}` models to the `{value, label}` dropdown shape. */
function toValueLabel(models) {
    return models.map(m => ({ value: m.id, label: m.name || m.id }));
}

/** Same as toValueLabel, keeping `context_length` for the tooltip. */
function toValueLabelWithContext(models) {
    return models.map(m => ({ value: m.id, label: m.name || m.id, context_length: m.context_length }));
}

/**
 * Shared success path: populate the dropdown, log, lock the .env model when
 * the provider opts in, restore the saved selection, refresh the model
 * recommendation, publish the available models and mark the provider as
 * connected.
 */
function applyLoadedModels(ctx, data) {
    const { provider, cfg } = ctx;
    MessageLogger.showMessage('', '');
    const models = cfg.formatModels ? cfg.formatModels(data.models) : data.models;
    const envModelApplied = populateModelSelect(models, data.default, provider);
    MessageLogger.addLog(t(cfg.successLogKey, cfg.successLogParams ? cfg.successLogParams(data) : { count: data.count }));

    // If .env model was found and applied, lock it in
    if (cfg.lockEnvModel && envModelApplied && data.default) {
        SettingsManager.markEnvModelApplied();
    }

    // Apply saved model preference if any (skipped if .env model was applied)
    SettingsManager.applyPendingModelSelection();
    ModelDetector.checkAndShowRecommendation();

    StateManager.setState('models.availableModels', cfg.availableModels ? cfg.availableModels(models) : models);
    StatusManager.setConnected(provider, data.count);
}

/** Publish a static fallback list in state and keep the provider "connected". */
function publishFallbackList(provider, fallback) {
    StateManager.setState('models.availableModels', fallback.models.map(m => m.value));
    StatusManager.setConnected(provider, fallback.models.length);
}

/**
 * Default handler for an empty / error-carrying response:
 *   - providers with a `fallback` config switch to their static list,
 *   - the others surface the error and leave a "no models" placeholder.
 */
function handleEmptyResponse(ctx, data) {
    const { provider, cfg, modelSelect } = ctx;
    const errorMessage = data.error || t(cfg.emptyDefaultKey);
    if (cfg.fallback) {
        MessageLogger.showMessage(t(cfg.fallback.emptyMessageKey, { message: errorMessage }), 'warning');
        populateModelSelect(cfg.fallback.models, cfg.fallback.defaultModel, provider);
        MessageLogger.addLog(t(cfg.fallback.emptyLogKey));
        publishFallbackList(provider, cfg.fallback);
        return;
    }
    MessageLogger.showMessage(t('settings:no_models_warning', { message: errorMessage }), 'error');
    setPlaceholderOption(modelSelect, 'settings:search_models_no_models_available');
    if (cfg.emptyLogKey) MessageLogger.addLog(t(cfg.emptyLogKey));
    StatusManager.setError(t('settings:status_no_models'));
}

/**
 * Default handler for a failed request: fallback list when the provider has
 * one, otherwise an error message plus an "error" placeholder.
 */
function handleRequestError(ctx, error) {
    const { provider, cfg, modelSelect } = ctx;
    const params = { error: error.message };
    if (cfg.fallback) {
        MessageLogger.showMessage(t(cfg.fallback.errorMessageKey, params), 'warning');
        MessageLogger.addLog(t(cfg.fallback.errorLogKey, params));
        populateModelSelect(cfg.fallback.models, cfg.fallback.defaultModel, provider);
        publishFallbackList(provider, cfg.fallback);
        return;
    }
    MessageLogger.showMessage(t(cfg.errorMessageKey, params), 'error');
    if (cfg.errorLogKey) MessageLogger.addLog(t(cfg.errorLogKey, params));
    setPlaceholderOption(modelSelect, 'settings:search_models_error');
    StatusManager.setError(error.message);
}

/** Last-resort static list for the official OpenAI host. */
function applyOpenAIStaticFallback() {
    populateModelSelect(OPENAI_MODELS, null, 'openai');
    MessageLogger.addLog(t('settings:openai_models_common_log'));

    SettingsManager.applyPendingModelSelection();
    ModelDetector.checkAndShowRecommendation();

    StateManager.setState('models.availableModels', OPENAI_MODELS.map(m => m.value));
    StatusManager.setConnected('openai', OPENAI_MODELS.length);
}

/**
 * Per-provider model-loading configuration consumed by
 * ProviderManager.loadModels(). Fields:
 *   loadingKey        i18n key of the "loading" placeholder
 *   apiKeyInput       id of the API-key input (read via ApiKeyUtils)
 *   missingKeyMessage when set, an empty API key aborts the load with this
 *                     warning and an "enter key first" placeholder
 *   prepare(ctx)      optional synchronous hook run before the request
 *   requestOptions(ctx, apiKey)  options passed to ApiClient.getModels
 *   formatModels      optional mapping of `data.models` before rendering
 *   availableModels   optional mapping of the rendered list into state
 *   lockEnvModel      call SettingsManager.markEnvModelApplied() when the
 *                     .env default model was found
 *   successLogKey / successLogParams  log line after a successful load
 *   emptyDefaultKey   message used when an empty response carries no error
 *   emptyLogKey       optional log line for an empty response (no fallback)
 *   errorMessageKey / errorLogKey     messages for a failed request (no fallback)
 *   fallback          static list used on empty response or request failure
 *   beforeSuccess(ctx) optional hook run before the shared success path
 *   onEmpty(ctx, data) optional replacement of handleEmptyResponse
 *   onData(ctx, data)  optional replacement of the whole response handling
 *   onError(ctx, error) optional replacement of handleRequestError
 */
const MODEL_LOADERS = {
    ollama: {
        loadingKey: 'settings:search_models_loading',
        requestOptions: (ctx) => {
            ctx.apiEndpoint = DomHelpers.getValue('apiEndpoint');
            return { apiEndpoint: ctx.apiEndpoint };
        },
        lockEnvModel: true,
        successLogKey: 'settings:models_loaded_ollama_log',
        successLogParams: (data) => ({ count: data.count, default: data.default }),
        // Success - stop auto-retry
        beforeSuccess: (ctx) => ctx.manager.stopOllamaAutoRetry(),
        onEmpty: (ctx, data) => {
            // No models available - start auto-retry
            const errorMessage = data.error || t('settings:no_models_default');

            // Show message only after several retries
            if (ollamaRetryCount >= OLLAMA_MAX_SILENT_RETRIES) {
                MessageLogger.showMessage(t('settings:no_models_warning', { message: errorMessage }), 'error');
                MessageLogger.addLog(t('settings:waiting_for_ollama_log', { endpoint: ctx.apiEndpoint, interval: OLLAMA_RETRY_INTERVAL / 1000 }));
            }

            setPlaceholderOption(ctx.modelSelect, 'settings:search_models_waiting_ollama');
            StatusManager.setWaiting(t('settings:search_models_waiting_ollama'));
            ctx.manager.startOllamaAutoRetry();
        },
        onError: (ctx) => {
            // Connection error - start auto-retry
            if (ollamaRetryCount >= OLLAMA_MAX_SILENT_RETRIES) {
                MessageLogger.showMessage(t('settings:waiting_for_ollama_msg'), 'warning');
                MessageLogger.addLog(t('settings:ollama_not_accessible_log', { interval: OLLAMA_RETRY_INTERVAL / 1000 }));
            }

            setPlaceholderOption(ctx.modelSelect, 'settings:search_models_waiting_ollama');
            StatusManager.setDisconnected(t('settings:status_not_accessible'));
            ctx.manager.startOllamaAutoRetry();
        },
    },

    gemini: {
        loadingKey: 'settings:search_models_loading_gemini',
        apiKeyInput: 'geminiApiKey',
        requestOptions: (ctx, apiKey) => ({ apiKey }),
        lockEnvModel: true,
        successLogKey: 'settings:models_loaded_gemini_log',
        emptyDefaultKey: 'settings:no_models_gemini',
        emptyLogKey: 'settings:models_no_gemini_log',
        errorMessageKey: 'settings:gemini_fetch_error',
        errorLogKey: 'settings:gemini_fetch_error_log',
    },

    /**
     * OpenAI-compatible endpoints. For api.openai.com, falls back to a static
     * cloud list if the live fetch fails. For any custom endpoint, surfaces
     * the error instead of falling back, since the cloud model names would
     * not exist on a local server.
     */
    openai: {
        loadingKey: 'settings:search_models_loading',
        apiKeyInput: 'openaiApiKey',
        prepare: (ctx) => {
            ctx.apiEndpoint = DomHelpers.getValue('openaiEndpoint') || 'https://api.openai.com/v1/chat/completions';
            ctx.isOfficialOpenAI = /^https?:\/\/api\.openai\.com(\/|$)/i.test(ctx.apiEndpoint);
            const isLocalHttps = /^https:\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])/i.test(ctx.apiEndpoint);
            if (isLocalHttps) {
                MessageLogger.addLog(t('settings:openai_local_https_warning_log', { host: ctx.apiEndpoint.slice(8) }));
            }
        },
        requestOptions: (ctx, apiKey) => ({ apiKey, apiEndpoint: ctx.apiEndpoint }),
        formatModels: toValueLabel,
        availableModels: (models) => models.map(m => m.value),
        lockEnvModel: true,
        successLogKey: 'settings:openai_models_loaded_log',
        onData: (ctx, data) => {
            const hasModels = data.models && data.models.length > 0;

            if (data.status === 'openai_connected' && hasModels) {
                applyLoadedModels(ctx, data);
                return;
            }

            if (data.status === 'openai_error') {
                const errorMsg = data.error || t('settings:openai_endpoint_error_default');
                MessageLogger.showMessage(t('settings:openai_endpoint_error_warning', { error: errorMsg }), 'warning');
                MessageLogger.addLog(t('settings:openai_endpoint_error_log', { error: errorMsg }));
                setPlaceholderOption(ctx.modelSelect, 'settings:search_models_no_check_endpoint');
                StateManager.setState('models.availableModels', []);
                StatusManager.setError(errorMsg);
                return;
            }

            if (data.status === 'openai_static' && hasModels) {
                const reason = data.error ? ` (${data.error})` : '';
                MessageLogger.showMessage(t('settings:openai_static_warning', { reason }), 'warning');
                MessageLogger.addLog(t('settings:openai_static_log', { reason }));

                const formattedModels = toValueLabel(data.models);
                populateModelSelect(formattedModels, data.default, 'openai');
                SettingsManager.applyPendingModelSelection();
                ModelDetector.checkAndShowRecommendation();

                StateManager.setState('models.availableModels', formattedModels.map(m => m.value));
                StatusManager.setConnected('openai', data.count);
                return;
            }

            // Any other answer: last-resort static list
            applyOpenAIStaticFallback();
        },
        onError: (ctx, error) => {
            MessageLogger.showMessage(t('settings:openai_cannot_reach_endpoint', { error: error.message }), 'warning');
            MessageLogger.addLog(t('settings:openai_connection_error_log', { error: error.message }));

            if (!ctx.isOfficialOpenAI) {
                setPlaceholderOption(ctx.modelSelect, 'settings:search_models_no_check_endpoint');
                StateManager.setState('models.availableModels', []);
                StatusManager.setError(error.message);
                return;
            }

            // Last-resort static fallback, only for the official OpenAI host
            applyOpenAIStaticFallback();
        },
    },

    // Text-only models, sorted by price
    openrouter: {
        loadingKey: 'settings:search_models_loading_openrouter',
        apiKeyInput: 'openrouterApiKey',
        requestOptions: (ctx, apiKey) => ({ apiKey }),
        availableModels: (models) => models.map(m => m.id),
        lockEnvModel: true,
        successLogKey: 'settings:openrouter_models_loaded_log',
        emptyDefaultKey: 'settings:openrouter_default_error',
        fallback: {
            models: OPENROUTER_FALLBACK_MODELS,
            defaultModel: 'anthropic/claude-sonnet-4',
            emptyMessageKey: 'settings:openrouter_fallback_warn',
            emptyLogKey: 'settings:openrouter_fallback_log',
            errorMessageKey: 'settings:openrouter_fetch_error_msg',
            errorLogKey: 'settings:openrouter_fetch_error_log',
        },
    },

    mistral: {
        loadingKey: 'settings:search_models_loading_mistral',
        apiKeyInput: 'mistralApiKey',
        missingKeyMessage: 'settings:mistral_key_required',
        requestOptions: (ctx, apiKey) => ({ apiKey }),
        formatModels: toValueLabelWithContext,
        availableModels: (models) => models.map(m => m.value),
        successLogKey: 'settings:mistral_models_loaded_log',
        emptyDefaultKey: 'settings:mistral_no_models_default',
        errorMessageKey: 'settings:mistral_error',
    },

    deepseek: {
        loadingKey: 'settings:search_models_loading_deepseek',
        apiKeyInput: 'deepseekApiKey',
        missingKeyMessage: 'settings:deepseek_key_required',
        requestOptions: (ctx, apiKey) => ({ apiKey }),
        formatModels: toValueLabelWithContext,
        availableModels: (models) => models.map(m => m.value),
        successLogKey: 'settings:deepseek_models_loaded_log',
        emptyDefaultKey: 'settings:deepseek_default_error',
        fallback: {
            models: DEEPSEEK_FALLBACK_MODELS,
            defaultModel: 'deepseek-v4-pro',
            emptyMessageKey: 'settings:deepseek_fallback_msg',
            emptyLogKey: 'settings:deepseek_fallback_log',
            errorMessageKey: 'settings:deepseek_error_fallback_msg',
            errorLogKey: 'settings:deepseek_error_fallback_log',
        },
    },

    nim: {
        loadingKey: 'settings:search_models_loading_nim',
        apiKeyInput: 'nimApiKey',
        missingKeyMessage: 'settings:nim_key_required',
        requestOptions: (ctx, apiKey) => ({ apiKey }),
        formatModels: toValueLabelWithContext,
        availableModels: (models) => models.map(m => m.value),
        successLogKey: 'settings:nim_models_loaded_log',
        emptyDefaultKey: 'settings:nim_default_error',
        fallback: {
            models: NIM_FALLBACK_MODELS,
            defaultModel: 'meta/llama-3.1-70b-instruct',
            // NIM and Poe reuse the generic DeepSeek-named fallback messages.
            emptyMessageKey: 'settings:deepseek_fallback_msg',
            emptyLogKey: 'settings:nim_fallback_log',
            errorMessageKey: 'settings:deepseek_error_fallback_msg',
            errorLogKey: 'settings:nim_error_fallback_log',
        },
    },

    poe: {
        loadingKey: 'settings:search_models_loading_poe',
        apiKeyInput: 'poeApiKey',
        missingKeyMessage: 'settings:poe_key_required',
        requestOptions: (ctx, apiKey) => ({ apiKey }),
        // Models are passed through as-is (same format as OpenRouter)
        availableModels: (models) => models.map(m => m.id),
        successLogKey: 'settings:poe_models_loaded_log',
        emptyDefaultKey: 'settings:poe_default_error',
        fallback: {
            models: POE_FALLBACK_MODELS,
            defaultModel: 'Claude-Sonnet-4',
            emptyMessageKey: 'settings:deepseek_fallback_msg',
            emptyLogKey: 'settings:poe_fallback_log',
            errorMessageKey: 'settings:deepseek_error_fallback_msg',
            errorLogKey: 'settings:poe_error_fallback_log',
        },
    },
};

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
                        this.loadModels('openai');
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
            this.applyServerDefaultProvider(serverConfig);
            this.toggleProviderSettings(true);
        } else {
            // Listen for server config to be loaded, THEN load models with correct endpoint
            console.log('[ProviderManager] Waiting for defaultConfigLoaded event');
            window.addEventListener('defaultConfigLoaded', () => {
                console.log('[ProviderManager] Server config loaded, now loading models with correct endpoint');
                this.applyServerDefaultProvider(StateManager.getState('ui.defaultConfig'));
                this.toggleProviderSettings(true);
            }, { once: true });
        }
    },

    /**
     * Select the provider configured on the server (LLM_PROVIDER) when this
     * browser has no saved choice. Without it the select keeps its first
     * option (Ollama) even when the .env configures another provider.
     * A provider the user picked before (lastProvider) always wins.
     * @param {Object} serverConfig - Response of /api/config
     */
    applyServerDefaultProvider(serverConfig) {
        const provider = serverConfig && serverConfig.llm_provider;
        if (!provider || SettingsManager.getLocalPreferences().lastProvider) {
            return;
        }

        const providerSelect = DomHelpers.getElement('llmProvider');
        if (!providerSelect || !Array.from(providerSelect.options).some(opt => opt.value === provider)) {
            return;
        }

        providerSelect.value = provider;
        const instance = SearchableSelectFactory.get('llmProvider');
        if (instance) {
            instance.refresh();
        }
        this.updateProviderDisplay(provider);
        console.log('[ProviderManager] Applying LLM_PROVIDER from server:', provider);
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
                placeholderKey: 'settings:search_providers_placeholder',
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
                placeholderKey: 'settings:search_models_placeholder',
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
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadModels('ollama');
        } else if (provider === 'poe') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'block';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadModels('poe');
        } else if (provider === 'gemini') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'block';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadModels('gemini');
        } else if (provider === 'openai') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'block';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'block';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadModels('openai');
        } else if (provider === 'openrouter') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'block';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadModels('openrouter');
        } else if (provider === 'mistral') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'block';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadModels('mistral');
        } else if (provider === 'deepseek') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'block';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'none';
            if (loadModels) this.loadModels('deepseek');
        } else if (provider === 'nim') {
            DomHelpers.hide('ollamaSettings');
            if (geminiSettings) geminiSettings.style.display = 'none';
            if (openaiApiKeyGroup) openaiApiKeyGroup.style.display = 'none';
            if (openaiEndpointRow) openaiEndpointRow.style.display = 'none';
            if (openrouterSettings) openrouterSettings.style.display = 'none';
            if (mistralSettings) mistralSettings.style.display = 'none';
            if (deepseekSettings) deepseekSettings.style.display = 'none';
            if (poeSettings) poeSettings.style.display = 'none';
            if (nimSettings) nimSettings.style.display = 'block';
            if (loadModels) this.loadModels('nim');
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
        // Unknown providers are ignored by loadModels()
        this.loadModels(DomHelpers.getValue('llmProvider'));
    },

    /**
     * Load the model list for `provider` using its MODEL_LOADERS config.
     * Only the latest request may write to the DOM / state: an older
     * response or error is dropped once a newer load has started or the
     * selected provider no longer matches.
     * @param {string} provider - Provider id ('ollama', 'gemini', ...)
     */
    async loadModels(provider) {
        const cfg = MODEL_LOADERS[provider];
        if (!cfg) return;

        const modelSelect = DomHelpers.getElement('model');
        if (!modelSelect) return;

        const token = ++latestModelLoadToken;
        const thisRequest = { token, provider };
        StateManager.setState('models.currentLoadRequest', thisRequest);
        const isStale = () =>
            token !== latestModelLoadToken || DomHelpers.getValue('llmProvider') !== provider;

        const ctx = { provider, cfg, modelSelect, manager: this };

        setPlaceholderOption(modelSelect, cfg.loadingKey);
        StatusManager.setChecking();
        if (cfg.prepare) cfg.prepare(ctx);

        try {
            // ApiKeyUtils returns '__USE_ENV__' when the key is configured in .env
            const apiKey = cfg.apiKeyInput ? ApiKeyUtils.getValue(cfg.apiKeyInput) : undefined;
            if (cfg.missingKeyMessage && !apiKey) {
                MessageLogger.showMessage(t(cfg.missingKeyMessage), 'warning');
                setPlaceholderOption(modelSelect, 'settings:search_models_enter_key_first');
                StatusManager.setError(t('settings:status_no_api_key'));
                return;
            }

            const data = await ApiClient.getModels(provider, cfg.requestOptions(ctx, apiKey));

            if (isStale()) {
                console.log(`[ProviderManager] Discarding stale ${provider} model list response`);
                return;
            }

            if (cfg.onData) {
                cfg.onData(ctx, data);
            } else if (data.models && data.models.length > 0) {
                if (cfg.beforeSuccess) cfg.beforeSuccess(ctx);
                applyLoadedModels(ctx, data);
            } else {
                (cfg.onEmpty || handleEmptyResponse)(ctx, data);
            }
        } catch (error) {
            if (isStale()) {
                console.log(`[ProviderManager] Discarding stale ${provider} model list error`);
                return;
            }
            (cfg.onError || handleRequestError)(ctx, error);
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
                this.loadModels('ollama');
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
