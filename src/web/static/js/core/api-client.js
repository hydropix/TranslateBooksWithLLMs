/**
 * API Client - Centralized API communication layer
 *
 * Provides a clean abstraction for all backend API calls
 * with consistent error handling and response processing.
 */

let API_BASE_URL = window.location.origin;

/**
 * Append the per-session API token (issue #210) as a query parameter.
 *
 * Used only for URLs reached by a top-level navigation or an anchor download
 * (file download, glossary export), which cannot send the X-API-Token header
 * the fetch-based calls rely on.
 * @param {string} url - Absolute or relative URL
 * @returns {string} URL carrying the token query parameter
 */
function withToken(url) {
    const token = window.__API_TOKEN__;
    if (!token) return url;
    const sep = url.includes('?') ? '&' : '?';
    return `${url}${sep}token=${encodeURIComponent(token)}`;
}

/**
 * Handle API errors consistently
 *
 * The backend pairs a short `error` label with a `message` holding the actual
 * reason and, usually, the fix ("add the host to LLM_ENDPOINT_ALLOWLIST").
 * Keeping only the label left users with an unactionable one-liner and nothing
 * in the logs (issue #263), so both parts are surfaced.
 *
 * @param {Response} response - Fetch response
 * @returns {Promise<Object>} Parsed error data
 */
async function handleApiError(response) {
    let errorData;
    try {
        errorData = await response.json();
    } catch {
        errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
    }
    const label = errorData.error || '';
    const detail = errorData.message || '';
    const parts = [label, detail !== label ? detail : ''].filter(Boolean);
    throw new Error(parts.join(' — ') || `Request failed with status ${response.status}`);
}

/**
 * Make API request with error handling
 * @param {string} endpoint - API endpoint path
 * @param {Object} [options] - Fetch options
 * @returns {Promise<Object>} Response data
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url, {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    });

    if (!response.ok) {
        await handleApiError(response);
    }

    // Handle non-JSON responses (like file downloads)
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
        return await response.json();
    }

    return response;
}

/**
 * API Client interface
 */
export const ApiClient = {
    /**
     * Set the base URL for API requests
     * @param {string} url - Base URL
     */
    setBaseUrl(url) {
        API_BASE_URL = url;
    },

    /**
     * Get current base URL
     * @returns {string} Current base URL
     */
    getBaseUrl() {
        return API_BASE_URL;
    },

    // ========================================
    // Health & Configuration
    // ========================================

    /**
     * Check server health
     * @returns {Promise<Object>} Health status
     */
    async healthCheck() {
        return await apiRequest('/api/health');
    },

    /**
     * Get server configuration
     * @returns {Promise<Object>} Configuration object
     */
    async getConfig() {
        return await apiRequest('/api/config');
    },

    // ========================================
    // Translation Operations
    // ========================================

    /**
     * Start a new translation
     * @param {Object} config - Translation configuration
     * @returns {Promise<Object>} Translation job info
     */
    async startTranslation(config) {
        return await apiRequest('/api/translate', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    },

    /**
     * Get translation status
     * @param {string} translationId - Translation ID
     * @returns {Promise<Object>} Translation status
     */
    async getTranslationStatus(translationId) {
        return await apiRequest(`/api/translation/${translationId}`);
    },

    /**
     * Get all active translations
     * @returns {Promise<Object>} Active translations list
     */
    async getActiveTranslations() {
        return await apiRequest('/api/translations');
    },

    /**
     * Interrupt a translation
     * @param {string} translationId - Translation ID
     * @returns {Promise<Object>} Interruption result
     */
    async interruptTranslation(translationId) {
        return await apiRequest(`/api/translation/${translationId}/interrupt`, {
            method: 'POST'
        });
    },

    // ========================================
    // File Upload & Management
    // ========================================

    /**
     * Upload a file
     * @param {File} file - File to upload
     * @returns {Promise<Object>} Upload result with file_path and file_type
     */
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            await handleApiError(response);
        }

        return await response.json();
    },

    /**
     * Get list of managed files
     * @returns {Promise<Object>} Files list
     */
    async getFileList() {
        return await apiRequest('/api/files');
    },

    /**
     * Download a single file
     * @param {string} filename - Filename to download
     * @returns {string} Download URL
     */
    getFileDownloadUrl(filename) {
        return withToken(`${API_BASE_URL}/api/files/${encodeURIComponent(filename)}`);
    },

    /**
     * Delete a single file
     * @param {string} filename - Filename to delete
     * @returns {Promise<Object>} Delete result
     */
    async deleteFile(filename) {
        return await apiRequest(`/api/files/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
    },

    /**
     * Batch download files as zip
     * @param {string[]} filenames - Array of filenames
     * @returns {Promise<Blob>} Zip file blob
     */
    async batchDownloadFiles(filenames) {
        const response = await apiRequest('/api/files/batch/download', {
            method: 'POST',
            body: JSON.stringify({ filenames })
        });

        return await response.blob();
    },

    /**
     * Batch delete files
     * @param {string[]} filenames - Array of filenames
     * @returns {Promise<Object>} Delete result
     */
    async batchDeleteFiles(filenames) {
        return await apiRequest('/api/files/batch/delete', {
            method: 'POST',
            body: JSON.stringify({ filenames })
        });
    },

    /**
     * Open a local file
     * @param {string} filename - Filename to open
     * @returns {Promise<Object>} Open result
     */
    async openLocalFile(filename) {
        return await apiRequest(`/api/files/${encodeURIComponent(filename)}/open`, {
            method: 'POST'
        });
    },

    /**
     * Reveal a local file in the system's file explorer
     * @param {string} filename - Filename to reveal
     * @returns {Promise<Object>} Reveal result
     */
    async revealLocalFile(filename) {
        return await apiRequest(`/api/files/${encodeURIComponent(filename)}/reveal`, {
            method: 'POST'
        });
    },

    /**
     * Open the translations output folder in the system's file explorer
     * @returns {Promise<Object>} Open result with folder_path
     */
    async openOutputFolder() {
        return await apiRequest('/api/folders/output/open', {
            method: 'POST'
        });
    },

    /**
     * Clear uploaded files
     * @param {string[]} filePaths - Array of file paths to delete
     * @returns {Promise<Object>} Clear result
     */
    async clearUploadedFiles(filePaths) {
        return await apiRequest('/api/uploads/clear', {
            method: 'POST',
            body: JSON.stringify({ file_paths: filePaths })
        });
    },

    /**
     * Verify which uploaded files still exist on the server
     * @param {string[]} filePaths - Array of file paths to verify
     * @returns {Promise<Object>} Object with existing and missing file paths
     */
    async verifyUploadedFiles(filePaths) {
        return await apiRequest('/api/uploads/verify', {
            method: 'POST',
            body: JSON.stringify({ file_paths: filePaths })
        });
    },

    /**
     * Detect language from an uploaded file
     * @param {string} filePath - Path to the uploaded file
     * @returns {Promise<Object>} Detection result with detected_language and language_confidence
     */
    async detectLanguage(filePath) {
        return await apiRequest('/api/detect-language', {
            method: 'POST',
            body: JSON.stringify({ file_path: filePath })
        });
    },

    // ========================================
    // Model Management
    // ========================================

    /**
     * Get available models for a provider
     * @param {string} provider - Provider name ('ollama', 'gemini', 'openai', 'openrouter')
     * @param {Object} [options] - Additional options (apiEndpoint, apiKey)
     * @returns {Promise<Object>} Models list
     */
    async getModels(provider, options = {}) {
        if (provider === 'ollama') {
            // Ollama: GET request (no API key needed)
            const params = new URLSearchParams();
            if (options.apiEndpoint) {
                params.append('api_endpoint', options.apiEndpoint);
            }
            return await apiRequest(`/api/models?${params.toString()}`);
        }

        // Gemini/OpenRouter/OpenAI: POST request (API key in body - more secure)
        const body = {
            provider: provider,
            api_key: options.apiKey || '__USE_ENV__'
        };

        // Include endpoint for OpenAI-compatible providers (llama.cpp, LM Studio, vLLM, etc.)
        if (provider === 'openai' && options.apiEndpoint) {
            body.api_endpoint = options.apiEndpoint;
        }

        return await apiRequest('/api/models', {
            method: 'POST',
            body: JSON.stringify(body)
        });
    },

    // ========================================
    // Resumable Jobs
    // ========================================

    /**
     * Get resumable jobs list
     * @returns {Promise<Object>} Resumable jobs
     */
    async getResumableJobs() {
        return await apiRequest('/api/resumable');
    },

    /**
     * Resume a paused job, optionally overriding model/provider for the
     * remaining chunks. With no overrides the body is empty and the server
     * keeps the original config (backward compatible).
     * @param {string} translationId - Translation ID to resume
     * @param {Object} [overrides] - Optional {model, llm_provider, llm_api_endpoint, api_key, context_window}
     * @returns {Promise<Object>} Resume result
     */
    async resumeJob(translationId, overrides = null) {
        const options = { method: 'POST' };
        if (overrides && Object.keys(overrides).length > 0) {
            options.body = JSON.stringify(overrides);
        }
        return await apiRequest(`/api/resume/${translationId}`, options);
    },

    /**
     * Delete a checkpoint
     * @param {string} translationId - Translation ID
     * @returns {Promise<Object>} Delete result
     */
    async deleteCheckpoint(translationId) {
        return await apiRequest(`/api/checkpoint/${translationId}`, {
            method: 'DELETE'
        });
    },

    // ========================================
    // Job History
    // ========================================

    /**
     * Get the list of completed jobs kept for history
     * @param {number} [limit] - Maximum number of entries to return
     * @returns {Promise<Object>} Job history
     */
    async getJobHistory(limit) {
        return await apiRequest('/api/history' + (limit ? '?limit=' + encodeURIComponent(limit) : ''));
    },

    // ========================================
    // Settings Management
    // ========================================

    /**
     * Get current user settings
     * @returns {Promise<Object>} Current settings
     */
    async getSettings() {
        return await apiRequest('/api/settings');
    },

    /**
     * Save user settings to .env file
     * @param {Object} settings - Settings to save
     * @returns {Promise<Object>} Save result
     */
    async saveSettings(settings) {
        return await apiRequest('/api/settings', {
            method: 'POST',
            body: JSON.stringify(settings)
        });
    },

    /**
     * Get the shared UI preferences document (synced across devices)
     * @returns {Promise<Object>} Stored preferences ({} when nothing is stored)
     */
    async getPreferences() {
        const data = await apiRequest('/api/preferences');
        return data.preferences || {};
    },

    /**
     * Replace the shared UI preferences document (PUT is a full replacement)
     * @param {Object} prefs - Flat preferences object to store
     * @returns {Promise<Object>} Save result with the stored preferences
     */
    async savePreferences(prefs) {
        return await apiRequest('/api/preferences', {
            method: 'PUT',
            body: JSON.stringify(prefs)
        });
    },

    /**
     * Get available custom instruction files
     * @returns {Promise<Object>} Custom instructions list with files array
     */
    async getCustomInstructions() {
        return await apiRequest('/api/custom-instructions');
    },

    /**
     * Open the Custom_Instructions folder in the system file explorer
     * @returns {Promise<Object>} Result with success status
     */
    async openCustomInstructionsFolder() {
        return await apiRequest('/api/custom-instructions/open-folder', {
            method: 'POST'
        });
    },

    /**
     * Get a single custom instruction preset (full extended-schema mapping)
     * @param {string} filename - Preset filename
     * @returns {Promise<Object>} Preset payload from read_preset()
     */
    async getCustomInstruction(filename) {
        return await apiRequest(`/api/custom-instructions/${encodeURIComponent(filename)}`);
    },

    /**
     * Create a new custom instruction preset
     * @param {Object} payload - {name, description?, context?, mode?, source_files?, rules?, translation?, refinement?, overwrite?}
     * @returns {Promise<Object>} {filename, display_name}
     */
    async createCustomInstruction(payload) {
        return await apiRequest('/api/custom-instructions', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    },

    /**
     * Update (partial) an existing custom instruction preset
     * @param {string} filename - Preset filename
     * @param {Object} payload - Any subset of {description, context, mode, source_files, rules, translation, refinement, manual}
     * @returns {Promise<Object>} {filename, display_name}
     */
    async updateCustomInstruction(filename, payload) {
        return await apiRequest(`/api/custom-instructions/${encodeURIComponent(filename)}`, {
            method: 'PUT',
            body: JSON.stringify(payload)
        });
    },

    /**
     * Delete a custom instruction preset
     * @param {string} filename - Preset filename
     * @returns {Promise<Object>} {deleted: true}
     */
    async deleteCustomInstruction(filename) {
        return await apiRequest(`/api/custom-instructions/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
    },

    /**
     * Duplicate a custom instruction preset
     * @param {string} filename - Source preset filename
     * @param {string} [name] - Explicit name for the duplicate (server picks a default when omitted)
     * @returns {Promise<Object>} {filename, display_name}
     */
    async duplicateCustomInstruction(filename, name) {
        return await apiRequest(`/api/custom-instructions/${encodeURIComponent(filename)}/duplicate`, {
            method: 'POST',
            body: JSON.stringify(name ? { name } : {})
        });
    },

    /**
     * Assemble a rule list into the translation/refinement prose blocks
     * @param {string} mode - 'source' | 'model'
     * @param {Array<Object>} rules - [{dimension, instruction}, ...]
     * @param {string} [context] - narrative setting (period, tech level, social frame)
     * @returns {Promise<Object>} {translation, refinement, flags}
     */
    async assembleCustomInstruction(mode, rules, context) {
        const body = { mode, rules };
        if (context !== undefined) body.context = context;
        return await apiRequest('/api/custom-instructions/assemble', {
            method: 'POST',
            body: JSON.stringify(body)
        });
    },

    /**
     * Sample uploaded documents and ask an LLM to characterize their writing style
     * @param {FormData} formData - Multipart body: files[], mode, source_lang, target_lang,
     *   max_chars, sample_count, provider, model, api_key, api_endpoint
     * @returns {Promise<Object>} {rules, summary, suggested_name, assembled, mode, warnings, ...}
     */
    async extractStyle(formData) {
        const response = await fetch(`${API_BASE_URL}/api/custom-instructions/extract-style`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            await handleApiError(response);
        }

        return await response.json();
    },

    /**
     * Get the export URL for a custom instruction preset (YAML download)
     * @param {string} filename - Preset filename
     * @returns {string} Download URL
     */
    exportCustomInstructionUrl(filename) {
        return withToken(`${API_BASE_URL}/api/custom-instructions/${encodeURIComponent(filename)}/export`);
    },

    // ========================================
    // TTS (Text-to-Speech) Operations
    // ========================================

    /**
     * Generate TTS audio from an existing file
     * @param {Object} config - TTS configuration
     * @param {string} config.filename - File to generate audio from
     * @param {string} config.target_language - Target language for voice selection
     * @param {string} [config.tts_voice] - Specific voice (auto-select if empty)
     * @param {string} [config.tts_rate] - Speech rate (default: +0%)
     * @param {string} [config.tts_format] - Output format (opus/mp3)
     * @param {string} [config.tts_bitrate] - Audio bitrate
     * @returns {Promise<Object>} TTS job info
     */
    async generateTTS(config) {
        return await apiRequest('/api/tts/generate', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    },

    /**
     * Get TTS job status
     * @param {string} jobId - TTS job ID
     * @returns {Promise<Object>} Job status
     */
    async getTTSStatus(jobId) {
        return await apiRequest(`/api/tts/status/${jobId}`);
    },

    /**
     * Get available TTS voices
     * @returns {Promise<Object>} Available voices by language
     */
    async getTTSVoices() {
        return await apiRequest('/api/tts/voices');
    },

    /**
     * Get available TTS providers and their status
     * @returns {Promise<Object>} Providers information
     */
    async getTTSProviders() {
        return await apiRequest('/api/tts/providers');
    },

    /**
     * Get available Chatterbox voices/languages
     * @returns {Promise<Object>} Chatterbox languages and availability
     */
    async getChatterboxVoices() {
        return await apiRequest('/api/tts/voices/chatterbox');
    },

    /**
     * Get GPU status for TTS
     * @returns {Promise<Object>} GPU status information
     */
    async getTTSGPUStatus() {
        return await apiRequest('/api/tts/gpu-status');
    },

    /**
     * Upload a voice prompt file for voice cloning
     * @param {File} file - Audio file to upload
     * @returns {Promise<Object>} Upload result with path
     */
    async uploadTTSVoicePrompt(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/api/tts/voice-prompt/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            await handleApiError(response);
        }

        return await response.json();
    },

    /**
     * Get list of available voice prompts
     * @returns {Promise<Object>} Voice prompts list
     */
    async getTTSVoicePrompts() {
        return await apiRequest('/api/tts/voice-prompts');
    },

    /**
     * Delete a voice prompt file
     * @param {string} filename - Filename to delete
     * @returns {Promise<Object>} Delete result
     */
    async deleteTTSVoicePrompt(filename) {
        return await apiRequest(`/api/tts/voice-prompt/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
    },

    // ========================================
    // Glossary Management
    // ========================================

    async getGlossaries() {
        return await apiRequest('/api/glossaries');
    },

    async getGlossary(gid) {
        return await apiRequest(`/api/glossaries/${gid}`);
    },

    async createGlossary(payload) {
        return await apiRequest('/api/glossaries', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    },

    async updateGlossary(gid, payload) {
        return await apiRequest(`/api/glossaries/${gid}`, {
            method: 'PUT',
            body: JSON.stringify(payload)
        });
    },

    async deleteGlossary(gid) {
        return await apiRequest(`/api/glossaries/${gid}`, {
            method: 'DELETE'
        });
    },

    async addGlossaryTerm(gid, term) {
        return await apiRequest(`/api/glossaries/${gid}/terms`, {
            method: 'POST',
            body: JSON.stringify(term)
        });
    },

    async updateGlossaryTerm(gid, tid, term) {
        return await apiRequest(`/api/glossaries/${gid}/terms/${tid}`, {
            method: 'PUT',
            body: JSON.stringify(term)
        });
    },

    async deleteGlossaryTerm(gid, tid) {
        return await apiRequest(`/api/glossaries/${gid}/terms/${tid}`, {
            method: 'DELETE'
        });
    },

    async importGlossaryTerms(gid, file) {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch(`${API_BASE_URL}/api/glossaries/${gid}/import`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            await handleApiError(response);
        }
        return await response.json();
    },

    getGlossaryExportUrl(gid, format = 'json') {
        return withToken(`${API_BASE_URL}/api/glossaries/${gid}/export?format=${encodeURIComponent(format)}`);
    },

    async suggestGlossaryTerms(gid, payload) {
        const url = `${API_BASE_URL}/api/glossaries/${gid}/suggest-terms`;
        const isFormData = (typeof FormData !== 'undefined') && (payload instanceof FormData);
        const response = await fetch(url, {
            method: 'POST',
            body: isFormData ? payload : JSON.stringify(payload),
            headers: isFormData ? {} : { 'Content-Type': 'application/json' }
        });
        if (!response.ok) {
            await handleApiError(response);
        }
        return await response.json();
    },

    async duplicateGlossary(gid, payload = {}) {
        return await apiRequest(`/api/glossaries/${gid}/duplicate`, {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    },

    async bulkGlossaryTerms(gid, payload) {
        return await apiRequest(`/api/glossaries/${gid}/terms/bulk`, {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    },

    async previewGlossaryBlock(gid, text) {
        return await apiRequest(`/api/glossaries/${gid}/preview-block`, {
            method: 'POST',
            body: JSON.stringify({ text: text || '' })
        });
    },

    // ========================================
    // Cost Estimation
    // ========================================

    async getPricingDefaults() {
        return await apiRequest('/api/pricing/defaults');
    },

    async estimateCost(payload, { signal } = {}) {
        return await apiRequest('/api/cost/estimate', {
            method: 'POST',
            body: JSON.stringify(payload),
            signal,
        });
    }
};

