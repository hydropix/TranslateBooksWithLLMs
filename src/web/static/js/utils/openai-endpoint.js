/**
 * Whether an OpenAI-provider endpoint needs an API key on the client side.
 *
 * Mirrors the backend guard (src/api/services/credential_guard.py): only the
 * official OpenAI API requires a key. Any other endpoint is a self-hosted
 * OpenAI-compatible server (llama.cpp, LM Studio, vLLM, Ollama's /v1 route)
 * that may run keyless on localhost, a LAN address or a LAN hostname; the
 * server's EndpointValidator decides whether that host is allowed and answers
 * 400 endpoint_rejected otherwise. This module has no imports so it can be
 * tested outside the browser.
 *
 * @param {string} endpoint - Endpoint typed in the OpenAI endpoint field
 * @returns {boolean} True when a missing key must block the request
 */
export function openaiEndpointRequiresKey(endpoint) {
    const value = (endpoint || '').trim();
    // An empty field means the default endpoint, i.e. the official API.
    if (!value) return true;
    let host = '';
    try {
        host = new URL(value).hostname.toLowerCase();
    } catch (e) {
        return value.toLowerCase().includes('api.openai.com');
    }
    return host === 'api.openai.com';
}
