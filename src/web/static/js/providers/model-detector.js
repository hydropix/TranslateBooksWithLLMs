/**
 * Model Detector - Detect model size, thinking behavior, and show recommendations
 *
 * Analyzes model names to determine parameter size.
 * Also checks for uncontrollable thinking models and displays warnings.
 */

import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';

/**
 * Extract parameter size from model name
 * @param {string} modelName - Model name (e.g., "qwen3:14b", "llama-12b")
 * @returns {number|null} Size in billions of parameters, or null if not detected
 */
function extractModelSize(modelName) {
    if (!modelName) return null;

    const lowerName = modelName.toLowerCase();

    // Pattern to detect model size (e.g., "7b", "12b", "3b", "1.5b")
    const sizeMatch = lowerName.match(/(\d+(?:\.\d+)?)b/);

    if (sizeMatch) {
        return parseFloat(sizeMatch[1]);
    }

    return null;
}

/**
 * Check if model is an uncontrollable thinking model and show warning
 * @param {string} modelName - Model name
 */
async function checkThinkingModelWarning(modelName) {
    const warningDiv = document.getElementById('thinkingModelWarning');
    const warningText = document.getElementById('thinkingModelWarningText');

    if (!warningDiv || !warningText) {
        return;
    }

    if (!modelName) {
        warningDiv.style.display = 'none';
        return;
    }

    try {
        // Get API endpoint for cache differentiation
        const endpoint = DomHelpers.getValue('apiEndpoint') || '';

        const data = await ApiClient.getModelWarning(modelName, endpoint);

        if (data.warning && data.is_uncontrollable) {
            warningText.textContent = data.warning;
            warningDiv.style.display = 'block';
        } else {
            warningDiv.style.display = 'none';
        }
    } catch (error) {
        console.warn('Failed to check thinking model warning:', error);
        warningDiv.style.display = 'none';
    }
}

/**
 * Determine if a model is considered "small" (≤12B parameters)
 * @param {string} modelName - Model name
 * @returns {boolean} True if model is small
 */
function isSmallModel(modelName) {
    const size = extractModelSize(modelName);

    if (size === null) {
        return false; // Unknown size, assume not small
    }

    return size <= 12;
}

export const ModelDetector = {
    /**
     * Check for thinking model warning
     * Called when model selection changes
     */
    checkAndShowRecommendation() {
        const modelName = DomHelpers.getValue('model');
        // Check for thinking model warning (async, non-blocking)
        checkThinkingModelWarning(modelName);
    },



    /**
     * Check if a model is small (≤12B parameters)
     * @param {string} modelName - Model name
     * @returns {boolean} True if model is small
     */
    isSmallModel(modelName) {
        return isSmallModel(modelName);
    },

    /**
     * Initialize model detector event listeners
     */
    initialize() {
        const modelSelect = DomHelpers.getElement('model');

        if (modelSelect) {
            modelSelect.addEventListener('change', () => {
                this.checkAndShowRecommendation();
            });
        }
    }
};
