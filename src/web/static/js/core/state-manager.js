/**
 * State Manager - Centralized state management with pub/sub pattern
 *
 * Provides a single source of truth for application state with
 * reactive updates via subscriber notifications.
 */

/**
 * Application state schema
 */
const state = {
    files: {
        toProcess: [],           // Files queued for translation
        selected: new Set(),     // Selected files in file manager
        managed: []              // Files displayed in file manager
    },
    translation: {
        currentJob: null,        // Current processing job { fileRef, translationId }
        isBatchActive: false,    // Whether a batch is currently running
        activeJobs: [],          // List of active translation jobs
        hasActive: false,        // Quick flag for active translations
        // Translation ids, most-recent last, that this browser tab started or
        // adopted. Used to ignore terminal events belonging to another tab.
        ownedJobIds: []
    },
    ui: {
        currentProvider: 'ollama',  // Selected LLM provider
        currentModel: null,         // Selected model
        isAdvancedOpen: false,      // Advanced settings panel state
        messages: []                // UI messages history
    },
    models: {
        currentLoadRequest: null,   // Track current model load request
        availableModels: []         // Cached available models
    }
};

/**
 * Subscribers registry
 * Map<string, Array<Function>>
 */
const subscribers = new Map();

/**
 * State Manager API
 */
export const StateManager = {
    /**
     * Get state value by key path (dot notation supported)
     * @param {string} [key] - State key path (e.g., 'files.toProcess' or 'translation')
     * @returns {any} State value or entire state if no key
     */
    getState(key) {
        if (!key) return state;

        const keys = key.split('.');
        let value = state;
        for (const k of keys) {
            value = value?.[k];
            if (value === undefined) return undefined;
        }
        return value;
    },

    /**
     * Set state value by key path
     * @param {string} key - State key path (e.g., 'files.toProcess')
     * @param {any} value - New value
     */
    setState(key, value) {
        const keys = key.split('.');
        const lastKey = keys.pop();

        // Navigate to parent object
        let target = state;
        for (const k of keys) {
            if (!target[k]) target[k] = {};
            target = target[k];
        }

        const oldValue = target[lastKey];
        target[lastKey] = value;

        // Notify subscribers
        this.notify(key, value, oldValue);
    },

    /**
     * Subscribe to state changes
     * @param {string} key - State key to watch
     * @param {Function} callback - Callback(newValue, oldValue)
     * @returns {Function} Unsubscribe function
     */
    subscribe(key, callback) {
        if (!subscribers.has(key)) {
            subscribers.set(key, []);
        }
        subscribers.get(key).push(callback);

        // Return unsubscribe function
        return () => {
            const callbacks = subscribers.get(key);
            if (callbacks) {
                const index = callbacks.indexOf(callback);
                if (index > -1) {
                    callbacks.splice(index, 1);
                }
            }
        };
    },

    /**
     * Unsubscribe from state changes
     * @param {string} key - State key
     * @param {Function} callback - Callback to remove
     */
    unsubscribe(key, callback) {
        if (subscribers.has(key)) {
            const callbacks = subscribers.get(key);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    },

    /**
     * Notify subscribers of state change
     * @param {string} key - State key that changed
     * @param {any} newValue - New value
     * @param {any} oldValue - Previous value
     */
    notify(key, newValue, oldValue) {
        if (subscribers.has(key)) {
            subscribers.get(key).forEach(callback => {
                try {
                    callback(newValue, oldValue);
                } catch (error) {
                    console.error(`Error in state subscriber for "${key}":`, error);
                }
            });
        }

        // Also notify parent keys (e.g., 'files' when 'files.toProcess' changes)
        const parts = key.split('.');
        if (parts.length > 1) {
            for (let i = parts.length - 1; i > 0; i--) {
                const parentKey = parts.slice(0, i).join('.');
                if (subscribers.has(parentKey)) {
                    const parentValue = this.getState(parentKey);
                    subscribers.get(parentKey).forEach(callback => {
                        try {
                            callback(parentValue, parentValue);
                        } catch (error) {
                            console.error(`Error in parent state subscriber for "${parentKey}":`, error);
                        }
                    });
                }
            }
        }
    },

    /**
     * Reset specific state key or entire state
     * @param {string} [key] - State key to reset, or reset all if omitted
     */
    reset(key) {
        if (!key) {
            // Reset entire state
            state.files = { toProcess: [], selected: new Set(), managed: [] };
            state.translation = { currentJob: null, isBatchActive: false, activeJobs: [], hasActive: false, ownedJobIds: [] };
            state.ui = { currentProvider: 'ollama', currentModel: null, isAdvancedOpen: false, messages: [] };
            state.models = { currentLoadRequest: null, availableModels: [] };

            this.notify('state', state, state);
        } else {
            // Reset specific key to default
            const defaultValues = {
                'files.toProcess': [],
                'files.selected': new Set(),
                'files.managed': [],
                'translation.currentJob': null,
                'translation.isBatchActive': false,
                'translation.activeJobs': [],
                'translation.hasActive': false,
                'translation.ownedJobIds': [],
                'ui.messages': []
            };

            if (defaultValues.hasOwnProperty(key)) {
                this.setState(key, defaultValues[key]);
            }
        }
    },

    /**
     * Get debug information
     * @returns {Object} Debug information
     */
    debug() {
        return {
            state: JSON.parse(JSON.stringify({
                ...state,
                files: {
                    ...state.files,
                    selected: Array.from(state.files.selected)
                }
            })),
            subscriberCount: Array.from(subscribers.entries()).map(([key, cbs]) => ({
                key,
                count: cbs.length
            }))
        };
    }
};

// Make state manager available globally for debugging
if (typeof window !== 'undefined') {
    window.__STATE_MANAGER__ = StateManager;
}
