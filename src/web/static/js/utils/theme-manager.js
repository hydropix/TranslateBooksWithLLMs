/**
 * Theme Manager
 *
 * Handles dark/light mode theme switching and persistence.
 */

class ThemeManager {
    constructor() {
        this.htmlElement = document.documentElement;
        this.themeIcon = document.getElementById('themeIcon');
        this.STORAGE_KEY = 'tbl-theme-preference';

        document.getElementById('themeToggleBtn')
            ?.addEventListener('click', () => this.toggleTheme());

        // Initialize theme from localStorage or system preference
        this.initializeTheme();
    }

    /**
     * Initialize theme based on saved preference or system default
     */
    initializeTheme() {
        const savedTheme = localStorage.getItem(this.STORAGE_KEY);

        if (savedTheme) {
            this.setTheme(savedTheme);
        } else {
            // Check system preference
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            this.setTheme(prefersDark ? 'dark' : 'light');
        }

        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            // Only auto-switch if user hasn't manually set a preference
            if (!localStorage.getItem(this.STORAGE_KEY)) {
                this.setTheme(e.matches ? 'dark' : 'light');
            }
        });
    }

    /**
     * Get current theme
     * @returns {string} 'light' or 'dark'
     */
    getCurrentTheme() {
        return this.htmlElement.classList.contains('dark') ? 'dark' : 'light';
    }

    /**
     * Set theme
     * @param {string} theme - 'light' or 'dark'
     */
    setTheme(theme) {
        if (theme === 'dark') {
            this.htmlElement.classList.add('dark');
            if (this.themeIcon) {
                this.themeIcon.textContent = 'light_mode';
            }
        } else {
            this.htmlElement.classList.remove('dark');
            if (this.themeIcon) {
                this.themeIcon.textContent = 'dark_mode';
            }
        }

        // Save preference
        localStorage.setItem(this.STORAGE_KEY, theme);
    }

    /**
     * Toggle between light and dark themes
     */
    toggleTheme() {
        const currentTheme = this.getCurrentTheme();
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }
}

// Global instance
let themeManagerInstance = null;

/**
 * Initialize theme manager
 * @returns {ThemeManager}
 */
export function initializeThemeManager() {
    if (!themeManagerInstance) {
        themeManagerInstance = new ThemeManager();
    }
    return themeManagerInstance;
}

/**
 * Get theme manager instance
 * @returns {ThemeManager}
 */
function getThemeManager() {
    return themeManagerInstance;
}

/**
 * Global toggle, kept for callers that use the window API. The header button
 * is wired by ThemeManager itself.
 */
window.toggleTheme = function() {
    const manager = getThemeManager();
    if (manager) {
        manager.toggleTheme();
    }
};
