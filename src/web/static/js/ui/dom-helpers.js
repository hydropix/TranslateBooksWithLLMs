/**
 * DOM Helpers - Utility functions for DOM manipulation
 *
 * Provides clean abstractions for common DOM operations
 */

export const DomHelpers = {
    /**
     * Escape HTML to prevent XSS
     * Numbers and booleans are stringified; null, undefined and objects
     * yield an empty string.
     * @param {string|number|boolean} unsafe - Unsafe value
     * @returns {string} Escaped HTML
     */
    escapeHtml(unsafe) {
        if (typeof unsafe === 'number' || typeof unsafe === 'boolean') {
            unsafe = String(unsafe);
        }
        if (typeof unsafe !== 'string') return '';

        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    },

    /**
     * Get element by ID
     * @param {string} id - Element ID
     * @returns {HTMLElement|null} Element or null
     */
    getElement(id) {
        return document.getElementById(id);
    },

    /**
     * Get elements by selector
     * @param {string} selector - CSS selector
     * @param {HTMLElement} [parent=document] - Parent element
     * @returns {NodeList} Node list
     */
    getElements(selector, parent = document) {
        return parent.querySelectorAll(selector);
    },

    /**
     * Get single element by selector
     * @param {string} selector - CSS selector
     * @param {HTMLElement} [parent=document] - Parent element
     * @returns {HTMLElement|null} Element or null
     */
    getOne(selector, parent = document) {
        return parent.querySelector(selector);
    },

    /**
     * Show element (remove 'hidden' class)
     * @param {HTMLElement|string} element - Element or element ID
     */
    show(element) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) {
            el.classList.remove('hidden');
            // Force immediate display update to ensure visibility
            if (el.style.display === 'none') {
                el.style.display = '';
            }
        }
    },

    /**
     * Hide element (add 'hidden' class)
     * @param {HTMLElement|string} element - Element or element ID
     */
    hide(element) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.classList.add('hidden');
    },

    /**
     * Toggle element visibility
     * @param {HTMLElement|string} element - Element or element ID
     * @param {boolean} [force] - Force show (true) or hide (false)
     */
    toggle(element, force) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.classList.toggle('hidden', force);
    },

    /**
     * Add class to element
     * @param {HTMLElement|string} element - Element or element ID
     * @param {string} className - Class name
     */
    addClass(element, className) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.classList.add(className);
    },

    /**
     * Remove class from element
     * @param {HTMLElement|string} element - Element or element ID
     * @param {string} className - Class name
     */
    removeClass(element, className) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.classList.remove(className);
    },

    /**
     * Toggle class on element
     * @param {HTMLElement|string} element - Element or element ID
     * @param {string} className - Class name
     * @param {boolean} [force] - Force add (true) or remove (false)
     */
    toggleClass(element, className, force) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.classList.toggle(className, force);
    },

    /**
     * Set text content
     * @param {HTMLElement|string} element - Element or element ID
     * @param {string} text - Text content
     */
    setText(element, text) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.textContent = text;
    },

    /**
     * Set HTML content (use with caution, prefer setText)
     * @param {HTMLElement|string} element - Element or element ID
     * @param {string} html - HTML content
     */
    setHtml(element, html) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.innerHTML = html;
    },

    /**
     * Get value from input element
     * @param {HTMLElement|string} element - Element or element ID
     * @returns {string} Input value
     */
    getValue(element) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        return el ? el.value : '';
    },

    /**
     * Set value on input element
     * @param {HTMLElement|string} element - Element or element ID
     * @param {string} value - Value to set
     */
    setValue(element, value) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.value = value;
    },

    /**
     * Enable element
     * @param {HTMLElement|string} element - Element or element ID
     */
    enable(element) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.disabled = false;
    },

    /**
     * Disable element
     * @param {HTMLElement|string} element - Element or element ID
     */
    disable(element) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.disabled = true;
    },

    /**
     * Set disabled state on element
     * @param {HTMLElement|string} element - Element or element ID
     * @param {boolean} disabled - Whether to disable (true) or enable (false)
     */
    setDisabled(element, disabled) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.disabled = disabled;
    },

    /**
     * Set element attribute
     * @param {HTMLElement|string} element - Element or element ID
     * @param {string} attr - Attribute name
     * @param {string} value - Attribute value
     */
    setAttr(element, attr, value) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.setAttribute(attr, value);
    },

    /**
     * Remove element attribute
     * @param {HTMLElement|string} element - Element or element ID
     * @param {string} attr - Attribute name
     */
    removeAttr(element, attr) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.removeAttribute(attr);
    },

    /**
     * Create element with properties
     * @param {string} tag - HTML tag name
     * @param {Object} [props] - Properties (className, id, textContent, etc.)
     * @param {Array<HTMLElement>|HTMLElement} [children] - Child elements
     * @returns {HTMLElement} Created element
     */
    createElement(tag, props = {}, children = []) {
        const element = document.createElement(tag);

        Object.entries(props).forEach(([key, value]) => {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'textContent') {
                element.textContent = value;
            } else if (key === 'innerHTML') {
                element.innerHTML = value;
            } else if (key.startsWith('on')) {
                // Event listeners
                const eventName = key.substring(2).toLowerCase();
                element.addEventListener(eventName, value);
            } else {
                element.setAttribute(key, value);
            }
        });

        const childArray = Array.isArray(children) ? children : [children];
        childArray.forEach(child => {
            if (child instanceof HTMLElement) {
                element.appendChild(child);
            } else if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            }
        });

        return element;
    },

    /**
     * Remove all children from element
     * @param {HTMLElement|string} element - Element or element ID
     */
    clearChildren(element) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) {
            while (el.firstChild) {
                el.removeChild(el.firstChild);
            }
        }
    },

    /**
     * Scroll element into view
     * @param {HTMLElement|string} element - Element or element ID
     * @param {Object} [options] - Scroll options
     */
    scrollIntoView(element, options = { behavior: 'smooth' }) {
        const el = typeof element === 'string' ? this.getElement(element) : element;
        if (el) el.scrollIntoView(options);
    }
};
