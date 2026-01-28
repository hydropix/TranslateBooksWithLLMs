/**
 * SearchableSelect - Dropdown with search/filter functionality
 *
 * A reusable component that enhances standard select elements with:
 * - Clear visual display of selected value
 * - Search/filter input in dropdown header
 * - Keyboard navigation
 * - Grouping support
 * - Model count badge
 */

import { DomHelpers } from './dom-helpers.js';

/**
 * SearchableSelect class
 */
export class SearchableSelect {
    /**
     * Create a searchable select
     * @param {HTMLSelectElement|string} selectElement - Original select element or its ID
     * @param {Object} options - Configuration options
     */
    constructor(selectElement, options = {}) {
        this.originalSelect = typeof selectElement === 'string'
            ? DomHelpers.getElement(selectElement)
            : selectElement;

        if (!this.originalSelect) {
            console.error('SearchableSelect: Select element not found');
            return;
        }

        this.options = {
            placeholder: options.placeholder || 'Search...',
            noSelectionText: options.noSelectionText || 'Select an option...',
            allowCustomValue: options.allowCustomValue || false,
            onSelect: options.onSelect || null,
            renderOption: options.renderOption || null,
            showBadge: options.showBadge !== false // Show count badge by default
        };

        this.isOpen = false;
        this.highlightedIndex = -1;
        this.filteredOptions = [];
        this.allOptions = [];
        this.currentValue = null;

        this._init();
    }

    /**
     * Initialize the searchable select
     */
    _init() {
        this._createWrapper();
        this._createDisplay();
        this._createDropdown();
        this._bindEvents();
        this._updateFromOriginalSelect();

        // Hide original select
        this.originalSelect.style.display = 'none';
    }

    /**
     * Create wrapper element
     */
    _createWrapper() {
        this.wrapper = document.createElement('div');
        this.wrapper.className = 'searchable-select';
        this.originalSelect.parentNode.insertBefore(this.wrapper, this.originalSelect);
        this.wrapper.appendChild(this.originalSelect);
    }

    /**
     * Create the display area (shows selected value)
     */
    _createDisplay() {
        this.displayWrapper = document.createElement('div');
        this.displayWrapper.className = 'searchable-select-display';

        // Selected value text
        this.displayText = document.createElement('span');
        this.displayText.className = 'searchable-select-value';
        this.displayText.textContent = this.options.noSelectionText;

        // Badge for count
        this.badge = document.createElement('span');
        this.badge.className = 'searchable-select-badge';
        this.badge.style.display = 'none';

        // Dropdown arrow
        this.arrow = document.createElement('span');
        this.arrow.className = 'searchable-select-arrow material-symbols-outlined';
        this.arrow.textContent = 'expand_more';

        this.displayWrapper.appendChild(this.displayText);
        this.displayWrapper.appendChild(this.badge);
        this.displayWrapper.appendChild(this.arrow);
        this.wrapper.appendChild(this.displayWrapper);
    }

    /**
     * Create dropdown with search
     */
    _createDropdown() {
        this.dropdown = document.createElement('div');
        this.dropdown.className = 'searchable-select-dropdown';

        // Search header
        this.searchHeader = document.createElement('div');
        this.searchHeader.className = 'searchable-select-search';

        this.searchIcon = document.createElement('span');
        this.searchIcon.className = 'searchable-select-search-icon material-symbols-outlined';
        this.searchIcon.textContent = 'search';

        this.searchInput = document.createElement('input');
        this.searchInput.type = 'text';
        this.searchInput.className = 'searchable-select-input';
        this.searchInput.placeholder = this.options.placeholder;
        this.searchInput.autocomplete = 'off';

        this.clearBtn = document.createElement('button');
        this.clearBtn.type = 'button';
        this.clearBtn.className = 'searchable-select-clear';
        this.clearBtn.innerHTML = '<span class="material-symbols-outlined">close</span>';
        this.clearBtn.style.display = 'none';

        this.searchHeader.appendChild(this.searchIcon);
        this.searchHeader.appendChild(this.searchInput);
        this.searchHeader.appendChild(this.clearBtn);

        // Options list
        this.optionsList = document.createElement('ul');
        this.optionsList.className = 'searchable-select-options';

        this.dropdown.appendChild(this.searchHeader);
        this.dropdown.appendChild(this.optionsList);
        this.wrapper.appendChild(this.dropdown);
    }

    /**
     * Bind event listeners
     */
    _bindEvents() {
        // Display click opens dropdown
        this.displayWrapper.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this._toggle();
        });

        // Search input events
        this.searchInput.addEventListener('input', () => this._onInput());
        this.searchInput.addEventListener('keydown', (e) => this._onKeyDown(e));

        // Clear button
        this.clearBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.searchInput.value = '';
            this.clearBtn.style.display = 'none';
            this._filterOptions('');
            this.searchInput.focus();
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!this.wrapper.contains(e.target)) {
                this._close();
            }
        });

        // Watch for programmatic changes to original select
        const observer = new MutationObserver(() => {
            this._updateFromOriginalSelect();
        });
        observer.observe(this.originalSelect, {
            childList: true,
            subtree: true,
            attributes: true
        });
    }

    /**
     * Handle input event
     */
    _onInput() {
        const query = this.searchInput.value.toLowerCase().trim();
        this.clearBtn.style.display = query ? 'flex' : 'none';
        this._filterOptions(query);
    }

    /**
     * Handle keydown event
     * @param {KeyboardEvent} e
     */
    _onKeyDown(e) {
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                if (!this.isOpen) {
                    this._open();
                } else {
                    this._highlightNext();
                }
                break;
            case 'ArrowUp':
                e.preventDefault();
                this._highlightPrev();
                break;
            case 'Enter':
                e.preventDefault();
                if (this.highlightedIndex >= 0 && this.filteredOptions[this.highlightedIndex]) {
                    this._selectOption(this.filteredOptions[this.highlightedIndex]);
                } else if (this.options.allowCustomValue && this.searchInput.value.trim()) {
                    this._selectCustomValue(this.searchInput.value.trim());
                }
                break;
            case 'Escape':
                this._close();
                break;
            case 'Tab':
                this._close();
                break;
        }
    }

    /**
     * Filter options based on query
     * @param {string} query
     */
    _filterOptions(query) {
        if (!query) {
            this.filteredOptions = [...this.allOptions];
        } else {
            this.filteredOptions = this.allOptions.filter(opt => {
                const searchText = (opt.label + ' ' + opt.value + ' ' + (opt.group || '')).toLowerCase();
                return searchText.includes(query);
            });
        }

        this._renderOptions();
        this.highlightedIndex = -1;

        // Auto-highlight first result when searching
        if (query && this.filteredOptions.length > 0) {
            this.highlightedIndex = 0;
            this._updateHighlight();
        }
    }

    /**
     * Render filtered options
     */
    _renderOptions() {
        this.optionsList.innerHTML = '';

        if (this.filteredOptions.length === 0) {
            const noResults = document.createElement('li');
            noResults.className = 'searchable-select-no-results';
            noResults.innerHTML = this.options.allowCustomValue
                ? '<span class="material-symbols-outlined">add_circle</span> Press Enter to use "<span class="custom-value"></span>"'
                : '<span class="material-symbols-outlined">search_off</span> No results found';
            if (this.options.allowCustomValue) {
                noResults.querySelector('.custom-value').textContent = this.searchInput.value;
            }
            this.optionsList.appendChild(noResults);
            return;
        }

        let currentGroup = null;

        this.filteredOptions.forEach((opt, index) => {
            // Add group header if needed
            if (opt.group && opt.group !== currentGroup) {
                currentGroup = opt.group;
                const groupHeader = document.createElement('li');
                groupHeader.className = 'searchable-select-group';
                groupHeader.textContent = currentGroup;
                this.optionsList.appendChild(groupHeader);
            }

            const li = document.createElement('li');
            li.className = 'searchable-select-option';
            li.dataset.index = index;
            li.dataset.value = opt.value;

            if (opt.value === this.currentValue) {
                li.classList.add('selected');
            }

            // Custom renderer or default
            if (this.options.renderOption) {
                li.innerHTML = this.options.renderOption(opt);
            } else {
                const checkmark = opt.value === this.currentValue
                    ? '<span class="option-check material-symbols-outlined">check</span>'
                    : '<span class="option-check"></span>';
                li.innerHTML = `
                    ${checkmark}
                    <span class="option-content">
                        <span class="option-label">${DomHelpers.escapeHtml(opt.label)}</span>
                        ${opt.description ? `<span class="option-description">${DomHelpers.escapeHtml(opt.description)}</span>` : ''}
                    </span>
                `;
            }

            li.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this._selectOption(opt);
            });
            li.addEventListener('mouseenter', () => this._highlightOption(index));

            this.optionsList.appendChild(li);
        });
    }

    /**
     * Select an option
     * @param {Object} option
     */
    _selectOption(option) {
        this.currentValue = option.value;
        this.displayText.textContent = option.label;
        this.displayText.classList.remove('placeholder');

        // Update original select
        this.originalSelect.value = option.value;
        this.originalSelect.dispatchEvent(new Event('change', { bubbles: true }));

        this._close();

        if (this.options.onSelect) {
            this.options.onSelect(option);
        }
    }

    /**
     * Select a custom value (when allowCustomValue is true)
     * @param {string} value
     */
    _selectCustomValue(value) {
        this.currentValue = value;
        this.displayText.textContent = value;
        this.displayText.classList.remove('placeholder');

        // Add option to original select if not exists
        let optionEl = this.originalSelect.querySelector(`option[value="${CSS.escape(value)}"]`);
        if (!optionEl) {
            optionEl = document.createElement('option');
            optionEl.value = value;
            optionEl.textContent = value;
            this.originalSelect.appendChild(optionEl);
        }

        this.originalSelect.value = value;
        this.originalSelect.dispatchEvent(new Event('change', { bubbles: true }));

        this._close();

        if (this.options.onSelect) {
            this.options.onSelect({ value, label: value, custom: true });
        }
    }

    /**
     * Highlight next option
     */
    _highlightNext() {
        if (this.filteredOptions.length === 0) return;

        this.highlightedIndex = Math.min(
            this.highlightedIndex + 1,
            this.filteredOptions.length - 1
        );
        this._updateHighlight();
    }

    /**
     * Highlight previous option
     */
    _highlightPrev() {
        if (this.filteredOptions.length === 0) return;

        this.highlightedIndex = Math.max(this.highlightedIndex - 1, 0);
        this._updateHighlight();
    }

    /**
     * Highlight option at index
     * @param {number} index
     */
    _highlightOption(index) {
        this.highlightedIndex = index;
        this._updateHighlight();
    }

    /**
     * Update highlight styling
     */
    _updateHighlight() {
        const options = this.optionsList.querySelectorAll('.searchable-select-option');
        options.forEach((opt, i) => {
            opt.classList.toggle('highlighted', i === this.highlightedIndex);
        });

        // Scroll highlighted into view
        if (this.highlightedIndex >= 0 && options[this.highlightedIndex]) {
            options[this.highlightedIndex].scrollIntoView({
                block: 'nearest',
                behavior: 'smooth'
            });
        }
    }

    /**
     * Open dropdown
     */
    _open() {
        if (this.isOpen) return;

        this.isOpen = true;
        this.wrapper.classList.add('open');
        this.arrow.textContent = 'expand_less';

        // Reset search and show all options
        this.searchInput.value = '';
        this.clearBtn.style.display = 'none';
        this._filterOptions('');

        // Focus search input
        setTimeout(() => {
            this.searchInput.focus();
        }, 50);

        // Scroll to selected option
        setTimeout(() => {
            const selected = this.optionsList.querySelector('.selected');
            if (selected) {
                selected.scrollIntoView({ block: 'center', behavior: 'instant' });
            }
        }, 100);
    }

    /**
     * Close dropdown
     */
    _close() {
        if (!this.isOpen) return;

        this.isOpen = false;
        this.wrapper.classList.remove('open');
        this.arrow.textContent = 'expand_more';
        this.highlightedIndex = -1;
    }

    /**
     * Toggle dropdown
     */
    _toggle() {
        if (this.isOpen) {
            this._close();
        } else {
            this._open();
        }
    }

    /**
     * Update badge with count
     */
    _updateBadge() {
        if (this.options.showBadge && this.allOptions.length > 0) {
            this.badge.textContent = `${this.allOptions.length} models`;
            this.badge.style.display = 'inline-flex';
        } else {
            this.badge.style.display = 'none';
        }
    }

    /**
     * Update options from original select element
     */
    _updateFromOriginalSelect() {
        this.allOptions = [];

        const options = this.originalSelect.querySelectorAll('option');
        let currentGroup = null;

        options.forEach(opt => {
            if (opt.value === '' && (opt.textContent.includes('Loading') || opt.textContent.includes('Enter'))) {
                return; // Skip loading/placeholder
            }

            // Check for optgroup
            if (opt.parentElement.tagName === 'OPTGROUP') {
                currentGroup = opt.parentElement.label;
            } else {
                currentGroup = null;
            }

            this.allOptions.push({
                value: opt.value,
                label: opt.textContent,
                description: opt.title || '',
                group: currentGroup,
                selected: opt.selected
            });

            if (opt.selected && opt.value) {
                this.currentValue = opt.value;
                this.displayText.textContent = opt.textContent;
                this.displayText.classList.remove('placeholder');
            }
        });

        // If no selection, show placeholder
        if (!this.currentValue && this.allOptions.length > 0) {
            // Auto-select first option
            const firstOpt = this.allOptions[0];
            this.currentValue = firstOpt.value;
            this.displayText.textContent = firstOpt.label;
            this.displayText.classList.remove('placeholder');
            this.originalSelect.value = firstOpt.value;
        } else if (this.allOptions.length === 0) {
            this.displayText.textContent = this.options.noSelectionText;
            this.displayText.classList.add('placeholder');
        }

        this.filteredOptions = [...this.allOptions];
        this._updateBadge();

        // Re-render if dropdown is open
        if (this.isOpen) {
            this._renderOptions();
        }
    }

    /**
     * Set value programmatically
     * @param {string} value
     */
    setValue(value) {
        const option = this.allOptions.find(o => o.value === value);
        if (option) {
            this._selectOption(option);
        } else if (this.options.allowCustomValue) {
            this._selectCustomValue(value);
        }
    }

    /**
     * Get current value
     * @returns {string}
     */
    getValue() {
        return this.currentValue;
    }

    /**
     * Refresh options from select
     */
    refresh() {
        this._updateFromOriginalSelect();
    }

    /**
     * Destroy the searchable select and restore original
     */
    destroy() {
        this.originalSelect.style.display = '';
        this.wrapper.parentNode.insertBefore(this.originalSelect, this.wrapper);
        this.wrapper.remove();
    }
}

/**
 * Factory function to create SearchableSelect instances
 */
export const SearchableSelectFactory = {
    instances: new Map(),

    /**
     * Create or get instance for a select element
     * @param {HTMLSelectElement|string} selectElement
     * @param {Object} options
     * @returns {SearchableSelect}
     */
    create(selectElement, options = {}) {
        const el = typeof selectElement === 'string'
            ? DomHelpers.getElement(selectElement)
            : selectElement;

        if (!el) return null;

        const id = el.id || `searchable-${Date.now()}`;

        if (this.instances.has(id)) {
            return this.instances.get(id);
        }

        const instance = new SearchableSelect(el, options);
        this.instances.set(id, instance);
        return instance;
    },

    /**
     * Get instance by select ID
     * @param {string} id
     * @returns {SearchableSelect|null}
     */
    get(id) {
        return this.instances.get(id) || null;
    },

    /**
     * Destroy instance
     * @param {string} id
     */
    destroy(id) {
        const instance = this.instances.get(id);
        if (instance) {
            instance.destroy();
            this.instances.delete(id);
        }
    },

    /**
     * Destroy all instances
     */
    destroyAll() {
        this.instances.forEach(instance => instance.destroy());
        this.instances.clear();
    }
};
