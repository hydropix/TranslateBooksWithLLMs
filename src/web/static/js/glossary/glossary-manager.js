/**
 * Glossary Manager
 *
 * Facade for the glossary UI. Wires up the dropdown on the Translate tab
 * (glossary-dropdown.js), the list and editor views on the Glossaries tab
 * (glossary-list.js, glossary-editor.js), the NER auto-extract modal
 * (glossary-ner-modal.js) and the prompt-block preview modal
 * (glossary-preview-modal.js). Shared state and helpers live in
 * glossary-shared.js. The top-tab router itself lives in ui/tab-router.js;
 * this facade only registers the list refresh for the Glossaries tab.
 */

import { onTabShown } from '../ui/tab-router.js';
import { $, glossaryState } from './glossary-shared.js';
import { refreshDropdown, wireTranslateTabDropdown } from './glossary-dropdown.js';
import { loadList, wireListView } from './glossary-list.js';
import { openEditor, wireEditorView } from './glossary-editor.js';
import { wireNerModal } from './glossary-ner-modal.js';
import { wirePreviewModal } from './glossary-preview-modal.js';

// Reload the glossary list every time the Glossaries tab is shown.
onTabShown('glossaries', loadList);

// ========================================
// Public API
// ========================================

export const GlossaryManager = {
    initialize() {
        wireTranslateTabDropdown();
        wireListView();
        wireEditorView();
        wireNerModal();
        wirePreviewModal();
        refreshDropdown().catch((err) => {
            console.error('Initial glossary dropdown refresh failed:', err);
        });

        // Re-render dynamic glossary content when the UI locale changes:
        //  - the Translate-tab dropdown carries t()-built option labels,
        //  - the list view tbody is JS-rendered (term counts, "—" fallbacks),
        //  - an open editor has translated category dropdowns per row.
        // Each helper is a no-op if its container isn't currently mounted.
        window.addEventListener('localeChanged', () => {
            refreshDropdown().catch(() => { /* swallow — UI text only */ });
            const listTable = $('glossaryListTable');
            if (listTable && !listTable.classList.contains('hidden')) {
                loadList().catch(() => { /* swallow — UI text only */ });
            }
            const editorView = $('glossaryEditorView');
            if (editorView && !editorView.classList.contains('hidden') && glossaryState.currentGlossaryId) {
                openEditor(glossaryState.currentGlossaryId).catch(() => { /* swallow */ });
            }
        });
    },
    refreshDropdown() {
        return refreshDropdown();
    },
};
