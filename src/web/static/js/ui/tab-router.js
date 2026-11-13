/**
 * Top-tab router
 *
 * Shows one top-level tab (Translate, Sample, Settings, Glossaries, Styles,
 * Files), highlights its button, and runs the refresh hooks for the tab
 * that was just shown. Feature modules register their hook through
 * onTabShown() instead of being imported here, so the router has no
 * dependency on them and cannot close an import cycle. The Files and
 * Styles tabs still go through the window.refreshFileList /
 * window.refreshStyleList globals that index.js installs.
 *
 * window.switchTopTab is kept for the template's inline onclick handlers
 * and for modules that call it through the global.
 */

const _tabShownCallbacks = new Map();

/**
 * Register a callback to run every time the given tab is shown.
 * @param {string} name - Tab name, as passed to switchTopTab()
 * @param {Function} callback
 */
export function onTabShown(name, callback) {
    if (!_tabShownCallbacks.has(name)) _tabShownCallbacks.set(name, []);
    _tabShownCallbacks.get(name).push(callback);
}

function $(id) {
    return document.getElementById(id);
}

export function switchTopTab(name) {
    const translateTab = $('tab-translate');
    const settingsTab = $('tab-settings');
    const glossariesTab = $('tab-glossaries');
    const filesTab = $('tab-files');
    const sampleTab = $('tab-sample');
    const stylesTab = $('tab-styles');

    if (translateTab) translateTab.classList.toggle('hidden', name !== 'translate');
    if (settingsTab) settingsTab.classList.toggle('hidden', name !== 'settings');
    if (glossariesTab) glossariesTab.classList.toggle('hidden', name !== 'glossaries');
    if (filesTab) filesTab.classList.toggle('hidden', name !== 'files');
    if (sampleTab) sampleTab.classList.toggle('hidden', name !== 'sample');
    if (stylesTab) stylesTab.classList.toggle('hidden', name !== 'styles');

    const buttons = document.querySelectorAll('#topTabNav .tab-btn');
    buttons.forEach((btn) => {
        const isActive = btn.dataset.tab === name;
        btn.classList.toggle('tab-btn-active', isActive);
    });

    const callbacks = _tabShownCallbacks.get(name);
    if (callbacks) callbacks.forEach((callback) => callback());

    if (name === 'files' && typeof window.refreshFileList === 'function') {
        window.refreshFileList();
    } else if (name === 'styles' && typeof window.refreshStyleList === 'function') {
        window.refreshStyleList();
    }
}

window.switchTopTab = switchTopTab;
