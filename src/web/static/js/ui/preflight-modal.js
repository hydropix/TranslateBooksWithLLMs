/**
 * Pre-flight Modal - the confirmation dialog that gates the Start button.
 *
 * Second entry point for the advice already shown inline by preflight-zone.js:
 * the zone is passive (it sits under the button and waits), this dialog is the
 * last-chance interruption. Both read the *same* rules from preflight-rules.js
 * and the same dismissal store, so there is no second notion of what to advise —
 * loadDismissed() and actOnRule() are imported from the zone rather than
 * reimplemented here. Suppressing a tip for good is the zone's job, not this
 * dialog's; it only reads the result.
 *
 * When nothing applies, guard() launches straight through: the dialog must never
 * add a click to a run that has nothing to be told.
 */

import { ModalStack } from './modal-stack.js';
import { DomHelpers } from './dom-helpers.js';
import { StateManager } from '../core/state-manager.js';
import { t } from '../i18n/i18n.js';
import { buildContext, evaluateRules } from './preflight-rules.js';
import { actOnRule, loadDismissed } from './preflight-zone.js';

// The rules currently painted, so the delegated click handler can recover the
// rule behind a button without re-evaluating the context mid-interaction.
let visibleRules = new Map();

// Resolver for the promise confirm() handed back. Held only while the dialog is
// open; every closing path settles it exactly once, so a caller awaiting the
// answer can never be left hanging.
let pendingResolve = null;

// Release callback from ModalStack while the dialog is open (Escape abandons).
let releaseEscape = null;

function isOpen() {
    const modal = DomHelpers.getElement('preflightModal');
    return !!modal && !modal.classList.contains('hidden');
}

/** @returns {import('./preflight-rules.js').PreflightRule[]} Rules worth showing */
function activeRules() {
    const dismissed = loadDismissed();
    return evaluateRules(buildContext()).filter(rule => !dismissed.includes(rule.id));
}

// The fix is derived from the control the rule points at, not from a list of
// rule ids: a checkbox can be flipped from inside the dialog, anything else
// needs the user to go and edit it. That keeps rule knowledge in
// preflight-rules.js and stops this module from drifting when a rule is added.
function fixableInPlace(el) {
    return !!(el && el.tagName === 'INPUT' && el.type === 'checkbox');
}

function renderRow(rule) {
    const id = DomHelpers.escapeHtml(rule.id);
    // t() output carries a model name and a file extension, both user-influenced,
    // and i18n.js runs with escapeValue: false.
    const reason = DomHelpers.escapeHtml(t(rule.reasonKey, rule.params));
    const fixKey = fixableInPlace(DomHelpers.getElement(rule.focusId))
        ? 'translation:preflight_modal_enable'
        : 'translation:preflight_adjust';
    const fixLabel = DomHelpers.escapeHtml(t(fixKey));
    // Two escalation levels above a plain suggestion: amber for a risky run,
    // brick red for one that is simply wrong (source language === target).
    const variant = rule.severity === 'danger' ? ' preflight-modal__rule--danger'
        : rule.severity === 'warning' ? ' preflight-modal__rule--warning'
        : '';

    // No dismiss control here on purpose: this dialog is a one-off interruption,
    // and suppressing a tip for good belongs to the inline zone's chips, which
    // the user sees on every visit. Dismissals made there are still honoured —
    // activeRules() filters on the same store.
    return `<li class="preflight-modal__rule${variant}">`
        + `<span class="preflight-modal__rule-text">${reason}</span>`
        + `<span class="preflight-modal__rule-actions">`
        + `<button type="button" class="btn btn-sm"`
        + ` data-preflight-modal-fix="${id}">${fixLabel}</button>`
        + `</span>`
        + `</li>`;
}

/**
 * Repaint the rule list. Every label is resolved with t() here, on each pass,
 * which is what lets a locale switch land on an open dialog.
 * @param {import('./preflight-rules.js').PreflightRule[]} [rules] - Defaults to a fresh evaluation
 */
function renderList(rules) {
    const list = DomHelpers.getElement('preflightModalList');
    if (!list) return;
    const current = rules || activeRules();
    visibleRules = new Map(current.map(rule => [rule.id, rule]));
    list.innerHTML = current.map(renderRow).join('');
}

function openDialog(rules) {
    renderList(rules);
    DomHelpers.show('preflightModal');
    if (!releaseEscape) releaseEscape = ModalStack.push(abandon);
    // Land on the confirm button: the user already asked to start, so Enter
    // should mean "launch as is" rather than re-trigger the first suggestion.
    DomHelpers.getElement('preflightModalLaunch')?.focus();
}

function closeDialog() {
    DomHelpers.hide('preflightModal');
    if (releaseEscape) {
        releaseEscape();
        releaseEscape = null;
    }
    visibleRules = new Map();
    // Hand focus back to the button that opened the dialog so keyboard users are
    // not dropped at the top of the document. Callers that navigate elsewhere
    // (actOnRule) focus their own target afterwards and win, since focusElement
    // defers by a timeout.
    DomHelpers.getElement('translateBtn')?.focus();
}

/**
 * Settle the pending confirm() exactly once.
 * @param {boolean} proceed - Whether the caller should go ahead with the launch
 */
function settle(proceed) {
    const resolve = pendingResolve;
    pendingResolve = null;
    closeDialog();
    if (resolve) resolve(proceed);
}

/** Close without launching: Cancel, the close button, Escape, backdrop, Adjust. */
function abandon() {
    settle(false);
}

function applyFix(rule) {
    if (!rule) return;
    const el = DomHelpers.getElement(rule.focusId);

    if (fixableInPlace(el)) {
        el.checked = true;
        // The `change` event is the fix, not a formality: the settings
        // auto-save, the settings summary and the inline zone all hang off it.
        // Writing localStorage from here would fork that chain.
        el.dispatchEvent(new Event('change', { bubbles: true }));
        // Repaint in place and stay open — the user still has to confirm. An
        // empty list is a valid end state; it is not an implicit launch.
        renderList();
        return;
    }

    // The control lives outside this dialog, so the fix is an edit the user has
    // to make. Abandon the launch: they will press Start again when done.
    settle(false);
    actOnRule(rule);
}

// One delegated listener on the overlay. The list is repainted wholesale, so
// per-row listeners would leak a closure per render.
function handleClick(event) {
    // Backdrop only: any click inside .modal-content has a closer ancestor.
    if (event.target === DomHelpers.getElement('preflightModal')) {
        abandon();
        return;
    }

    const fix = event.target.closest('[data-preflight-modal-fix]');
    if (fix) {
        applyFix(visibleRules.get(fix.getAttribute('data-preflight-modal-fix')));
        return;
    }

    if (event.target.closest('#preflightModalLaunch')) {
        settle(true);
        return;
    }

    if (event.target.closest('#preflightModalCancel')
        || event.target.closest('#preflightModalClose')) {
        abandon();
    }
}

export const PreflightModal = {
    /** Wire the dialog. Called from index.js right after PreflightZone.initialize(). */
    initialize() {
        const modal = DomHelpers.getElement('preflightModal');
        if (modal) modal.addEventListener('click', handleClick);

        // Only while open: the rows are JS-owned text with no data-i18n
        // attribute, so applyToDOM() cannot re-translate them.
        window.addEventListener('localeChanged', () => {
            if (isOpen()) renderList();
        });
    },

    /**
     * Ask the user to confirm the launch, but only when there is something worth
     * saying. Called by batch-controller *after* its own validation has passed,
     * so a misconfigured run reports the real problem instead of a suggestion
     * list the user would have to dismiss first.
     * @returns {Promise<boolean>} true to proceed with the launch, false to abort
     */
    confirm() {
        // Never gate mid-run: the options in the dialog cannot influence a job
        // that is already in flight, and the zone is locked for the same reason.
        if (StateManager.getState('translation.isBatchActive')) return Promise.resolve(true);

        const rules = activeRules();
        // Nothing to say, no extra click. Also the fallback when the overlay is
        // missing from the DOM: a launch must never be lost to a render gap.
        if (!rules.length || !DomHelpers.getElement('preflightModal')) return Promise.resolve(true);

        // A second call while the dialog is already up would orphan the first
        // resolver; abandon it so its caller unwinds instead of hanging.
        if (pendingResolve) settle(false);

        return new Promise(resolve => {
            pendingResolve = resolve;
            openDialog(rules);
        });
    },
};
