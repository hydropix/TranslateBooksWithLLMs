/**
 * Progress Manager - Translation progress tracking and display
 *
 * Manages progress bar updates and statistics display for active translations.
 * Supports different file types (text, EPUB, SRT) with appropriate stat labels.
 */

import { DomHelpers } from '../ui/dom-helpers.js';
import { t } from '../i18n/i18n.js';
import { navigateToSetting } from '../ui/settings-summary.js';

/**
 * Is a stats key actually present in the payload? Precedence in this module is
 * by presence, not truthiness — the same discipline as `_unfinished()` in
 * src/api/completion_status.py and `runCounter()` in translation-tracker.js: a
 * key that is present and 0 is trusted, and only an absent one falls back to an
 * older counter (legacy payloads, formats that never emit it).
 *
 * @param {Object} stats - Job stats payload
 * @param {string} key - Key to test
 * @returns {boolean}
 */
function statPresent(stats, key) {
    return !!stats && stats[key] !== undefined && stats[key] !== null;
}

// State for tracking chunk completion times (for ETA calculation)
let chunkCompletionTimes = [];
let lastCompletedChunks = 0;
let lastElapsedTime = 0;
const MAX_SAMPLES = 10; // Number of recent chunks to average for ETA

/**
 * Format elapsed time in a human-readable format
 * - Under 60s: shows seconds (e.g., "45.2s")
 * - Under 1h: shows minutes and seconds (e.g., "5m 23s")
 * - 1h+: shows hours, minutes and seconds (e.g., "1h 23m 45s")
 * @param {number} seconds - Elapsed time in seconds
 * @returns {string} Formatted time string
 */
export function formatElapsedTime(seconds) {
    if (seconds < 60) {
        return seconds.toFixed(1) + 's';
    }

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    }

    return `${minutes}m ${secs}s`;
}

/**
 * Calculate and update estimated time remaining
 * Uses a moving average of the last N chunk completion times
 * @param {number} completedChunks - Number of completed chunks
 * @param {number} totalChunks - Total number of chunks
 * @param {number} elapsedTime - Total elapsed time in seconds
 */
function updateEstimatedTimeRemaining(completedChunks, totalChunks, elapsedTime) {
    // Track time per chunk when a new chunk is completed
    if (completedChunks > lastCompletedChunks && elapsedTime > lastElapsedTime) {
        const chunksCompleted = completedChunks - lastCompletedChunks;
        const timeTaken = elapsedTime - lastElapsedTime;
        const timePerChunk = timeTaken / chunksCompleted;

        chunkCompletionTimes.push(timePerChunk);

        // Keep only the last N samples
        if (chunkCompletionTimes.length > MAX_SAMPLES) {
            chunkCompletionTimes.shift();
        }
    }

    lastCompletedChunks = completedChunks;
    lastElapsedTime = elapsedTime;

    // Calculate ETA only if we have samples
    if (chunkCompletionTimes.length === 0 || completedChunks === 0) {
        DomHelpers.setText('estimatedTimeRemaining', '--');
        return;
    }

    const remainingChunks = totalChunks - completedChunks;

    if (remainingChunks <= 0) {
        DomHelpers.setText('estimatedTimeRemaining', '0s');
        return;
    }

    // Calculate average time per chunk from recent samples
    const avgTimePerChunk = chunkCompletionTimes.reduce((a, b) => a + b, 0) / chunkCompletionTimes.length;
    const estimatedRemaining = avgTimePerChunk * remainingChunks;

    DomHelpers.setText('estimatedTimeRemaining', formatElapsedTime(estimatedRemaining));
}

/**
 * Reset ETA tracking state
 */
function resetEtaTracking() {
    chunkCompletionTimes = [];
    lastCompletedChunks = 0;
    lastElapsedTime = 0;
}

/**
 * Update progress bar
 * @param {number} percent - Progress percentage (0-100)
 */
function updateProgressBar(percent) {
    const progressBar = DomHelpers.getElement('progressBar');

    if (progressBar) {
        progressBar.style.width = percent + '%';
        progressBar.textContent = Math.round(percent) + '%';
    }
}

/**
 * Resolve the localized operation label ("Translating" / "Refining (2/2)" / etc.)
 * for the current phase of the workflow.
 *
 * @param {Object} stats - Server stats. May contain auto_prep_active,
 *   enable_refinement, current_phase, refine_only.
 * @returns {string} Label to show in the progress section title.
 */
export function resolveOperationLabel(stats) {
    if (!stats) return t('translation:translating');

    // Auto glossary/style run before the first chunk exists: the bar has
    // nothing to advance yet, so name the work instead of claiming to
    // translate a document that has not been chunked.
    if (stats.auto_prep_active) return t('translation:preparing_auto');

    if (stats.enable_refinement) {
        return stats.current_phase === 2
            ? t('translation:refining_step', { step: 2, total: 2, defaultValue: 'Refining (2/2)' })
            : t('translation:translating_step', { step: 1, total: 2, defaultValue: 'Translating (1/2)' });
    }

    if (stats.term_fix) {
        return t('translation:fixing_terms');
    }

    if (stats.refine_only) {
        return t('translation:refining');
    }

    return t('translation:translating');
}

/**
 * Update the operation label inside the current-file title without rebuilding
 * the surrounding DOM (icon, filename, languages). The element is created by
 * `updateTranslationTitle` in the file controllers and tagged with id
 * `progressOperationLabel`.
 *
 * @param {Object} stats - Server stats
 */
function updateOperationLabel(stats) {
    const labelEl = DomHelpers.getElement('progressOperationLabel');
    if (!labelEl) return;

    const text = resolveOperationLabel(stats);
    // While auto prep runs, the bar has no chunk to advance: the spinner is the
    // only moving thing telling the user the job is alive rather than hung.
    const wantSpinner = !!(stats && stats.auto_prep_active);
    const hasSpinner = !!labelEl.querySelector('.inline-spinner');

    // Auto prep re-emits every couple of seconds to keep the panel alive.
    // Rebuilding identical content on each of those ticks would restart the
    // CSS animation, so the spinner would visibly stutter. Only touch the DOM
    // when something actually changed.
    if (labelEl.dataset.labelText === text && hasSpinner === wantSpinner) return;

    labelEl.textContent = text;
    labelEl.dataset.labelText = text;

    if (wantSpinner) {
        const spinner = document.createElement('span');
        spinner.className = 'inline-spinner';
        spinner.setAttribute('aria-hidden', 'true');
        labelEl.appendChild(spinner);
    }
}

// The Fallbacks card has a single active state — "critical" (red). Any
// fallback at all is treated as critical because placeholder loss already
// degrades the output; there is no intermediate "warning" tier any more.
//   false → nothing to surface (card neutral, panel hidden)
//   true  → red card + red panel with the full recommendation block
function isAlertActive(stats, fallbacks) {
    if (fallbacks > 0) return true;
    // Backend's one-shot quality flag and our client-side threshold check
    // are kept as additional ways to flip the card on even when fallbacks
    // are still 0 but placeholder errors are accumulating fast.
    if (stats && stats.quality_warning_fired) return true;
    return exceedsQualityThreshold(stats);
}

// Mirror of TranslationMetrics._QUALITY_* thresholds, kept here so the UI can
// detect the failure state from the cumulative stats payload even when no
// single XHTML file ever reached the per-file 5-chunk minimum (e.g. an EPUB
// split into many short chapters).
const QUALITY_MIN_PROCESSED = 5;
const QUALITY_RETRY_RATE_THRESHOLD = 0.30;
const QUALITY_FALLBACK_RATE_THRESHOLD = 0.10;
const QUALITY_AVG_ERRORS_THRESHOLD = 1.0;

function exceedsQualityThreshold(stats) {
    if (!stats) return false;
    const processed = stats.processed_chunks || 0;
    if (processed < QUALITY_MIN_PROCESSED) return false;
    const fallbackCount = (stats.token_alignment_used || 0) + (stats.fallback_used || 0);
    const notFirstTry = (stats.successful_after_retry || 0) + fallbackCount;
    const retryRate = notFirstTry / processed;
    const fallbackRate = fallbackCount / processed;
    const avgErrors = (stats.placeholder_errors || 0) / processed;
    return (
        retryRate > QUALITY_RETRY_RATE_THRESHOLD
        || fallbackRate > QUALITY_FALLBACK_RATE_THRESHOLD
        || avgErrors > QUALITY_AVG_ERRORS_THRESHOLD
    );
}

/**
 * Compute the rate metrics the critical intro line surfaces. We re-derive them
 * client-side from the cumulative stats payload so the numbers stay in sync
 * across multi-file EPUB runs (the per-file backend warning text would drift).
 *
 * The `run_*` counters are preferred over their accumulated twins whenever the
 * backend sends them. The accumulated ones are deliberately rehydrated across
 * resume passes (issue #180: the Fallbacks card must not reset to zero on
 * resume), so both the denominator and the numerators would carry work from
 * earlier passes — that is what produced "Across 34 chunks: 100% needed
 * retries" for a 12-chunk book. Percentages must describe one run.
 *
 * Precedence is by presence, not truthiness — the same discipline as
 * `_unfinished()` in src/api/completion_status.py: a `run_*` key that is
 * present and 0 is trusted (that run really did zero retries); only an absent
 * one falls back to the accumulated field (legacy payloads, formats that never
 * emit it).
 */
export function deriveRateContext(stats) {
    const s = stats || {};
    const present = (key) => statPresent(s, key);
    const pick = (runKey, accKey) => (present(runKey) ? (s[runKey] || 0) : (s[accKey] || 0));

    const processed = present('run_processed_chunks')
        ? (s.run_processed_chunks || 0)
        : (s.processed_chunks || s.completed_chunks || 0);
    if (processed <= 0) {
        return { retryPct: 0, fallbackPct: 0, avgErrors: '0.0', processed: 0 };
    }
    const fallbackCount = pick('run_token_alignment_used', 'token_alignment_used')
        + pick('run_fallback_used', 'fallback_used');
    const notFirstTry = pick('run_successful_after_retry', 'successful_after_retry')
        + fallbackCount;
    const placeholderErrors = pick('run_placeholder_errors', 'placeholder_errors');
    return {
        retryPct: Math.round((notFirstTry / processed) * 100),
        fallbackPct: Math.round((fallbackCount / processed) * 100),
        avgErrors: (placeholderErrors / processed).toFixed(1),
        processed,
    };
}

/**
 * Build a list item that interpolates a "{{link}}" placeholder in the given
 * template with a clickable link that jumps to the matching setting.
 */
function buildTipWithLink(template, linkText, action) {
    const li = document.createElement('li');
    const idx = template.indexOf('{{link}}');
    if (idx === -1) {
        // Defensive: if the i18n string lost its placeholder, fall back to
        // appending the link at the end so the user still has the affordance.
        li.appendChild(document.createTextNode(template + ' '));
    } else {
        li.appendChild(document.createTextNode(template.slice(0, idx)));
    }
    const link = document.createElement('a');
    link.href = '#';
    link.className = 'recommendation-link';
    link.textContent = linkText;
    link.addEventListener('click', (event) => {
        event.preventDefault();
        navigateToSetting(action);
    });
    li.appendChild(link);
    if (idx !== -1) {
        li.appendChild(document.createTextNode(template.slice(idx + '{{link}}'.length)));
    }
    return li;
}

/**
 * Build the recommendation content (intro line + tip list) into the given
 * container element. Shared between the live in-progress panel and the
 * post-translation completion card so the wording and ordering stay in sync.
 *
 * Caller owns the container styling (color palette, hidden state, etc.).
 *
 * @param {HTMLElement} container - Empty (or to-be-emptied) target element.
 * @param {Object} [context] - Numbers used by the intro line
 *   (retryPct, fallbackPct, avgErrors, processed).
 * @param {string} [introKey] - i18n key for the intro line. Defaults to the
 *   live-progress critical wording; the completion card overrides it with a
 *   past-tense variant.
 */
export function buildRecommendationContent(container, context, introKey) {
    if (!container) return;
    container.textContent = '';

    const ctx = context || { retryPct: 0, fallbackPct: 0, avgErrors: '0.0', processed: 0 };
    const intro = document.createElement('strong');
    intro.textContent = t(introKey || 'translation:fallback_panel_intro_critical', ctx);
    container.appendChild(intro);

    const list = document.createElement('ul');
    list.className = 'recommendation-list';

    const llmTip = document.createElement('li');
    llmTip.textContent = t('translation:fallback_panel_tip_llm');
    list.appendChild(llmTip);

    list.appendChild(buildTipWithLink(
        t('translation:fallback_panel_tip_plain_text_mode'),
        t('translation:fallback_panel_link_plain_text_mode'),
        'plainText',
    ));

    container.appendChild(list);
}

function renderRecommendationPanel(context) {
    buildRecommendationContent(
        DomHelpers.getElement('fallbackRecommendationPanel'),
        context,
    );
}

/**
 * Open/close the inline recommendation panel. When opening (or refreshing
 * while already open) we re-render with the latest context so the rate
 * numbers in the intro stay live.
 */
function toggleRecommendationPanel({ forceState, context } = {}) {
    const card = DomHelpers.getElement('fallbackStatCard');
    const panel = DomHelpers.getElement('fallbackRecommendationPanel');
    if (!card || !panel) return;
    const willOpen = forceState !== undefined
        ? forceState
        : panel.hasAttribute('hidden');
    if (willOpen) {
        renderRecommendationPanel(context);
        panel.removeAttribute('hidden');
        card.setAttribute('aria-expanded', 'true');
    } else {
        panel.setAttribute('hidden', '');
        card.setAttribute('aria-expanded', 'false');
    }
}

// Latest derived rate context (retry %, fallback %, avg errors), so the click
// handler can re-render the panel without re-receiving the stats payload.
let _lastSeverityContext = null;

let _fallbackCardClickBound = false;

function bindFallbackCardClick() {
    if (_fallbackCardClickBound) return;
    const card = DomHelpers.getElement('fallbackStatCard');
    if (!card) return;
    card.addEventListener('click', () => {
        // Only react when the card is in the alert state.
        if (!card.classList.contains('stat-card-critical')) return;
        toggleRecommendationPanel({ context: _lastSeverityContext });
    });
    _fallbackCardClickBound = true;
}

/**
 * Apply the alert highlight to the Fallbacks card and keep the inline panel
 * in sync. Only one active state exists ("critical", red palette):
 *  - inactive → card neutral, panel hidden
 *  - active   → red card, red panel with live retry/fallback/avg-error rates
 *
 * The panel stays open if the user already opened it; we just refresh its
 * content. When dropping back to inactive, any open panel is closed.
 */
function updateFallbackHighlight(count, stats) {
    const card = DomHelpers.getElement('fallbackStatCard');
    if (!card) return;

    const active = isAlertActive(stats, count);
    _lastSeverityContext = active ? deriveRateContext(stats || {}) : null;

    card.classList.toggle('stat-card-critical', active);
    card.title = t(active
        ? 'translation:stat_fallbacks_tooltip_critical'
        : 'translation:stat_fallbacks_tooltip');

    if (!active) {
        toggleRecommendationPanel({ forceState: false });
        return;
    }

    bindFallbackCardClick();
    const panel = DomHelpers.getElement('fallbackRecommendationPanel');
    if (panel && !panel.hasAttribute('hidden')) {
        renderRecommendationPanel(_lastSeverityContext);
    }
}

// Re-translate the Fallbacks card tooltip and the open recommendation panel
// when the UI locale changes. Both pieces are rendered imperatively (no
// data-i18n marker), so applyToDOM doesn't touch them — without this listener
// they would stay in the boot locale until the next stats update.
window.addEventListener('localeChanged', () => {
    const card = DomHelpers.getElement('fallbackStatCard');
    if (card) {
        const isCritical = card.classList.contains('stat-card-critical');
        card.title = t(isCritical
            ? 'translation:stat_fallbacks_tooltip_critical'
            : 'translation:stat_fallbacks_tooltip');
    }
    const panel = DomHelpers.getElement('fallbackRecommendationPanel');
    if (panel && !panel.hasAttribute('hidden') && _lastSeverityContext) {
        renderRecommendationPanel(_lastSeverityContext);
    }
});

/**
 * The chunk pair the live panel must display: the scope of the pass in flight.
 *
 * A repair pass (the "Fix these N chunks" button on a `partial` job) retries a
 * handful of chunks of an otherwise finished book. Reported against the book it
 * reads "12 TOTAL / 10 COMPLETED, 83%", creeps to 100% in tiny steps and hands
 * `updateEstimatedTimeRemaining` a denominator that has nothing to do with the
 * work being done. `run_is_repair` / `run_total_chunks` (src/core/epub/translator.py,
 * `_global_stats_payload`) describe that pass, so the panel reports it as the
 * little job it is: "1 TOTAL / 1 COMPLETED".
 *
 * Deliberately NOT applied to a plain interrupted resume (`run_is_repair` is
 * False there): someone resuming a book at 50% expects 50→100% of the book, not
 * 0→100% of the remainder.
 *
 * `total_chunks` / `completed_chunks` keep their book-level meaning everywhere
 * else - they feed the checkpoint and the resumable-job card's
 * "Progress: X/Y chunks (Z%)", which must never read "1/1 (100%)" for a
 * 12-chunk book.
 *
 * Precedence is by presence, not truthiness (see `statPresent`): a present
 * `run_processed_chunks` of 0 is trusted (the pass really has done nothing
 * yet), and an absent key - legacy payloads, and the TXT/SRT/DOCX path which
 * has no repair concept and emits neither flag - leaves the whole-book pair in
 * place, i.e. today's behaviour.
 *
 * @param {Object} stats - Job stats payload
 * @returns {{total: number, completed: number, isRepair: boolean}}
 */
function resolveDisplayChunks(stats) {
    const s = stats || {};
    const runTotal = statPresent(s, 'run_total_chunks') ? Number(s.run_total_chunks) : NaN;
    if (s.run_is_repair === true && Number.isFinite(runTotal) && runTotal > 0) {
        const done = statPresent(s, 'run_processed_chunks') ? (s.run_processed_chunks || 0) : 0;
        return { total: runTotal, completed: Math.min(done, runTotal), isRepair: true };
    }
    return {
        total: s.total_chunks || 0,
        completed: s.completed_chunks || 0,
        isRepair: false,
    };
}

/**
 * Update statistics display based on file type
 * All file types (txt, epub, srt) show stats uniformly
 * @param {Object} stats - Statistics object from server
 * @param {string} fileType - File type ('txt', 'epub', 'srt')
 */
function updateStatistics(stats, fileType) {
    if (!stats) return;

    DomHelpers.show('statsGrid');

    // Book-level by default, pass-level on a repair pass (see
    // resolveDisplayChunks). `failedChunks` stays book-level: it counts the
    // damage in the document, not the work in flight.
    const scope = resolveDisplayChunks(stats);
    DomHelpers.setText('totalChunks', scope.total || '0');
    DomHelpers.setText('completedChunks', scope.completed || '0');
    DomHelpers.setText('failedChunks', stats.failed_chunks || '0');

    if (fileType === 'srt') {
        // SRT does not use placeholders → Fallbacks stays at 0.
        DomHelpers.setText('fallbackChunks', '0');
        updateFallbackHighlight(0, stats);
    } else {
        // Live count of degraded chunks: Phase 2 (token alignment, an accumulated
        // tally — those chunks are translated and are never retried) plus the
        // chunks *currently* sitting in their source text after Phase 3.
        //
        // `untranslated_chunks` is a live projection of the per-chunk statuses,
        // so a retry that heals a chunk makes this card count down, while
        // `fallback_used` is an accumulated cross-pass tally that only ever
        // grows and would stay frozen at its historical value. Precedence is by
        // presence: present and 0 is trusted (a retry healed everything), and
        // only an absent key — legacy payloads, and the TXT/SRT/DOCX path which
        // has no chunk statuses — falls back to the historical counter.
        const untranslated = statPresent(stats, 'untranslated_chunks')
            ? (stats.untranslated_chunks || 0)
            : (stats.fallback_used || 0);
        const fallbacks = (stats.token_alignment_used || 0) + untranslated;
        DomHelpers.setText('fallbackChunks', String(fallbacks));
        updateFallbackHighlight(fallbacks, stats);
    }

    if (stats.elapsed_time !== undefined) {
        DomHelpers.setText('elapsedTime', formatElapsedTime(stats.elapsed_time));
        // Same pair as the two cards above, so the ETA describes the work in
        // flight instead of a book whose remaining 2 chunks would be paced by
        // the 10 it already had.
        updateEstimatedTimeRemaining(
            scope.completed,
            scope.total,
            stats.elapsed_time
        );
    }
}

export const ProgressManager = {
    /**
     * Update progress display
     * @param {number} percent - Progress percentage (0-100)
     */
    updateProgress(percent) {
        updateProgressBar(percent);
    },

    /**
     * Update statistics display
     * @param {string} fileType - File type ('txt', 'epub', 'srt')
     * @param {Object} stats - Statistics object from server
     */
    updateStats(fileType, stats) {
        updateStatistics(stats, fileType);
    },

    /**
     * Update progress and statistics together
     * @param {Object} data - Update data from server
     * @param {Object} data.stats - Statistics object
     * @param {string} fileType - File type ('txt', 'epub', 'srt')
     */
    update(data, fileType) {
        if (!data.stats) return;

        const stats = data.stats;
        const scope = resolveDisplayChunks(stats);
        const completed = scope.completed;
        const total = scope.total;

        // The bar value is server-authoritative: the backend emits a single
        // canonical `percent` (see src/core/progress) with the phase mapping
        // and monotonic floor already applied. Display it verbatim; the chunk
        // ratio is only a defensive fallback should `percent` ever be absent.
        //
        // The one exception is a repair pass, where the canonical `percent`
        // answers a different question (how much of the *book* is done, which
        // is what the checkpoint and the resumable-job card need) than the one
        // the panel is asking. It is overridden here rather than in
        // `snapshot_from_legacy_stats`, so the server-side contract - the
        // canonical percent, its monotonic floor in handlers.py, and the
        // DB-facing `progress_percentage` - keeps its book-level meaning. The
        // bar is only ever written from this one site per socket update, so the
        // two routes cannot fight each other.
        const globalPercent = scope.isRepair
            ? (total > 0 ? Math.min(100, (completed / total) * 100) : 0)
            : (typeof stats.percent === 'number'
                ? stats.percent
                : (total > 0 ? (completed / total) * 100 : 0));
        updateProgressBar(globalPercent);

        updateOperationLabel(stats);
        updateStatistics(stats, fileType);
    },

    /**
     * Reset progress display to initial state
     */
    reset() {
        updateProgressBar(0);
        DomHelpers.setText('totalChunks', '0');
        DomHelpers.setText('completedChunks', '0');
        DomHelpers.setText('failedChunks', '0');
        DomHelpers.setText('fallbackChunks', '0');
        _lastSeverityContext = null;
        updateFallbackHighlight(0);
        DomHelpers.setText('elapsedTime', '0s');
        DomHelpers.setText('estimatedTimeRemaining', '--');
        resetEtaTracking();
        DomHelpers.hide('statsGrid');
    },

    /**
     * Show progress section
     */
    show() {
        DomHelpers.show('progressSection');
    },

    /**
     * Hide progress section
     */
    hide() {
        DomHelpers.hide('progressSection');
    },

    /**
     * Set progress to complete (100%)
     */
    complete() {
        updateProgressBar(100);
    },

    /**
     * Get current progress percentage
     * @returns {number} Current progress (0-100)
     */
    getCurrentProgress() {
        const progressBar = DomHelpers.getElement('progressBar');

        if (!progressBar) return 0;

        const widthStyle = progressBar.style.width;
        const match = widthStyle.match(/(\d+(?:\.\d+)?)/);

        return match ? parseFloat(match[1]) : 0;
    }
};
