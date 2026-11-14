/**
 * Completion Card - Persistent per-file card rendered when a translation ends
 *
 * Builds the success / partial card (thumbnail, title, stat chips, warning
 * block, "retranslate these chunks" affordances, quick actions) and keeps every
 * visible card in sync with the UI locale and the active-translation state.
 *
 * Must not import translation-tracker.js: the tracker imports this module.
 */

import { StateManager } from '../core/state-manager.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { FileActions } from '../files/file-actions.js';
import { formatElapsedTime, deriveRateContext, buildRecommendationContent } from './progress-manager.js';
import { getFileIcon, createGenericEPUBIcon } from '../files/file-icons.js';
import { t } from '../i18n/i18n.js';
import {
    overridePanelHtml,
    openOverridePanel,
    toggleOverridePanel,
    readOverrideConfig,
    destroyOverridePickers,
} from './model-override-panel.js';

// Whether the locale / active-state listeners have been bound (lazily, on the
// first rendered card).
let completionLocaleListenerBound = false;

/**
 * Read a per-run counter from a finished job's stats, falling back to its
 * accumulated twin only when the `run_*` key is absent.
 *
 * Same presence-based precedence as `_unfinished()` in
 * src/api/completion_status.py and `deriveRateContext` in progress-manager.js:
 * a `run_*` key that is present and 0 is trusted (that pass really did zero),
 * and only a missing key falls back to the accumulated field (legacy payloads,
 * or a format that never learned to emit the twin).
 *
 * @param {Object} stats - Job stats payload
 * @param {string} runKey - Per-run key (e.g. 'run_placeholder_errors')
 * @param {string} accumulatedKey - Cross-pass twin (e.g. 'placeholder_errors')
 * @returns {number}
 */
function runCounter(stats, runKey, accumulatedKey) {
    const s = stats || {};
    return (s[runKey] !== undefined && s[runKey] !== null)
        ? (s[runKey] || 0)
        : (s[accumulatedKey] || 0);
}

/**
 * Number of chunks the delivered file still owes (issue #261, design decision
 * D9). Trust `unfinished_chunks` whenever the key is present — including when
 * it is 0 while `fallback_used` is not, which is exactly what a fully healed
 * retry pass looks like — and fall back to the historical `fallback_used`
 * counter only for legacy payloads that never emitted it.
 *
 * @param {Object} stats - Job stats payload
 * @returns {number}
 */
function unfinishedChunkCount(stats) {
    const s = stats || {};
    return typeof s.unfinished_chunks === 'number'
        ? s.unfinished_chunks
        : (s.fallback_used || 0);
}

/**
 * Number of chunks the delivered file carries with only approximate tag
 * positions — translated by the token-alignment repair (Phase 2), tags
 * reinserted proportionally.
 *
 * Read from the live map the EPUB pipeline emits (`degraded_files`, total
 * `degraded_chunks`), NOT from `token_alignment_used`: that counter is an
 * accumulated, cross-pass tally of Phase 2 *events*, so it stays positive after
 * those chunks have been retranslated successfully and would make a clean book
 * claim defects — the same class of bug as reading `fallback_used` for
 * "still untranslated".
 *
 * Same presence-based precedence as `unfinishedChunkCount` above and
 * `_unfinished()` in src/api/completion_status.py: a key that is present and 0
 * is trusted, and only an absent pair falls back to the accumulated counter
 * (legacy payloads, and formats that never emit the map).
 *
 * @param {Object} stats - Job stats payload
 * @returns {number}
 */
function degradedChunkCount(stats) {
    const s = stats || {};
    if (typeof s.degraded_chunks === 'number') return s.degraded_chunks;
    const files = s.degraded_files;
    if (files && typeof files === 'object' && !Array.isArray(files)) {
        return countMapIndices(files);
    }
    return s.token_alignment_used || 0;
}

/**
 * True when `map` is a plain, non-empty `{file_href: [chunk_index, ...]}`
 * object (the shape of `degraded_files` / `unfinished_files`).
 * @param {any} map
 * @returns {boolean}
 */
function isNonEmptyFileMap(map) {
    return !!map
        && typeof map === 'object'
        && !Array.isArray(map)
        && Object.keys(map).length > 0;
}

/**
 * Total number of chunk indices listed in a `{file_href: [chunk_index, ...]}`
 * map, ignoring entries that are not arrays.
 * @param {Object} map
 * @returns {number}
 */
function countMapIndices(map) {
    return Object.values(map).reduce(
        (sum, indices) => sum + (Array.isArray(indices) ? indices.length : 0), 0);
}

/**
 * Render a persistent success card for a completed file, with quick actions
 * to locate it on disk.
 * @param {Object} file - The file that just finished
 * @param {Object} resultData - Final payload from the server (output_filename, output_dir)
 */
export function renderCompletionCard(file, resultData) {
    const container = DomHelpers.getElement('completionCardsContainer');
    if (!container) return;

    const card = document.createElement('div');
    card.className = 'completion-card';
    populateCompletionCard(card, file, resultData);
    container.appendChild(card);
    ensureCompletionCardsLocaleListener();

    DomHelpers.hide('progressSection');
}

/**
 * Fill (or rebuild) an existing completion card with localized content.
 * Pulled out of `renderCompletionCard` so the same DOM tree can be
 * re-rendered on `localeChanged` without dropping the card from the page.
 *
 * Stashes the source payload on the element itself so the locale listener
 * can rebuild without coordinating extra storage.
 */
function populateCompletionCard(card, file, resultData) {
    card._tblPayload = { file, resultData };

    // The card is rebuilt in place (locale switch, active-state change), so
    // any override picker created by a previous render is about to lose its
    // DOM. Release it first, otherwise its SearchableSelect registrations
    // leak and a second picker gets registered for the same job.
    destroyOverridePickers(card);

    const outputFilename = resultData.output_filename || file.outputFilename || file.name;
    const safeFilename = DomHelpers.escapeHtml(outputFilename);
    const statsHtml = buildCompletionStatsHtml(file, resultData);
    const dismissLabel = t('translation:completion_card_dismiss');
    const isPartial = resultData.status === 'partial';
    const titleIcon = isPartial ? 'warning' : 'check_circle';
    const titleText = t(isPartial
        ? 'translation:translation_partial_card_title'
        : 'translation:translation_completed_card_title');

    card.innerHTML = '';

    const topRow = document.createElement('div');
    topRow.className = 'completion-card__top';
    topRow.appendChild(buildCompletionThumb(file));

    const main = document.createElement('div');
    main.className = 'completion-card__main';
    main.innerHTML = `
            <div class="completion-card__header">
                <h3 class="completion-card__title">
                    <span class="material-symbols-outlined">${titleIcon}</span>
                    <span>${titleText}${statsHtml}</span>
                </h3>
                <button type="button" class="completion-card__close" title="${dismissLabel}" aria-label="${dismissLabel}">
                    <span class="material-symbols-outlined">close</span>
                </button>
            </div>
            <div class="completion-card__filename" title="${safeFilename}">${safeFilename}</div>
        `;
    topRow.appendChild(main);
    card.appendChild(topRow);

    const warningBlock = buildCompletionWarningBlock(file, resultData);
    if (warningBlock) {
        card.appendChild(warningBlock);
    }

    const actionsGroup = FileActions.createActionGroup({
        actions: ['download', 'open', 'reveal', 'files-tab'],
        filename: outputFilename,
        variant: 'labeled'
    });
    actionsGroup.classList.add('completion-card__actions');
    card.appendChild(actionsGroup);

    card.querySelector('.completion-card__close').addEventListener('click', () => {
        destroyOverridePickers(card);
        card.remove();
    });
}

/**
 * Re-render every visible completion card whenever the user switches
 * locale, so the dynamically interpolated strings (title, stat badges,
 * warning block, action labels) stay in sync with the rest of the UI.
 * Bound once, lazily, the first time a card is rendered.
 */
function ensureCompletionCardsLocaleListener() {
    if (completionLocaleListenerBound) return;
    completionLocaleListenerBound = true;
    window.addEventListener('localeChanged', () => refreshCompletionCards());

    // The "fix these chunks" affordance mirrors the resumable-job card's
    // active-translation guard, and that guard is only correct at render
    // time. The resumable list solves it by reloading on this same state
    // key; do the equivalent here so a card rendered while the batch was
    // still running does not keep a permanently disabled button (or offer
    // a resume while another job is running).
    // `translation.hasActive` is re-set on every poll of the active-jobs
    // endpoint, so only a real transition triggers a rebuild — otherwise an
    // open panel would be torn down and re-created for nothing.
    StateManager.subscribe('translation.hasActive', (hasActive, wasActive) => {
        if (hasActive === wasActive) return;
        refreshCompletionCards();
    });
}

/**
 * Rebuild every visible completion card from its stashed payload.
 *
 * A rebuild replaces the model-override panel, so an open one is reopened
 * afterwards (with a fresh picker — the old one is destroyed by
 * `populateCompletionCard`, never left orphaned). The picker is only
 * created once the card is back in the document, which is why this lives
 * here rather than in the builder.
 */
function refreshCompletionCards() {
    const container = DomHelpers.getElement('completionCardsContainer');
    if (!container) return;
    container.querySelectorAll('.completion-card').forEach((card) => {
        if (!card._tblPayload) return;
        const previousPanel = card.querySelector('.completion-override');
        const wasOpen = !!previousPanel && previousPanel.style.display !== 'none';

        populateCompletionCard(card, card._tblPayload.file, card._tblPayload.resultData);

        if (!wasOpen) return;
        const panel = card.querySelector('.completion-override');
        if (!panel) return;
        panel.style.display = 'block';
        openOverridePanel(panel);
    });
}

/**
 * Build the thumbnail element for the completion card.
 * Uses the book cover for EPUBs (with SVG fallback), generic icon otherwise.
 * @param {Object} file - File object (fileType, thumbnail)
 * @returns {HTMLElement} Thumb wrapper element
 */
function buildCompletionThumb(file) {
    const wrap = document.createElement('div');
    wrap.className = 'completion-card__thumb';

    if (file.fileType === 'epub' && file.thumbnail) {
        const img = document.createElement('img');
        img.src = `/api/thumbnails/${encodeURIComponent(file.thumbnail)}`;
        img.alt = 'Cover';
        img.onerror = () => {
            wrap.innerHTML = createGenericEPUBIcon();
        };
        wrap.appendChild(img);
    } else {
        wrap.innerHTML = getFileIcon(file.fileType);
    }

    return wrap;
}

/**
 * Build the stats block HTML for the completion card.
 *
 * The chips describe the state of the file that was just delivered, not
 * the accumulated history of the job (issue #261). Concretely:
 *   - the approximate-tag chip counts the chunks that are STILL only
 *     approximately tagged (`degraded_files` / `degraded_chunks`), not the
 *     accumulated `token_alignment_used` tally of Phase 2 events: those
 *     chunks are never retried automatically (design decision D3), but the
 *     card now offers to retry them explicitly, and a book whose chunks
 *     were all repaired that way must stop reporting them.
 *   - Phase 3 fallbacks are no longer merged into that number. A chunk left
 *     in the source language is not "approximately tagged", and once it has
 *     been retried successfully it is neither — it gets its own chip driven
 *     by the current unfinished count (D9).
 *   - `placeholder_errors` is read per-run: retry-attempt noise from an
 *     earlier pass is not a property of the output.
 *
 * @param {Object} file - File object (for fileType)
 * @param {Object} resultData - Final payload (contains stats)
 * @returns {string} HTML for the stats block (empty string if no stats)
 */
function buildCompletionStatsHtml(file, resultData) {
    const stats = resultData.stats || {};

    const failed = stats.failed_chunks || 0;
    const elapsed = stats.elapsed_time;
    const isSrt = !!(file && file.fileType === 'srt');
    const fallbacks = isSrt ? 0 : degradedChunkCount(stats);
    const placeholderErrors = isSrt
        ? 0
        : runCounter(stats, 'run_placeholder_errors', 'placeholder_errors');

    // Content still in the source language in the delivered file. On the
    // TXT/SRT/DOCX path every unfinished unit is also counted in
    // `failed_chunks` (which has its own chip), so subtracting avoids
    // reporting the same chunks twice in the header.
    const untranslated = Math.max(0, unfinishedChunkCount(stats) - failed);

    // Plain Text Mode paragraph alignment (issue #253). Zero on every other
    // path, so no gating by file type is needed: a segment the model
    // returned with the wrong paragraph count was re-translated until the
    // count matched, and whatever is left over stayed misaligned.
    const paragraphMismatches = stats.paragraph_count_mismatches || 0;
    const paragraphRepairFailed = stats.paragraph_repair_failed || 0;
    const paragraphRealigned = Math.max(0, paragraphMismatches - paragraphRepairFailed);

    const cost = stats.openrouter_cost || 0;
    const promptTokens = stats.openrouter_prompt_tokens || 0;
    const completionTokens = stats.openrouter_completion_tokens || 0;
    const totalTokens = promptTokens + completionTokens;

    const items = [];

    if (typeof elapsed === 'number' && elapsed > 0) {
        items.push(formatElapsedTime(elapsed));
    }

    if (failed > 0) {
        items.push(`<span class="completion-card__stat--error">${t('translation:completion_failed_chunks', { count: failed })}</span>`);
    }

    // Missing content, not a formatting blemish: same severity styling as
    // the failed-chunks chip rather than the `--warn` formatting ones.
    if (untranslated > 0) {
        items.push(`<span class="completion-card__stat--error">${t('translation:completion_warning_untranslated', { count: untranslated })}</span>`);
    }

    if (fallbacks > 0) {
        items.push(`<span class="completion-card__stat--warn">${t('translation:completion_fallback_chunks', { count: fallbacks })}</span>`);
    }

    if (placeholderErrors > 0) {
        items.push(`<span class="completion-card__stat--warn">${t('translation:completion_placeholder_errors', { count: placeholderErrors })}</span>`);
    }

    if (paragraphRealigned > 0) {
        items.push(t('translation:completion_paragraph_realigned', { count: paragraphRealigned }));
    }

    if (paragraphRepairFailed > 0) {
        items.push(`<span class="completion-card__stat--warn">${t('translation:completion_paragraph_repair_failed', { count: paragraphRepairFailed })}</span>`);
    }

    if (cost > 0 || totalTokens > 0) {
        items.push(t('translation:completion_cost_tokens', {
            cost: `${stats.openrouter_cost_estimated ? '~' : ''}$${Number(cost).toFixed(4)}`,
            tokens: Number(totalTokens).toLocaleString(),
        }));
    }

    if (items.length === 0) return '';

    return `<span class="completion-card__stats"> - ${items.join(' · ')}</span>`;
}

/**
 * Build the warning block surfaced beneath the title when something is
 * still worth saying about the delivered file: work the job still owes,
 * chunks that failed, chunks whose tags are only approximate, or
 * placeholder trouble in this run. Mirrors the live recommendation panel
 * from progress-manager so the post-translation advice stays in sync with
 * what was shown during the run.
 *
 * Issue #261: the block claims only what is currently true of the output.
 * A healed retry pass must not be told to "use a more capable LLM" for
 * chunks it just fixed, so the rate-based advice is gated on this pass's
 * own counters — the same `run_*` values `deriveRateContext` divides by.
 *
 * Sections, in DOM order: heading, breakdown line, unfinished-files line,
 * unfinished-chunks fix affordance, degraded-chunks fix affordance, then the
 * recommendations (which host the token-alignment note when present).
 *
 * @param {Object} file - File object (used to gate by file type)
 * @param {Object} resultData - Final payload (contains stats)
 * @returns {HTMLElement|null} Warning block element, or null when there is
 *   nothing worth surfacing.
 */
function buildCompletionWarningBlock(file, resultData) {
    const stats = resultData.stats || {};
    if (file && file.fileType === 'srt') {
        return buildSrtCompletionWarningBlock(stats);
    }

    const signals = deriveWarningSignals(stats);
    if (!signals.hasCurrentIssue && !signals.showRecommendations) {
        return null;
    }

    const block = document.createElement('div');
    block.className = 'completion-card__warning';

    block.appendChild(buildWarningHeading(warningHeadingKey(signals)));

    const breakdown = buildWarningBreakdown(signals);
    if (breakdown) {
        block.appendChild(breakdown);
    }

    if (signals.hasUnfinishedFilesMap) {
        block.appendChild(buildUnfinishedFilesLine(signals.unfinishedFiles));
    }

    const translationId = resultData.translation_id || (file && file.translationId) || '';
    appendUnfinishedFixAffordance(block, resultData, translationId, signals);
    appendDegradedFixAffordance(block, resultData, translationId, signals);

    appendRecommendations(
        block,
        stats,
        signals.showRecommendations,
        buildTokenAlignmentNote(file, signals.tokenAlignment),
    );

    return block;
}

/**
 * Derive every counter and gate the (non-SRT) warning block needs from the
 * final stats payload.
 *
 * @param {Object} stats - Final stats payload
 * @returns {Object} Signals consumed by the warning block section builders
 */
function deriveWarningSignals(stats) {
    // What is still true of the delivered file. The approximate-tag count
    // comes from the live degraded map, never from the accumulated
    // `token_alignment_used` tally (see `degradedChunkCount`): a book whose
    // approximately-tagged chunks were all retranslated must stop claiming
    // them. `placeholder_errors` becomes per-run — an earlier pass's retry
    // noise says nothing about the output.
    const placeholderErrors = runCounter(stats, 'run_placeholder_errors', 'placeholder_errors');
    const failed = stats.failed_chunks || 0;
    const tokenAlignment = degradedChunkCount(stats);
    // The map itself: {file_href: [chunk_index, ...]}. Non-empty is the one
    // gate for the explicit retry affordance below — it is exactly what
    // handlers.py reads to decide to KEEP the checkpoint that retry needs,
    // so the button can never be offered without something behind it.
    const hasDegradedMap = isNonEmptyFileMap(stats.degraded_files);

    // What THIS pass struggled with. Only these may gate the rate-based
    // recommendation, whose percentages `deriveRateContext` already derives
    // from the same `run_*` values: advising "use a more capable LLM" after
    // a flawless retry pass — on chunks that pass just healed — is exactly
    // the defect being fixed here.
    const runFallbacks = runCounter(stats, 'run_token_alignment_used', 'token_alignment_used')
        + runCounter(stats, 'run_fallback_used', 'fallback_used');

    // Issue #261: `stats.unfinished_chunks` / `stats.unfinished_files` are
    // the authoritative "work still pending" signals. They cover both
    // outright failures and Phase 3 fallbacks, and — unlike the counters
    // above — they go back to 0 once those chunks have been retried
    // successfully. `fallback_used` is a historical tally of what happened
    // during the run: reading it here would claim a fully healed job still
    // holds source-language content. Same precedence as `_unfinished()` in
    // src/api/completion_status.py (D9): trust `unfinished_chunks` whenever
    // the key is present — including when it is 0 — and fall back to
    // `fallback_used` only when it is absent (legacy payloads).
    const unfinishedFiles = stats.unfinished_files;
    const hasUnfinishedFilesMap = isNonEmptyFileMap(unfinishedFiles);
    const unfinishedNow = unfinishedChunkCount(stats);

    // The part of the remaining work to attribute to "kept in source
    // language". On the TXT/SRT/DOCX path every unfinished unit is also
    // counted in `failed_chunks`, so subtracting avoids reporting the same
    // chunks twice in the breakdown below.
    const untranslated = Math.max(0, unfinishedNow - failed);

    // The rate-based advice describes the pass that just ran, so it is
    // gated on the per-run signals only.
    const showRecommendations = runFallbacks > 0 || placeholderErrors > 0;
    // The block itself renders when there is something currently true to
    // say: work still owed, chunks that failed, approximate tags still in
    // the book, or placeholder trouble in this run. `showRecommendations` is
    // ORed in so the advice can never be computed without a home.
    const hasCurrentIssue = unfinishedNow > 0 || hasUnfinishedFilesMap
        || failed > 0 || tokenAlignment > 0 || placeholderErrors > 0;

    return {
        placeholderErrors,
        failed,
        tokenAlignment,
        hasDegradedMap,
        unfinishedFiles,
        hasUnfinishedFilesMap,
        unfinishedNow,
        untranslated,
        showRecommendations,
        hasCurrentIssue,
    };
}

/**
 * Pick the warning heading's i18n key.
 *
 * When chunks are still in the source language or outright failed, the
 * optimistic "translations are correct" heading is misleading — surface
 * the missing-content message instead. A run whose fallbacks were all
 * retried successfully has no remaining work, so it keeps the
 * optimistic heading (issue #261).
 *
 * @param {Object} signals - Output of `deriveWarningSignals`
 * @returns {string} i18n key
 */
function warningHeadingKey(signals) {
    const hasUntranslatedContent = signals.unfinishedNow > 0 || signals.failed > 0
        || signals.hasUnfinishedFilesMap;
    return hasUntranslatedContent
        ? 'translation:completion_warning_heading_untranslated'
        : 'translation:completion_warning_heading';
}

/**
 * Build the warning block heading (warning icon + localized text). Shared by
 * the regular and the SRT warning blocks.
 *
 * @param {string} headingKey - i18n key for the heading text
 * @returns {HTMLElement}
 */
function buildWarningHeading(headingKey) {
    const heading = document.createElement('div');
    heading.className = 'completion-card__warning-heading';
    const icon = document.createElement('span');
    icon.className = 'material-symbols-outlined';
    icon.textContent = 'warning';
    heading.appendChild(icon);
    const headingText = document.createElement('span');
    headingText.textContent = t(headingKey);
    heading.appendChild(headingText);
    return heading;
}

/**
 * Build the one-line breakdown (approximate tags · untranslated ·
 * placeholder errors · failed), or null when every count is zero.
 *
 * @param {Object} signals - Output of `deriveWarningSignals`
 * @returns {HTMLElement|null}
 */
function buildWarningBreakdown(signals) {
    const { tokenAlignment, untranslated, placeholderErrors, failed } = signals;
    const breakdownItems = [];
    if (tokenAlignment > 0) {
        breakdownItems.push(t('translation:completion_warning_token_alignment', { count: tokenAlignment }));
    }
    if (untranslated > 0) {
        breakdownItems.push(t('translation:completion_warning_untranslated', { count: untranslated }));
    }
    if (placeholderErrors > 0) {
        breakdownItems.push(t('translation:completion_warning_placeholder_errors', { count: placeholderErrors }));
    }
    if (failed > 0) {
        breakdownItems.push(t('translation:completion_warning_failed', { count: failed }));
    }
    if (breakdownItems.length === 0) {
        return null;
    }
    const breakdown = document.createElement('div');
    breakdown.className = 'completion-card__warning-breakdown';
    breakdown.textContent = breakdownItems.join(' · ');
    return breakdown;
}

/**
 * Name the files still holding untranslated content (issue #261), so
 * the user knows where to look instead of just a count. Reuses the
 * breakdown line's style rather than introducing a new class.
 *
 * @param {Object} unfinishedFiles - Non-empty `{file_href: [chunk_index, ...]}` map
 * @returns {HTMLElement}
 */
function buildUnfinishedFilesLine(unfinishedFiles) {
    const fileHrefs = Object.keys(unfinishedFiles);
    const shown = fileHrefs.slice(0, 5);
    const extra = fileHrefs.length - shown.length;
    const filesLine = document.createElement('div');
    filesLine.className = 'completion-card__warning-breakdown';
    filesLine.textContent = t('translation:completion_warning_untranslated_files', {
        files: shown.join(', '),
        // A neutral numeric suffix ("(+N)") rather than a worded
        // "N more" — it needs no translation of its own and is simply
        // '' when nothing was truncated (5 or fewer files).
        more: extra > 0 ? ` (+${extra})` : '',
    });
    return filesLine;
}

/**
 * Offer to retranslate the chunks a `partial` job left unfinished.
 *
 * Issue #261: those chunks are kept and retryable, so the card can offer
 * the fix instead of only advising "use a more capable LLM".
 *
 * CRITICAL: gated on `partial`. A `completed` job whose chunks are all
 * finished has had its chunks and upload directory pruned by
 * checkpoint_manager.prune_job_data (see src/api/handlers.py),
 * so there is nothing left to resume and the button would only produce
 * an error. Without a translation id there is nothing to act on either
 * — in both cases the block stays exactly as it was before this feature
 * (breakdown, file list, recommendations). The degraded affordance
 * (`appendDegradedFixAffordance`) has its own, narrower gate for the
 * `completed` case.
 *
 * @param {HTMLElement} block - Warning block to append to
 * @param {Object} resultData - Final payload
 * @param {string} translationId - Job id ('' when unknown)
 * @param {Object} signals - Output of `deriveWarningSignals`
 */
function appendUnfinishedFixAffordance(block, resultData, translationId, signals) {
    const unfinishedFromMap = signals.hasUnfinishedFilesMap
        ? countMapIndices(signals.unfinishedFiles)
        : 0;
    const retryableChunks = signals.unfinishedNow > 0 ? signals.unfinishedNow : unfinishedFromMap;
    if (resultData.status === 'partial' && translationId && retryableChunks > 0) {
        appendCompletionFixAffordance(block, {
            translationId,
            count: retryableChunks,
            resultData,
        });
    }
}

/**
 * Second, separate affordance: retranslate the chunks that ARE
 * translated but whose inline tags were only approximately
 * repositioned. Opt-in and reachable ONLY from this card while it is on
 * screen — the checkpoint behind it is kept for a `completed` job
 * precisely because these chunks exist (src/api/handlers.py), and the
 * job deliberately never appears in the resumable-jobs list: a finished
 * book is not unfinished work.
 *
 * Gated on the degraded MAP, not on `token_alignment_used`: that
 * counter never goes back down, so it would keep offering a repair for
 * chunks that have already been repaired. Unlike the unfinished-chunks
 * affordance this one is not restricted to `partial` — its whole point is
 * the `completed` job whose only imperfection is tag placement.
 *
 * @param {HTMLElement} block - Warning block to append to
 * @param {Object} resultData - Final payload
 * @param {string} translationId - Job id ('' when unknown)
 * @param {Object} signals - Output of `deriveWarningSignals`
 */
function appendDegradedFixAffordance(block, resultData, translationId, signals) {
    if (translationId && signals.hasDegradedMap && signals.tokenAlignment > 0) {
        appendCompletionFixAffordance(block, {
            translationId,
            count: signals.tokenAlignment,
            resultData,
            adviceKey: 'translation:completion_degraded_advice',
            toggleKey: 'translation:completion_degraded_toggle',
            toggleTitleKey: 'translation:completion_degraded_toggle_title',
            // The apply button says "Retranslate these chunks" in both
            // affordances, so its key is shared rather than duplicated.
            extraOverrides: { retry_token_aligned: true },
        });
    }
}

/**
 * Expert-level note about the Phase 2 trade-off
 * (EPUB_TOKEN_ALIGNMENT_ENABLED, src/config.py): these chunks were
 * salvaged with approximate tag placement. Turning the setting off
 * would leave them untranslated instead — a defensible choice, since
 * untranslated chunks are retryable on any resume (issue #261).
 * Only shown when it is actionable: EPUB, and at least one chunk that
 * is STILL approximately tagged (the degraded map, not the accumulated
 * Phase 2 counter — a repaired book has nothing left to explain).
 * Built separately from the recommendations sub-block so it survives a
 * pass that earns no advice: it describes chunks still in the book, not
 * this run.
 *
 * @param {Object} file - File object (used to gate by file type)
 * @param {number} tokenAlignment - Chunks still approximately tagged
 * @returns {HTMLElement|null}
 */
function buildTokenAlignmentNote(file, tokenAlignment) {
    if (!(file && file.fileType === 'epub' && tokenAlignment > 0)) {
        return null;
    }
    const note = document.createElement('p');
    note.className = 'completion-card__warning-note';
    note.textContent = t('translation:completion_token_alignment_note', { count: tokenAlignment });
    return note;
}

/**
 * Append the rate-based recommendations (hosting the token-alignment note),
 * or the note alone when this pass earned no advice.
 *
 * Only renew the rate-based recommendations when this pass actually
 * produced fallbacks or placeholder issues — a run with only
 * `failed_chunks` (e.g. provider errors) is not really a "tune the LLM"
 * situation, and neither is a retry pass that healed everything.
 *
 * @param {HTMLElement} block - Warning block to append to
 * @param {Object} stats - Final stats payload
 * @param {boolean} showRecommendations - Per-run gate from `deriveWarningSignals`
 * @param {HTMLElement|null} tokenAlignmentNote - From `buildTokenAlignmentNote`
 */
function appendRecommendations(block, stats, showRecommendations, tokenAlignmentNote) {
    if (showRecommendations) {
        const recommendations = document.createElement('div');
        recommendations.className = 'completion-card__warning-recommendations';
        buildRecommendationContent(
            recommendations,
            deriveRateContext(stats),
            'translation:completion_warning_intro',
        );
        if (tokenAlignmentNote) {
            recommendations.appendChild(tokenAlignmentNote);
        }
        block.appendChild(recommendations);
    } else if (tokenAlignmentNote) {
        block.appendChild(tokenAlignmentNote);
    }
}

/**
 * Append an inline "retranslate these chunks with a better model"
 * affordance to a completion warning block: one advice sentence, a
 * disclosure button, and the shared model-override panel.
 *
 * Two callers, one wiring: the `partial` job's unfinished chunks (default
 * keys) and the `completed` job's approximately-tagged chunks, which add
 * `retry_token_aligned: true` to the overrides via `extraOverrides`. Only
 * the copy and that extra field differ, so the panel, the active-job guard
 * and the card teardown are shared rather than duplicated.
 *
 * Applying calls `window.resumeJob` (bound in index.js) rather than
 * importing ResumeManager: resume-manager.js already imports the tracker,
 * which imports this module, so an import here would close the cycle.
 *
 * @param {HTMLElement} block - Warning block to append to
 * @param {Object} opts
 * @param {string} opts.translationId - Job to resume
 * @param {number} opts.count - Number of retryable chunks
 * @param {Object} opts.resultData - Final payload (model/provider seed)
 * @param {string} [opts.adviceKey] - i18n key for the advice sentence
 * @param {string} [opts.toggleKey] - i18n key for the disclosure label
 * @param {string} [opts.toggleTitleKey] - i18n key for its tooltip
 * @param {string} [opts.applyLabelKey] - i18n key for the apply button
 * @param {Object} [opts.extraOverrides] - Extra fields merged into the
 *   resume body (the model overrides always win nothing here: these are
 *   pass-scoped flags the endpoint validates on its own).
 */
function appendCompletionFixAffordance(block, {
    translationId,
    count,
    resultData,
    adviceKey = 'translation:completion_fix_advice',
    toggleKey = 'translation:completion_fix_toggle',
    toggleTitleKey = 'translation:completion_fix_toggle_title',
    applyLabelKey = 'translation:completion_fix_apply_btn',
    extraOverrides = null,
}) {
    // Same source of truth as the resumable-job card: a resume is refused
    // server-side while another job runs, so the affordance is disabled
    // rather than failing on click. The card is re-rendered when this state
    // flips (see ensureCompletionCardsLocaleListener).
    const hasActiveTranslation = StateManager.getState('translation.hasActive') || false;

    const advice = document.createElement('div');
    advice.className = 'completion-card__warning-breakdown';
    advice.textContent = t(adviceKey, { count });
    block.appendChild(advice);

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'completion-card__fix-toggle';
    toggle.title = hasActiveTranslation
        ? t('translation:cannot_resume_in_progress_title')
        : t(toggleTitleKey);
    const toggleIcon = document.createElement('span');
    toggleIcon.className = 'material-symbols-outlined';
    toggleIcon.textContent = 'tune';
    toggle.appendChild(toggleIcon);
    const toggleLabel = document.createElement('span');
    toggleLabel.textContent = t(toggleKey, { count });
    toggle.appendChild(toggleLabel);
    toggle.disabled = hasActiveTranslation;
    block.appendChild(toggle);

    const holder = document.createElement('div');
    holder.innerHTML = overridePanelHtml({
        tid: translationId,
        provider: resultData.llm_provider,
        model: resultData.model,
        endpoint: resultData.llm_api_endpoint,
        applyLabelKey,
        panelClass: 'completion-override',
    });
    const panel = holder.firstElementChild;
    if (!panel) return;
    block.appendChild(panel);

    toggle.addEventListener('click', () => {
        if (toggle.disabled) return;
        toggleOverridePanel(panel);
    });

    const applyBtn = panel.querySelector('.resume-apply');
    if (!applyBtn) return;
    applyBtn.disabled = hasActiveTranslation;
    if (hasActiveTranslation) {
        applyBtn.title = t('translation:cannot_resume_in_progress_title');
    }
    applyBtn.addEventListener('click', () => {
        if (applyBtn.disabled) return;
        const picked = readOverrideConfig(panel);
        // `undefined` means the panel refused the input (no model picked)
        // and already told the user; `null` means "resume as configured".
        if (picked === undefined) return;
        // A pass-scoped flag (retry_token_aligned) travels in the same body
        // as the model overrides, so "resume as configured" still has to
        // send one. ApiClient.resumeJob omits the body only for an empty
        // object, which is exactly the no-flag no-override case.
        const overrides = extraOverrides
            ? Object.assign({}, picked || {}, extraOverrides)
            : picked;
        if (typeof window.resumeJob !== 'function') {
            console.error('window.resumeJob is not available; cannot retry unfinished chunks');
            return;
        }

        // This card is superseded the moment the retry starts: the resumed
        // run overwrites the very file its Download button points at, and a
        // fresh completion card is rendered when the retry finishes. Two
        // cards for one book — the stale one first — is exactly what we do
        // not want, so drop it once the resume is really under way.
        //
        // "Really" matters: ResumeManager.resumeJob returns undefined in
        // every case (it is async and reports nothing), and it can bail out
        // before doing anything — another job is active, or the user
        // dismisses its confirm() dialog — in which case this card must
        // survive. Its only success signal is the `translationResumed`
        // event, dispatched after the server accepted the resume, so that
        // is what we listen for.
        const card = block.closest('.completion-card');
        const onResumed = (event) => {
            if (!event.detail || event.detail.translationId !== translationId) return;
            window.removeEventListener('translationResumed', onResumed);
            if (!card) return;
            // Same discipline as populateCompletionCard and the dismiss
            // handler: release the override picker before its DOM goes
            // away, or its SearchableSelect registration leaks.
            destroyOverridePickers(card);
            card.remove();
        };
        window.addEventListener('translationResumed', onResumed);
        Promise.resolve(window.resumeJob(translationId, overrides))
            .catch(() => { /* resumeJob reports its own errors */ })
            .finally(() => {
                // Cancelled or rejected: the event never fired, so tear the
                // listener down instead of leaving it on the window. On the
                // success path onResumed already removed it.
                window.removeEventListener('translationResumed', onResumed);
            });
    });
}

/**
 * SRT variant of the completion warning block. Shown when subtitle
 * blocks still failed after the automatic marker-validation retries:
 * the affected cues kept the source-language text. Mirrors the EPUB
 * fallback panel structure (heading + breakdown + advice list) with
 * SRT-specific recommendations.
 *
 * @param {Object} stats - Final stats payload
 * @returns {HTMLElement|null} Warning block, or null when nothing failed
 */
function buildSrtCompletionWarningBlock(stats) {
    const failed = stats.failed_chunks || 0;
    if (failed === 0) {
        return null;
    }

    const block = document.createElement('div');
    block.className = 'completion-card__warning';

    block.appendChild(buildWarningHeading('translation:srt_completion_warning_heading'));

    const breakdown = document.createElement('div');
    breakdown.className = 'completion-card__warning-breakdown';
    breakdown.textContent = t('translation:srt_completion_warning_blocks', { count: failed });
    block.appendChild(breakdown);

    const recommendations = document.createElement('div');
    recommendations.className = 'completion-card__warning-recommendations';
    const intro = document.createElement('strong');
    intro.textContent = t('translation:srt_completion_warning_intro');
    recommendations.appendChild(intro);

    const list = document.createElement('ul');
    list.className = 'recommendation-list';
    const llmTip = document.createElement('li');
    llmTip.textContent = t('translation:fallback_panel_tip_llm');
    list.appendChild(llmTip);
    const blockSizeTip = document.createElement('li');
    blockSizeTip.textContent = t('translation:srt_completion_tip_block_size');
    list.appendChild(blockSizeTip);
    recommendations.appendChild(list);
    block.appendChild(recommendations);

    return block;
}
