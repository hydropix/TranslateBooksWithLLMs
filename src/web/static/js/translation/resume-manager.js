/**
 * Resume Manager - Resumable jobs management
 *
 * Handles loading, resuming, and deleting interrupted translation checkpoints.
 * Manages resumable jobs UI and state synchronization.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { ProgressManager } from './progress-manager.js';
import { TranslationTracker } from './translation-tracker.js';
import { t, getCurrentLocale, applyToDOM } from '../i18n/i18n.js';
import { createProviderModelPicker } from '../providers/provider-model-picker.js';

// Live picker instances, keyed by translation_id, so the override panel keeps
// its state while open and can be cleaned up on the next list render.
const overridePickers = new Map();

function destroyOverridePickers() {
    overridePickers.forEach((p) => p.destroy?.());
    overridePickers.clear();
}

/**
 * Format resumable job card HTML
 * @param {Object} job - Job data
 * @param {boolean} hasActiveTranslation - Whether there's an active translation
 * @param {string} activeNames - Names of active translations
 * @returns {string} HTML for job card
 */
function formatJobCard(job, hasActiveTranslation, activeNames) {
    const progress = job.progress || {};
    const completedChunks = progress.completed_chunks || 0;
    const totalChunks = progress.total_chunks || 0;
    const failedChunks = progress.failed_chunks || 0;
    const progressPercent = job.progress_percentage || 0;
    const fileType = (job.file_type || 'txt').toUpperCase();
    const isPartial = job.status === 'partial';

    const failedBadgeLabel = t('translation:job_card_failed_badge', { count: failedChunks });
    const statusBadge = isPartial
        ? `<span style="display: inline-block; margin-left: 8px; padding: 2px 8px; font-size: 11px; font-weight: 600; color: #92400e; background: #fef3c7; border: 1px solid #f59e0b; border-radius: 4px;" title="${t('translation:job_card_partial_title', { count: failedChunks })}">${failedBadgeLabel}</span>`
        : (failedChunks > 0
            ? `<span style="display: inline-block; margin-left: 8px; padding: 2px 8px; font-size: 11px; font-weight: 600; color: #92400e; background: #fef3c7; border: 1px solid #f59e0b; border-radius: 4px;" title="${t('translation:job_card_failed_title', { count: failedChunks })}">${failedBadgeLabel}</span>`
            : '');

    const naText = t('translation:job_card_na');
    const dateLocale = getCurrentLocale();
    const createdDate = job.created_at ? new Date(job.created_at).toLocaleString(dateLocale) : naText;
    const pausedDate = job.paused_at ? new Date(job.paused_at).toLocaleString(dateLocale) :
                       job.updated_at ? new Date(job.updated_at).toLocaleString(dateLocale) : naText;

    // Extract original filename (remove 16-char hash prefix + underscore)
    const unknownText = t('translation:job_card_unknown');
    const inputFilename = job.input_filename || unknownText;
    const outputFilename = job.output_filename || unknownText;

    // Extract hash and original name from input filename
    const inputMatch = inputFilename.match(/^([a-f0-9]{16})_(.+)$/);
    const inputHash = inputMatch ? inputMatch[1] : null;
    const inputOriginalName = inputMatch ? inputMatch[2] : inputFilename;

    // Format the display name (capitalize first letter, remove extension for display)
    const displayName = inputOriginalName.replace(/\.[^.]+$/, '');
    const displayNameFormatted = displayName.charAt(0).toUpperCase() + displayName.slice(1);

    const idValue = inputHash || job.translation_id.replace('trans_', '');
    const typeIdLine = t('translation:job_card_type_id', { type: fileType, id: idValue });
    const resumeTitle = hasActiveTranslation
        ? t('translation:cannot_resume_in_progress_title')
        : t('translation:resume_btn_title');

    // Original model/provider, used to seed the override picker and to show what
    // the resumed portion would switch away from. Keys were already stripped
    // server-side from job.config.
    const cfg = job.config || {};
    const origProvider = cfg.llm_provider || 'ollama';
    const origModel = cfg.model || '';
    const origEndpoint = cfg.llm_api_endpoint || '';

    return `
        <div class="resumable-job-card" style="border: 1px solid #e5e7eb; padding: 20px; margin-bottom: 15px; border-radius: 8px; background: #f9fafb;">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px; gap: 15px;">
                <div style="flex: 1; min-width: 0;">
                    <div style="font-size: 18px; font-weight: 600; color: #1f2937; margin-bottom: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${DomHelpers.escapeHtml(displayNameFormatted)}">
                        ${DomHelpers.escapeHtml(displayNameFormatted)}
                    </div>
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="→ ${DomHelpers.escapeHtml(outputFilename)}">
                        → ${DomHelpers.escapeHtml(outputFilename)}
                    </div>
                    <div style="font-size: 12px; color: #9ca3af; margin-top: 8px;">
                        ${typeIdLine}${statusBadge}
                    </div>
                </div>

                <div style="display: flex; gap: 10px; flex-shrink: 0;">
                    <button class="btn btn-primary" onclick="resumeJob('${job.translation_id}')"
                            title="${resumeTitle}"
                            ${hasActiveTranslation ? 'disabled style="opacity: 0.5; cursor: not-allowed;"' : ''}>
                        ${t('translation:job_card_resume_btn')}
                    </button>
                    <button class="btn btn-danger" onclick="deleteCheckpoint('${job.translation_id}')" title="${t('translation:job_card_delete_title')}">
                        ${t('translation:job_card_delete_btn')}
                    </button>
                </div>
            </div>

            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 13px; color: #6b7280; margin-bottom: 5px;">
                    <span>${t('translation:job_card_progress', { completed: completedChunks, total: totalChunks, percent: progressPercent })}</span>
                </div>
                <div style="width: 100%; background: #e5e7eb; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="width: ${progressPercent}%; background: #3b82f6; height: 100%; transition: width 0.3s;"></div>
                </div>
            </div>

            <div style="display: flex; align-items: center; justify-content: space-between; gap: 20px; font-size: 12px; color: #9ca3af;">
                <div style="display: flex; gap: 20px;">
                    <span>${t('translation:job_card_created', { date: createdDate })}</span>
                    <span>${t('translation:job_card_paused', { date: pausedDate })}</span>
                </div>
                <button class="resume-change-model" data-tid="${job.translation_id}"
                        title="${t('translation:resume_change_model_title')}"
                        ${hasActiveTranslation ? 'disabled' : ''}
                        style="display: inline-flex; align-items: center; gap: 4px; background: none; border: none; padding: 2px 4px; font-size: 12px; color: #3b82f6; cursor: pointer; white-space: nowrap;${hasActiveTranslation ? ' opacity: 0.5; cursor: not-allowed;' : ''}">
                    <span class="material-symbols-outlined" style="font-size: 0.95rem;">tune</span>
                    <span data-i18n="translation:resume_change_model">Change model</span>
                </button>
            </div>

            <div class="resume-override" data-tid="${job.translation_id}"
                 data-provider="${DomHelpers.escapeHtml(origProvider)}"
                 data-model="${DomHelpers.escapeHtml(origModel)}"
                 data-endpoint="${DomHelpers.escapeHtml(origEndpoint)}"
                 style="display: none; margin-top: 15px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 10px;">
                    <span data-i18n="translation:resume_original_model">Original model</span>:
                    <strong>${DomHelpers.escapeHtml(origProvider)} / ${DomHelpers.escapeHtml(origModel || '—')}</strong>
                </div>
                <div class="resume-picker-mount"></div>
                <div class="resume-style-warning" style="display: none; margin-top: 10px; font-size: 12px; color: #92400e; background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px; padding: 8px;">
                    ⚠️ <span data-i18n="translation:resume_style_warning">Switching model mid-book may cause a style break between the already-translated and remaining chunks.</span>
                </div>
                <div style="margin-top: 12px;">
                    <button class="btn btn-primary resume-apply" data-tid="${job.translation_id}" data-i18n="translation:resume_apply_btn">
                        Resume with this model
                    </button>
                </div>
            </div>
        </div>
    `;
}

/**
 * Create warning banner HTML if active translations exist
 * @param {Array} activeJobs - Active translation jobs
 * @returns {string} Warning banner HTML or empty string
 */
function createWarningBanner(activeJobs) {
    if (!activeJobs || activeJobs.length === 0) return '';

    const activeNames = activeJobs.map(job => job.output_filename || t('translation:job_card_unknown')).join(', ');

    return `
        <div class="active-translation-warning" style="background: #fef3c7; border: 1px solid #f59e0b; padding: 12px; margin-bottom: 15px; border-radius: 6px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 20px;">⚠️</span>
                <div style="flex: 1;">
                    <strong style="color: #92400e;">${t('translation:active_translation_warning_title')}</strong>
                    <p style="margin: 5px 0 0 0; font-size: 13px; color: #78350f;">
                        ${t('translation:active_translation_warning_desc', { names: DomHelpers.escapeHtml(activeNames) })}
                    </p>
                </div>
            </div>
        </div>
    `;
}

/**
 * Lazily build the provider+model picker the first time a job's override panel
 * is opened. Seeds it with the job's original config and toggles the style-break
 * warning whenever the chosen model/provider diverges from the original.
 */
function ensureOverridePicker(panel) {
    const tid = panel.dataset.tid;
    if (overridePickers.has(tid)) return;
    const mount = panel.querySelector('.resume-picker-mount');
    const warning = panel.querySelector('.resume-style-warning');
    if (!mount) return;

    const origProvider = panel.dataset.provider || 'ollama';
    const origModel = panel.dataset.model || '';
    const origEndpoint = panel.dataset.endpoint || '';

    const picker = createProviderModelPicker(mount, {
        config: { provider: origProvider, model: origModel, api_endpoint: origEndpoint },
        onChange: (cfg) => {
            const changed = (cfg.provider !== origProvider) || (cfg.model && cfg.model !== origModel);
            if (warning) warning.style.display = changed ? 'block' : 'none';
        },
    });
    overridePickers.set(tid, picker);
}

/**
 * Wire the "Change model" toggles and "Resume with this model" buttons inside a
 * freshly rendered resumable-jobs list. Pickers are created on first open only.
 */
function wireResumeOverrides(container) {
    container.querySelectorAll('.resume-change-model').forEach((btn) => {
        if (btn.disabled) return;
        btn.addEventListener('click', () => {
            const card = btn.closest('.resumable-job-card');
            const panel = card && card.querySelector('.resume-override');
            if (!panel) return;
            if (panel.style.display !== 'none') {
                panel.style.display = 'none';
                return;
            }
            panel.style.display = 'block';
            ensureOverridePicker(panel);
        });
    });

    container.querySelectorAll('.resume-apply').forEach((btn) => {
        btn.addEventListener('click', () => {
            const tid = btn.dataset.tid;
            const picker = overridePickers.get(tid);
            const cfg = picker ? picker.getConfig() : null;
            if (cfg && !cfg.model) {
                MessageLogger.showMessage(t('translation:resume_no_model_selected'), 'error');
                return;
            }
            // Map the picker's generic field names onto the backend override
            // schema (llm_provider / llm_api_endpoint).
            let overrides = null;
            if (cfg) {
                overrides = { model: cfg.model, llm_provider: cfg.provider };
                if (cfg.api_endpoint) overrides.llm_api_endpoint = cfg.api_endpoint;
                if (cfg.api_key) overrides.api_key = cfg.api_key;
            }
            ResumeManager.resumeJob(tid, overrides);
        });
    });
}

export const ResumeManager = {
    /**
     * Load and display resumable jobs
     */
    async loadResumableJobs() {
        const section = DomHelpers.getElement('resumableJobsSection');
        const loading = DomHelpers.getElement('resumableJobsLoading');
        const listContainer = DomHelpers.getElement('resumableJobsList');
        const emptyMessage = DomHelpers.getElement('resumableJobsEmpty');

        // Show loading, hide list and empty message (use inline style to override)
        if (loading) loading.style.display = 'block';
        if (listContainer) listContainer.style.display = 'none';
        if (emptyMessage) emptyMessage.style.display = 'none';

        try {
            const data = await ApiClient.getResumableJobs();
            const jobs = data.resumable_jobs || [];

            // Get active translation state
            const hasActiveTranslation = StateManager.getState('translation.hasActive') || false;
            const activeJobs = StateManager.getState('translation.activeJobs') || [];

            // Hide loading
            if (loading) loading.style.display = 'none';

            if (jobs.length === 0) {
                // Hide section if no jobs (use inline style to override)
                if (section) section.style.display = 'none';
                if (emptyMessage) emptyMessage.style.display = 'block';
                return;
            }

            // Show section and populate jobs (use inline style to override)
            if (section) section.style.display = 'block';
            if (listContainer) listContainer.style.display = 'block';

            // Build warning banner if active translation exists
            const warningBanner = createWarningBanner(hasActiveTranslation ? activeJobs : null);

            // Build jobs HTML
            const jobsHtml = jobs.map(job => formatJobCard(job, hasActiveTranslation, activeJobs)).join('');

            if (!listContainer) {
                console.error('Error: resumableJobsList element not found');
                return;
            }

            // Drop pickers from the previous render before wiping their DOM.
            destroyOverridePickers();
            listContainer.innerHTML = warningBanner + jobsHtml;
            // Translate the freshly injected data-i18n markup, then wire the
            // override toggles / apply buttons.
            applyToDOM(listContainer);
            wireResumeOverrides(listContainer);

            MessageLogger.addLog(t('translation:paused_count_log', { count: jobs.length }));

        } catch (error) {
            // Hide loading, show error message
            if (loading) loading.style.display = 'none';
            if (emptyMessage) {
                emptyMessage.style.display = 'block';
                emptyMessage.innerHTML = `<p style="color: #ef4444;">${t('translation:paused_load_error', { error: DomHelpers.escapeHtml(error.message) })}</p>`;
            }
            // Hide section on error
            if (section) section.style.display = 'none';
            console.error('Error loading resumable jobs:', error);
        }
    },

    /**
     * Resume a paused translation job
     * @param {string} translationId - Translation ID to resume
     * @param {Object} [overrides] - Optional model/provider overrides for the remaining chunks
     */
    async resumeJob(translationId, overrides = null) {
        // Check if there's an active translation
        const hasActive = StateManager.getState('translation.hasActive') || false;
        const activeJobs = StateManager.getState('translation.activeJobs') || [];

        if (hasActive) {
            const activeNames = activeJobs.map(job => job.output_filename || t('translation:job_card_unknown')).join(', ');
            MessageLogger.showMessage(
                t('translation:cannot_resume_active', { names: activeNames }),
                'error'
            );
            return;
        }

        if (!confirm(t('translation:confirm_resume'))) {
            return;
        }

        try {
            MessageLogger.addLog(t('translation:resuming_log', { id: translationId }));
            MessageLogger.showMessage(t('translation:resuming_msg'), 'info');

            const data = await ApiClient.resumeJob(translationId, overrides);

            MessageLogger.showMessage(
                t('translation:resume_success', { chunk: data.resume_from_chunk }),
                'success'
            );
            MessageLogger.addLog(t('translation:resume_success_log', { id: translationId, chunk: data.resume_from_chunk }));

            // Fetch job details to get filename and file type
            const jobData = await ApiClient.getTranslationStatus(translationId);

            // Set up current processing job in state
            StateManager.setState('translation.currentJob', {
                translationId: translationId,
                fileRef: {
                    name: jobData.config?.output_filename || t('translation:resumed_translation_default'),
                    fileType: jobData.config?.file_type || 'txt'
                }
            });
            TranslationTracker.registerOwnedJob(translationId);

            // Mark as batch active
            StateManager.setState('translation.isBatchActive', true);

            // Clear any stale progress/ETA state left over from a previous run
            // before showing the section, so the resumed job's ETA is not
            // computed from another job's chunk timings.
            ProgressManager.reset();

            // Show progress section
            ProgressManager.show();
            const progressSection = DomHelpers.getElement('progressSection');
            if (progressSection) {
                progressSection.scrollIntoView({ behavior: 'smooth' });
            }

            // Update title with actual filename
            const fileName = jobData.config?.output_filename || t('translation:resumed_translation_default');
            DomHelpers.setText('currentFileProgressTitle', t('translation:resuming_file', { name: fileName }));

            // Show stats grid
            DomHelpers.show('statsGrid');

            // Show interrupt button
            const interruptBtn = DomHelpers.getElement('interruptBtn');
            if (interruptBtn) {
                DomHelpers.show('interruptBtn');
                interruptBtn.disabled = false;
            }

            // Seed the bar with the canonical percent from the checkpoint when
            // available, so it doesn't flash 0% before the first live update.
            // (The legacy top-level `jobData.progress` was always 0.)
            const resumedPercent = (typeof jobData.stats?.percent === 'number')
                ? jobData.stats.percent
                : (jobData.progress || 0);
            ProgressManager.updateProgress(resumedPercent);

            // Emit event for translation started
            const event = new CustomEvent('translationResumed', { detail: { translationId, jobData } });
            window.dispatchEvent(event);

            // Refresh resumable jobs list after a delay
            setTimeout(() => {
                this.loadResumableJobs();
            }, 1000);

        } catch (error) {
            // Enhanced error message for active translation conflicts
            if (error.status === 409 && error.data?.active_translations) {
                const activeList = error.data.active_translations
                    .map(item => `• ${item.output_filename} (${item.status})`)
                    .join('\n');
                MessageLogger.showMessage(
                    t('translation:cannot_resume_with_list', { list: activeList }),
                    'error'
                );
                MessageLogger.addLog(`⚠️ ${error.data.message}`);
            } else {
                MessageLogger.showMessage(t('translation:resume_error', { error: error.message }), 'error');
                MessageLogger.addLog(t('translation:resume_network_error_log', { error: error.message }));
            }
            console.error('Error resuming job:', error);
        }
    },

    /**
     * Delete a checkpoint
     * @param {string} translationId - Translation ID to delete
     */
    async deleteCheckpoint(translationId) {
        if (!confirm(t('translation:confirm_delete_checkpoint'))) {
            return;
        }

        try {
            MessageLogger.addLog(t('translation:deleting_checkpoint_log', { id: translationId }));

            await ApiClient.deleteCheckpoint(translationId);

            MessageLogger.showMessage(t('translation:checkpoint_deleted'), 'success');
            MessageLogger.addLog(t('translation:checkpoint_deleted_log', { id: translationId }));

            // Refresh resumable jobs list
            this.loadResumableJobs();

        } catch (error) {
            MessageLogger.showMessage(t('translation:checkpoint_delete_error', { error: error.message }), 'error');
            MessageLogger.addLog(t('translation:resume_network_error_log', { error: error.message }));
            console.error('Error deleting checkpoint:', error);
        }
    },

    /**
     * Initialize resume manager
     */
    initialize() {
        // Load resumable jobs on initialization
        this.loadResumableJobs();

        // Listen for translation state changes
        StateManager.subscribe('translation.hasActive', (hasActive) => {
            // Refresh job list when active state changes
            this.loadResumableJobs();
        });
    }
};
