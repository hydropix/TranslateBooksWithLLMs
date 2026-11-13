/**
 * TTS Status
 *
 * Renders WebSocket `tts_update` events into the TTS progress section and
 * exposes the FFmpeg installer used by the "Install FFmpeg" button
 * (window.installFFmpeg, called from an inline onclick).
 */

import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { MessageLogger } from '../ui/message-logger.js';
import { t, i18nSpan } from '../i18n/i18n.js';

/** Link to the FFmpeg download page, shown when the automatic install is not an option. */
function ffmpegManualLink() {
    return `<a href="https://ffmpeg.org/download.html" target="_blank" rel="noopener" class="btn btn-secondary" style="text-decoration: none;">${i18nSpan('tts:ffmpeg_manual')}</a>`;
}

/**
 * Handle TTS update events from WebSocket
 * @param {Object} data - TTS update data
 */
export function handleTtsUpdate(data) {
    const { status, progress, message, audio_filename, error, current_chunk, total_chunks } = data;

    // Update TTS progress section
    const ttsProgressSection = DomHelpers.getElement('ttsProgressSection');
    const ttsProgressBar = DomHelpers.getElement('ttsProgressBar');
    const ttsStatusText = DomHelpers.getElement('ttsStatusText');

    switch (status) {
        case 'started':
            if (ttsProgressSection) {
                ttsProgressSection.style.display = 'block';
            }
            if (ttsProgressBar) {
                ttsProgressBar.style.width = '0%';
                ttsProgressBar.textContent = '0%';
            }
            if (ttsStatusText) {
                ttsStatusText.textContent = t('tts:starting_status');
            }
            MessageLogger.addLog(t('tts:starting_log'));
            break;

        case 'processing':
            if (ttsProgressBar) {
                ttsProgressBar.style.width = `${progress}%`;
                ttsProgressBar.textContent = `${progress}%`;
            }
            if (ttsStatusText) {
                const chunkInfo = current_chunk && total_chunks
                    ? ` (${current_chunk}/${total_chunks})`
                    : '';
                ttsStatusText.textContent = `🔊 ${message || t('tts:processing_default')}${chunkInfo}`;
            }
            break;

        case 'completed':
            if (ttsProgressBar) {
                ttsProgressBar.style.width = '100%';
                ttsProgressBar.textContent = '100%';
            }
            if (ttsStatusText) {
                ttsStatusText.textContent = t('tts:completed_status', { name: audio_filename || t('tts:audio_file_default') });
            }
            MessageLogger.addLog(t('tts:completed_log', { name: audio_filename || t('tts:audio_file_default') }));

            setTimeout(() => {
                if (ttsProgressSection) {
                    ttsProgressSection.style.display = 'none';
                }
            }, 5000);
            break;

        case 'failed':
            if (ttsProgressBar) {
                ttsProgressBar.style.width = '0%';
                ttsProgressBar.textContent = t('tts:failed_label');
                ttsProgressBar.style.background = '#ef4444';
            }

            const errorText = error || message || t('errors:unknown_error');
            const isFFmpegError = errorText.toLowerCase().includes('ffmpeg');

            if (ttsStatusText) {
                if (isFFmpegError) {
                    ttsStatusText.innerHTML = `
                        <span style="color: #ef4444;">❌ ${i18nSpan('tts:ffmpeg_required')}</span>
                        <div style="margin-top: 10px;">
                            <button id="installFFmpegBtn" class="btn btn-primary" style="margin-right: 10px;" onclick="window.installFFmpeg()">
                                <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle;">download</span>
                                ${i18nSpan('tts:ffmpeg_install')}
                            </button>
                            ${ffmpegManualLink()}
                        </div>
                        <p style="margin-top: 8px; font-size: 0.8rem; color: var(--text-secondary);">
                            ${i18nSpan('tts:ffmpeg_restart_hint')}
                        </p>
                    `;
                } else {
                    ttsStatusText.textContent = t('tts:failed_status', { error: errorText });
                }
            }
            MessageLogger.addLog(t('tts:failed_log', { error: errorText }));
            break;
    }
}

/**
 * Install FFmpeg via winget (Windows)
 */
window.installFFmpeg = async function() {
    const btn = document.getElementById('installFFmpegBtn');
    const ttsStatusText = DomHelpers.getElement('ttsStatusText');

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `
            <span class="material-symbols-outlined rotating" style="font-size: 18px; vertical-align: middle;">sync</span>
            ${i18nSpan('tts:ffmpeg_installing')}
        `;
    }

    const renderFailure = (key, error) => {
        if (!ttsStatusText) return;
        ttsStatusText.innerHTML = `
            <span style="color: #ef4444;">❌ ${i18nSpan(key, { error })}</span>
            <div style="margin-top: 10px;">
                ${ffmpegManualLink()}
            </div>
        `;
    };

    try {
        const result = await ApiClient.installFFmpeg();
        if (ttsStatusText) {
            // result.message is server text: escape it.
            ttsStatusText.innerHTML = `
                <span style="color: #22c55e;">✅ ${DomHelpers.escapeHtml(result.message || '')}</span>
                <p style="margin-top: 8px; font-size: 0.8rem; color: var(--text-secondary);">
                    ${i18nSpan('tts:ffmpeg_restart_needed')}
                </p>
            `;
        }
        MessageLogger.addLog(t('tts:ffmpeg_installed_log'));
    } catch (err) {
        if (err.status) {
            // The server answered with a failure (400: no winget, 500: install failed).
            renderFailure('tts:ffmpeg_install_failed', err.message);
            MessageLogger.addLog(t('tts:ffmpeg_install_failed_log', { error: err.message }));
        } else {
            renderFailure('tts:ffmpeg_install_error', err.message);
            MessageLogger.addLog(t('tts:ffmpeg_install_error_log', { error: err.message }));
        }
    }
}
