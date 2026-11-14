/**
 * TTS Audiobook Modal
 *
 * Audiobook generation dialog: exposes window.createAudiobook (called by the
 * file list's audiobook button) which opens the provider/voice configuration
 * modal and starts the TTS job.
 */

import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { MessageLogger } from '../ui/message-logger.js';
import { ModalStack } from '../ui/modal-stack.js';
import { t, i18nSpan } from '../i18n/i18n.js';

/**
 * Show TTS configuration modal and start audiobook generation
 * @param {string} filename - File to generate audio from
 * @param {string} filepath - Full path to the file
 */
window.createAudiobook = async function(filename, filepath) {
    // Show TTS modal
    showTTSModal(filename, filepath);
};

/**
 * Show TTS configuration modal with provider selection
 */
async function showTTSModal(filename, filepath) {
    // Remove existing modal if present
    const existingModal = document.getElementById('ttsModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Get providers info and voice prompts
    let providersInfo = {};
    let voicePrompts = [];
    let gpuStatus = { cuda_available: false };

    try {
        [providersInfo, voicePrompts, gpuStatus] = await Promise.all([
            ApiClient.getTTSProviders().catch(() => ({ providers: {} })),
            ApiClient.getTTSVoicePrompts().catch(() => ({ voice_prompts: [] })),
            ApiClient.getTTSGPUStatus().catch(() => ({ cuda_available: false }))
        ]);
        providersInfo = providersInfo.providers || {};
        voicePrompts = voicePrompts.voice_prompts || [];
    } catch {
    }

    const isChatterboxAvailable = providersInfo.chatterbox?.available || false;

    // Build voice prompts options
    const voicePromptsOptions = voicePrompts.map(vp =>
        `<option value="${DomHelpers.escapeHtml(vp.path)}">${DomHelpers.escapeHtml(vp.filename)}</option>`
    ).join('');

    const chatterboxKey = isChatterboxAvailable
        ? 'tts:provider_chatterbox_full_local'
        : 'tts:provider_chatterbox_full_unavailable';

    // Create modal HTML with provider selection. Every label carries a
    // data-i18n* marker so applyToDOM() re-translates it on a locale switch.
    // Language option labels are left as-is: they are language names with
    // autonyms whose value selects the voice, like the main language pickers.
    const modalHtml = `
        <div id="ttsModal" class="modal-overlay">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 data-i18n="tts:modal_title_emoji">${t('tts:modal_title_emoji')}</h3>
                    <button class="close-btn" id="ttsModalClose">&times;</button>
                </div>
                <div class="modal-body">
                    <p style="margin: 0 0 20px 0; color: var(--text-muted-light); font-size: 14px;">
                        ${i18nSpan('tts:modal_subtitle', { filename })}
                    </p>

                    <!-- Provider Selection -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                        <div class="form-group" style="margin-bottom: 0;">
                            <label style="font-size: 13px;" data-i18n="tts:provider">${t('tts:provider')}</label>
                            <select id="ttsModalProvider" class="form-control" style="font-size: 13px;">
                                <option value="edge-tts" data-i18n="tts:provider_edge">${t('tts:provider_edge')}</option>
                                <option value="chatterbox" ${!isChatterboxAvailable ? 'disabled' : ''} data-i18n="${chatterboxKey}">${t(chatterboxKey)}</option>
                            </select>
                        </div>

                        <!-- GPU Status (shown when Chatterbox selected) -->
                        <div id="ttsModalGpuStatus" class="form-group" style="margin-bottom: 0; display: none;">
                            <label style="font-size: 13px;" data-i18n="tts:gpu_status">${t('tts:gpu_status')}</label>
                            <div class="gpu-status ${gpuStatus.cuda_available ? 'gpu-available' : 'gpu-unavailable'}">
                                <span class="status-dot ${gpuStatus.cuda_available ? 'available' : 'unavailable'}"></span>
                                ${gpuStatus.cuda_available && gpuStatus.gpu_name ? `<span>${DomHelpers.escapeHtml(gpuStatus.gpu_name)}</span>` : i18nSpan(gpuStatus.cuda_available ? 'tts:gpu_cuda' : 'tts:gpu_cpu')}
                            </div>
                        </div>
                    </div>

                    <!-- Edge-TTS Options -->
                    <div id="ttsModalEdgeOptions">
                        <div style="display: grid; gap: 15px;">
                            <div class="form-group" style="margin-bottom: 0;">
                                <label style="font-size: 13px;" data-i18n="tts:target_language">${t('tts:target_language')}</label>
                                <select id="ttsModalLanguage" class="form-control" style="font-size: 13px;">
                                    <!-- Most Common -->
                                    <option value="Chinese">Chinese (中文)</option>
                                    <option value="English">English</option>
                                    <option value="French">French (Français)</option>
                                    <option value="Spanish">Spanish (Español)</option>
                                    <option value="German">German (Deutsch)</option>
                                    <option value="Japanese">Japanese (日本語)</option>
                                    <option value="Korean">Korean (한국어)</option>
                                    <option value="Portuguese (Brazil)">Portuguese (Brazil) (Português)</option>
                                    <option value="Portuguese (Portugal)">Portuguese (Portugal) (Português)</option>
                                    <option value="Russian">Russian (Русский)</option>
                                    <option value="Arabic">Arabic (العربية)</option>
                                    <!-- European -->
                                    <option value="Italian">Italian (Italiano)</option>
                                    <option value="Dutch">Dutch (Nederlands)</option>
                                    <option value="Polish">Polish (Polski)</option>
                                    <option value="Swedish">Swedish (Svenska)</option>
                                    <option value="Norwegian">Norwegian (Norsk)</option>
                                    <option value="Danish">Danish (Dansk)</option>
                                    <option value="Finnish">Finnish (Suomi)</option>
                                    <option value="Greek">Greek (Ελληνικά)</option>
                                    <option value="Czech">Czech (Čeština)</option>
                                    <option value="Hungarian">Hungarian (Magyar)</option>
                                    <option value="Romanian">Romanian (Română)</option>
                                    <option value="Turkish">Turkish (Türkçe)</option>
                                    <option value="Ukrainian">Ukrainian (Українська)</option>
                                    <option value="Bulgarian">Bulgarian (Български)</option>
                                    <option value="Croatian">Croatian (Hrvatski)</option>
                                    <option value="Slovak">Slovak (Slovenčina)</option>
                                    <option value="Slovenian">Slovenian (Slovenščina)</option>
                                    <option value="Lithuanian">Lithuanian (Lietuvių)</option>
                                    <option value="Latvian">Latvian (Latviešu)</option>
                                    <option value="Estonian">Estonian (Eesti)</option>
                                    <!-- Asian -->
                                    <option value="Hindi">Hindi (हिन्दी)</option>
                                    <option value="Vietnamese">Vietnamese (Tiếng Việt)</option>
                                    <option value="Thai">Thai (ไทย)</option>
                                    <option value="Indonesian">Indonesian (Bahasa Indonesia)</option>
                                    <option value="Malay">Malay (Bahasa Melayu)</option>
                                    <option value="Filipino">Filipino (Tagalog)</option>
                                    <option value="Bengali">Bengali (বাংলা)</option>
                                    <option value="Tamil">Tamil (தமிழ்)</option>
                                    <option value="Telugu">Telugu (తెలుగు)</option>
                                    <!-- Middle Eastern -->
                                    <option value="Hebrew">Hebrew (עברית)</option>
                                    <option value="Persian">Persian/Farsi (فارسی)</option>
                                    <option value="Urdu">Urdu (اردو)</option>
                                </select>
                            </div>

                            <div class="form-group" style="margin-bottom: 0;">
                                <label style="font-size: 13px;" data-i18n="tts:voice_optional">${t('tts:voice_optional')}</label>
                                <input type="text" id="ttsModalVoice" class="form-control" placeholder="${DomHelpers.escapeHtml(t('tts:voice_placeholder'))}" data-i18n-attr="placeholder:tts:voice_placeholder" style="font-size: 13px;">
                                <small style="color: var(--text-muted-light);" data-i18n="tts:voice_auto_hint">${t('tts:voice_auto_hint')}</small>
                            </div>

                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                                <div class="form-group" style="margin-bottom: 0;">
                                    <label style="font-size: 13px;" data-i18n="tts:speech_rate">${t('tts:speech_rate')}</label>
                                    <select id="ttsModalRate" class="form-control" style="font-size: 13px;">
                                        <option value="-20%" data-i18n="tts:rate_slower">${t('tts:rate_slower')}</option>
                                        <option value="-10%" data-i18n="tts:rate_slightly_slower">${t('tts:rate_slightly_slower')}</option>
                                        <option value="+0%" selected data-i18n="tts:rate_normal">${t('tts:rate_normal')}</option>
                                        <option value="+10%" data-i18n="tts:rate_slightly_faster">${t('tts:rate_slightly_faster')}</option>
                                        <option value="+20%" data-i18n="tts:rate_faster">${t('tts:rate_faster')}</option>
                                        <option value="+30%" data-i18n="tts:rate_much_faster">${t('tts:rate_much_faster')}</option>
                                    </select>
                                </div>

                                <div class="form-group" style="margin-bottom: 0;">
                                    <label style="font-size: 13px;" data-i18n="tts:audio_format">${t('tts:audio_format')}</label>
                                    <select id="ttsModalFormat" class="form-control" style="font-size: 13px;">
                                        <option value="opus" selected data-i18n="tts:audio_format_opus">${t('tts:audio_format_opus')}</option>
                                        <option value="mp3" data-i18n="tts:audio_format_mp3">${t('tts:audio_format_mp3')}</option>
                                    </select>
                                </div>
                            </div>

                            <div class="form-group" style="margin-bottom: 0;">
                                <label style="font-size: 13px;" data-i18n="tts:audio_bitrate">${t('tts:audio_bitrate')}</label>
                                <select id="ttsModalBitrate" class="form-control" style="font-size: 13px;">
                                    <option value="48k" data-i18n="tts:bitrate_48">${t('tts:bitrate_48')}</option>
                                    <option value="64k" selected data-i18n="tts:bitrate_64">${t('tts:bitrate_64')}</option>
                                    <option value="96k" data-i18n="tts:bitrate_96">${t('tts:bitrate_96')}</option>
                                    <option value="128k" data-i18n="tts:bitrate_128">${t('tts:bitrate_128')}</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <!-- Chatterbox Options (hidden by default) -->
                    <div id="ttsModalChatterboxOptions" style="display: none;">
                        <div class="tts-voice-cloning" style="background: var(--background-inset); border-radius: 8px; padding: 15px; margin-bottom: 15px; border: 1px solid var(--warning-light-border);">
                            <h4 style="margin: 0 0 12px 0; font-size: 14px; color: var(--warning-light-text);" data-i18n="tts:voice_cloning_heading">${t('tts:voice_cloning_heading')}</h4>
                            <div class="form-group" style="margin-bottom: 0;">
                                <label style="font-size: 13px;" data-i18n="tts:voice_prompt">${t('tts:voice_prompt')}</label>
                                <select id="ttsModalVoicePrompt" class="form-control" style="font-size: 13px;">
                                    <option value="" data-i18n="tts:voice_prompt_default">${t('tts:voice_prompt_default')}</option>
                                    ${voicePromptsOptions}
                                </select>
                                <small style="color: var(--text-muted-light);" data-i18n="tts:voice_prompt_hint">${t('tts:voice_prompt_hint')}</small>
                            </div>
                        </div>

                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                            <div class="form-group" style="margin-bottom: 0;">
                                <label style="font-size: 13px;">
                                    ${i18nSpan('tts:exaggeration')} <span id="ttsModalExaggerationValue" style="color: var(--warning-light-text);">0.50</span>
                                </label>
                                <input type="range" id="ttsModalExaggeration" min="0" max="1" step="0.05" value="0.5" class="tts-slider">
                                <small style="color: var(--text-muted-light);" data-i18n="tts:exaggeration_hint">${t('tts:exaggeration_hint')}</small>
                            </div>
                            <div class="form-group" style="margin-bottom: 0;">
                                <label style="font-size: 13px;">
                                    ${i18nSpan('tts:cfg_weight')} <span id="ttsModalCfgValue" style="color: var(--warning-light-text);">0.50</span>
                                </label>
                                <input type="range" id="ttsModalCfgWeight" min="0" max="1" step="0.05" value="0.5" class="tts-slider">
                                <small style="color: var(--text-muted-light);" data-i18n="tts:cfg_weight_hint">${t('tts:cfg_weight_hint')}</small>
                            </div>
                        </div>

                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div class="form-group" style="margin-bottom: 0;">
                                <label style="font-size: 13px;" data-i18n="tts:target_language">${t('tts:target_language')}</label>
                                <select id="ttsModalChatterboxLang" class="form-control" style="font-size: 13px;">
                                    <!-- Most Common -->
                                    <option value="en">English</option>
                                    <option value="zh">Chinese (中文)</option>
                                    <option value="es">Spanish (Español)</option>
                                    <option value="fr">French (Français)</option>
                                    <option value="de">German (Deutsch)</option>
                                    <option value="it">Italian (Italiano)</option>
                                    <option value="ja">Japanese (日本語)</option>
                                    <option value="ko">Korean (한국어)</option>
                                    <option value="pt-br">Portuguese (Brazil) (Português)</option>
                                    <option value="pt-pt">Portuguese (Portugal) (Português)</option>
                                    <option value="ru">Russian (Русский)</option>
                                    <option value="ar">Arabic (العربية)</option>
                                    <!-- European -->
                                    <option value="pl">Polish (Polski)</option>
                                    <option value="tr">Turkish (Türkçe)</option>
                                    <option value="nl">Dutch (Nederlands)</option>
                                    <option value="cs">Czech (Čeština)</option>
                                    <option value="sv">Swedish (Svenska)</option>
                                    <option value="da">Danish (Dansk)</option>
                                    <option value="fi">Finnish (Suomi)</option>
                                    <option value="hu">Hungarian (Magyar)</option>
                                    <!-- Asian -->
                                    <option value="hi">Hindi (हिन्दी)</option>
                                    <option value="vi">Vietnamese (Tiếng Việt)</option>
                                    <option value="id">Indonesian (Bahasa Indonesia)</option>
                                    <!-- Other -->
                                    <option value="el">Greek (Ελληνικά)</option>
                                </select>
                            </div>
                            <div class="form-group" style="margin-bottom: 0;">
                                <label style="font-size: 13px;" data-i18n="tts:audio_format">${t('tts:audio_format')}</label>
                                <select id="ttsModalChatterboxFormat" class="form-control" style="font-size: 13px;">
                                    <option value="wav" data-i18n="tts:audio_format_wav">${t('tts:audio_format_wav')}</option>
                                    <option value="mp3" selected data-i18n="tts:audio_format_mp3">${t('tts:audio_format_mp3')}</option>
                                    <option value="opus" data-i18n="tts:audio_format_opus">${t('tts:audio_format_opus')}</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="modal-footer">
                    <button id="ttsModalCancel" class="btn btn-secondary" data-i18n="common:cancel">${t('common:cancel')}</button>
                    <button id="ttsModalGenerate" class="btn btn-primary" style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);" data-i18n="tts:generate_audio_with_icon">${t('tts:generate_audio_with_icon')}</button>
                </div>
            </div>
        </div>
    `;

    // Add modal to DOM
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Get modal elements
    const modal = document.getElementById('ttsModal');
    const closeBtn = document.getElementById('ttsModalClose');
    const cancelBtn = document.getElementById('ttsModalCancel');
    const generateBtn = document.getElementById('ttsModalGenerate');
    const providerSelect = document.getElementById('ttsModalProvider');
    const edgeOptions = document.getElementById('ttsModalEdgeOptions');
    const chatterboxOptions = document.getElementById('ttsModalChatterboxOptions');
    const gpuStatusDiv = document.getElementById('ttsModalGpuStatus');

    // Slider value updates
    const exaggerationSlider = document.getElementById('ttsModalExaggeration');
    const cfgSlider = document.getElementById('ttsModalCfgWeight');
    const exaggerationValue = document.getElementById('ttsModalExaggerationValue');
    const cfgValue = document.getElementById('ttsModalCfgValue');

    if (exaggerationSlider && exaggerationValue) {
        exaggerationSlider.addEventListener('input', () => {
            exaggerationValue.textContent = parseFloat(exaggerationSlider.value).toFixed(2);
        });
    }
    if (cfgSlider && cfgValue) {
        cfgSlider.addEventListener('input', () => {
            cfgValue.textContent = parseFloat(cfgSlider.value).toFixed(2);
        });
    }

    // Provider change handler
    providerSelect.addEventListener('change', () => {
        const isChatterbox = providerSelect.value === 'chatterbox';
        edgeOptions.style.display = isChatterbox ? 'none' : 'block';
        chatterboxOptions.style.display = isChatterbox ? 'block' : 'none';
        gpuStatusDiv.style.display = isChatterbox ? 'block' : 'none';
    });

    // Close handlers
    let releaseEscape = null;
    const closeModal = () => {
        modal.remove();
        if (releaseEscape) releaseEscape();
    };
    releaseEscape = ModalStack.push(closeModal);
    closeBtn.addEventListener('click', closeModal);
    cancelBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    // Generate audio
    generateBtn.addEventListener('click', async () => {
        const provider = providerSelect.value;

        // Build config based on provider
        let config = {
            filename: filename,
            tts_provider: provider
        };

        if (provider === 'edge-tts') {
            config.target_language = document.getElementById('ttsModalLanguage').value;
            config.tts_voice = document.getElementById('ttsModalVoice').value;
            config.tts_rate = document.getElementById('ttsModalRate').value;
            config.tts_format = document.getElementById('ttsModalFormat').value;
            config.tts_bitrate = document.getElementById('ttsModalBitrate').value;
        } else {
            // Chatterbox
            config.target_language = document.getElementById('ttsModalChatterboxLang').value;
            config.tts_voice_prompt_path = document.getElementById('ttsModalVoicePrompt').value;
            config.tts_exaggeration = parseFloat(document.getElementById('ttsModalExaggeration').value);
            config.tts_cfg_weight = parseFloat(document.getElementById('ttsModalCfgWeight').value);
            config.tts_format = document.getElementById('ttsModalChatterboxFormat').value;
        }

        // Disable button and show loading
        generateBtn.disabled = true;
        generateBtn.textContent = t('tts:starting');

        try {
            const result = await ApiClient.generateTTS(config);

            MessageLogger.showMessage(t('tts:tts_started', { filename }), 'success');
            MessageLogger.addLog(t('tts:tts_started_log', { provider, filename, jobId: result.job_id }));

            closeModal();

            const ttsProgressSection = DomHelpers.getElement('ttsProgressSection');
            if (ttsProgressSection) {
                ttsProgressSection.style.display = 'block';
            }

        } catch (error) {
            MessageLogger.showMessage(t('tts:tts_error', { error: error.message }), 'error');
            generateBtn.disabled = false;
            generateBtn.textContent = t('tts:generate_audio_with_icon');
        }
    });
}
