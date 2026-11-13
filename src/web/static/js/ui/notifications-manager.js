/**
 * Notifications Manager — wires the Notifications section in Settings.
 *
 * Backend lives in src/utils/notifier.py and is configured via NOTIFY_* keys
 * in .env. This module only deals with the UI: presets, collapsible toggle,
 * and the "Send test" button. Persistence is handled by SettingsManager.
 */

import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from './dom-helpers.js';
import { MessageLogger } from './message-logger.js';
import { t } from '../i18n/i18n.js';

const PRESETS = {
    ntfy: {
        url: 'https://ntfy.sh/tbl-CHANGE-ME-K8x9p2',
        method: 'POST',
        headers: '',
        payload: '',
        noteKey: 'settings:preset_ntfy_note'
    },
    gotify: {
        url: 'https://gotify.example.com/message?token=YOUR_TOKEN_HERE',
        method: 'POST',
        headers: '',
        payload: '',
        noteKey: 'settings:preset_gotify_note'
    },
    discord: {
        url: 'https://discord.com/api/webhooks/XXXXX/YYYYY',
        method: 'POST',
        headers: '',
        payload: '{"content":"Translation **{event}**: `{file}` in {duration_seconds:.0f}s"}',
        noteKey: 'settings:preset_discord_note'
    },
    slack: {
        url: 'https://hooks.slack.com/services/XXX/YYY/ZZZ',
        method: 'POST',
        headers: '',
        payload: '{"text":"Translation {event}: {file} ({duration_seconds:.0f}s)"}',
        noteKey: 'settings:preset_slack_note'
    },
    healthchecks: {
        url: 'https://hc-ping.com/your-uuid-here/{event}',
        method: 'GET',
        headers: '',
        payload: '',
        noteKey: 'settings:preset_healthchecks_note'
    },
    clear: {
        url: '',
        method: 'POST',
        headers: '',
        payload: '',
        noteKey: 'settings:preset_clear_note'
    }
};

export const NotificationsManager = {
    initialize() {
        this._wirePresetButtons();
        this._wireDirtyTracking();
        this._autoExpandIfConfigured();
    },

    _wirePresetButtons() {
        document.querySelectorAll('[data-notif-preset]').forEach((btn) => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const preset = btn.getAttribute('data-notif-preset');
                this.applyPreset(preset);
            });
        });
    },

    _wireDirtyTracking() {
        const ids = [
            { id: 'notifyWebhookUrl', event: 'input' },
            { id: 'notifyWebhookMethod', event: 'change' },
            { id: 'notifyWebhookHeaders', event: 'input' },
            { id: 'notifyWebhookPayload', event: 'input' },
            { id: 'notifyOnSuccess', event: 'change' },
            { id: 'notifyOnFailure', event: 'change' },
            { id: 'notifyOnInterruption', event: 'change' },
            { id: 'notifyTimeoutSeconds', event: 'input' }
        ];
        ids.forEach(({ id, event }) => {
            const el = DomHelpers.getElement(id);
            if (el) {
                el.addEventListener(event, () => {
                    const btn = DomHelpers.getElement('saveSettingsBtn');
                    if (btn) btn.disabled = false;
                });
            }
        });
    },

    _autoExpandIfConfigured() {
        // Open the section automatically when a URL is already set, so users
        // see right away that notifications are active and what's configured.
        window.addEventListener('defaultConfigLoaded', () => {
            const url = DomHelpers.getValue('notifyWebhookUrl');
            if (url && url.trim()) {
                const section = DomHelpers.getElement('notificationOptionsSection');
                const icon = DomHelpers.getElement('notificationOptionsIcon');
                if (section && section.classList.contains('hidden')) {
                    section.classList.remove('hidden');
                    if (icon) icon.style.transform = 'rotate(180deg)';
                }
            }
        }, { once: true });
    },

    toggleOptions() {
        const section = DomHelpers.getElement('notificationOptionsSection');
        const icon = DomHelpers.getElement('notificationOptionsIcon');
        if (!section) return;
        section.classList.toggle('hidden');
        if (icon) {
            icon.style.transform = section.classList.contains('hidden')
                ? 'rotate(0deg)'
                : 'rotate(180deg)';
        }
    },

    applyPreset(name) {
        const preset = PRESETS[name];
        if (!preset) return;

        DomHelpers.setValue('notifyWebhookUrl', preset.url);
        DomHelpers.setValue('notifyWebhookMethod', preset.method);
        DomHelpers.setValue('notifyWebhookHeaders', preset.headers);
        DomHelpers.setValue('notifyWebhookPayload', preset.payload);

        // Mark Save button dirty so the user is reminded to persist
        const btn = DomHelpers.getElement('saveSettingsBtn');
        if (btn) btn.disabled = false;

        const result = DomHelpers.getElement('notifyTestResult');
        if (result) {
            result.style.color = 'var(--text-muted-light)';
            result.textContent = preset.noteKey ? t(preset.noteKey) : '';
        }

        MessageLogger.addLog(t('common:notifications_preset_applied', { name }));
    },

    async testNotification() {
        const eventSelect = DomHelpers.getElement('notifyTestEvent');
        const resultEl = DomHelpers.getElement('notifyTestResult');
        const btn = DomHelpers.getElement('notifyTestBtn');
        const event = eventSelect ? eventSelect.value : 'success';

        if (resultEl) {
            resultEl.style.color = 'var(--text-muted-light)';
            resultEl.textContent = t('common:sending');
        }
        if (btn) btn.disabled = true;

        const showFailure = (err) => {
            if (resultEl) {
                resultEl.style.color = 'var(--danger-light)';
                resultEl.textContent = err;
            }
            MessageLogger.showMessage(t('common:notification_test_failed', { error: err }), 'error');
        };

        try {
            const data = await ApiClient.testNotification(event);

            if (data && data.success) {
                if (resultEl) {
                    resultEl.style.color = 'var(--success-light)';
                    resultEl.textContent = data.message || t('common:test_sent_default');
                }
                MessageLogger.showMessage(t('common:notification_test_sent', { event }), 'success');
            } else {
                // 2xx without success (not emitted by the current route).
                showFailure((data && data.error) || 'HTTP 200');
            }
        } catch (e) {
            if (e.status) {
                // The server answered (empty URL, disabled event, webhook
                // failure): show its reason.
                showFailure(e.body?.error || `HTTP ${e.status}`);
            } else {
                if (resultEl) {
                    resultEl.style.color = 'var(--danger-light)';
                    resultEl.textContent = t('errors:network_error', { error: e.message });
                }
                MessageLogger.showMessage(t('common:notification_test_failed', { error: e.message }), 'error');
            }
        } finally {
            if (btn) btn.disabled = false;
        }
    }
};
