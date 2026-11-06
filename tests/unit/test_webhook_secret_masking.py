"""Webhook secrets must be masked by GET /api/config and survive a round trip.

Regression tests for GitHub issue #214. All secrets below are inert placeholders.
"""
import json

import pytest
from dotenv import dotenv_values
from flask import Flask

import src.config as cfg
from src.api.blueprints import config_routes
from src.utils.webhook_secrets import (
    MASK,
    mask_webhook_headers,
    mask_webhook_url,
    unmask_webhook_headers,
    unmask_webhook_url,
)

DISCORD_URL = "https://discord.com/api/webhooks/123456/SECRET_TOKEN_PLACEHOLDER"
GOTIFY_URL = "https://gotify.example.com/message?token=SECRET_TOKEN_PLACEHOLDER&priority=5"
HEADERS = '{"Authorization":"Bearer SECRET_TOKEN_PLACEHOLDER","X-Other":"OTHER_SECRET_PLACEHOLDER"}'


class TestMaskHelpers:
    def test_discord_path_token_masked(self):
        masked = mask_webhook_url(DISCORD_URL)
        assert "SECRET_TOKEN_PLACEHOLDER" not in masked
        assert "123456" not in masked
        assert masked == f"https://discord.com/api/webhooks/{MASK}/{MASK}"

    def test_query_values_masked_keys_kept(self):
        masked = mask_webhook_url(GOTIFY_URL)
        assert "SECRET_TOKEN_PLACEHOLDER" not in masked
        assert f"token={MASK}" in masked and f"priority={MASK}" in masked
        assert masked.startswith("https://gotify.example.com/message?")

    def test_userinfo_masked(self):
        masked = mask_webhook_url("https://user:PASSWORD_PLACEHOLDER@host.example.com/ping/{event}")
        assert "PASSWORD_PLACEHOLDER" not in masked and "user" not in masked
        assert masked == f"https://{MASK}@host.example.com/ping/{{event}}"

    def test_ntfy_topic_masked(self):
        assert mask_webhook_url("https://ntfy.sh/my-private-topic") == f"https://ntfy.sh/{MASK}"

    def test_empty_and_garbage(self):
        assert mask_webhook_url("") == ""
        assert mask_webhook_url(None) == ""
        assert mask_webhook_url("not a url with SECRET") == MASK

    def test_headers_names_visible_values_masked(self):
        masked = json.loads(mask_webhook_headers(HEADERS))
        assert masked == {"Authorization": MASK, "X-Other": MASK}
        assert mask_webhook_headers("") == ""
        assert mask_webhook_headers("not json SECRET") == MASK

    def test_unmask_url_unchanged_and_edited(self):
        assert unmask_webhook_url(mask_webhook_url(DISCORD_URL), DISCORD_URL) == DISCORD_URL
        assert unmask_webhook_url(mask_webhook_url(GOTIFY_URL), GOTIFY_URL) == GOTIFY_URL
        # Edit one query value, keep the other masked
        edited = f"https://gotify.example.com/message?token={MASK}&priority=8"
        assert unmask_webhook_url(edited, GOTIFY_URL) == \
            "https://gotify.example.com/message?token=SECRET_TOKEN_PLACEHOLDER&priority=8"
        # A fully new URL replaces the stored one
        new = "https://ntfy.sh/new-topic"
        assert unmask_webhook_url(new, DISCORD_URL) == new

    def test_unmask_headers_partial_edit(self):
        submitted = json.dumps({"Authorization": MASK, "X-Other": "NEW_VALUE_PLACEHOLDER"})
        merged = json.loads(unmask_webhook_headers(submitted, HEADERS))
        assert merged == {
            "Authorization": "Bearer SECRET_TOKEN_PLACEHOLDER",
            "X-Other": "NEW_VALUE_PLACEHOLDER",
        }


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Flask client whose .env lives in tmp_path and whose reload reads it back."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr(config_routes, "get_config_path", lambda: str(tmp_path))

    def fake_reload():
        values = dotenv_values(env_file)
        for key in ("NOTIFY_WEBHOOK_URL", "NOTIFY_WEBHOOK_HEADERS"):
            monkeypatch.setattr(cfg, key, values.get(key) or "")

    monkeypatch.setattr(config_routes, "reload_config", fake_reload)
    monkeypatch.setattr(cfg, "NOTIFY_WEBHOOK_URL", DISCORD_URL)
    monkeypatch.setattr(cfg, "NOTIFY_WEBHOOK_HEADERS", HEADERS)

    app = Flask(__name__)
    app.register_blueprint(config_routes.create_config_blueprint(server_session_id=1))
    with app.test_client() as c:
        c.env_file = env_file
        yield c


class TestConfigEndpoint:
    def test_get_config_masks_webhook_secrets(self, client):
        data = client.get("/api/config").get_json()
        raw = json.dumps(data)
        assert "SECRET_TOKEN_PLACEHOLDER" not in raw
        assert "OTHER_SECRET_PLACEHOLDER" not in raw
        assert data["notify_webhook_url"] == f"https://discord.com/api/webhooks/{MASK}/{MASK}"
        assert json.loads(data["notify_webhook_headers"]) == {"Authorization": MASK, "X-Other": MASK}
        assert data["notify_configured"] is True

    def test_saving_masked_values_keeps_originals(self, client):
        data = client.get("/api/config").get_json()
        resp = client.post("/api/settings", json={
            "NOTIFY_WEBHOOK_URL": data["notify_webhook_url"],
            "NOTIFY_WEBHOOK_HEADERS": data["notify_webhook_headers"],
        })
        assert resp.status_code == 200
        assert cfg.NOTIFY_WEBHOOK_URL == DISCORD_URL
        assert json.loads(cfg.NOTIFY_WEBHOOK_HEADERS) == json.loads(HEADERS)

    def test_saving_new_values_replaces_originals(self, client):
        new_url = "https://ntfy.sh/NEW_TOPIC_PLACEHOLDER"
        new_headers = '{"Authorization":"Bearer NEW_TOKEN_PLACEHOLDER"}'
        resp = client.post("/api/settings", json={
            "NOTIFY_WEBHOOK_URL": new_url,
            "NOTIFY_WEBHOOK_HEADERS": new_headers,
        })
        assert resp.status_code == 200
        assert cfg.NOTIFY_WEBHOOK_URL == new_url
        assert json.loads(cfg.NOTIFY_WEBHOOK_HEADERS) == json.loads(new_headers)

    def test_clearing_values_still_works(self, client):
        resp = client.post("/api/settings", json={
            "NOTIFY_WEBHOOK_URL": "",
            "NOTIFY_WEBHOOK_HEADERS": "",
        })
        assert resp.status_code == 200
        assert cfg.NOTIFY_WEBHOOK_URL == ""
        assert cfg.NOTIFY_WEBHOOK_HEADERS == ""
