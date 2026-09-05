"""Unit tests for the `input_filename` field recorded in a translation job's config.

Covers Phase 1 of the multi-device sharing plan (issue #271): `/api/translate`
must record a human-readable `input_filename` in the job config, whether the
client supplied one, the server must derive it from the validated upload
path, or (for a text job with none of the above) omit the key entirely so
`get_translation_summaries()` keeps emitting `null` as before.
"""
import pytest
from flask import Flask

import src.config as _config
from src.api.blueprints.translation_routes import (
    _display_input_filename,
    create_translation_blueprint,
)

OPENAI_DEFAULT_ENDPOINT = 'https://api.openai.com/v1/chat/completions'


# ---------------------------------------------------------------------------
# Fixture pattern copied locally from tests/unit/test_endpoint_allowlist.py
# (lines 244-272) — do not import test modules from each other.
# ---------------------------------------------------------------------------

class _RecordingStateManager:
    """Minimal stand-in that records the config a job was created with."""

    def __init__(self):
        self.created = []

    def create_translation(self, translation_id, config):
        self.created.append((translation_id, dict(config)))


@pytest.fixture
def translate_app(tmp_path):
    """Flask app exposing only the translation blueprint (no auth gate)."""
    state_manager = _RecordingStateManager()
    started = []

    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state_manager,
        lambda translation_id, config: started.append(translation_id),
        str(tmp_path),
    ))

    with app.test_client() as client:
        yield client, state_manager, started


def _payload(**overrides):
    body = {
        'text': 'Hello world.',
        'source_language': 'English',
        'target_language': 'French',
        'model': 'gpt-4o',
        'llm_api_endpoint': OPENAI_DEFAULT_ENDPOINT,
        'output_filename': 'out.txt',
    }
    body.update(overrides)
    return body


@pytest.fixture(autouse=True)
def deterministic_openai_endpoint(monkeypatch):
    """Pin the OpenAI default so 'override' is unambiguous regardless of .env."""
    monkeypatch.setattr(_config, 'OPENAI_API_ENDPOINT', OPENAI_DEFAULT_ENDPOINT)


# ---------------------------------------------------------------------------
# POST /api/translate — text jobs
# ---------------------------------------------------------------------------

def test_text_job_records_requested_input_filename(translate_app):
    client, state_manager, _started = translate_app
    resp = client.post('/api/translate', json=_payload(input_filename='chapter one.txt'))
    assert resp.status_code == 200
    _translation_id, config = state_manager.created[0]
    assert config['input_filename'] == 'chapter one.txt'


def test_text_job_strips_path_traversal_from_input_filename(translate_app):
    client, state_manager, _started = translate_app
    resp = client.post('/api/translate', json=_payload(input_filename='../../etc/passwd'))
    assert resp.status_code == 200
    _translation_id, config = state_manager.created[0]
    assert config['input_filename'] == 'passwd'


def test_text_job_without_input_filename_omits_key(translate_app):
    client, state_manager, _started = translate_app
    resp = client.post('/api/translate', json=_payload())
    assert resp.status_code == 200
    _translation_id, config = state_manager.created[0]
    assert 'input_filename' not in config


def test_input_filename_is_truncated_to_200_chars(translate_app):
    client, state_manager, _started = translate_app
    long_name = 'a' * 250
    resp = client.post('/api/translate', json=_payload(input_filename=long_name))
    assert resp.status_code == 200
    _translation_id, config = state_manager.created[0]
    assert config['input_filename'] == 'a' * 200


# ---------------------------------------------------------------------------
# POST /api/translate — file jobs
# ---------------------------------------------------------------------------

def test_file_job_falls_back_to_stripped_upload_name(translate_app, tmp_path):
    client, state_manager, _started = translate_app
    uploads_dir = tmp_path / 'uploads'
    uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_path = uploads_dir / '0123456789abcdef_book.epub'
    upload_path.write_bytes(b'fake epub content')

    resp = client.post('/api/translate', json=_payload(
        file_path=str(upload_path),
        file_type='epub',
        output_filename='out.epub',
    ))
    assert resp.status_code == 200
    _translation_id, config = state_manager.created[0]
    assert config['input_filename'] == 'book.epub'


# ---------------------------------------------------------------------------
# Direct unit tests of _display_input_filename
# ---------------------------------------------------------------------------

def test_display_input_filename_prefers_requested_basename():
    assert _display_input_filename('  some/dir/chapter one.txt  ', None) == 'chapter one.txt'


def test_display_input_filename_falls_back_to_stripped_safe_path(tmp_path):
    safe_path = tmp_path / '0123456789abcdef_book.epub'
    safe_path.write_bytes(b'x')
    assert _display_input_filename(None, safe_path) == 'book.epub'


def test_display_input_filename_keeps_full_name_when_prefix_does_not_match(tmp_path):
    safe_path = tmp_path / 'not_a_hex_prefix_book.epub'
    safe_path.write_bytes(b'x')
    assert _display_input_filename('', safe_path) == 'not_a_hex_prefix_book.epub'


def test_display_input_filename_returns_none_without_either_source():
    assert _display_input_filename(None, None) is None
    assert _display_input_filename('   ', None) is None
