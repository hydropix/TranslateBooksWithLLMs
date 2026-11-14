"""Regression tests: client-chosen output names and the "open file" action.

`output_filename` from /api/translate used to be joined to the output
directory unchanged, so '../../x' or an absolute path wrote outside it; and
POST /api/files/<name>/open launched any file type through the OS handler.
"""
import os
from pathlib import Path

import pytest
from flask import Flask

from src.api import handlers
from src.api.blueprints.file_routes import create_file_blueprint
from src.api.blueprints.translation_routes import create_translation_blueprint
from src.api.services.file_service import FileService
from src.api.services.path_validator import PathValidator


@pytest.mark.parametrize('requested, expected', [
    ('book (French).epub', 'book (French).epub'),
    ('../../evil.txt', 'evil.txt'),
    ('..\\..\\evil.txt', 'evil.txt'),
    ('/etc/cron.d/evil.txt', 'evil.txt'),
    ('C:\\Windows\\evil.txt', 'evil.txt'),
    ('C:evil.txt', 'evil.txt'),
    ('payload.bat', 'payload.bat.txt'),
    ('payload.hta', 'payload.hta.txt'),
    ('noext', 'noext.txt'),
    ('notes.MD', 'notes.MD'),
])
def test_safe_output_filename(requested, expected):
    assert PathValidator.safe_output_filename(requested) == expected


@pytest.mark.parametrize('requested', ['', '   ', '..', '../', '/', None, 42])
def test_unusable_output_filename_is_rejected(requested):
    assert PathValidator.safe_output_filename(requested) is None


@pytest.mark.parametrize('requested', ['../../escape.txt', str(Path('/abs/escape.txt'))])
def test_resolve_output_path_stays_inside_output_dir(tmp_path, requested):
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    for is_resume in (False, True):
        resolved = Path(handlers.resolve_output_path(str(out_dir), requested, is_resume))
        assert resolved.parent.resolve() == out_dir.resolve()
        assert resolved.name == 'escape.txt'


def test_resolve_output_path_rejects_empty_name(tmp_path):
    with pytest.raises(ValueError):
        handlers.resolve_output_path(str(tmp_path), '..', False)


class _RecordingStateManager:
    def __init__(self):
        self.created = []

    def create_translation(self, translation_id, config):
        self.created.append(dict(config))


def _translate_client(tmp_path):
    state_manager = _RecordingStateManager()
    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state_manager, lambda tid, config: None, str(tmp_path)))
    return app.test_client(), state_manager


def _body(output_filename):
    return {
        'text': 'Hello.', 'source_language': 'English', 'target_language': 'French',
        'model': 'm', 'llm_api_endpoint': 'http://localhost:11434/api/generate',
        'output_filename': output_filename,
    }


def test_translate_route_stores_sanitized_output_filename(tmp_path):
    client, state_manager = _translate_client(tmp_path)
    resp = client.post('/api/translate', json=_body('../../../startup/run.bat'))
    assert resp.status_code == 200
    assert state_manager.created[0]['output_filename'] == 'run.bat.txt'


def test_translate_route_rejects_unusable_output_filename(tmp_path):
    client, state_manager = _translate_client(tmp_path)
    resp = client.post('/api/translate', json=_body('../..'))
    assert resp.status_code == 400
    assert state_manager.created == []


@pytest.fixture
def no_launch(monkeypatch):
    launched = []
    monkeypatch.setattr(os, 'startfile', lambda p: launched.append(p), raising=False)
    # open_file imports platform lazily, so patch the module attribute.
    monkeypatch.setattr('platform.system', lambda: 'Windows')
    return launched


def test_open_refuses_non_document_types(tmp_path, no_launch):
    (tmp_path / 'payload.bat').write_text('echo hi', encoding='utf-8')
    ok, message, _ = FileService(str(tmp_path)).open_file('payload.bat')
    assert not ok and 'not allowed' in message
    assert no_launch == []

    app = Flask(__name__)
    app.register_blueprint(create_file_blueprint(str(tmp_path)))
    resp = app.test_client().post('/api/files/payload.bat/open')
    assert resp.status_code == 403
    assert no_launch == []


def test_open_allows_translated_documents(tmp_path, no_launch):
    (tmp_path / 'book (French).epub').write_bytes(b'PK')
    ok, _message, abs_path = FileService(str(tmp_path)).open_file('book (French).epub')
    assert ok
    assert no_launch == [abs_path]
