"""
Tests for the web side of the term fix: `term_fix` validation on
/api/translate and the /api/term-fix/scan preview endpoint.
"""
import pytest
from flask import Flask

import src.config as _config
from src.api.blueprints.translation_routes import create_translation_blueprint

OPENAI_DEFAULT_ENDPOINT = 'https://api.openai.com/v1/chat/completions'


class _RecordingStateManager:
    def __init__(self):
        self.created = []

    def create_translation(self, translation_id, config):
        self.created.append((translation_id, dict(config)))


@pytest.fixture(autouse=True)
def deterministic_openai_endpoint(monkeypatch):
    monkeypatch.setattr(_config, 'OPENAI_API_ENDPOINT', OPENAI_DEFAULT_ENDPOINT)


@pytest.fixture
def app_client(tmp_path):
    state_manager = _RecordingStateManager()
    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state_manager, lambda translation_id, config: None, str(tmp_path),
    ))
    (tmp_path / 'uploads').mkdir()
    with app.test_client() as client:
        yield client, state_manager, tmp_path


def _payload(**overrides):
    body = {
        'text': 'Ashe came.',
        'source_language': 'English',
        'target_language': 'English',
        'model': 'gpt-4o',
        'llm_api_endpoint': OPENAI_DEFAULT_ENDPOINT,
        'output_filename': 'out.txt',
    }
    body.update(overrides)
    return body


def test_term_fix_is_recorded_and_disables_refine(app_client):
    client, state_manager, _ = app_client
    resp = client.post('/api/translate', json=_payload(
        refine_only=True, term_fix={'rules_text': 'Ashe -> Askan', 'mode': 'literal'}))
    assert resp.status_code == 200
    _id, config = state_manager.created[0]
    assert config['term_fix'] == {'rules_text': 'Ashe -> Askan', 'mode': 'literal'}
    assert config['refine_only'] is False and config['refine_after'] is False


def test_no_term_fix_by_default(app_client):
    client, state_manager, _ = app_client
    assert client.post('/api/translate', json=_payload(term_fix=None)).status_code == 200
    assert state_manager.created[0][1]['term_fix'] is None


@pytest.mark.parametrize('term_fix,message', [
    ({'rules_text': 'broken line', 'mode': 'smart'}, "line 1"),
    ({'rules_text': '   ', 'mode': 'smart'}, "no correction rule"),
    ({'rules_text': 'Drak* -> Eld', 'mode': 'literal'}, "wildcard"),
    ({'rules_text': 'Ashe -> Askan', 'mode': 'fancy'}, "unknown mode"),
    ('Ashe -> Askan', "must be an object"),
])
def test_invalid_term_fix_is_rejected_before_the_job(app_client, term_fix, message):
    client, state_manager, _ = app_client
    resp = client.post('/api/translate', json=_payload(term_fix=term_fix))
    assert resp.status_code == 400
    assert message in resp.get_json()['error']
    assert state_manager.created == []


def test_scan_counts_matches(app_client):
    client, _state, tmp_path = app_client
    upload = tmp_path / 'uploads' / 'book.txt'
    upload.write_text('Ashe one. Ashen two. Ashe three.', encoding='utf-8')
    resp = client.post('/api/term-fix/scan', json={
        'file_path': str(upload), 'rules_text': 'Ashe|Ashen|Asha -> Askan'})
    assert resp.status_code == 200
    [rule] = resp.get_json()['rules']
    assert rule['total'] == 3 and rule['replacement'] == 'Askan'
    assert [a['count'] for a in rule['alternatives']] == [2, 1, 0]
    assert rule['samples'][0] == 'Ashe one.'


def test_scan_accepts_wildcards(app_client):
    client, _state, tmp_path = app_client
    upload = tmp_path / 'uploads' / 'book.txt'
    upload.write_text('Drakbranden. Drakbrandens.', encoding='utf-8')
    resp = client.post('/api/term-fix/scan', json={
        'file_path': str(upload), 'rules_text': 'Drakbrand* -> Eldsdraken'})
    assert resp.status_code == 200
    assert resp.get_json()['rules'][0]['total'] == 2


def test_scan_rejects_path_outside_uploads(app_client):
    client, _state, tmp_path = app_client
    outside = tmp_path / 'secret.txt'
    outside.write_text('Ashe', encoding='utf-8')
    resp = client.post('/api/term-fix/scan', json={
        'file_path': str(outside), 'rules_text': 'Ashe -> Askan'})
    assert resp.status_code == 403


def test_scan_rejects_bad_rules(app_client):
    client, _state, tmp_path = app_client
    upload = tmp_path / 'uploads' / 'book.txt'
    upload.write_text('Ashe', encoding='utf-8')
    resp = client.post('/api/term-fix/scan', json={
        'file_path': str(upload), 'rules_text': 'nope'})
    assert resp.status_code == 400
