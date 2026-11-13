"""Regression tests: POST /api/settings rewrites .env atomically and serially.

The .env writer truncated and rewrote the file in place with no lock, so a
failure mid-write left a truncated .env (every API key lost) and two
concurrent saves could drop each other's keys. Tests only ever touch a
temporary .env, never the project's real one.
"""
import os
import threading

import pytest
from flask import Flask

from src.api.blueprints import config_routes

ORIGINAL = "# test env\nGEMINI_MODEL=model-a\nNOTIFY_ON_SUCCESS=false\n"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config_routes, 'get_config_path', lambda: str(tmp_path))
    monkeypatch.setattr(config_routes, 'reload_config', lambda: None)
    env_file = tmp_path / '.env'
    env_file.write_text(ORIGINAL, encoding='utf-8')
    app = Flask(__name__)
    app.register_blueprint(config_routes.create_config_blueprint())
    return app, env_file, tmp_path


def test_failed_write_leaves_env_untouched(settings, monkeypatch):
    app, env_file, tmp_path = settings

    def broken_replace(src, dst):
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(config_routes.os, 'replace', broken_replace)
    resp = app.test_client().post('/api/settings', json={'GEMINI_MODEL': 'model-b'})

    assert resp.status_code == 500
    assert env_file.read_text(encoding='utf-8') == ORIGINAL
    assert sorted(p.name for p in tmp_path.iterdir()) == ['.env']


def test_successful_write_updates_and_keeps_other_lines(settings):
    app, env_file, _tmp = settings
    resp = app.test_client().post('/api/settings', json={'GEMINI_MODEL': 'model-b'})
    assert resp.status_code == 200
    text = env_file.read_text(encoding='utf-8')
    assert 'GEMINI_MODEL=model-b' in text
    assert 'NOTIFY_ON_SUCCESS=false' in text
    assert text.startswith('# test env')


def test_concurrent_saves_do_not_lose_updates(settings, monkeypatch):
    app, env_file, _tmp = settings
    keys = ['GEMINI_MODEL', 'OPENROUTER_MODEL', 'MISTRAL_MODEL', 'DEEPSEEK_MODEL',
            'POE_MODEL', 'NIM_MODEL', 'DEFAULT_MODEL', 'NOTIFY_ON_FAILURE']

    # Widen the read-modify-write window so an unserialised writer loses keys.
    real_open = open

    def slow_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        if str(args[0]).endswith('.env') and 'r' in (args[1] if len(args) > 1 else kwargs.get('mode', 'r')):
            threading.Event().wait(0.02)
        return handle

    monkeypatch.setattr('builtins.open', slow_open)
    errors = []

    def save(key):
        resp = app.test_client().post('/api/settings', json={key: f'value-{key.lower()}'})
        if resp.status_code != 200:
            errors.append(resp.status_code)

    threads = [threading.Thread(target=save, args=(k,)) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    text = env_file.read_text(encoding='utf-8')
    for key in keys:
        assert f'{key}=value-{key.lower()}' in text
    assert not [p for p in os.listdir(env_file.parent) if p.endswith('.tmp')]
