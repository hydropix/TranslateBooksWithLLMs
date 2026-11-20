"""Extra regression pins for the Phase 1 security fixes.

Complements test_output_path_safety.py, test_upload_limits_and_containment.py,
test_zip_slip.py, test_api_auth.py and test_env_file_atomic_write.py with the
cases they do not cover: UNC and device-namespace output names, Windows
reserved device names and NTFS alternate data streams, the full list of
executable extensions refused by the "open file" action, /api/uploads/verify
with symlinks, zip-slip uploads for DOCX as well as EPUB, the token gate over
the *whole* app (every Flask rule and Socket.IO handler), and the .env writer
following the working directory.

Tests marked DEFECT failed before PathValidator.safe_output_filename
replaced Windows-invalid characters and prefixed reserved device names.
"""
import io
import os
import zipfile
from pathlib import Path
from unittest import mock

import pytest
from flask import Flask

from src.api.blueprints.file_routes import create_file_blueprint
from src.api.blueprints.security_routes import create_security_blueprint
from src.api.services.file_service import FileService
from src.api.services.path_validator import PathValidator, is_windows_device_name

BS = '\\'


# ---------------------------------------------------------------------------
# output_filename
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('requested, expected', [
    (BS * 2 + 'server' + BS + 'share' + BS + 'evil.txt', 'evil.txt'),
    ('//server/share/evil.txt', 'evil.txt'),
    (BS * 2 + '?' + BS + 'C:' + BS + 'evil.txt', 'evil.txt'),
    (BS * 2 + '.' + BS + 'C:' + BS + 'evil.txt', 'evil.txt'),
    ('c:/Windows/System32/evil.epub', 'evil.epub'),
    ('payload.EXE', 'payload.EXE.txt'),
    ('payload.bat .', 'payload.bat.txt'),
])
def test_output_filename_drops_every_directory_form(requested, expected):
    assert PathValidator.safe_output_filename(requested) == expected


@pytest.mark.parametrize('requested', ['CON', 'nul.txt', 'com1', 'COM1.epub', 'LPT9.docx', 'AUX',
                                       'CONIN$', 'nul .txt'])
def test_DEFECT_output_filename_is_never_a_windows_device_name(requested):
    """On Windows 10 and older, 'NUL.txt' / 'CON.txt' / 'COM1.epub' open the
    device, not a file: the translation is silently discarded (or written to
    a serial port). Windows 11 treats them as plain files, but they remain
    hard to handle in Explorer and scripts."""
    name = PathValidator.safe_output_filename(requested)
    assert name is not None
    # Same device-name rule the validator applies; PureWindowsPath.is_reserved()
    # cannot be used here: it was removed in Python 3.15.
    assert not is_windows_device_name(name), name


@pytest.mark.parametrize('requested', ['notes:hidden.txt', 'x.txt:evil.bat', 'x.txt::$DATA',
                                       'a<b>.txt', 'a|b.txt', 'a?.txt', 'a*.txt', 'a".txt',
                                       'tab\tname.txt'])
def test_DEFECT_output_filename_has_no_windows_invalid_characters(requested):
    """':' after the first two characters is kept, so 'notes:hidden.txt'
    writes the translation into an NTFS alternate data stream of a 0-byte
    file 'notes' (the listing shows 'notes', 0 bytes). '<>|?*"' and control
    characters make the final write fail with EINVAL after the whole job ran."""
    name = PathValidator.safe_output_filename(requested)
    assert name is not None
    assert not any(c in name for c in ':<>|?*"') and all(ord(c) >= 32 for c in name), name


# ---------------------------------------------------------------------------
# POST /api/files/<name>/open
# ---------------------------------------------------------------------------

@pytest.fixture
def no_launch(monkeypatch):
    launched = []
    monkeypatch.setattr(os, 'startfile', lambda p: launched.append(p), raising=False)
    monkeypatch.setattr('platform.system', lambda: 'Windows')
    return launched


EXECUTABLE_NAMES = [
    'x.exe', 'x.bat', 'x.cmd', 'x.com', 'x.ps1', 'x.psm1', 'x.lnk', 'x.url', 'x.js', 'x.jse',
    'x.vbs', 'x.vbe', 'x.wsf', 'x.wsh', 'x.hta', 'x.html', 'x.htm', 'x.svg', 'x.scr', 'x.msi',
    'x.msc', 'x.jar', 'x.py', 'x.pyw', 'x.reg', 'x.cpl', 'x.chm', 'x.appref-ms', 'x.library-ms',
    'x.settingcontent-ms', 'x.csv', 'x.xml', 'X.EXE', 'x.Bat', 'x.txt.exe', 'x.exe.', 'x.exe ',
]


@pytest.mark.parametrize('name', EXECUTABLE_NAMES)
def test_open_refuses_active_content(tmp_path, no_launch, name):
    try:
        (tmp_path / name).write_bytes(b'MZ')
    except OSError:
        pass  # names Windows cannot create are still refused before any lookup
    app = Flask(__name__)
    app.register_blueprint(create_file_blueprint(str(tmp_path)))
    resp = app.test_client().post(f'/api/files/{name}/open')
    assert resp.status_code in (400, 403, 404)
    assert no_launch == []


@pytest.mark.parametrize('name', ['../x.txt', '..%2Fx.txt', 'sub/../../x.txt', 'C:/x.txt'])
def test_open_refuses_paths(tmp_path, no_launch, name):
    app = Flask(__name__)
    app.register_blueprint(create_file_blueprint(str(tmp_path / 'out')))
    (tmp_path / 'x.txt').write_text('outside', encoding='utf-8')
    resp = app.test_client().post(f'/api/files/{name}/open')
    assert resp.status_code in (400, 403, 404)
    assert no_launch == []


def test_open_refuses_a_document_name_that_links_to_an_executable(tmp_path, no_launch):
    target = tmp_path / 'payload.exe'
    target.write_bytes(b'MZ')
    out = tmp_path / 'out'
    out.mkdir()
    try:
        os.symlink(target, out / 'book.txt')
    except (OSError, NotImplementedError):
        pytest.skip('symlinks need privileges on this host')
    ok, _message, _path = FileService(str(out)).open_file('book.txt')
    assert not ok
    assert no_launch == []


# ---------------------------------------------------------------------------
# /api/uploads/verify
# ---------------------------------------------------------------------------

@pytest.fixture
def security_app(tmp_path):
    out = tmp_path / 'translated_files'
    uploads = out / 'uploads'
    uploads.mkdir(parents=True)
    app = Flask(__name__)
    app.register_blueprint(create_security_blueprint(str(out)))
    return app.test_client(), out, uploads


def test_verify_never_confirms_paths_outside_uploads(security_app, tmp_path):
    client, out, uploads = security_app
    inside = uploads / 'ok.txt'
    inside.write_text('x', encoding='utf-8')
    secret = tmp_path / 'secret.txt'
    secret.write_text('x', encoding='utf-8')
    sibling = tmp_path / 'translated_files' / 'uploads-evil'
    sibling.mkdir()
    (sibling / 'a.txt').write_text('x', encoding='utf-8')
    probes = [
        str(secret),
        str(uploads / '..' / '..' / 'secret.txt'),
        '../../secret.txt',
        str(sibling / 'a.txt'),
        BS * 2 + '?' + BS + str(secret),
        str(secret).replace(BS, '/'),
    ]
    resp = client.post('/api/uploads/verify', json={'file_paths': [str(inside)] + probes})
    body = resp.get_json()
    assert body['existing'] == [str(inside)]
    assert sorted(body['missing']) == sorted(probes)


def test_verify_does_not_follow_a_symlink_out_of_uploads(security_app, tmp_path):
    client, _out, uploads = security_app
    secret = tmp_path / 'secret.txt'
    secret.write_text('x', encoding='utf-8')
    link = uploads / 'link.txt'
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip('symlinks need privileges on this host')
    resp = client.post('/api/uploads/verify', json={'file_paths': [str(link)]})
    assert resp.get_json()['existing'] == []


# ---------------------------------------------------------------------------
# Zip slip through POST /api/upload (EPUB and DOCX)
# ---------------------------------------------------------------------------

def _zip_bytes(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


EPUB_BASE = [('mimetype', 'application/epub+zip'),
             ('META-INF/container.xml', '<container/>'),
             ('OEBPS/content.opf', '<package/>')]
DOCX_BASE = [('[Content_Types].xml', '<Types/>'), ('word/document.xml', '<w:document/>'),
             ('_rels/.rels', '<Relationships/>')]

EVIL_NAMES = ['../../evil.txt', '..' + BS + '..' + BS + 'evil.txt', '/abs/evil.txt',
              'C:/evil.txt', 'C:evil.txt', BS * 2 + 'server' + BS + 'share' + BS + 'evil.txt',
              'OEBPS/../../evil.txt']


@pytest.mark.parametrize('ext, base', [('epub', EPUB_BASE), ('docx', DOCX_BASE)])
@pytest.mark.parametrize('evil', EVIL_NAMES)
def test_upload_refuses_zip_slip_archives(security_app, tmp_path, monkeypatch, ext, base, evil):
    from src.utils.security import RateLimiter
    monkeypatch.setattr(RateLimiter, 'is_allowed', lambda self, ip: True)
    client, out, uploads = security_app
    data = _zip_bytes(base + [(evil, 'pwned')])
    resp = client.post('/api/upload', data={'file': (io.BytesIO(data), f'book.{ext}')},
                       content_type='multipart/form-data')
    assert resp.status_code == 400
    assert not any(p.name == 'evil.txt' for p in tmp_path.rglob('*'))
    assert list(uploads.iterdir()) == []


# ---------------------------------------------------------------------------
# Token gate over the whole application
# ---------------------------------------------------------------------------

@pytest.fixture
def full_app(tmp_path, monkeypatch):
    from flask_socketio import SocketIO
    from src.api.auth import register_auth
    from src.api.routes import configure_routes
    from src.api.websocket import configure_websocket_handlers
    monkeypatch.chdir(tmp_path)
    app = Flask(__name__)
    socketio = SocketIO(app, async_mode='threading')
    register_auth(app)
    state = mock.MagicMock()
    state.server_session_id = '1700000000'
    configure_routes(app, state, str(tmp_path / 'out'), lambda *a, **k: None, socketio)
    configure_websocket_handlers(socketio, state)
    return app, socketio


# Reachable without the token, on purpose (see src/api/auth.py).
PUBLIC_RULES = {'/', '/api/health', '/static/<path:filename>'}


def _fill(rule):
    url = rule.rule
    for arg in rule.arguments:
        url = url.replace(f'<int:{arg}>', '1').replace(f'<path:{arg}>', 'x.txt').replace(f'<{arg}>', 'x')
    return url


def test_every_route_except_the_public_ones_requires_the_token(full_app):
    app, _socketio = full_app
    client = app.test_client()
    ungated = []
    for rule in app.url_map.iter_rules():
        if rule.rule in PUBLIC_RULES:
            continue
        for method in sorted(rule.methods - {'HEAD', 'OPTIONS'}):
            resp = client.open(_fill(rule), method=method)
            if resp.status_code != 401:
                ungated.append((rule.rule, method, resp.status_code))
    assert ungated == []
    public = {r.rule for r in app.url_map.iter_rules()} & PUBLIC_RULES
    assert public == PUBLIC_RULES


def test_every_route_lives_under_api_or_is_public(full_app):
    """The gate only looks at '/api/' paths: a new blueprint mounted
    elsewhere would be silently public."""
    app, _socketio = full_app
    outside = {r.rule for r in app.url_map.iter_rules() if not r.rule.startswith('/api/')}
    assert outside <= PUBLIC_RULES


def test_socketio_only_has_the_gated_connect_handler(full_app):
    from src.api.auth import API_TOKEN
    app, socketio = full_app
    if isinstance(socketio, mock.Mock):
        # tests/unit/test_state_management.py replaces sys.modules['flask_socketio']
        # with a MagicMock at import time for the rest of the session.
        pytest.skip('flask_socketio is stubbed by another test module in this session')
    handlers = {ns: set(evs) for ns, evs in socketio.server.handlers.items()}
    assert handlers == {'/': {'connect', 'disconnect'}}, (
        'a new Socket.IO handler must re-check the token or rely on the connect gate')
    assert not socketio.test_client(app).is_connected()
    assert not socketio.test_client(app, auth={'token': 'wrong'}).is_connected()
    assert socketio.test_client(app, auth={'token': API_TOKEN}).is_connected()


# ---------------------------------------------------------------------------
# POST /api/settings writes the .env of the working directory, atomically
# ---------------------------------------------------------------------------

def test_settings_write_goes_to_the_working_directory_env(tmp_path, monkeypatch):
    from src.api.blueprints import config_routes
    repo_env = Path(config_routes.__file__).resolve().parents[3] / '.env'
    repo_env_before = repo_env.read_bytes() if repo_env.exists() else None
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_routes, 'reload_config', lambda: None)
    app = Flask(__name__)
    app.register_blueprint(config_routes.create_config_blueprint())
    resp = app.test_client().post('/api/settings', json={'GEMINI_MODEL': 'model-verif'})
    assert resp.status_code == 200
    assert 'GEMINI_MODEL=model-verif' in (tmp_path / '.env').read_text(encoding='utf-8')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['.env']  # no temp file left behind
    after = repo_env.read_bytes() if repo_env.exists() else None
    assert after == repo_env_before


def test_settings_strip_line_breaks_so_no_extra_key_is_injected(tmp_path, monkeypatch):
    from src.api.blueprints import config_routes
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_routes, 'reload_config', lambda: None)
    app = Flask(__name__)
    app.register_blueprint(config_routes.create_config_blueprint())
    app.test_client().post('/api/settings', json={
        'GEMINI_MODEL': 'm\nOPENAI_API_ENDPOINT=https://attacker.example.com/v1',
        'POE_MODEL': 'p\rLLM_ENDPOINT_ALLOWLIST=attacker.example.com'})
    keys = [line.split('=', 1)[0] for line in (tmp_path / '.env').read_text(encoding='utf-8').splitlines()
            if line and not line.startswith('#')]
    assert 'OPENAI_API_ENDPOINT' not in keys and 'LLM_ENDPOINT_ALLOWLIST' not in keys
