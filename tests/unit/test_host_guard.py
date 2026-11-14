"""
Regression tests for the DNS-rebinding Host-header guard (src/api/host_guard.py).

A DNS-rebinding page reaches the loopback server under the attacker's domain
name, so the Host header is the only signal left. These tests pin the policy:
a loopback bind accepts only loopback names plus ALLOWED_HOSTS, a non-loopback
bind (Docker / LAN) is unchecked unless ALLOWED_HOSTS is set, and the check
covers both Flask routes and the /socket.io/ handshake.
"""
import pytest
from flask import Flask
from flask_socketio import SocketIO

from src.api.host_guard import (
    is_loopback_bind,
    parse_host,
    register_host_guard,
)

SOCKETIO_HANDSHAKE = '/socket.io/?EIO=4&transport=polling'


def _make_app(bind_host, allowed_hosts=''):
    app = Flask(__name__)
    socketio = SocketIO(app, async_mode='threading')

    @app.route('/')
    def index():
        return 'page'

    @socketio.on('connect')
    def _connect(auth=None):
        return True

    register_host_guard(app, bind_host, allowed_hosts)
    return app


def _get(app, path, host):
    return app.test_client().get(path, headers={'Host': host})


@pytest.mark.parametrize('path', ['/', SOCKETIO_HANDSHAKE])
def test_loopback_bind_rejects_foreign_host(path):
    app = _make_app('127.0.0.1')
    resp = _get(app, path, 'evil.example.com')
    assert resp.status_code == 403
    assert resp.mimetype == 'text/plain'
    assert b'ALLOWED_HOSTS' in resp.data


@pytest.mark.parametrize('path', ['/', SOCKETIO_HANDSHAKE])
def test_loopback_bind_rejects_foreign_host_with_port(path):
    app = _make_app('127.0.0.1')
    assert _get(app, path, 'evil.example.com:5000').status_code == 403


def test_rejection_is_logged_with_the_host(caplog):
    app = _make_app('127.0.0.1')
    with caplog.at_level('WARNING', logger='src.api.host_guard'):
        _get(app, '/', 'evil.example.com')
    assert any('evil.example.com' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('bind', ['127.0.0.1', 'localhost', '::1', '[::1]'])
@pytest.mark.parametrize('host', [
    'localhost:5000', '127.0.0.1:5000', '[::1]:5000',
    'localhost', 'LOCALHOST:5000', 'localhost.:5000', '[::1]',
])
@pytest.mark.parametrize('path', ['/', SOCKETIO_HANDSHAKE])
def test_loopback_bind_allows_loopback_names(bind, host, path):
    app = _make_app(bind)
    assert _get(app, path, host).status_code == 200


def test_flask_test_client_default_host_is_allowed():
    app = _make_app('127.0.0.1')
    assert app.test_client().get('/').status_code == 200


def test_loopback_bind_rejects_missing_host():
    app = _make_app('127.0.0.1')
    environ = {
        'REQUEST_METHOD': 'GET', 'PATH_INFO': '/', 'SERVER_NAME': 'localhost',
        'SERVER_PORT': '5000', 'wsgi.url_scheme': 'http',
    }
    status = []
    body = app.wsgi_app(environ, lambda s, h: status.append(s))
    assert status == ['403 Forbidden']
    assert b''.join(body).startswith(b'Forbidden')


@pytest.mark.parametrize('host', ['', '::1', '[::1', 'localhost:abc', 'a b'])
def test_loopback_bind_rejects_malformed_host(host):
    app = _make_app('127.0.0.1')
    assert _get(app, '/', host).status_code == 403


def test_allowed_hosts_extends_loopback_bind():
    app = _make_app('127.0.0.1', 'myhost.lan, other.example')
    assert _get(app, '/', 'myhost.lan:5000').status_code == 200
    assert _get(app, SOCKETIO_HANDSHAKE, 'myhost.lan:5000').status_code == 200
    assert _get(app, '/', 'MyHost.Lan.').status_code == 200
    assert _get(app, '/', 'other.example').status_code == 200
    assert _get(app, '/', 'evil.example.com').status_code == 403


def test_allowed_hosts_entry_with_port_pins_the_port():
    app = _make_app('127.0.0.1', 'myhost.lan:8080')
    assert _get(app, '/', 'myhost.lan:8080').status_code == 200
    assert _get(app, '/', 'myhost.lan:5000').status_code == 403


@pytest.mark.parametrize('path', ['/', SOCKETIO_HANDSHAKE])
@pytest.mark.parametrize('host', ['evil.example.com', '192.168.1.20:5000', 'localhost:5000'])
def test_wildcard_bind_without_allowed_hosts_allows_any_host(path, host):
    app = _make_app('0.0.0.0')
    assert _get(app, path, host).status_code == 200


@pytest.mark.parametrize('path', ['/', SOCKETIO_HANDSHAKE])
def test_wildcard_bind_with_allowed_hosts_enforces_it(path):
    app = _make_app('0.0.0.0', 'myhost.lan')
    assert _get(app, path, 'myhost.lan:5000').status_code == 200
    assert _get(app, path, 'localhost:5000').status_code == 200
    assert _get(app, path, 'evil.example.com').status_code == 403


@pytest.mark.parametrize('value, expected', [
    ('LocalHost:5000', ('localhost', '5000')),
    ('[::1]:5000', ('::1', '5000')),
    ('[::1]', ('::1', None)),
    ('example.com.', ('example.com', None)),
    ('', None),
    (None, None),
    ('::1', None),
    ('host:', None),
    ('[::1]x', None),
])
def test_parse_host(value, expected):
    assert parse_host(value) == expected


@pytest.mark.parametrize('bind, expected', [
    ('127.0.0.1', True), ('localhost', True), ('::1', True), ('[::1]', True),
    ('0.0.0.0', False), ('', False), ('192.168.1.20', False), ('::', False),
])
def test_is_loopback_bind(bind, expected):
    assert is_loopback_bind(bind) is expected


def test_translation_api_installs_the_guard():
    """The real app must be wrapped, outside the Socket.IO middleware."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / 'translation_api.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    calls = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert 'register_host_guard' in calls
    assert source.index('register_host_guard(app') > source.index('SocketIO(app')
