"""Regression tests: request size cap, single containment check, verify scope.

There was no MAX_CONTENT_LENGTH; path containment was implemented four
different ways, two with a string startswith that treats '/uploads-evil' as
inside '/uploads'; and /api/uploads/verify told any caller whether an
arbitrary server path exists.
"""
import pytest
from flask import Flask, request

from src.api import routes
from src.api.blueprints.security_routes import create_security_blueprint
from src.api.services.file_service import FileService
from src.api.services.path_validator import PathValidator
from src.utils.security import SecureFileHandler, SecurityError, is_within_directory


# ---------------------------------------------------------------------------
# MAX_CONTENT_LENGTH
# ---------------------------------------------------------------------------

def test_default_request_cap_is_200_mb(monkeypatch):
    monkeypatch.delenv('MAX_REQUEST_SIZE_MB', raising=False)
    assert routes.max_request_size_bytes() == 200 * 1024 * 1024


@pytest.mark.parametrize('raw, expected_mb', [('500', 500), ('abc', 200), ('0', 200), ('-5', 200)])
def test_request_cap_is_configurable(monkeypatch, raw, expected_mb):
    monkeypatch.setenv('MAX_REQUEST_SIZE_MB', raw)
    assert routes.max_request_size_bytes() == expected_mb * 1024 * 1024


def test_oversized_request_gets_413(monkeypatch):
    monkeypatch.setenv('MAX_REQUEST_SIZE_MB', '1')
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = routes.max_request_size_bytes()
    routes._register_error_handlers(app)

    @app.route('/echo', methods=['POST'])
    def echo():
        return {'size': len(request.get_data())}

    client = app.test_client()
    assert client.post('/echo', data=b'x' * 1024).status_code == 200
    resp = client.post('/echo', data=b'x' * (2 * 1024 * 1024))
    assert resp.status_code == 413
    assert resp.get_json()['error'] == 'Request too large'


# ---------------------------------------------------------------------------
# One containment check
# ---------------------------------------------------------------------------

def test_sibling_prefix_directory_is_not_inside(tmp_path):
    uploads = tmp_path / 'uploads'
    evil = tmp_path / 'uploads-evil'
    uploads.mkdir()
    evil.mkdir()
    target = evil / 'secret.txt'
    target.write_text('x', encoding='utf-8')

    assert not is_within_directory(target, uploads)
    assert not PathValidator.is_within_directory(target, uploads)
    assert is_within_directory(uploads / 'a.txt', uploads)


def test_delete_uploaded_file_refuses_sibling_prefix_dir(tmp_path):
    out = tmp_path / 'out'
    (out / 'uploads').mkdir(parents=True)
    sibling = out / 'uploads-evil'
    sibling.mkdir()
    victim = sibling / 'keep.txt'
    victim.write_text('x', encoding='utf-8')

    ok, message = FileService(str(out)).delete_uploaded_file(str(victim))
    assert not ok and 'Security' in message
    assert victim.exists()


def test_secure_file_handler_path_check(tmp_path):
    handler = SecureFileHandler(tmp_path / 'uploads')
    with pytest.raises(SecurityError):
        handler._get_secure_path('../uploads-evil/x.txt')
    assert handler._get_secure_path('a.txt').name == 'a.txt'


# ---------------------------------------------------------------------------
# /api/uploads/verify and /api/detect-language
# ---------------------------------------------------------------------------

@pytest.fixture
def security_client(tmp_path):
    app = Flask(__name__)
    app.register_blueprint(create_security_blueprint(str(tmp_path)))
    uploads = tmp_path / 'uploads'
    uploads.mkdir(exist_ok=True)
    with app.test_client() as client:
        yield client, tmp_path, uploads


def test_verify_only_answers_for_the_uploads_directory(security_client):
    client, root, uploads = security_client
    inside = uploads / 'book.txt'
    inside.write_text('x', encoding='utf-8')
    outside = root / 'outside.txt'
    outside.write_text('x', encoding='utf-8')
    sibling = root / 'uploads-evil'
    sibling.mkdir()
    (sibling / 'x.txt').write_text('x', encoding='utf-8')

    paths = [str(inside), str(outside), str(sibling / 'x.txt'),
             str(uploads / '..' / 'outside.txt'), 42, '']
    resp = client.post('/api/uploads/verify', json={'file_paths': paths})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body['existing'] == [str(inside)]
    assert len(body['missing']) == len(paths) - 1


def test_detect_language_checks_containment_before_existence(security_client):
    client, root, _uploads = security_client
    resp_missing = client.post('/api/detect-language',
                               json={'file_path': str(root / 'nope.txt')})
    existing = root / 'present.txt'
    existing.write_text('Hello world, this is English.', encoding='utf-8')
    resp_present = client.post('/api/detect-language', json={'file_path': str(existing)})
    # Same answer whether or not the outside file exists.
    assert resp_missing.status_code == resp_present.status_code == 403
