"""
Regression tests for the per-session API token gate (issue #210).

Before this layer the server enabled wildcard CORS and had no authentication,
so any website the user visited could read and trigger the local API. These
tests pin the new behavior: /api/ routes require a valid token, while the page
and static assets stay public.
"""
import pytest
from flask import Flask, jsonify

from src.api import auth
from src.api.auth import register_auth, is_authorized, API_TOKEN


@pytest.fixture
def client():
    app = Flask(__name__)
    register_auth(app)

    @app.route('/')
    def serve_interface():
        return "page"

    @app.route('/api/secret')
    def secret():
        return jsonify({"ok": True})

    @app.route('/api/danger', methods=['DELETE'])
    def danger():
        return jsonify({"deleted": True})

    # The real page endpoint is named config.serve_interface; the gate exempts
    # that name. Mirror it here so the exemption is exercised.
    app.view_functions['config.serve_interface'] = serve_interface
    app.add_url_rule('/page', endpoint='config.serve_interface')

    # /api/health is the unauthenticated liveness probe (config.health_check),
    # hit by the Docker HEALTHCHECK and CI with no token. Mirror its endpoint
    # name so the exemption is exercised through the /api/ prefix.
    def health_check():
        return jsonify({"status": "ok"})

    app.view_functions['config.health_check'] = health_check
    app.add_url_rule('/api/health', endpoint='config.health_check')

    with app.test_client() as c:
        yield c


def test_api_route_without_token_is_rejected(client):
    resp = client.get('/api/secret')
    assert resp.status_code == 401
    assert resp.get_json()['code'] == 'missing_or_invalid_token'


def test_api_route_with_bad_token_is_rejected(client):
    resp = client.get('/api/secret', headers={'X-API-Token': 'nope'})
    assert resp.status_code == 401


def test_api_route_with_valid_header_token_passes(client):
    resp = client.get('/api/secret', headers={'X-API-Token': API_TOKEN})
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}


def test_api_route_with_valid_query_token_passes(client):
    resp = client.get(f'/api/secret?token={API_TOKEN}')
    assert resp.status_code == 200


def test_state_changing_route_is_gated(client):
    assert client.delete('/api/danger').status_code == 401
    assert client.delete(
        '/api/danger', headers={'X-API-Token': API_TOKEN}
    ).status_code == 200


def test_page_endpoint_is_exempt(client):
    assert client.get('/page').status_code == 200


def test_non_api_path_is_not_gated(client):
    # The root page carries no /api/ prefix and must remain reachable.
    assert client.get('/').status_code == 200


def test_health_endpoint_is_exempt(client):
    # The liveness probe must answer without a token: the Docker HEALTHCHECK
    # and CI smoke test call it unauthenticated, and it exposes no secrets.
    resp = client.get('/api/health')
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_is_authorized_constant_time_compare():
    assert is_authorized(API_TOKEN) is True
    assert is_authorized('') is False
    assert is_authorized(None) is False
    assert is_authorized(API_TOKEN + 'x') is False
