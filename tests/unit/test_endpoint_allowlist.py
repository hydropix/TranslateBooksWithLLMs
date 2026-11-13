"""Unit tests for the LLM endpoint allowlist and the endpoint/key pairing rule.

Two guards are covered:

1. `EndpointValidator` — a request-supplied endpoint must be loopback/private
   or an allowlisted host.
2. The pairing rule in `translation_routes` — an endpoint that differs from the
   server default must carry its own API key, so the `.env` key never travels
   to a host the request chose.
"""
import socket

import pytest
from flask import Flask

import src.config as _config
from src.api.api_keys import resolve_api_key
from src.api.blueprints.translation_routes import create_translation_blueprint
from src.api.services import endpoint_validator as _endpoint_validator
from src.api.services.endpoint_validator import EndpointValidator

# Names the stub resolver knows about. Everything else is treated as NXDOMAIN,
# which is what a public host that nobody points at a LAN box behaves like.
_FAKE_DNS = {
    'ai-server.example.com': ['192.168.1.50'],
    'llm.example.org': ['10.2.3.4', '10.2.3.5'],
    'split.example.net': ['10.0.0.9', '93.184.216.34'],  # private + public
    'public.example.com': ['93.184.216.34'],
}


@pytest.fixture(autouse=True)
def stub_resolver(monkeypatch):
    """Keep the resolution fallback hermetic: no test may touch real DNS."""
    def fake_getaddrinfo(host, *_args, **_kwargs):
        addresses = _FAKE_DNS.get((host or '').lower())
        if not addresses:
            raise socket.gaierror(-2, 'Name or service not known')
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (a, 0))
                for a in addresses]

    monkeypatch.setattr(socket, 'getaddrinfo', fake_getaddrinfo)
    _endpoint_validator._resolution_cache.clear()
    yield
    _endpoint_validator._resolution_cache.clear()

# Inert placeholder — never a real credential.
FAKE_KEY = 'sk-xxxxxxxx'
OPENAI_DEFAULT_ENDPOINT = 'https://api.openai.com/v1/chat/completions'
# Allowlisted by the subdomain rule, but not the configured default.
OPENAI_ALT_ENDPOINT = 'https://eu.api.openai.com/v1/chat/completions'


# ---------------------------------------------------------------------------
# EndpointValidator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('endpoint', [None, '', '   '])
def test_absent_endpoint_is_accepted(endpoint):
    """No endpoint means 'use the server default', which is not an override."""
    assert EndpointValidator.validate(endpoint) == (True, None)


@pytest.mark.parametrize('endpoint', [
    'http://localhost:11434/api/generate',
    'http://127.0.0.1:1234/v1/chat/completions',
    'http://192.168.1.50:11434/api/generate',
    'http://host.docker.internal:11434/api/generate',
    'http://10.0.0.7:8080/v1/chat/completions',
    'http://[::1]:11434/api/generate',
    'http://ollama.localhost/api/generate',
])
def test_local_and_private_endpoints_are_allowed(endpoint):
    assert EndpointValidator.validate(endpoint) == (True, None)


@pytest.mark.parametrize('endpoint', [
    # A container / LXC / compose service name: no dot, so only an internal
    # resolver can answer it.
    'http://ollama:11434/api/generate',
    'http://nas:1234/v1/chat/completions',
    # Names a home router or an mDNS responder hands out.
    'http://ollama.local:11434/api/generate',
    'http://ollama.local.:11434/api/generate',  # trailing dot = FQDN form
    'http://nas.lan:11434/api/generate',
    'http://pve.home.arpa:11434/api/generate',
    'http://gpu.home:11434/api/generate',
    'http://llm.internal:11434/api/generate',
    'http://llm.intranet:11434/api/generate',
    'http://llm.corp:11434/api/generate',
    'http://llm.private:11434/api/generate',
    # Tailscale: MagicDNS name, shared-address-space IPv4 (RFC 6598), and the
    # tailnet's IPv6 range.
    'http://ollama.tail1234.ts.net:11434/api/generate',
    'http://100.101.102.103:11434/api/generate',
    'http://[fd7a:115c:a1e0::1]:11434/api/generate',
])
def test_private_network_hostnames_are_allowed(endpoint):
    """Issue #263: a self-hosted backend reached by name, not by literal IP.

    v1.5.0 accepted only literal private addresses, which broke every LAN,
    container and tailnet deployment that addresses its LLM by hostname.
    """
    assert EndpointValidator.validate(endpoint) == (True, None)


@pytest.mark.parametrize('endpoint', [
    'http://8.8.8.8/api/generate',
    'https://ollama.example.com/api/generate',
    'https://llm.evil.io:11434/api/generate',
    'https://public.example.com/api/generate',      # resolves, but to a public IP
    'https://split.example.net/api/generate',       # one private answer is not enough
])
def test_public_hosts_stay_rejected(endpoint):
    """Widening the local rules must not open the public internet."""
    ok, message = EndpointValidator.validate(endpoint)
    assert ok is False
    assert 'LLM_ENDPOINT_ALLOWLIST' in message


@pytest.mark.parametrize('endpoint', [
    'http://ai-server.example.com:11434/api/generate',
    'http://llm.example.org:11434/api/generate',
])
def test_hostname_resolving_to_the_lan_is_allowed(endpoint):
    """Issue #263: a LAN box named under a domain the operator owns.

    Nothing about the name says "local", so only resolution can tell. The
    endpoint is reachable and private, so the job must be allowed to start.
    """
    assert EndpointValidator.validate(endpoint) == (True, None)


def test_resolution_is_only_a_last_resort(monkeypatch):
    """The fast paths must never wait on a resolver."""
    def explode(*_args, **_kwargs):
        raise AssertionError('resolver must not be consulted')

    monkeypatch.setattr(socket, 'getaddrinfo', explode)
    for endpoint in (
        'http://localhost:11434/api/generate',        # syntactic local rule
        'http://192.168.1.50:11434/api/generate',     # literal private address
        'http://ollama.local:11434/api/generate',     # private-network suffix
        'https://api.openai.com/v1/chat/completions',  # allowlisted host
        'ftp://api.openai.com/x',                     # rejected before any lookup
    ):
        EndpointValidator.validate(endpoint)


def test_resolution_verdict_is_cached():
    """A polled endpoint must not trigger a lookup per request."""
    calls = []
    real = socket.getaddrinfo

    def counting(host, *args, **kwargs):
        calls.append(host)
        return real(host, *args, **kwargs)

    socket.getaddrinfo = counting
    try:
        for _ in range(3):
            assert EndpointValidator.validate(
                'http://ai-server.example.com:11434/api/generate') == (True, None)
    finally:
        socket.getaddrinfo = real
    assert len(calls) == 1


def test_unresolvable_host_is_rejected_without_raising():
    ok, message = EndpointValidator.validate('http://does-not-exist.example.com/x')
    assert ok is False
    assert 'LLM_ENDPOINT_ALLOWLIST' in message


@pytest.mark.parametrize('endpoint', [
    OPENAI_DEFAULT_ENDPOINT,
    OPENAI_ALT_ENDPOINT,  # subdomain rule
    'https://openrouter.ai/api/v1/chat/completions',
    'https://integrate.api.nvidia.com/v1/chat/completions',
])
def test_known_provider_hosts_are_allowed(endpoint):
    assert EndpointValidator.validate(endpoint) == (True, None)


def test_unknown_public_host_is_rejected():
    ok, message = EndpointValidator.validate('https://evil.example.com/v1/chat/completions')
    assert ok is False
    assert 'evil.example.com' in message
    assert 'LLM_ENDPOINT_ALLOWLIST' in message


def test_non_http_scheme_is_rejected():
    ok, message = EndpointValidator.validate('ftp://api.openai.com/x')
    assert ok is False
    assert 'http or https' in message


def test_embedded_credentials_are_rejected():
    ok, message = EndpointValidator.validate('https://user:pw@api.openai.com/v1')
    assert ok is False
    assert 'credentials' in message


def test_env_allowlist_extends_accepted_hosts(monkeypatch):
    """The allowlist is read at call time so reload_config() is honoured."""
    rejected, _ = EndpointValidator.validate('https://llm.internal.example/v1')
    assert rejected is False

    monkeypatch.setattr(_config, 'LLM_ENDPOINT_ALLOWLIST', ('llm.internal.example',))
    assert EndpointValidator.validate('https://llm.internal.example/v1') == (True, None)
    # Subdomains of an allowlisted host are covered too.
    assert EndpointValidator.validate('https://eu.llm.internal.example/v1') == (True, None)


def test_configured_server_default_is_always_allowed(monkeypatch):
    monkeypatch.setattr(_config, 'OPENAI_API_ENDPOINT', 'https://gateway.example.org/v1/chat/completions')
    assert EndpointValidator.validate('https://gateway.example.org/v1/chat/completions') == (True, None)


# ---------------------------------------------------------------------------
# resolve_api_key(allow_env_fallback=...)
# ---------------------------------------------------------------------------

def test_env_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', FAKE_KEY)
    assert resolve_api_key('__USE_ENV__', 'OPENAI_API_KEY', allow_env_fallback=False) == ''
    assert resolve_api_key('', 'OPENAI_API_KEY', allow_env_fallback=False) == ''


def test_env_fallback_is_on_by_default(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', FAKE_KEY)
    assert resolve_api_key('__USE_ENV__', 'OPENAI_API_KEY') == FAKE_KEY


def test_explicit_key_survives_disabled_fallback(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
    assert resolve_api_key(FAKE_KEY, 'OPENAI_API_KEY', allow_env_fallback=False) == FAKE_KEY


# ---------------------------------------------------------------------------
# POST /api/translate
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


def test_disallowed_endpoint_is_rejected_before_job_creation(translate_app):
    client, state_manager, started = translate_app
    resp = client.post('/api/translate', json=_payload(
        llm_provider='openai',
        llm_api_endpoint='https://evil.example.com/v1/chat/completions',
        openai_api_key=FAKE_KEY,
    ))
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'Endpoint not allowed'
    assert state_manager.created == []
    assert started == []


def test_endpoint_override_without_own_key_is_rejected(translate_app, monkeypatch):
    client, state_manager, _started = translate_app
    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='openai',
        llm_api_endpoint=OPENAI_ALT_ENDPOINT,
        openai_api_key='__USE_ENV__',
    ))
    assert resp.status_code == 400
    # The allowlist accepted the host; it is the pairing rule that refused.
    assert resp.get_json()['error'] == 'Endpoint override requires its own API key'
    assert state_manager.created == []


def test_endpoint_override_with_own_key_is_accepted(translate_app, monkeypatch):
    client, state_manager, _started = translate_app
    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='openai',
        llm_api_endpoint=OPENAI_ALT_ENDPOINT,
        openai_api_key=FAKE_KEY,
    ))
    assert resp.status_code == 200
    assert len(state_manager.created) == 1
    _translation_id, config = state_manager.created[0]
    assert config['openai_api_key'] == FAKE_KEY


def test_gemini_endpoint_field_is_inert(translate_app, monkeypatch):
    """Gemini ignores llm_api_endpoint, so the pairing guard must not fire."""
    client, state_manager, _started = translate_app
    monkeypatch.setenv('GEMINI_API_KEY', 'env-gemini-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='gemini',
        model='gemini-2.0-flash',
        llm_api_endpoint=OPENAI_ALT_ENDPOINT,
        gemini_api_key='__USE_ENV__',
    ))
    assert resp.status_code == 200
    assert len(state_manager.created) == 1


def test_nim_endpoint_field_is_inert(translate_app, monkeypatch):
    """NIM ignores llm_api_endpoint: create_llm_client() builds it without one.

    The frontend sends the Ollama endpoint field unconditionally, so a stale
    value used to trigger the pairing rule and reject valid NIM jobs with
    "Endpoint override requires its own API key" — over a value that never
    reached an HTTP client.
    """
    client, state_manager, _started = translate_app
    monkeypatch.setenv('NIM_API_KEY', 'env-nim-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='nim',
        model='nvidia/nemotron-3-ultra-550b-a55b',
        llm_api_endpoint='http://localhost:11434/api/generate',
        nim_api_key='__USE_ENV__',
    ))
    assert resp.status_code == 200
    assert len(state_manager.created) == 1
    _translation_id, config = state_manager.created[0]
    assert config['llm_provider'] == 'nim'


@pytest.mark.parametrize('endpoint', [
    'https://ollama.example.com/api/generate',  # not allowlisted
    'ollama.local:11434',                       # not even a URL
])
def test_inert_endpoint_field_never_blocks_a_cloud_job(translate_app, monkeypatch, endpoint):
    """Issue #263: the frontend sends llm_api_endpoint for every provider.

    Gemini never reads it, so a stale value in the Ollama field must not fail
    the job before it starts. This is the path where the UI looked healthy
    (the model list comes from the provider API) yet the translation 400'd.
    """
    client, state_manager, _started = translate_app
    monkeypatch.setenv('GEMINI_API_KEY', 'env-gemini-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='gemini',
        model='gemini-2.0-flash',
        llm_api_endpoint=endpoint,
        gemini_api_key='__USE_ENV__',
    ))
    assert resp.status_code == 200
    assert len(state_manager.created) == 1


def test_rejected_endpoint_reply_carries_an_actionable_message(translate_app):
    """The label alone is unactionable, so the reason must travel with it."""
    client, _state_manager, _started = translate_app
    resp = client.post('/api/translate', json=_payload(
        llm_provider='ollama',
        model='qwen3:14b',
        llm_api_endpoint='https://ollama.example.com/api/generate',
    ))
    assert resp.status_code == 400
    body = resp.get_json()
    assert body['error'] == 'Endpoint not allowed'
    assert 'ollama.example.com' in body['message']
    assert 'LLM_ENDPOINT_ALLOWLIST' in body['message']


def test_hostname_ollama_endpoint_starts_a_job(translate_app):
    """The exact configuration reported in issue #263."""
    client, state_manager, started = translate_app
    resp = client.post('/api/translate', json=_payload(
        llm_provider='ollama',
        model='qwen3:14b',
        llm_api_endpoint='http://ollama.local:11434/api/generate',
    ))
    assert resp.status_code == 200
    assert len(state_manager.created) == 1
    assert len(started) == 1


def test_default_ollama_endpoint_needs_no_key(translate_app):
    """The common local path must not regress."""
    client, state_manager, _started = translate_app
    resp = client.post('/api/translate', json=_payload(
        llm_provider='ollama',
        model='qwen3:14b',
        llm_api_endpoint=_config.API_ENDPOINT,
    ))
    assert resp.status_code == 200
    assert len(state_manager.created) == 1


def test_keyless_local_openai_server_is_accepted(translate_app, monkeypatch):
    """LM Studio / llama.cpp / vLLM run as provider 'openai' with no stored key.

    Nothing can leak, so the pairing rule must not fire.
    """
    client, state_manager, _started = translate_app
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    resp = client.post('/api/translate', json=_payload(
        llm_provider='openai',
        model='qwen3-14b',
        llm_api_endpoint='http://localhost:1234/v1/chat/completions',
        openai_api_key='__USE_ENV__',
    ))
    assert resp.status_code == 200
    _translation_id, config = state_manager.created[0]
    assert config['openai_api_key'] == ''


def test_stored_key_never_reaches_a_local_override(translate_app, monkeypatch):
    """A private-range host is allowlisted, so the pairing rule is the only
    thing standing between the .env key and the operator's LAN."""
    client, state_manager, _started = translate_app
    monkeypatch.setenv('OPENAI_API_KEY', 'env-only-key')
    resp = client.post('/api/translate', json=_payload(
        llm_provider='openai',
        llm_api_endpoint='http://192.168.1.50:1234/v1/chat/completions',
        openai_api_key='__USE_ENV__',
    ))
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'Endpoint override requires its own API key'
    assert state_manager.created == []


def test_custom_ollama_endpoint_needs_no_key(translate_app):
    """Ollama has no stored credential to leak, so an override is fine."""
    client, state_manager, _started = translate_app
    resp = client.post('/api/translate', json=_payload(
        llm_provider='ollama',
        model='qwen3:14b',
        llm_api_endpoint='http://192.168.1.50:11434/api/generate',
    ))
    assert resp.status_code == 200
    assert len(state_manager.created) == 1
