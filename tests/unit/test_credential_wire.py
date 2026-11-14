"""Wire-level checks: the server's stored key never reaches a request-chosen host.

tests/unit/test_credential_guard.py and test_endpoint_allowlist.py pin the
guard's *decision* (400 or accepted). These tests drive every HTTP entry point
that lets the request pick an LLM host all the way to the outgoing HTTP
request, through a recording fake transport, while a stored key sits in the
environment, and assert on the request that would hit the wire:

- GET /api/models (openai listing, uses `requests`)
- POST /api/glossaries/<gid>/suggest-terms (NER, provider via httpx)
- POST /api/custom-instructions/extract-style (style extraction, httpx)
- sample routes: the per-column provider builder (httpx)
- POST /api/translate: the JobConfig it builds, turned into a provider (httpx)

No network: httpx.AsyncClient gets a MockTransport, requests.get is replaced
and DNS resolution raises.
"""
import io
import json
import os
import socket
import tempfile

import httpx
import pytest
from flask import Flask

import src.config as _config
from src.api.blueprints import config_routes
from src.api.blueprints import custom_instruction_routes as cir
from src.api.blueprints import sample_routes
from src.api.blueprints.glossary_routes import create_glossary_blueprint
from src.api.services import endpoint_validator as _endpoint_validator

STORED_KEY = 'sk-DUMMYKEY-verif-1234'
OWN_KEY = 'sk-xxxxxxxx-own'
OPENAI_DEFAULT = 'https://api.openai.com/v1/chat/completions'
OLLAMA_DEFAULT = 'http://localhost:11434/api/generate'
LAN_OPENAI = 'http://192.168.1.50:1234/v1/chat/completions'
LAN_OLLAMA = 'http://192.168.1.60:11434/api/generate'
EVIL = 'https://attacker.example.com/v1/chat/completions'

NER_REPLY = json.dumps({"terms": []})
CHAT_REPLY = {
    "id": "x", "object": "chat.completion",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant",
                             "content": "<TRANSLATION>ok</TRANSLATION>" + NER_REPLY}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


class Wire:
    """Every outgoing request, whichever HTTP library sent it."""

    def __init__(self):
        self.requests = []

    def record(self, method, url, headers, body):
        self.requests.append({'method': method, 'url': str(url),
                              'headers': {k.lower(): v for k, v in (headers or {}).items()},
                              'body': body or ''})

    def httpx_handler(self, request):
        self.record(request.method, request.url, dict(request.headers),
                    request.content.decode('utf-8', 'replace'))
        path = request.url.path
        if path.endswith('/api/generate'):
            return httpx.Response(200, json={"response": "<TRANSLATION>ok</TRANSLATION>", "done": True})
        if path.endswith('/api/chat'):
            return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})
        if request.method == 'GET':
            return httpx.Response(200, json={"data": [], "models": []})
        return httpx.Response(200, json=CHAT_REPLY)

    def leaked(self, key=STORED_KEY):
        """Requests carrying ``key`` anywhere (URL, headers, body)."""
        return [r for r in self.requests
                if key in r['url'] or key in r['body'] or any(key in v for v in r['headers'].values())]

    def hosts(self):
        return {httpx.URL(r['url']).host for r in self.requests}


@pytest.fixture
def wire(monkeypatch):
    rec = Wire()
    real_async_client = httpx.AsyncClient

    class RecordingAsyncClient(real_async_client):
        def __init__(self, *args, **kwargs):
            kwargs['transport'] = httpx.MockTransport(rec.httpx_handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, 'AsyncClient', RecordingAsyncClient)

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"data": [{"id": "m1"}]}

    def fake_requests_get(url, headers=None, **_kwargs):
        rec.record('GET', url, headers, '')
        return _Resp()

    monkeypatch.setattr(config_routes.requests, 'get', fake_requests_get)
    return rec


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    def no_dns(*_args, **_kwargs):
        raise socket.gaierror(-2, 'Name or service not known')

    monkeypatch.setattr(socket, 'getaddrinfo', no_dns)
    _endpoint_validator._resolution_cache.clear()
    monkeypatch.setattr(_config, 'OPENAI_API_ENDPOINT', OPENAI_DEFAULT)
    monkeypatch.setattr(_config, 'OLLAMA_API_ENDPOINT', OLLAMA_DEFAULT)
    monkeypatch.setattr(_config, 'API_ENDPOINT', OLLAMA_DEFAULT)
    monkeypatch.setattr(_config, 'LLM_ENDPOINT_ALLOWLIST', ())
    monkeypatch.setenv('OPENAI_API_KEY', STORED_KEY)
    monkeypatch.setenv('NIM_API_KEY', STORED_KEY)
    monkeypatch.setattr(_config, 'OPENAI_API_KEY', STORED_KEY)
    monkeypatch.setattr(_config, 'NIM_API_KEY', STORED_KEY)
    yield
    _endpoint_validator._resolution_cache.clear()


# ---------------------------------------------------------------------------
# GET /api/models (openai listing goes through `requests`)
# ---------------------------------------------------------------------------

@pytest.fixture
def models_client():
    app = Flask(__name__)
    app.register_blueprint(config_routes.create_config_blueprint())
    with app.test_client() as client:
        yield client


@pytest.mark.parametrize('sent_key', [None, '', '__USE_ENV__'])
def test_models_listing_on_lan_host_never_gets_the_stored_key(models_client, wire, sent_key):
    query = {'provider': 'openai', 'api_endpoint': LAN_OPENAI}
    if sent_key is not None:
        query['api_key'] = sent_key
    models_client.get('/api/models', query_string=query)
    assert wire.requests, 'the listing should reach the LAN server'
    assert wire.hosts() == {'192.168.1.50'}
    assert wire.leaked() == []


def test_models_listing_on_foreign_host_sends_nothing(models_client, wire):
    resp = models_client.get('/api/models', query_string={
        'provider': 'openai', 'api_endpoint': EVIL, 'api_key': '__USE_ENV__'})
    assert resp.status_code == 400
    assert wire.requests == []


def test_models_listing_default_host_keeps_working(models_client, wire):
    models_client.get('/api/models', query_string={'provider': 'openai', 'api_key': '__USE_ENV__'})
    assert wire.hosts() == {'api.openai.com'}
    assert wire.requests[0]['headers'].get('authorization') == f'Bearer {STORED_KEY}'


def test_ollama_models_listing_rejects_foreign_host(models_client, wire):
    resp = models_client.get('/api/models', query_string={'provider': 'ollama', 'api_endpoint': EVIL})
    assert resp.status_code >= 400
    assert wire.requests == []


# ---------------------------------------------------------------------------
# POST /api/glossaries/<gid>/suggest-terms
# ---------------------------------------------------------------------------

@pytest.fixture
def ner_client():
    from src.core.glossary.store import GlossaryStore
    db = os.path.join(tempfile.mkdtemp(prefix='wire_ner_'), 'g.db')
    store = GlossaryStore(db_path=db)
    gid = store.create_glossary(name='wire', source_language='English',
                                target_language='French').id
    app = Flask(__name__)
    app.register_blueprint(create_glossary_blueprint(store=store))
    with app.test_client() as client:
        yield client, gid
    store.close_all()


@pytest.mark.parametrize('provider, endpoint', [('openai', LAN_OPENAI), ('nim', LAN_OPENAI)])
@pytest.mark.parametrize('sent_key', [None, '', '__USE_ENV__'])
def test_ner_override_without_own_key_sends_nothing(ner_client, wire, provider, endpoint, sent_key):
    client, gid = ner_client
    body = {'text': 'Alice met Bob in Paris.', 'provider': provider, 'model': 'm',
            'api_endpoint': endpoint}
    if sent_key is not None:
        body['api_key'] = sent_key
    resp = client.post(f'/api/glossaries/{gid}/suggest-terms', json=body)
    assert resp.status_code == 400
    assert wire.requests == []


def test_ner_override_with_own_key_sends_only_that_key(ner_client, wire):
    client, gid = ner_client
    client.post(f'/api/glossaries/{gid}/suggest-terms', json={
        'text': 'Alice met Bob in Paris.', 'provider': 'openai', 'model': 'm',
        'api_endpoint': LAN_OPENAI, 'api_key': OWN_KEY})
    posts = [r for r in wire.requests if r['method'] == 'POST']
    assert posts, 'the NER request should reach the LAN server'
    assert wire.hosts() == {'192.168.1.50'}
    assert all(r['headers'].get('authorization') == f'Bearer {OWN_KEY}' for r in posts)
    assert wire.leaked() == []


def test_ner_foreign_ollama_host_sends_nothing(ner_client, wire):
    client, gid = ner_client
    resp = client.post(f'/api/glossaries/{gid}/suggest-terms', json={
        'text': 'Alice met Bob.', 'provider': 'ollama', 'model': 'm', 'api_endpoint': EVIL})
    assert resp.status_code == 400
    assert wire.requests == []


@pytest.mark.parametrize('provider', ['gemini', 'mistral', 'deepseek', 'poe', 'openrouter'])
def test_ner_endpoint_field_is_inert_for_cloud_providers(ner_client, wire, monkeypatch, provider):
    """The endpoint field is ignored for these providers: the key may only
    travel to the provider's own host, never to the request-chosen one."""
    env_var = f'{provider.upper()}_API_KEY'
    monkeypatch.setenv(env_var, STORED_KEY)
    monkeypatch.setattr(_config, env_var, STORED_KEY, raising=False)
    client, gid = ner_client
    client.post(f'/api/glossaries/{gid}/suggest-terms', json={
        'text': 'Alice met Bob.', 'provider': provider, 'model': 'm',
        'api_endpoint': EVIL, 'api_key': '__USE_ENV__'})
    assert 'attacker.example.com' not in wire.hosts()
    assert all('attacker.example.com' not in r['url'] for r in wire.requests)


# ---------------------------------------------------------------------------
# POST /api/custom-instructions/extract-style
# ---------------------------------------------------------------------------

@pytest.fixture
def style_client(tmp_path, monkeypatch):
    monkeypatch.setattr(cir, 'get_config_path', lambda: str(tmp_path))
    app = Flask(__name__)
    app.register_blueprint(cir.create_custom_instruction_blueprint())
    with app.test_client() as client:
        yield client


def _post_style(client, **fields):
    data = {'files': [(io.BytesIO(('It was a dark night. ' * 40).encode()), 'novel.txt')]}
    data.update(fields)
    return client.post('/api/custom-instructions/extract-style', data=data,
                       content_type='multipart/form-data')


@pytest.mark.parametrize('sent_key', ['', '__USE_ENV__'])
def test_style_override_without_own_key_sends_nothing(style_client, wire, sent_key):
    resp = _post_style(style_client, provider='openai', model='m',
                       api_endpoint=LAN_OPENAI, api_key=sent_key)
    assert resp.status_code == 400
    assert wire.requests == []


def test_style_override_with_own_key_sends_only_that_key(style_client, wire):
    _post_style(style_client, provider='openai', model='m',
                api_endpoint=LAN_OPENAI, api_key=OWN_KEY)
    posts = [r for r in wire.requests if r['method'] == 'POST']
    assert posts
    assert wire.hosts() == {'192.168.1.50'}
    assert all(r['headers'].get('authorization') == f'Bearer {OWN_KEY}' for r in posts)
    assert wire.leaked() == []


# ---------------------------------------------------------------------------
# The legacy keyword spelling used by the glossary and style routes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('provider_type, expected_class', [
    ('openai', 'OpenAICompatibleProvider'),
    ('nim', 'OpenAICompatibleProvider'),
    ('gemini', 'GeminiProvider'),
    ('mistral', 'MistralProvider'),
    ('ollama', 'OllamaProvider'),
])
def test_factory_honours_provider_type_keyword(monkeypatch, provider_type, expected_class):
    """create_llm_provider(provider_type=...) is how glossary_routes and
    custom_instruction_routes call the factory. The refactor renamed the
    parameter to `provider` (default "ollama"), so the keyword was silently
    dropped and every NER / style-extraction request was sent by an Ollama
    client to the request endpoint, whatever provider the user picked."""
    from src.core.llm.factory import create_llm_provider
    for var in ('GEMINI_API_KEY', 'MISTRAL_API_KEY'):
        monkeypatch.setenv(var, OWN_KEY)
    provider = create_llm_provider(provider_type=provider_type, model='m',
                                   api_endpoint=LAN_OPENAI, api_key=OWN_KEY)
    assert type(provider).__name__ == expected_class


# ---------------------------------------------------------------------------
# Sample routes: the per-column provider, driven to the wire
# ---------------------------------------------------------------------------

def _generate(provider):
    import asyncio

    async def _run():
        try:
            return await provider.generate('Hello', system_prompt='sys')
        finally:
            await provider.close()

    return asyncio.run(_run())


@pytest.mark.parametrize('column', [
    {'provider': 'openai', 'model': 'm', 'api_endpoint': LAN_OPENAI, 'api_key': OWN_KEY},
    {'provider': 'nim', 'model': 'm', 'api_endpoint': LAN_OPENAI, 'api_key': OWN_KEY},
])
def test_sample_column_override_with_own_key(wire, column):
    _generate(sample_routes._instantiate_provider(column))
    assert wire.hosts() == {'192.168.1.50'}
    assert wire.leaked() == []
    assert wire.requests[-1]['headers'].get('authorization') == f'Bearer {OWN_KEY}'


@pytest.mark.parametrize('column', [
    {'provider': 'openai', 'model': 'm', 'api_endpoint': LAN_OPENAI},
    {'provider': 'openai', 'model': 'm', 'api_endpoint': LAN_OPENAI, 'api_key': ''},
    {'provider': 'openai', 'model': 'm', 'api_endpoint': LAN_OPENAI, 'api_key': '__USE_ENV__'},
    {'provider': 'OpenAI ', 'model': 'm', 'endpoint': LAN_OPENAI},
    {'provider': 'nim', 'model': 'm', 'api_endpoint': LAN_OPENAI},
    {'provider': 'openai', 'model': 'm', 'api_endpoint': EVIL, 'api_key': OWN_KEY},
])
def test_sample_column_never_pairs_stored_key_with_override(wire, column):
    with pytest.raises(ValueError):
        provider = sample_routes._instantiate_provider(column)
        _generate(provider)
    assert wire.leaked() == []


def test_sample_column_ollama_lan_host_carries_no_key(wire):
    _generate(sample_routes._instantiate_provider(
        {'provider': 'ollama', 'model': 'm', 'api_endpoint': LAN_OLLAMA}))
    assert wire.requests
    assert wire.leaked() == []
    assert all('authorization' not in r['headers'] for r in wire.requests)


# ---------------------------------------------------------------------------
# POST /api/translate: JobConfig -> provider -> wire
# ---------------------------------------------------------------------------

def _job_from_web_request(tmp_path, **fields):
    from src.core.job.config import JobConfig
    body = {'text': 'Hello.', 'source_language': 'English', 'target_language': 'French',
            'model': 'm', 'output_filename': 'out.txt'}
    body.update(fields)
    return JobConfig.from_web_request(body, output_dir=str(tmp_path),
                                      uploads_dir=str(tmp_path / 'uploads'))


def test_translate_keyless_lan_override_sends_no_key_when_none_is_stored(tmp_path, wire, monkeypatch):
    from src.core.llm.factory import create_llm_provider
    monkeypatch.delenv('OPENAI_API_KEY')
    monkeypatch.setattr(_config, 'OPENAI_API_KEY', '')
    job = _job_from_web_request(tmp_path, llm_provider='openai', llm_api_endpoint=LAN_OPENAI)
    _generate(create_llm_provider(job.provider_for()))
    assert wire.hosts() == {'192.168.1.50'}
    assert all('authorization' not in r['headers'] for r in wire.requests)


def test_translate_lan_override_with_own_key(tmp_path, wire):
    from src.core.llm.factory import create_llm_provider
    job = _job_from_web_request(tmp_path, llm_provider='openai', llm_api_endpoint=LAN_OPENAI,
                                openai_api_key=OWN_KEY)
    _generate(create_llm_provider(job.provider_for()))
    assert wire.hosts() == {'192.168.1.50'}
    assert wire.leaked() == []


def test_translate_persisted_form_never_carries_a_key(tmp_path):
    job = _job_from_web_request(tmp_path, llm_provider='openai', llm_api_endpoint=LAN_OPENAI,
                                openai_api_key=OWN_KEY)
    stored = json.dumps(job.to_persisted())
    assert OWN_KEY not in stored and STORED_KEY not in stored


# ---------------------------------------------------------------------------
# Defence in depth: the guard and the factory must read the stored key from
# the same place. The guard asks os.environ only; the factory's env fallback
# also reads src.config. When the two disagree, routes that pass api_key=None
# to the factory (glossary NER, style extraction, sample columns) get the key
# back from src.config and send it to the override host.
# ---------------------------------------------------------------------------

def test_hardening_guard_and_factory_agree_on_the_stored_key(ner_client, wire, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY')
    monkeypatch.setattr(_config, 'OPENAI_API_KEY', STORED_KEY)
    client, gid = ner_client
    client.post(f'/api/glossaries/{gid}/suggest-terms', json={
        'text': 'Alice met Bob in Paris.', 'provider': 'openai', 'model': 'm',
        'api_endpoint': LAN_OPENAI})
    assert wire.leaked() == []
