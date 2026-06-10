"""
Unit tests for the /api/glossaries/<gid>/suggest-terms endpoint.

Verifies that provider RateLimitError surfaces as HTTP 429 with a
Retry-After header instead of being swallowed as a generic 500.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

# Make the project importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.api.blueprints.glossary_routes import create_glossary_blueprint
from src.core.glossary.store import GlossaryStore
from src.core.llm.exceptions import RateLimitError


@pytest.fixture
def store():
    """Per-test temporary GlossaryStore on an isolated SQLite file."""
    db = os.path.join(
        tempfile.gettempdir(),
        f"glossary_ner_endpoint_{os.getpid()}_{id(object())}.db",
    )
    if os.path.exists(db):
        os.remove(db)
    s = GlossaryStore(db_path=db)
    try:
        yield s
    finally:
        s.close_all()
        try:
            os.remove(db)
        except OSError:
            pass


@pytest.fixture
def client(store):
    """Flask test client with the glossary blueprint mounted on a fresh store."""
    app = Flask(__name__)
    app.register_blueprint(create_glossary_blueprint(store=store))
    with app.test_client() as c:
        yield c


class _RateLimitedProvider:
    """Stand-in LLMProvider whose generate() always raises RateLimitError."""

    def __init__(self, retry_after=None, provider_name="testprov"):
        self._retry_after = retry_after
        self._provider_name = provider_name

    async def generate(self, user_prompt, system_prompt=None, **kwargs):
        raise RateLimitError(
            "rate limit reached",
            retry_after=self._retry_after,
            provider=self._provider_name,
        )


class TestSuggestTermsRateLimit:
    """The endpoint must translate RateLimitError into a 429 response."""

    def _create_glossary(self, store):
        return store.create_glossary(
            name="rl-test",
            source_language="English",
            target_language="French",
        )

    def test_rate_limit_returns_429_with_retry_after_header(self, client, store):
        glossary = self._create_glossary(store)

        provider = _RateLimitedProvider(retry_after=42, provider_name="ollama")
        with patch(
            "src.core.llm.factory.create_llm_provider",
            return_value=provider,
        ):
            response = client.post(
                f"/api/glossaries/{glossary.id}/suggest-terms",
                json={"text": "Some sample source text to analyze."},
            )

        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "42"
        body = response.get_json()
        assert body["provider"] == "ollama"
        assert body["retry_after"] == 42
        assert "rate limit" in body["error"].lower()

    def test_rate_limit_without_retry_after_omits_header(self, client, store):
        glossary = self._create_glossary(store)

        provider = _RateLimitedProvider(retry_after=None, provider_name="openrouter")
        with patch(
            "src.core.llm.factory.create_llm_provider",
            return_value=provider,
        ):
            response = client.post(
                f"/api/glossaries/{glossary.id}/suggest-terms",
                json={"text": "Some sample source text."},
            )

        assert response.status_code == 429
        assert "Retry-After" not in response.headers
        body = response.get_json()
        assert body["provider"] == "openrouter"
        assert body["retry_after"] is None


class _CapturingProvider:
    """Stand-in provider that records init kwargs and returns no candidates."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def suggest(self, *args, **kwargs):  # pragma: no cover - unused
        return [], []

    async def close(self):
        pass


class TestSuggestTermsApiKeyResolution:
    """The endpoint must resolve the '__USE_ENV__' sentinel to the env key.

    Regression for issue #200: with keys configured in .env, the frontend
    sends api_key='__USE_ENV__'. The endpoint used to forward that literal
    string to the provider, yielding 'API key not valid' from Gemini.
    """

    def _create_glossary(self, store):
        return store.create_glossary(
            name="env-key-test",
            source_language="English",
            target_language="French",
        )

    def test_use_env_sentinel_resolves_to_env_key(self, client, store):
        glossary = self._create_glossary(store)

        captured = {}

        def _fake_factory(**kwargs):
            captured.update(kwargs)
            return _CapturingProvider(**kwargs)

        async def _fake_suggest(*args, **kwargs):
            return [], []

        with patch(
            "src.core.llm.factory.create_llm_provider",
            side_effect=_fake_factory,
        ), patch(
            "src.api.blueprints.glossary_routes.ner_suggest_terms",
            side_effect=_fake_suggest,
        ), patch.dict(os.environ, {"GEMINI_API_KEY": "real-env-key"}):
            response = client.post(
                f"/api/glossaries/{glossary.id}/suggest-terms",
                json={
                    "text": "Some sample source text to analyze.",
                    "provider": "gemini",
                    "model": "gemini-3.1-flash-lite",
                    "api_key": "__USE_ENV__",
                },
            )

        assert response.status_code == 200, response.get_json()
        assert captured.get("api_key") == "real-env-key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
