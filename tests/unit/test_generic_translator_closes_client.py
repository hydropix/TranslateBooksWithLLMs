"""The generic translator must close the LLM client it creates.

It created an LLMClient per job and never closed it, leaking an httpx
connection pool per job in the long-running server.
"""

import pytest

import src.core.llm.factory as llm_factory_module
import src.core.translator as translator_module
from src.core.adapters.generic_translator import GenericTranslator
from src.core.adapters.txt_adapter import TxtAdapter
from src.core.llm.exceptions import RateLimitError
from src.persistence.checkpoint_manager import CheckpointManager
from tests.job_helpers import make_ctx, make_job

created = []


class _ClosingClient:
    def __init__(self, **kwargs):
        self.closed = False
        created.append(self)

    async def close(self):
        self.closed = True


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    yield manager
    manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "rate_limit"])
async def test_client_is_closed_on_every_exit(tmp_path, cm, monkeypatch, outcome):
    created.clear()
    src = tmp_path / "in.txt"
    src.write_text("Hello world.\n", encoding="utf-8")

    async def fake_request(main_content=None, **kwargs):
        if outcome == "rate_limit":
            raise RateLimitError("429", provider="test")
        return "Bonjour le monde."

    monkeypatch.setattr(llm_factory_module, "create_llm_provider", _fake_factory(_ClosingClient))
    monkeypatch.setattr(translator_module, "generate_translation_request", fake_request)

    translator = GenericTranslator(
        adapter=TxtAdapter(str(src), str(tmp_path / "out.txt"), {}),
        checkpoint_manager=cm, translation_id="job",
    )
    run = translator.translate(
        make_job(source_language="English", target_language="French", model="m", provider="openai"),
        make_ctx(),
    )
    if outcome == "rate_limit":
        with pytest.raises(RateLimitError):
            await run
    else:
        assert await run is True

    assert len(created) == 1 and created[0].closed


def _fake_factory(client_cls):
    """Stand-in for `create_llm_provider` that builds the test's fake client."""
    def build(cfg):
        return client_cls(provider_type=cfg.provider, model=cfg.model,
                          context_window=cfg.context_window, log_callback=cfg.log_callback)
    return build
