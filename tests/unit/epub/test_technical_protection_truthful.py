"""Regression tests: the technical-protection log and state must tell the truth.

The XHTML translation ran with protect_technical=False while logging
"Technical content protection active" and persisting protect_technical=True
in the partial state.
"""
import pytest
from lxml import etree

from src.core.epub import chunk_fallback
from src.core.epub import document_chunking
from tests.epub_engine_helpers import translate_xhtml
from src.persistence.checkpoint_manager import CheckpointManager

XHTML_NS = "http://www.w3.org/1999/xhtml"


@pytest.fixture
def checkpoint_manager(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir(parents=True, exist_ok=True)
    return manager


async def _run(monkeypatch, checkpoint_manager, enabled):
    async def echo(main_content, *args, **kwargs):
        return main_content

    monkeypatch.setattr(chunk_fallback, "generate_translation_request", echo)
    monkeypatch.setattr("src.config.TECHNICAL_PROTECTION_ENABLED", enabled)
    seen = {}

    def spy_preserve(body_html, tag_preserver, log_callback=None, protect_technical=False):
        seen["protect_technical"] = protect_technical
        return original_preserve(body_html, tag_preserver, log_callback, protect_technical)

    original_preserve = document_chunking.preserve_tags
    monkeypatch.setattr(document_chunking, "preserve_tags", spy_preserve)

    doc = etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><head><title>t</title></head>'
        f'<body><p>Run the command now, please.</p></body></html>'.encode()
    )
    events = []
    translation_id, href = f"job_a8_{enabled}", "OEBPS/ch.xhtml"
    success, _ = await translate_xhtml(
        doc, "English", "French", "fake", llm_client=object(),
        log_callback=lambda event, message: events.append((event, message)),
        checkpoint_manager=checkpoint_manager, translation_id=translation_id,
        file_href=href,
    )
    assert success is True
    state = checkpoint_manager.load_xhtml_partial_state(translation_id, href)
    return seen["protect_technical"], events, state


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_log_and_state_match_the_real_setting(monkeypatch, checkpoint_manager, enabled):
    used, events, state = await _run(monkeypatch, checkpoint_manager, enabled)
    assert used is enabled
    claims_active = any("protection active" in message for _, message in events)
    assert claims_active is enabled
    assert state is not None
    assert state.protect_technical is enabled
