"""The web adapter of the job runner (``src/api/handlers.py``), end to end.

``perform_actual_translation`` is driven with a real ``TranslationStateManager``
backed by a real ``CheckpointManager`` on tmp_path, a fake Socket.IO server
recording every emit and a fake ``translate_file`` engine patched onto
``src.core.adapters``. Nothing is mocked between the handler and the runner.
"""

import copy
import json

import pytest

from src.api import handlers
from src.api.translation_state import TranslationStateManager
from src.core import adapters
from src.core.job import JobConfig, JobResult
from src.core.llm.exceptions import RateLimitError
from src.core.llm.provider_config import ProviderConfig
from src.persistence.checkpoint_manager import CheckpointManager

# Inert placeholders, never real credentials.
JOB_KEY = "sk-test-not-a-real-key"
ENV_KEY = "sk-test-env-not-a-real-key"
TID = "trans_1700000000000_abc123"


class RecordingSocketIO:
    def __init__(self):
        self.events = []

    def emit(self, event, payload=None, namespace=None, **kwargs):
        self.events.append((event, copy.deepcopy(payload)))

    def payloads(self, event='translation_update'):
        return [payload for name, payload in self.events if name == event]


@pytest.fixture
def web(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.utils.notifier.notify", lambda *a, **k: None)
    output_dir = tmp_path / "translated_files"
    uploads = output_dir / "uploads"
    uploads.mkdir(parents=True)
    upload = uploads / "0123456789abcdef_book.txt"
    upload.write_text("one\n\ntwo\n\nthree\n", encoding="utf-8")
    manager = CheckpointManager(db_path=str(tmp_path / "data" / "jobs.db"))
    state = TranslationStateManager(checkpoint_manager=manager, server_session_id="test")
    return output_dir, upload, manager, state, RecordingSocketIO()


def _job(output_dir, upload, **changes):
    base = dict(
        operation='translate', file_type='txt', input_path=str(upload),
        output_path=str(output_dir / "book_fr.txt"), input_filename="book.txt",
        source_language="English", target_language="French",
        provider=ProviderConfig(provider="poe", model="fake-model", api_key=JOB_KEY),
        origin='web',
    )
    base.update(changes)
    return JobConfig(**base)


def _completing_engine(seen):
    async def translate_file(job, ctx):
        seen['api_key'] = job.provider.api_key
        seen['output_path'] = job.output_path
        ctx.log("chunk_done", "Translated chunk 1/3")
        ctx.emit({'total_chunks': 3, 'completed_chunks': 1, 'failed_chunks': 0})
        with open(job.output_path, "w", encoding="utf-8") as handle:
            handle.write("un\n\ndeux\n\ntrois\n")
        stats = {'total_chunks': 3, 'completed_chunks': 3, 'failed_chunks': 0,
                 'unfinished_chunks': 0}
        ctx.emit(stats)
        return JobResult.from_run(stats, job.output_path, engine_ok=True)
    return translate_file


def _assert_no_key_anywhere(state, socketio, *keys):
    stored = state.get_translation(TID)
    dumped_state = json.dumps(stored, default=str)
    dumped_emits = json.dumps(socketio.events, default=str)
    for key in keys:
        assert key not in dumped_state
        assert key not in dumped_emits
    config = stored['config']
    assert not [k for k in config if k == 'api_key' or k.endswith('_api_key')]


@pytest.mark.asyncio
async def test_completed_job_final_payload_state_and_secrets(web, monkeypatch):
    output_dir, upload, manager, state, socketio = web
    seen = {}
    monkeypatch.setattr(adapters, "translate_file", _completing_engine(seen))
    job = _job(output_dir, upload)
    state.create_translation(TID, job.to_persisted())

    result = await handlers.perform_actual_translation(TID, job, state, str(output_dir), socketio)

    assert result.ok
    # The engine got the in-memory key.
    assert seen['api_key'] == JOB_KEY
    final = socketio.payloads()[-1]
    assert final['status'] == 'completed'
    assert final['translation_id'] == TID
    assert final['output_filename'] == 'book_fr.txt'
    assert final['model'] == 'fake-model' and final['llm_provider'] == 'poe'
    assert final['stats']['total_chunks'] == 3
    assert final['stats']['completed_chunks'] == 3
    assert 'error' not in final

    assert state.get_translation_field(TID, 'status') == 'completed'
    assert state.get_translation_field(TID, 'output_filepath') == str(output_dir / "book_fr.txt")
    assert (output_dir / "book_fr.txt").read_text(encoding="utf-8").startswith("un")
    assert manager.get_job(TID)['status'] == 'completed'
    # The web upload is deleted once the job no longer needs it.
    assert not upload.exists()
    assert ('file_list_changed', {'reason': 'completed', 'filename': 'book_fr.txt'}) in socketio.events
    # The log line reached both the live view and the stored log.
    assert any(p.get('log') == "Translated chunk 1/3" for p in socketio.payloads())
    stored_logs = state.get_translation_field(TID, 'logs')
    assert any(isinstance(e, dict) and e.get('message') == "Translated chunk 1/3" for e in stored_logs)

    _assert_no_key_anywhere(state, socketio, JOB_KEY)
    assert JOB_KEY not in json.dumps(manager.get_job(TID)['config'])


@pytest.mark.asyncio
async def test_progress_is_monotonic_and_reaches_completion(web, monkeypatch):
    output_dir, upload, _manager, state, socketio = web
    monkeypatch.setattr(adapters, "translate_file", _completing_engine({}))
    job = _job(output_dir, upload)
    state.create_translation(TID, job.to_persisted())

    await handlers.perform_actual_translation(TID, job, state, str(output_dir), socketio)

    percents = [p['stats']['percent'] for p in socketio.payloads()
                if isinstance(p.get('stats'), dict) and 'percent' in p['stats']]
    assert percents, "no stats emitted"
    assert percents == sorted(percents)


@pytest.mark.asyncio
async def test_a_fresh_job_never_overwrites_an_existing_output(web, monkeypatch):
    output_dir, upload, _manager, state, socketio = web
    (output_dir / "book_fr.txt").write_text("previous run", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(adapters, "translate_file", _completing_engine(seen))
    job = _job(output_dir, upload)
    state.create_translation(TID, job.to_persisted())

    await handlers.perform_actual_translation(TID, job, state, str(output_dir), socketio)

    assert seen['output_path'] != str(output_dir / "book_fr.txt")
    assert (output_dir / "book_fr.txt").read_text(encoding="utf-8") == "previous run"
    final = socketio.payloads()[-1]
    assert final['output_filename'] == state.get_translation(TID)['config']['output_filename']
    assert final['output_filename'] != "book_fr.txt"


@pytest.mark.asyncio
async def test_persisted_dict_is_rebuilt_with_the_env_key(web, monkeypatch):
    output_dir, upload, manager, state, socketio = web
    monkeypatch.setenv("POE_API_KEY", ENV_KEY)
    seen = {}
    monkeypatch.setattr(adapters, "translate_file", _completing_engine(seen))
    persisted = _job(output_dir, upload).to_persisted()
    assert 'poe_api_key' not in persisted
    state.create_translation(TID, persisted)

    result = await handlers.perform_actual_translation(TID, persisted, state, str(output_dir), socketio)

    assert result.ok
    assert seen['api_key'] == ENV_KEY
    assert socketio.payloads()[-1]['status'] == 'completed'
    _assert_no_key_anywhere(state, socketio, ENV_KEY)


@pytest.mark.asyncio
async def test_crashing_engine_reports_error(web, monkeypatch):
    output_dir, upload, manager, state, socketio = web

    async def boom(job, ctx):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(adapters, "translate_file", boom)
    job = _job(output_dir, upload)
    state.create_translation(TID, job.to_persisted())

    result = await handlers.perform_actual_translation(TID, job, state, str(output_dir), socketio)

    assert not result.ok
    final = socketio.payloads()[-1]
    assert final['status'] == 'error'
    assert final['translation_id'] == TID
    assert "provider exploded" in final['error']
    assert state.get_translation_field(TID, 'status') == 'error'
    assert manager.get_job(TID)['status'] == 'error'
    # A failed job keeps its upload: it is the only copy for a retry.
    assert upload.exists()
    _assert_no_key_anywhere(state, socketio, JOB_KEY)


@pytest.mark.asyncio
async def test_rate_limit_pauses_the_job(web, monkeypatch):
    output_dir, upload, manager, state, socketio = web

    async def rate_limited(job, ctx):
        raise RateLimitError("429", retry_after=30, provider="poe")

    monkeypatch.setattr(adapters, "translate_file", rate_limited)
    job = _job(output_dir, upload, auto_pause_on_rate_limit=True)
    state.create_translation(TID, job.to_persisted())

    await handlers.perform_actual_translation(TID, job, state, str(output_dir), socketio)

    final = socketio.payloads()[-1]
    assert final['status'] == 'rate_limited'
    assert final['translation_id'] == TID
    assert state.get_translation_field(TID, 'status') == 'rate_limited'
    assert state.get_translation_field(TID, 'interrupted') is True
    assert manager.get_job(TID)['status'] == 'interrupted'
    assert any(name == 'checkpoint_created' and payload['status'] == 'rate_limited'
               for name, payload in socketio.events)
    _assert_no_key_anywhere(state, socketio, JOB_KEY)


@pytest.mark.asyncio
async def test_unknown_job_is_ignored(web, monkeypatch):
    output_dir, upload, _manager, state, socketio = web
    called = []

    async def never(job, ctx):
        called.append(job)

    monkeypatch.setattr(adapters, "translate_file", never)
    result = await handlers.perform_actual_translation(
        "trans_missing", _job(output_dir, upload), state, str(output_dir), socketio)

    assert result is None
    assert called == []
    assert socketio.events == []


def test_resolve_output_path_stays_inside_the_output_dir(tmp_path):
    (tmp_path / "out.txt").write_text("x", encoding="utf-8")
    fresh = handlers.resolve_output_path(str(tmp_path), "out.txt", is_resume=False)
    resumed = handlers.resolve_output_path(str(tmp_path), "out.txt", is_resume=True)
    assert resumed == str(tmp_path / "out.txt")
    assert fresh != resumed and fresh.startswith(str(tmp_path))
    traversal = handlers.resolve_output_path(str(tmp_path), "../escape.txt", is_resume=True)
    assert traversal.startswith(str(tmp_path))


@pytest.mark.asyncio
async def test_job_with_nothing_translated_is_an_error_with_a_code_and_stays_resumable(web, monkeypatch):
    output_dir, upload, manager, state, socketio = web

    async def translate_file(job, ctx):
        with open(job.output_path, "w", encoding="utf-8") as handle:
            handle.write("one\n\ntwo\n\nthree\n")  # source text only
        stats = {'total_chunks': 3, 'completed_chunks': 3, 'failed_chunks': 3,
                 'unfinished_chunks': 3}
        ctx.emit(stats)
        return JobResult.from_run(stats, job.output_path, engine_ok=False)

    monkeypatch.setattr(adapters, "translate_file", translate_file)
    job = _job(output_dir, upload)
    state.create_translation(TID, job.to_persisted())

    result = await handlers.perform_actual_translation(TID, job, state, str(output_dir), socketio)

    assert not result.ok
    final = socketio.payloads()[-1]
    assert final['status'] == 'error'
    assert final['error_code'] == 'nothing_translated'
    assert state.get_translation_field(TID, 'status') == 'error'
    assert state.get_translation_field(TID, 'error_code') == 'nothing_translated'
    assert manager.get_job(TID)['status'] == 'error'
    assert TID in [j['translation_id'] for j in manager.get_resumable_jobs()]


@pytest.mark.asyncio
async def test_end_of_run_sends_a_single_100_percent_progress_payload(web, monkeypatch):
    output_dir, upload, _manager, state, socketio = web
    monkeypatch.setattr(adapters, "translate_file", _completing_engine({}))
    job = _job(output_dir, upload)
    state.create_translation(TID, job.to_persisted())

    await handlers.perform_actual_translation(TID, job, state, str(output_dir), socketio)

    # Progress-only payloads (the final status payload carries its own stats).
    full = [p for p in socketio.payloads()
            if set(p) == {'stats', 'translation_id'} and p['stats'].get('percent') == 100.0]
    assert len(full) == 1, len(full)
