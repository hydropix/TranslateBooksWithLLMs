"""Regression tests: one worker per job on resume.

The resume route checked for active jobs and then started the worker with no
lock, and ignored 'rate_limited' (worker asleep, about to auto-resume), so a
double click or a manual resume during the auto-resume wait ran two workers
on one job.
"""
import pytest
from flask import Flask

from src.api import handlers
from src.api.blueprints.translation_routes import create_translation_blueprint
from src.api.translation_state import TranslationStateManager


class FakeCheckpointManager:
    def __init__(self, preserved):
        self.preserved = preserved
        self.has_checkpoint = True

    def load_checkpoint(self, translation_id):
        if not self.has_checkpoint:
            return None
        return {
            'job': {'config': {
                'model': 'm', 'llm_provider': 'ollama',
                'llm_api_endpoint': 'http://localhost:11434/api/generate',
                'output_filename': 'out.txt', 'file_type': 'txt',
                'preserved_input_path': str(self.preserved),
            }, 'progress': {}},
            'resume_from_index': 2,
        }

    def get_preserved_input_path(self, translation_id):
        return str(self.preserved)

    def mark_running(self, translation_id):
        return True


@pytest.fixture
def resume_app(tmp_path):
    preserved = tmp_path / 'book.txt'
    preserved.write_text('Hello.', encoding='utf-8')
    checkpoint_manager = FakeCheckpointManager(preserved)
    state = TranslationStateManager(checkpoint_manager=checkpoint_manager, server_session_id='s')
    started = []

    def start(translation_id, config):
        # Stands in for the thread start: the worker now owns the claim and
        # stays alive until the test releases it.
        state.mark_worker_active(translation_id)
        started.append(translation_id)

    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(state, start, str(tmp_path)))
    with app.test_client() as client:
        yield client, state, checkpoint_manager, started


def test_second_resume_while_worker_alive_is_refused(resume_app):
    client, state, _cm, started = resume_app
    assert client.post('/api/resume/trans_1').status_code == 200
    # Until the new worker flips the status to 'running', the in-memory
    # status still reads 'paused', so the old status check let a second
    # resume through: that window is the race.
    state.set_translation_field('trans_1', 'status', 'paused')
    resp = client.post('/api/resume/trans_1')
    assert resp.status_code == 409
    assert started == ['trans_1']

    state.release_worker('trans_1')
    assert client.post('/api/resume/trans_1').status_code == 200
    assert started == ['trans_1', 'trans_1']


def test_failed_resume_releases_the_claim(resume_app):
    client, state, checkpoint_manager, started = resume_app
    checkpoint_manager.has_checkpoint = False
    assert client.post('/api/resume/trans_1').status_code == 404
    assert not state.is_worker_active('trans_1')

    checkpoint_manager.has_checkpoint = True
    assert client.post('/api/resume/trans_1').status_code == 200
    assert started == ['trans_1']


def test_job_waiting_to_auto_resume_counts_as_active(resume_app):
    client, state, _cm, started = resume_app
    state.create_translation('trans_other', {'output_filename': 'other.txt'})
    state.set_translation_field('trans_other', 'status', 'rate_limited')
    assert client.post('/api/resume/trans_1').status_code == 409

    # Auto-paused (interrupted) is not active: the worker has exited.
    state.set_translation_field('trans_other', 'interrupted', True)
    assert client.post('/api/resume/trans_1').status_code == 200
    assert started == ['trans_1']


def test_worker_wrapper_releases_its_claim(monkeypatch):
    state = TranslationStateManager(checkpoint_manager=object(), server_session_id='s')
    seen = {}

    async def fake_job(translation_id, *args, **kwargs):
        seen['active_during_run'] = state.is_worker_active(translation_id)

    monkeypatch.setattr(handlers, 'perform_actual_translation', fake_job)
    assert state.try_claim_worker('trans_9')
    handlers.run_translation_async_wrapper('trans_9', {}, state, '.', None)
    assert seen['active_during_run'] is True
    assert not state.is_worker_active('trans_9')
