"""Regression tests: in-memory job logs are stored once, compactly, bounded.

Every log entry was appended twice (the WebSocket callback and the storage
callback both stored it), never trimmed, and full LLM prompts/responses were
kept in it, while the status endpoints deep-copy the whole job state.
"""
import pytest

from src.api import handlers
from src.api import translation_state
from src.api.translation_state import TranslationStateManager


class FakeCheckpointManager:
    def start_job(self, *args, **kwargs):
        return True

    def update_job_config(self, *args, **kwargs):
        return True

    def __getattr__(self, name):
        if name.startswith('mark_') or name == 'prune_job_data':
            return lambda translation_id: None
        raise AttributeError(name)

    def load_checkpoint(self, translation_id):
        return None


class FakeSocketIO:
    def emit(self, *args, **kwargs):
        pass


def test_append_log_is_bounded():
    state = TranslationStateManager(checkpoint_manager=object(), server_session_id='s')
    state.create_translation('t', {})
    total = translation_state.MAX_STORED_LOG_ENTRIES + 100
    for i in range(total):
        state.append_log('t', f'entry {i}')
    logs = state.get_translation_field('t', 'logs')
    assert len(logs) == translation_state.MAX_STORED_LOG_ENTRIES
    assert logs[-1] == f'entry {total - 1}'


@pytest.mark.asyncio
async def test_job_logs_are_stored_once_and_compact(tmp_path, monkeypatch):
    monkeypatch.setattr(handlers, 'notify', lambda *a, **k: None)
    emitted = []
    monkeypatch.setattr(handlers, 'emit_update',
                        lambda _sio, _tid, payload, _sm: emitted.append(payload))
    huge = 'x' * 50_000

    async def fake_translate_file(**kwargs):
        kwargs['log_callback']('llm_response', '', data={'type': 'llm_response', 'response': huge})
        kwargs['log_callback']('marker', 'unique marker line')
        with open(kwargs['output_filepath'], 'w', encoding='utf-8') as handle:
            handle.write('done')

    monkeypatch.setattr(handlers, 'translate_file', fake_translate_file)
    state = TranslationStateManager(checkpoint_manager=FakeCheckpointManager(), server_session_id='s')
    config = {
        'text': 'Hello.', 'file_type': 'txt', 'source_language': 'English',
        'target_language': 'French', 'model': 'm', 'llm_provider': 'ollama',
        'llm_api_endpoint': 'http://localhost:11434/api/generate',
        'output_filename': 'out.txt', 'prompt_options': {},
    }
    state.create_translation('trans_1', config)
    output_dir = tmp_path / 'out'
    output_dir.mkdir()

    await handlers.perform_actual_translation('trans_1', config, state, str(output_dir), FakeSocketIO())

    logs = [e for e in state.get_translation_field('trans_1', 'logs') if isinstance(e, dict)]
    assert [e['message'] for e in logs].count('unique marker line') == 1
    stored_response = next(e for e in logs if e['type'] == 'llm_response')['data']['response']
    assert len(stored_response) < 3000
    # The live WebSocket view still gets the full response.
    assert any(p.get('log_entry', {}).get('data', {}).get('response') == huge for p in emitted)
