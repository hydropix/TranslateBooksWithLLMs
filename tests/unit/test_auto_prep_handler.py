"""
Unit tests for the auto-mode wiring in `src.api.handlers`
(Phase 2 of plan/PLAN_AutoGlossaryStyle.md).

Fully offline: `auto_prep.build_auto_prompt_options` and `create_llm_client`
are monkeypatched on the `handlers` module, so no provider is contacted and no
Flask app is needed. The two call-site regressions (the suppressed
"📖 Loaded glossary" line and the resume guard) are exercised by driving
`perform_actual_translation` itself against fake state/checkpoint managers and
a fake `translate_file`, because both live inside that coroutine.

Each numbered section matches the matching item in the "Validation criteria
(Phase 2)" list of the plan.
"""
import asyncio
import time

import pytest

from src.api import handlers
from src.core import auto_prep


SOURCE_TEXT = (
    "The rain kept falling on the empty avenue, and nobody came looking for answers. "
) * 40

FRAGMENT = {
    "glossary_terms": {"Avenue": "Avenue des Ombres"},
    "glossary_term_metadata": {"Avenue": {"category": "place"}},
    "glossary_name": auto_prep.AUTO_GLOSSARY_NAME,
    "glossary_source": "auto",
    "custom_instructions": "## Style\n- Keep sentences short.",
}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeClient:
    """Stand-in for `LLMClient`: only `close()` is ever reached from handlers."""

    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


class ClientFactory:
    """Monkeypatched `create_llm_client`: records its calls, hands back a client."""

    def __init__(self, client=None):
        self.client = FakeClient() if client is None else client
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.client


class LogRecorder:
    """Matches `_log_message_callback(message_key, message_content="", data=None)`."""

    def __init__(self):
        self.entries = []

    def __call__(self, message_key, message_content="", data=None):
        self.entries.append((message_key, message_content))

    def keys(self):
        return [key for key, _ in self.entries]


def _base_config(prompt_options=None, **overrides):
    config = {
        'file_type': 'txt',
        'text': SOURCE_TEXT,
        'output_filename': 'out.txt',
        'source_language': 'English',
        'target_language': 'French',
        'model': 'test-model',
        'llm_provider': 'ollama',
        'llm_api_endpoint': 'http://localhost:11434/api/generate',
        'prompt_options': {} if prompt_options is None else prompt_options,
    }
    config.update(overrides)
    return config


@pytest.fixture
def patched(monkeypatch):
    """Patch the two seams `_apply_auto_prep` reaches out through.

    Returns a small namespace: `.factory` (the fake `create_llm_client`),
    `.builder_calls` (kwargs each `build_auto_prompt_options` call received)
    and `.fragment` / `.raises` to steer what the builder does.
    """

    class Patched:
        def __init__(self):
            self.factory = ClientFactory()
            self.builder_calls = []
            self.fragment = dict(FRAGMENT)
            self.raises = None

    state = Patched()

    async def fake_build(**kwargs):
        state.builder_calls.append(kwargs)
        if state.raises is not None:
            raise state.raises
        return dict(state.fragment)

    monkeypatch.setattr(handlers, 'create_llm_client', state.factory)
    monkeypatch.setattr(auto_prep, 'build_auto_prompt_options', fake_build)
    return state


# ---------------------------------------------------------------------------
# 1. _auto_prep_wants across the five documented shapes
# ---------------------------------------------------------------------------
class TestAutoPrepWants:
    def test_glossary_auto_alone(self):
        config = _base_config({'glossary_auto': True})
        assert handlers._auto_prep_wants(config) == (True, False)

    def test_explicit_glossary_id_wins(self):
        config = _base_config({'glossary_auto': True, 'glossary_id': 7})
        assert handlers._auto_prep_wants(config) == (False, False)

    def test_snapshotted_glossary_terms_win(self):
        # The selected-glossary snapshot ran just above the call site: a
        # non-empty snapshot means a real glossary already won.
        config = _base_config({'glossary_auto': True, 'glossary_terms': {'a': 'b'}})
        assert handlers._auto_prep_wants(config) == (False, False)

    def test_refine_only_skips_glossary(self):
        config = _base_config({'glossary_auto': True}, refine_only=True)
        assert handlers._auto_prep_wants(config) == (False, False)

    def test_explicit_preset_wins_over_style_auto(self):
        config = _base_config({'style_auto': True, 'custom_instruction_file': 'x.yaml'})
        assert handlers._auto_prep_wants(config) == (False, False)

    def test_sentinels_are_normalized(self):
        config = _base_config({
            'glossary_id': auto_prep.AUTO_SENTINEL,
            'custom_instruction_file': auto_prep.AUTO_SENTINEL,
        })
        assert handlers._auto_prep_wants(config) == (True, True)
        assert 'glossary_id' not in config['prompt_options']
        assert config['prompt_options']['custom_instruction_file'] == ''

    def test_missing_prompt_options_is_created(self):
        config = {'source_language': 'English'}
        assert handlers._auto_prep_wants(config) == (False, False)
        assert config['prompt_options'] == {}

    def test_refine_only_keeps_style_auto(self):
        config = _base_config({'glossary_auto': True, 'style_auto': True}, refine_only=True)
        assert handlers._auto_prep_wants(config) == (False, True)


# ---------------------------------------------------------------------------
# 2-5. _apply_auto_prep
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fragment_is_merged_and_nothing_else_is_touched(patched):
    config = _base_config({'glossary_auto': True, 'style_auto': True})
    before = {key: value for key, value in config.items() if key != 'prompt_options'}
    log = LogRecorder()

    await handlers._apply_auto_prep(config, log)

    for key, value in before.items():
        assert config[key] == value
    assert set(config) == set(before) | {'prompt_options'}
    for key, value in FRAGMENT.items():
        assert config['prompt_options'][key] == value
    # The intent flags stay in the persisted config (plan §3.3).
    assert config['prompt_options']['glossary_auto'] is True
    assert patched.factory.client.closed == 1

    kwargs = patched.builder_calls[0]
    assert kwargs['want_glossary'] is True
    assert kwargs['want_style'] is True
    assert kwargs['source_language'] == 'English'
    assert kwargs['target_language'] == 'French'
    assert kwargs['log'] is log
    assert kwargs['source_text'].startswith('The rain kept falling')


@pytest.mark.asyncio
async def test_builder_failure_is_swallowed_and_logged(patched):
    patched.raises = RuntimeError("provider exploded")
    config = _base_config({'glossary_auto': True})
    prompt_options_before = dict(config['prompt_options'])
    log = LogRecorder()

    await handlers._apply_auto_prep(config, log)

    assert config['prompt_options'] == prompt_options_before
    assert log.keys().count('auto_prep_error') == 1
    assert 'provider exploded' in log.entries[0][1]


@pytest.mark.asyncio
async def test_client_is_closed_even_when_the_builder_raises(patched):
    patched.raises = RuntimeError("boom")
    log = LogRecorder()

    await handlers._apply_auto_prep(_base_config({'style_auto': True}), log)

    assert patched.factory.client.closed == 1


@pytest.mark.asyncio
async def test_no_client_is_created_when_nothing_is_wanted(patched):
    config = _base_config({})
    log = LogRecorder()

    await handlers._apply_auto_prep(config, log)

    assert patched.factory.calls == []
    assert patched.builder_calls == []
    assert log.entries == []


@pytest.mark.asyncio
async def test_unreadable_source_logs_and_returns(patched):
    config = _base_config({'glossary_auto': True}, text=None, file_path=None)
    log = LogRecorder()

    await handlers._apply_auto_prep(config, log)

    assert patched.factory.calls == []
    assert log.keys() == ['auto_prep_no_text']


@pytest.mark.asyncio
async def test_unknown_provider_yields_no_llm_call(monkeypatch, patched):
    monkeypatch.setattr(handlers, 'create_llm_client', lambda *a, **k: None)
    config = _base_config({'glossary_auto': True})
    log = LogRecorder()

    await handlers._apply_auto_prep(config, log)

    assert patched.builder_calls == []
    assert log.keys() == ['auto_prep_client_warning']
    assert 'glossary_terms' not in config['prompt_options']


@pytest.mark.asyncio
async def test_client_is_built_with_the_jobs_own_keys(patched):
    config = _base_config(
        {'glossary_auto': True},
        llm_provider='openrouter',
        gemini_api_key='gk',
        openai_api_key='ok',
        openrouter_api_key='ork',
        mistral_api_key='mk',
        deepseek_api_key='dk',
        poe_api_key='pk',
        nim_api_key='nk',
    )

    await handlers._apply_auto_prep(config, LogRecorder())

    args, kwargs = patched.factory.calls[0]
    assert args == ('openrouter', 'gk', config['llm_api_endpoint'], 'test-model')
    assert kwargs == {
        'openai_api_key': 'ok',
        'openrouter_api_key': 'ork',
        'mistral_api_key': 'mk',
        'deepseek_api_key': 'dk',
        'poe_api_key': 'pk',
        'nim_api_key': 'nk',
        'context_window': auto_prep.AUTO_PREP_CONTEXT_WINDOW,
    }


@pytest.mark.asyncio
async def test_refine_only_runs_style_with_target_as_source(patched):
    config = _base_config(
        {'glossary_auto': True, 'style_auto': True},
        refine_only=True,
    )

    await handlers._apply_auto_prep(config, LogRecorder())

    kwargs = patched.builder_calls[0]
    assert kwargs['want_glossary'] is False
    assert kwargs['want_style'] is True
    # D7: the input is already in the target language.
    assert kwargs['source_language'] == 'French'
    assert kwargs['target_language'] == 'French'


# ---------------------------------------------------------------------------
# 6-7. Call-site regressions, driven through perform_actual_translation
# ---------------------------------------------------------------------------
class FakeCheckpointManager:
    def __init__(self):
        self.started = []

    def start_job(self, translation_id, file_type, config, input_file_path):
        self.started.append(dict(config.get('prompt_options', {})))

    def mark_completed(self, translation_id):
        return True

    def prune_job_data(self, translation_id):
        return True

    def mark_interrupted(self, translation_id):
        pass

    def mark_error(self, translation_id):
        pass

    def mark_partial(self, translation_id):
        pass

    def load_checkpoint(self, translation_id):
        return None


class FakeStateManager:
    def __init__(self, checkpoint_manager):
        self._checkpoint_manager = checkpoint_manager
        self.fields = {
            'logs': [],
            'stats': {'start_time': time.time(), 'total_chunks': 1, 'completed_chunks': 1},
        }

    def exists(self, translation_id):
        return True

    def set_translation_field(self, translation_id, field, value):
        self.fields[field] = value

    def get_translation_field(self, translation_id, field):
        return self.fields.get(field)

    def update_stats(self, translation_id, new_stats):
        self.fields.setdefault('stats', {}).update(new_stats)

    def get_checkpoint_manager(self):
        return self._checkpoint_manager


class FakeSocketIO:
    def emit(self, *args, **kwargs):
        pass


async def _run_job(config, tmp_path, monkeypatch):
    """Drive `perform_actual_translation` with every external seam faked out."""
    checkpoint_manager = FakeCheckpointManager()
    state_manager = FakeStateManager(checkpoint_manager)

    async def fake_translate_file(**kwargs):
        with open(kwargs['output_filepath'], 'w', encoding='utf-8') as handle:
            handle.write('translated')

    async def fake_refine_file(**kwargs):
        with open(kwargs['output_filepath'], 'w', encoding='utf-8') as handle:
            handle.write('refined')

    monkeypatch.setattr(handlers, 'translate_file', fake_translate_file)
    monkeypatch.setattr(handlers, 'refine_file', fake_refine_file)
    monkeypatch.setattr(handlers, 'emit_update', lambda *a, **k: None)
    monkeypatch.setattr(handlers, 'notify', lambda *a, **k: None)

    await handlers.perform_actual_translation(
        'job-1', config, state_manager, str(tmp_path), FakeSocketIO()
    )

    messages = [
        entry['message'] for entry in state_manager.fields.get('logs', [])
        if isinstance(entry, dict) and 'message' in entry
    ]
    return state_manager, checkpoint_manager, messages


@pytest.mark.asyncio
async def test_auto_glossary_suppresses_the_legacy_loaded_line(tmp_path, monkeypatch, patched):
    config = _base_config({'glossary_auto': True, 'style_auto': True})

    _state, checkpoint_manager, messages = await _run_job(config, tmp_path, monkeypatch)

    assert not any('📖 Loaded glossary' in message for message in messages)
    # The derived snapshot is persisted with the job config (D5).
    assert checkpoint_manager.started[0]['glossary_terms'] == FRAGMENT['glossary_terms']
    assert checkpoint_manager.started[0]['glossary_source'] == 'auto'
    # resolve_custom_instructions stayed a no-op, so the auto style survived.
    assert config['prompt_options']['custom_instructions'] == FRAGMENT['custom_instructions']


@pytest.mark.asyncio
async def test_real_glossary_still_prints_the_loaded_line(tmp_path, monkeypatch, patched):
    config = _base_config({
        'glossary_terms': {'Avenue': 'Avenue'},
        'glossary_name': 'My Glossary',
    })

    _state, _checkpoint_manager, messages = await _run_job(config, tmp_path, monkeypatch)

    assert any("📖 Loaded glossary 'My Glossary' (1 terms)" in message for message in messages)
    assert patched.factory.calls == []


@pytest.mark.asyncio
async def test_resume_never_recomputes_the_auto_snapshot(tmp_path, monkeypatch, patched):
    snapshot = {'Avenue': 'Avenue des Ombres'}
    config = _base_config(
        {
            'glossary_auto': True,
            'style_auto': True,
            'glossary_terms': dict(snapshot),
            'glossary_name': auto_prep.AUTO_GLOSSARY_NAME,
            'glossary_source': 'auto',
        },
        is_resume=True,
        resume_from_index=3,
    )

    _state, checkpoint_manager, messages = await _run_job(config, tmp_path, monkeypatch)

    assert patched.factory.calls == []
    assert patched.builder_calls == []
    assert config['prompt_options']['glossary_terms'] == snapshot
    # A resumed job does not re-open a checkpoint either.
    assert checkpoint_manager.started == []
    assert not any('📖 Loaded glossary' in message for message in messages)


# ---------------------------------------------------------------------------
# 8. Progress feedback while the auto passes run
#
# The passes happen before the first chunk exists, so nothing else emits for as
# long as they take. Without these signals the progress panel is a still image
# and the run looks hung.
# ---------------------------------------------------------------------------
class ProgressRecorder:
    """Matches the `progress_callback(active: bool)` seam."""

    def __init__(self):
        self.states = []

    def __call__(self, active):
        self.states.append(active)


@pytest.mark.asyncio
async def test_progress_callback_brackets_the_passes(patched):
    config = _base_config({'glossary_auto': True, 'style_auto': True})
    progress = ProgressRecorder()

    await handlers._apply_auto_prep(config, LogRecorder(), progress)

    assert progress.states[0] is True
    assert progress.states[-1] is False
    assert progress.states.count(False) == 1


@pytest.mark.asyncio
async def test_progress_callback_is_cleared_when_the_builder_raises(patched):
    patched.raises = RuntimeError("provider exploded")
    config = _base_config({'glossary_auto': True})
    progress = ProgressRecorder()

    await handlers._apply_auto_prep(config, LogRecorder(), progress)

    # A failure must not strand the panel on "Preparing…".
    assert progress.states[0] is True
    assert progress.states[-1] is False


@pytest.mark.asyncio
async def test_progress_callback_is_cleared_when_the_document_is_unreadable(patched):
    config = _base_config({'glossary_auto': True}, text='')
    config['file_path'] = None
    progress = ProgressRecorder()

    await handlers._apply_auto_prep(config, LogRecorder(), progress)

    assert progress.states == [True, False]
    assert patched.factory.calls == []


@pytest.mark.asyncio
async def test_no_progress_signal_when_no_pass_is_wanted(patched):
    config = _base_config({})
    progress = ProgressRecorder()

    await handlers._apply_auto_prep(config, LogRecorder(), progress)

    assert progress.states == []


@pytest.mark.asyncio
async def test_a_raising_progress_callback_never_breaks_the_job(patched):
    def boom(active):
        raise ValueError("emit is broken")

    config = _base_config({'style_auto': True})
    await handlers._apply_auto_prep(config, LogRecorder(), boom)

    assert config['prompt_options']['custom_instructions'] == FRAGMENT['custom_instructions']


@pytest.mark.asyncio
async def test_heartbeat_ticks_while_a_slow_pass_runs(patched, monkeypatch):
    monkeypatch.setattr(handlers, '_AUTO_PREP_HEARTBEAT_S', 0.01)

    async def slow_build(**kwargs):
        await asyncio.sleep(0.08)
        return dict(FRAGMENT)

    monkeypatch.setattr(auto_prep, 'build_auto_prompt_options', slow_build)
    progress = ProgressRecorder()

    await handlers._apply_auto_prep(
        _base_config({'glossary_auto': True}), LogRecorder(), progress
    )

    # Start signal + several heartbeats + the final clear.
    assert progress.states.count(True) >= 3
    assert progress.states[-1] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
