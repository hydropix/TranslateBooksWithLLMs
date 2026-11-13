"""
Unit tests for the auto-mode wiring of the job prepare phase
(``src.core.job.prepare``; Phase 2 of plan/PLAN_AutoGlossaryStyle.md).

Fully offline: ``auto_prep.build_auto_prompt_options`` and
``src.core.llm.factory.create_llm_provider`` are monkeypatched, so no provider
is contacted and no Flask app is needed. The call-site regressions (the
suppressed "📖 Loaded glossary" line and the resume guard) are exercised by
driving ``src.api.handlers.perform_actual_translation`` itself against fake
state/checkpoint managers and a fake ``translate_file`` engine, because both
live in the job runner it hands the job to.

Each numbered section matches the matching item in the "Validation criteria
(Phase 2)" list of the plan.
"""
import asyncio
import copy
import time

import pytest

from src.api import handlers
from src.core import adapters, auto_prep
from src.core.job import OP_REFINE, OP_TRANSLATE, JobConfig, JobResult
from src.core.job import prepare
from src.core.llm import factory
from src.core.llm.exceptions import LLMClientConfigError
from src.core.llm.provider_config import ProviderConfig


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

ENDPOINT = 'http://localhost:11434/api/generate'


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeClient:
    """Stand-in for an LLM provider: only `close()` is ever reached."""

    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


class ClientFactory:
    """Monkeypatched `create_llm_provider`: records its calls, hands back a client."""

    def __init__(self, client=None):
        self.client = FakeClient() if client is None else client
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.client


class LogRecorder:
    """Matches the job log callback `(message_key, message_content="", data=None)`."""

    def __init__(self):
        self.entries = []

    def __call__(self, message_key, message_content="", data=None):
        self.entries.append((message_key, message_content))

    def keys(self):
        return [key for key, _ in self.entries]


def _job(prompt_options=None, **changes):
    base = dict(
        operation=OP_TRANSLATE,
        file_type='txt',
        text=SOURCE_TEXT,
        output_path='out.txt',
        source_language='English',
        target_language='French',
        provider=ProviderConfig(provider='ollama', model='test-model', endpoint=ENDPOINT),
        prompt_options={} if prompt_options is None else prompt_options,
        origin='web',
    )
    base.update(changes)
    return JobConfig(**base)


async def _apply(job, log, progress=None):
    """Run the prepare helper on a private copy of the job's options, like the runner."""
    options = job.engine_prompt_options()
    await prepare.apply_auto_prep(job, options, log, progress)
    return options


@pytest.fixture
def patched(monkeypatch):
    """Patch the two seams `apply_auto_prep` reaches out through.

    Returns a small namespace: `.factory` (the fake `create_llm_provider`),
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

    monkeypatch.setattr(factory, 'create_llm_provider', state.factory)
    monkeypatch.setattr(auto_prep, 'build_auto_prompt_options', fake_build)
    return state


# ---------------------------------------------------------------------------
# 1. auto_prep_wants across the five documented shapes
# ---------------------------------------------------------------------------
class TestAutoPrepWants:
    @staticmethod
    def _wants(prompt_options, **changes):
        job = _job(prompt_options, **changes)
        return prepare.auto_prep_wants(job, prompt_options)

    def test_glossary_auto_alone(self):
        assert self._wants({'glossary_auto': True}) == (True, False)

    def test_explicit_glossary_id_wins(self):
        assert self._wants({'glossary_auto': True, 'glossary_id': 7}) == (False, False)

    def test_snapshotted_glossary_terms_win(self):
        # The selected-glossary snapshot ran just before: a non-empty
        # snapshot means a real glossary already won.
        assert self._wants({'glossary_auto': True, 'glossary_terms': {'a': 'b'}}) == (False, False)

    def test_refine_only_skips_glossary(self):
        assert self._wants({'glossary_auto': True}, operation=OP_REFINE) == (False, False)

    def test_explicit_preset_wins_over_style_auto(self):
        options = {'style_auto': True, 'custom_instruction_file': 'x.yaml'}
        assert self._wants(options) == (False, False)

    def test_sentinels_are_normalized(self):
        options = {
            'glossary_id': auto_prep.AUTO_SENTINEL,
            'custom_instruction_file': auto_prep.AUTO_SENTINEL,
        }
        assert self._wants(options) == (True, True)
        assert 'glossary_id' not in options
        assert options['custom_instruction_file'] == ''

    def test_empty_prompt_options_want_nothing(self):
        assert self._wants({}) == (False, False)

    def test_refine_only_keeps_style_auto(self):
        options = {'glossary_auto': True, 'style_auto': True}
        assert self._wants(options, operation=OP_REFINE) == (False, True)

    def test_term_fix_wants_nothing(self):
        options = {'glossary_auto': True, 'style_auto': True}
        assert self._wants(options, operation='term_fix',
                           term_fix={'mode': 'smart', 'rules_text': 'a -> b'}) == (False, False)


# ---------------------------------------------------------------------------
# 2-5. apply_auto_prep
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fragment_is_merged_and_nothing_else_is_touched(patched):
    job = _job({'glossary_auto': True, 'style_auto': True})
    before = copy.deepcopy(dict(job.prompt_options))
    log = LogRecorder()

    options = await _apply(job, log)

    # Only the prompt options handed in are mutated; the frozen job is not.
    assert dict(job.prompt_options) == before
    for key, value in FRAGMENT.items():
        assert options[key] == value
    # The intent flags stay in the persisted config (plan §3.3).
    assert options['glossary_auto'] is True
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
    job = _job({'glossary_auto': True})
    log = LogRecorder()

    options = await _apply(job, log)

    assert options == {'glossary_auto': True}
    assert log.keys().count('auto_prep_error') == 1
    assert 'provider exploded' in log.entries[0][1]


@pytest.mark.asyncio
async def test_client_is_closed_even_when_the_builder_raises(patched):
    patched.raises = RuntimeError("boom")

    await _apply(_job({'style_auto': True}), LogRecorder())

    assert patched.factory.client.closed == 1


@pytest.mark.asyncio
async def test_no_client_is_created_when_nothing_is_wanted(patched):
    log = LogRecorder()

    await _apply(_job({}), log)

    assert patched.factory.calls == []
    assert patched.builder_calls == []
    assert log.entries == []


@pytest.mark.asyncio
async def test_unreadable_source_logs_and_returns(patched):
    job = _job({'glossary_auto': True}, text=None, input_path=None)
    log = LogRecorder()

    await _apply(job, log)

    assert patched.factory.calls == []
    assert log.keys() == ['auto_prep_no_text']


@pytest.mark.asyncio
async def test_unknown_provider_yields_no_llm_call(monkeypatch, patched):
    def unknown_provider(*args, **kwargs):
        raise LLMClientConfigError("Unknown LLM provider 'nope'.")

    monkeypatch.setattr(factory, 'create_llm_provider', unknown_provider)
    log = LogRecorder()

    options = await _apply(_job({'glossary_auto': True}), log)

    assert patched.builder_calls == []
    assert log.keys() == ['auto_prep_client_warning']
    assert 'glossary_terms' not in options


@pytest.mark.asyncio
async def test_create_provider_argument_overrides_the_factory(patched):
    own = ClientFactory()
    job = _job({'style_auto': True})

    await prepare.apply_auto_prep(job, job.engine_prompt_options(), LogRecorder(),
                                  create_provider=own)

    assert len(own.calls) == 1
    assert patched.factory.calls == []
    assert own.client.closed == 1


@pytest.mark.asyncio
async def test_client_is_built_with_the_jobs_own_provider(patched):
    key = "sk-test-not-a-real-key"
    job = _job(
        {'glossary_auto': True},
        provider=ProviderConfig(provider='openrouter', model='test-model',
                                endpoint=ENDPOINT, api_key=key, context_window=4096),
    )

    await _apply(job, LogRecorder())

    args, kwargs = patched.factory.calls[0]
    assert kwargs == {}
    (provider,) = args
    assert isinstance(provider, ProviderConfig)
    assert provider.provider == 'openrouter'
    assert provider.model == 'test-model'
    assert provider.endpoint == ENDPOINT
    assert provider.api_key == key
    assert provider.context_window == auto_prep.AUTO_PREP_CONTEXT_WINDOW


@pytest.mark.asyncio
async def test_refine_only_runs_style_with_target_as_source(patched):
    job = _job({'glossary_auto': True, 'style_auto': True}, operation=OP_REFINE)

    await _apply(job, LogRecorder())

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
        self.started.append(copy.deepcopy(dict(config.get('prompt_options', {}))))

    def update_job_config(self, translation_id, config):
        return True

    def set_job_phase(self, translation_id, phase):
        return True

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


async def _run_job(job, tmp_path, monkeypatch, is_resume=False):
    """Drive `perform_actual_translation` with every external seam faked out."""
    checkpoint_manager = FakeCheckpointManager()
    state_manager = FakeStateManager(checkpoint_manager)
    seen = {}

    async def fake_translate_file(engine_job, ctx):
        seen['prompt_options'] = copy.deepcopy(dict(engine_job.prompt_options))
        with open(engine_job.output_path, 'w', encoding='utf-8') as handle:
            handle.write('translated')
        stats = {'total_chunks': 1, 'completed_chunks': 1, 'failed_chunks': 0}
        ctx.emit(stats)
        return JobResult.from_run(stats, engine_job.output_path, engine_ok=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(adapters, 'translate_file', fake_translate_file)
    monkeypatch.setattr(handlers, 'emit_update', lambda *a, **k: None)
    monkeypatch.setattr('src.utils.notifier.notify', lambda *a, **k: None)

    job = job.with_changes(output_path=str(tmp_path / 'out.txt'))
    await handlers.perform_actual_translation(
        'job-1', job, state_manager, str(tmp_path), FakeSocketIO(), is_resume=is_resume
    )

    messages = [
        entry['message'] for entry in state_manager.fields.get('logs', [])
        if isinstance(entry, dict) and 'message' in entry
    ]
    return state_manager, checkpoint_manager, messages, seen


@pytest.mark.asyncio
async def test_auto_glossary_suppresses_the_legacy_loaded_line(tmp_path, monkeypatch, patched):
    job = _job({'glossary_auto': True, 'style_auto': True})

    state, checkpoint_manager, messages, seen = await _run_job(job, tmp_path, monkeypatch)

    assert state.fields['status'] == 'completed'
    assert not any('📖 Loaded glossary' in message for message in messages)
    # The derived snapshot is persisted with the job config (D5).
    assert checkpoint_manager.started[0]['glossary_terms'] == FRAGMENT['glossary_terms']
    assert checkpoint_manager.started[0]['glossary_source'] == 'auto'
    # The custom-instructions snapshot stayed a no-op, so the auto style survived.
    assert seen['prompt_options']['custom_instructions'] == FRAGMENT['custom_instructions']


@pytest.mark.asyncio
async def test_real_glossary_still_prints_the_loaded_line(tmp_path, monkeypatch, patched):
    job = _job({
        'glossary_terms': {'Avenue': 'Avenue'},
        'glossary_name': 'My Glossary',
    })

    _state, _checkpoint_manager, messages, _seen = await _run_job(job, tmp_path, monkeypatch)

    assert any("📖 Loaded glossary 'My Glossary' (1 terms)" in message for message in messages)
    assert patched.factory.calls == []


@pytest.mark.asyncio
async def test_resume_never_recomputes_the_auto_snapshot(tmp_path, monkeypatch, patched):
    snapshot = {'Avenue': 'Avenue des Ombres'}
    job = _job({
        'glossary_auto': True,
        'style_auto': True,
        'glossary_terms': dict(snapshot),
        'glossary_name': auto_prep.AUTO_GLOSSARY_NAME,
        'glossary_source': 'auto',
    })

    _state, checkpoint_manager, messages, seen = await _run_job(
        job, tmp_path, monkeypatch, is_resume=True)

    assert patched.factory.calls == []
    assert patched.builder_calls == []
    assert seen['prompt_options']['glossary_terms'] == snapshot
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
    progress = ProgressRecorder()

    await _apply(_job({'glossary_auto': True, 'style_auto': True}), LogRecorder(), progress)

    assert progress.states[0] is True
    assert progress.states[-1] is False
    assert progress.states.count(False) == 1


@pytest.mark.asyncio
async def test_progress_callback_is_cleared_when_the_builder_raises(patched):
    patched.raises = RuntimeError("provider exploded")
    progress = ProgressRecorder()

    await _apply(_job({'glossary_auto': True}), LogRecorder(), progress)

    # A failure must not strand the panel on "Preparing…".
    assert progress.states[0] is True
    assert progress.states[-1] is False


@pytest.mark.asyncio
async def test_progress_callback_is_cleared_when_the_document_is_unreadable(patched):
    progress = ProgressRecorder()

    await _apply(_job({'glossary_auto': True}, text='', input_path=None), LogRecorder(), progress)

    assert progress.states == [True, False]
    assert patched.factory.calls == []


@pytest.mark.asyncio
async def test_no_progress_signal_when_no_pass_is_wanted(patched):
    progress = ProgressRecorder()

    await _apply(_job({}), LogRecorder(), progress)

    assert progress.states == []


@pytest.mark.asyncio
async def test_a_raising_progress_callback_never_breaks_the_job(patched):
    def boom(active):
        raise ValueError("emit is broken")

    options = await _apply(_job({'style_auto': True}), LogRecorder(), boom)

    assert options['custom_instructions'] == FRAGMENT['custom_instructions']


@pytest.mark.asyncio
async def test_heartbeat_ticks_while_a_slow_pass_runs(patched, monkeypatch):
    monkeypatch.setattr(prepare, 'AUTO_PREP_HEARTBEAT_S', 0.01)

    async def slow_build(**kwargs):
        await asyncio.sleep(0.08)
        return dict(FRAGMENT)

    monkeypatch.setattr(auto_prep, 'build_auto_prompt_options', slow_build)
    progress = ProgressRecorder()

    await _apply(_job({'glossary_auto': True}), LogRecorder(), progress)

    # Start signal + several heartbeats + the final clear.
    assert progress.states.count(True) >= 3
    assert progress.states[-1] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
