"""
The job runner: the only orchestration of a translation / refine / term-fix
job, shared by the web server and the CLI.

Phases, each persisted in the checkpoint (``job_phase``) so a resume re-enters
the one that was running:

1. **prepare** (fresh jobs only): glossary snapshot, auto glossary/style,
   custom-instructions snapshot, checkpoint ``start_job``, output naming.
2. **translate** (or **term_fix**): one engine call.
3. **refine**: when requested, exactly once, as its own phase with its own
   checkpoint namespace. ``prompt_options['refine']`` and ``refine_after``
   both request it; the engines never refine inline when driven from here.
4. **finalize**: the verdict (``classify_completion``), the checkpoint's
   terminal status, upload/temp cleanup, notifications, the TTS hand-off.

A rate limit either pauses the job (default) or, with auto-pause off, waits
and re-enters the interrupted phase from the checkpoint: a loop here, not a
recursive call. A terminal checkpoint status and the temp-file cleanup are
guaranteed by ``try/finally``.

The runner never talks to Flask, Socket.IO or the console: it reports through
a :class:`JobSinks` object the caller supplies (``src/api/handlers.py`` for
the web, ``translate.py`` for the CLI).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import OP_REFINE, OP_TERM_FIX, JobConfig
from .context import JobContext
from .prepare import apply_auto_prep, snapshot_custom_instructions, snapshot_glossary
from .result import JobResult, JobStatus

_logger = logging.getLogger(__name__)

PHASE_TRANSLATE = 'translate'
PHASE_REFINE = 'refine'
PHASE_TERM_FIX = 'term_fix'

# Consecutive auto-resume cycles without progress before the user is warned.
_AUTO_RESUME_STUCK_WARNING = 3


def degraded_chunk_count(stats) -> int:
    """Chunks translated with only approximate tag positions (EPUB).

    Read from the live ``degraded_files`` map (``degraded_chunks`` as its
    flat fallback), never from the cumulative ``token_alignment_used`` event
    counter. Same precedence as the completion card, so retention and the
    retry button it enables never disagree.
    """
    files = stats.get('degraded_files')
    if isinstance(files, dict):
        return sum(len(indices or ()) for indices in files.values())
    try:
        return max(0, int(stats.get('degraded_chunks') or 0))
    except (TypeError, ValueError):
        return 0


def format_stats_summary(job: JobConfig, stats, result: JobResult) -> str:
    """The ' | 8/10 chunks (1 failed) (1 untranslated)' log suffix."""
    total = stats.get('total_chunks', 0) or 0
    if total <= 0:
        return ""
    completed = stats.get('completed_chunks', 0)
    unit = 'subtitles' if job.file_type == 'srt' else 'chunks'
    summary = f" | {completed}/{total} {unit}"
    verdict = result.verdict
    failed = verdict.failed_chunks if verdict else result.units_failed
    unfinished = verdict.unfinished_chunks if verdict else result.units_untranslated
    if failed > 0:
        summary += f" ({failed} failed)"
    # unfinished (not fallback_used) is what is still untranslated now:
    # fallback_used is a historical tally (issue #261, D9/D10).
    if unfinished > 0:
        summary += f" ({unfinished} untranslated)"
    return summary


class JobSinks:
    """Where a running job reports. Every method has a harmless default.

    The web adapter forwards to the state manager and Socket.IO, the CLI
    adapter to the console logger.
    """

    def log(self, key: str, message: str = "", data: Optional[dict] = None) -> None:
        """The job log (same ``(key, message, data)`` shape the engines use)."""

    def emit_stats(self, stats: Dict[str, Any]) -> None:
        """A stats update, already tagged with the workflow phase."""

    def is_interrupted(self) -> bool:
        return False

    def current_stats(self) -> Dict[str, Any]:
        """The merged stats the caller holds (web state); {} when it holds none."""
        return {}

    def set_status(self, status: str, message: Optional[str] = None) -> None:
        """A transient status change (``running``, ``rate_limited``)."""

    def job_changed(self, job: JobConfig) -> None:
        """The job config changed (output renamed, preserved input recorded)."""

    def job_started(self, job: JobConfig, is_resume: bool) -> None:
        """The prepare phase is over and the work starts."""

    def on_cost(self, cost_data: Dict[str, Any]) -> None:
        """OpenRouter cost totals (called from the provider, no emit here)."""

    def job_finished(self, job: JobConfig, result: JobResult, elapsed: float) -> None:
        """Terminal: completed, partial, failed (verdict) or interrupted."""

    def job_rate_limited(self, job: JobConfig, message: str) -> None:
        """Terminal for this run: paused on a rate limit, resumable."""

    def job_failed(self, job: JobConfig, error: str) -> None:
        """Terminal: an exception aborted the job."""

    async def run_tts(self, job: JobConfig, output_path: str) -> None:
        """Generate audio from the finished output (``job.tts`` is set)."""
        from src.tts.tts_config import TTSConfig
        from src.utils.file_utils import generate_tts_for_translation
        self.log("tts_phase_start", "🔊 Starting TTS audio generation...")
        success, message, audio_path = await generate_tts_for_translation(
            translated_filepath=output_path,
            target_language=job.target_language,
            tts_config=TTSConfig.from_dict(dict(job.tts or {})),
            log_callback=self.log,
        )
        if success:
            self.log("tts_complete", f"✅ TTS audio generated: {os.path.basename(audio_path)}")
        else:
            self.log("tts_failed", f"❌ TTS generation failed: {message}")

    def notify(self, event: str, context: Dict[str, Any]) -> None:
        from src.utils.notifier import notify
        notify(event, context)


class JobRunner:
    """Run one job through its phases. Build one per job; call :meth:`run` once.

    Args:
        job: The job configuration (credentials included, in memory only).
        translation_id: Checkpoint/job id.
        checkpoint_manager: ``CheckpointManager``.
        sinks: Where to report (see :class:`JobSinks`).
        is_resume: Continue a stored job instead of starting one.
        glossary_store: Callable returning the ``GlossaryStore`` used for the
            glossary snapshot (None: no ``glossary_id`` support).
        custom_instructions_dir: Where presets are read from.
        temp_dir: Where an inline-text job writes its temporary input.
        engines: Optional overrides of ``translate_file`` / ``refine_file`` /
            ``term_fix_file`` (tests).
    """

    def __init__(self, job: JobConfig, *, translation_id: str, checkpoint_manager,
                 sinks: Optional[JobSinks] = None, is_resume: bool = False,
                 glossary_store: Optional[Callable[[], Any]] = None,
                 custom_instructions_dir: Optional[Path] = None,
                 temp_dir: Optional[str] = None,
                 engines: Optional[Dict[str, Callable]] = None):
        self.job = job
        self.translation_id = translation_id
        self.checkpoint_manager = checkpoint_manager
        self.sinks = sinks or JobSinks()
        self.is_resume = is_resume
        self.glossary_store = glossary_store
        self.custom_instructions_dir = custom_instructions_dir or (Path(os.getcwd()) / 'Custom_Instructions')
        self.temp_dir = temp_dir
        self._engines = engines or {}
        self._terminal = False
        self._temp_input: Optional[str] = None
        self._started_at = time.time()
        self._progress_meta: Dict[str, Any] = {}
        self._stuck = {'index': None, 'count': 0}

    # -- public ------------------------------------------------------------

    async def run(self) -> JobResult:
        from src.core.llm.cost_tracker import begin_job_cost_tracking, end_job_cost_tracking
        from src.core.llm.exceptions import RateLimitError
        from src.core.llm.policy import begin_job_retry_policy, end_job_retry_policy

        cost_token = None
        if self.job.llm_provider == 'openrouter':
            _, cost_token = begin_job_cost_tracking(self.sinks.on_cost)
        # Per-job retry/timeout policy: every provider built during the job.
        policy_token = begin_job_retry_policy(self.job.retry_policy())
        heartbeat = asyncio.ensure_future(self._heartbeat())
        try:
            if not self.is_resume:
                await self._prepare()
            await self._start()
            while True:
                try:
                    result = await self._execute()
                    return await self._finalize(result)
                except RateLimitError as e:
                    if await self._should_auto_resume(e):
                        continue
                    return self._pause_on_rate_limit(e)
        except Exception as e:
            return await self._fail(e)
        finally:
            heartbeat.cancel()
            if cost_token is not None:
                end_job_cost_tracking(cost_token)
            end_job_retry_policy(policy_token)
            self._remove_temp_input()
            if not self._terminal:
                # Cancelled or crashed before any terminal status: never leave
                # the row at 'running' (it would be excluded from the
                # resumable list until the next restart).
                self._mark('mark_error')

    # -- phases ------------------------------------------------------------

    async def _prepare(self) -> None:
        """Snapshot everything that must not change for the rest of the book,
        create the checkpoint, resolve the output name."""
        job = self.job
        options = job.engine_prompt_options()
        snapshot_glossary(options, self.glossary_store)
        tmp_job = job.with_changes(prompt_options=options)
        await apply_auto_prep(tmp_job, options, self.sinks.log, self._auto_prep_progress)
        snapshot_custom_instructions(options, self.custom_instructions_dir, self.sinks.log)
        job = job.with_changes(prompt_options=options)

        persisted = job.to_persisted()
        self.checkpoint_manager.start_job(self.translation_id, job.file_type, persisted,
                                          job.input_path)
        if persisted.get('preserved_input_path'):
            job = job.with_changes(extras={**dict(job.extras),
                                           'preserved_input_path': persisted['preserved_input_path']})

        output_path = self._resolve_output_path(job)
        if os.path.basename(output_path) != job.output_filename:
            self.sinks.log("output_filename_modified",
                           f"ℹ️ Output filename modified to avoid overwriting: "
                           f"{job.output_filename} → {os.path.basename(output_path)}")
        job = job.with_changes(output_path=output_path)
        # Persist the name this run actually writes, so a resume targets it.
        self.checkpoint_manager.update_job_config(self.translation_id, job.to_persisted())
        self.job = job
        self.sinks.job_changed(job)

    async def _start(self) -> None:
        """Common to fresh and resumed runs: inline-text input, input check,
        legacy instruction snapshot, glossary report, start log."""
        job = self.job
        if self.is_resume:
            self.job = job = job.with_changes(output_path=self._resolve_output_path(job))
            # No-op for jobs created with a snapshot; older jobs read the file.
            options = job.engine_prompt_options()
            snapshot_custom_instructions(options, self.custom_instructions_dir, self.sinks.log)
            self.job = job = job.with_changes(prompt_options=options)
            # A resume may switch model/provider (issue #183): show it.
            self.sinks.job_changed(job)

        self.sinks.job_started(job, self.is_resume)

        if job.file_type == 'txt' and job.text is not None and job.input_path is None:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                             suffix=".txt", dir=self.temp_dir) as tmp_f:
                tmp_f.write(job.text)
                self._temp_input = tmp_f.name
            self.job = job = job.with_changes(input_path=self._temp_input)
        if not job.input_path:
            self.sinks.log("error_no_path",
                           f"❌ {job.file_type.upper()} translation requires a file path from upload.")
            raise ValueError(f"{job.file_type.upper()} translation requires a file_path.")

        options = job.engine_prompt_options()
        glossary_load_error = options.pop('glossary_load_error', None)
        if options.get('glossary_terms'):
            # An auto glossary already announced itself ("🧠 Auto glossary:").
            if options.get('glossary_source') != 'auto':
                self.sinks.log("glossary_loaded",
                               f"📖 Loaded glossary '{options.get('glossary_name', '?')}' "
                               f"({len(options['glossary_terms'])} terms)")
        elif glossary_load_error:
            self.sinks.log("glossary_error", f"⚠️ Could not load glossary: {glossary_load_error}")
        if glossary_load_error is not None:
            self.job = job.with_changes(prompt_options=options)

    async def _execute(self) -> JobResult:
        """Run the work phases from the one the checkpoint says is next."""
        job = self.job
        checkpoint = self._load_checkpoint() if self.is_resume else None
        stored_phase = (checkpoint or {}).get('job_phase') or PHASE_TRANSLATE

        if job.is_term_fix:
            return await self._run_term_fix()

        if job.is_refine_only:
            self.sinks.log("refine_only_mode",
                           "✨ Refine-only mode: skipping translation, polishing the input file as-is.")
            if job.source_language and job.target_language and job.source_language != job.target_language:
                self.sinks.log("refine_only_lang_mismatch",
                               f"⚠️ source_language ({job.source_language}) ≠ target_language "
                               f"({job.target_language}). Refinement is monolingual; the file "
                               f"will be polished as {job.target_language}.")
            resume_index = 0
            if checkpoint:
                resume_index = checkpoint.get('refine_resume_index') or 0
                if not resume_index and stored_phase != PHASE_REFINE:
                    # A refine-only job checkpointed before the refine
                    # namespace existed stored its cues as translation chunks.
                    resume_index = checkpoint.get('resume_from_index') or 0
            return await self._run_refine(job, resume_index,
                                          meta={'enable_refinement': False, 'refine_only': True,
                                                'current_phase': 1})

        refine_after = job.wants_refine_after
        if refine_after and not job.refine_supported:
            self.sinks.log("refine_after_unsupported_format",
                           f"ℹ️ Refinement is not available for {job.file_type.upper()} files yet; "
                           f"the translated file is kept as is.")
            refine_after = False

        resume_refine = (checkpoint is not None and stored_phase == PHASE_REFINE
                         and refine_after and os.path.exists(job.output_path))
        if resume_refine:
            self.sinks.log("resume_phase_refine",
                           "↻ Translation already finished — resuming the refinement pass.")
            translated = None
        else:
            resume_index = (checkpoint or {}).get('resume_from_index', 0) if checkpoint else 0
            if self.is_resume:
                self.sinks.log("resume_model_info",
                               f"↻ Resuming from chunk {resume_index} using "
                               f"{job.llm_provider} / {job.model}.")
            translated = await self._run_translate(job, resume_index, refine_after)
            if not refine_after or translated.status is not JobStatus.COMPLETED:
                return translated

        self.sinks.log("refine_after_start", "✨ Translation done — running refinement pass on the output.")
        refine_index = (checkpoint or {}).get('refine_resume_index', 0) if resume_refine else 0
        refine_job = job.with_changes(input_path=job.output_path)
        return await self._run_refine(refine_job, refine_index,
                                      meta={'enable_refinement': True, 'current_phase': 2})

    async def _run_translate(self, job: JobConfig, resume_index: int, refine_after: bool) -> JobResult:
        translate_file = self._engine('translate_file')
        self._set_phase(PHASE_TRANSLATE)
        options = job.engine_prompt_options()
        # Refinement is the runner's refine phase, never the engine's inline pass.
        options['refine'] = False
        engine_job = job.with_changes(prompt_options=options)
        ctx = self._context(resume_index, {'enable_refinement': refine_after, 'current_phase': 1})
        return await translate_file(engine_job, ctx)

    async def _run_refine(self, job: JobConfig, resume_index: int, meta: Dict[str, Any]) -> JobResult:
        refine_file = self._engine('refine_file')
        self._set_phase(PHASE_REFINE)
        options = job.engine_prompt_options()
        options['refine'] = False
        ctx = self._context(resume_index, meta, checkpoint_phase=PHASE_REFINE)
        return await refine_file(job.with_changes(prompt_options=options), ctx)

    async def _run_term_fix(self) -> JobResult:
        from src.core.term_fix import load_rules_file, parse_rules_text
        term_fix_file = self._engine('term_fix_file')
        job = self.job
        spec = dict(job.term_fix or {})
        mode = spec.get('mode', 'smart')
        self._set_phase(PHASE_TERM_FIX)
        self.sinks.log("term_fix_mode",
                       f"🔤 Term-fix mode ({mode}): only the listed wrong renderings are changed, "
                       f"the rest of the file is kept as is.")
        if spec.get('rules_text'):
            rules = parse_rules_text(spec['rules_text'])
        else:
            rules = load_rules_file(spec['rules_file'])
        ctx = self._context(0, {'enable_refinement': False, 'refine_only': True,
                                'term_fix': True, 'current_phase': 1})
        return await term_fix_file(job, ctx, rules)

    async def _finalize(self, result: JobResult) -> JobResult:
        """Verdict, terminal checkpoint status, cleanup, notifications, TTS."""
        from src.utils.file_utils import find_partial_output_paths
        from src.utils.notifier import EVENT_FAILURE, EVENT_INTERRUPTION, EVENT_SUCCESS

        job = self.job
        output_path = result.output_path or job.output_path
        if result.status is JobStatus.INTERRUPTED and not os.path.exists(output_path):
            # A paused EPUB is saved with a `[partial NN%]` prefix.
            candidates = find_partial_output_paths(job.output_path)
            if candidates:
                output_path = max(candidates, key=os.path.getmtime)
        if output_path != job.output_path:
            self.job = job = job.with_changes(output_path=output_path)
            if result.status is JobStatus.INTERRUPTED:
                self.sinks.log("output_marked_partial", f"💾 Partial EPUB saved as: {job.output_filename}")
            self.sinks.job_changed(job)
        result = result.with_changes(output_path=output_path)

        elapsed = time.time() - self._started_at
        self.sinks.emit_stats({'elapsed_time': elapsed})
        stats = {**dict(result.stats or {}), **self.sinks.current_stats()}

        if result.status is JobStatus.INTERRUPTED:
            self.sinks.log("summary_interrupted",
                           f"🛑 Translation interrupted - partial result saved ({elapsed:.2f}s)")
            await asyncio.to_thread(self.sinks.notify, EVENT_INTERRUPTION, self._notice(elapsed))
            self._mark('mark_interrupted')
            self._cleanup_upload(interrupted=True)
        elif result.status is JobStatus.FAILED:
            self.sinks.log("summary_error_no_output",
                           f"❌ Translation finished without a usable output file in {elapsed:.2f}s"
                           f"{format_stats_summary(job, stats, result)} — {result.error}")
            self._mark('mark_error')
            await asyncio.to_thread(self.sinks.notify, EVENT_FAILURE,
                                    self._notice(elapsed, error=result.error))
        elif result.status is JobStatus.PARTIAL:
            failed = result.verdict.failed_chunks if result.verdict else result.units_failed
            unfinished = result.verdict.unfinished_chunks if result.verdict else result.units_untranslated
            self.sinks.log("summary_partial",
                           f"⚠️ Translation finished with {failed} failed and {unfinished} chunk(s) "
                           f"still untranslated in {elapsed:.2f}s"
                           f"{format_stats_summary(job, stats, result)} — checkpoint kept for retry")
            self._mark('mark_partial')
            self._cleanup_upload()
        else:
            self.sinks.log("summary_completed",
                           f"✅ Translation completed in {elapsed:.2f}s{format_stats_summary(job, stats, result)}")
            await asyncio.to_thread(self.sinks.notify, EVENT_SUCCESS, self._notice(elapsed))
            self._mark('mark_completed')
            degraded = degraded_chunk_count(stats)
            if degraded > 0:
                # The completion card offers "retry them with another model",
                # which needs the checkpoint, so it is kept (design D3).
                self.sinks.log("checkpoint_kept_degraded",
                               f"📌 Checkpoint kept: {degraded} chunk(s) have approximate tag "
                               f"placement and can be retried from the completion card while "
                               f"it is on screen")
            else:
                self._safe(lambda: self.checkpoint_manager.prune_job_data(self.translation_id))
            self._cleanup_upload()

        if job.llm_provider == 'openrouter':
            cost = stats.get('openrouter_cost', 0.0) or 0.0
            prompt_tokens = stats.get('openrouter_prompt_tokens', 0) or 0
            completion_tokens = stats.get('openrouter_completion_tokens', 0) or 0
            if cost > 0 or prompt_tokens + completion_tokens > 0:
                estimated = ' (estimate, usage.cost missing)' if stats.get('openrouter_cost_estimated') else ''
                self.sinks.log("openrouter_cost_final",
                               f"💰 OpenRouter Cost{estimated}: ${cost:.4f} | Tokens: "
                               f"{prompt_tokens + completion_tokens:,} ({prompt_tokens:,} prompt + "
                               f"{completion_tokens:,} completion)")

        if job.tts and result.status is JobStatus.COMPLETED:
            try:
                await self.sinks.run_tts(job, output_path)
            except Exception as e:
                self.sinks.log("tts_error", f"❌ TTS generation error: {e}")

        self.sinks.job_finished(job, result, elapsed)
        return result

    # -- rate limit --------------------------------------------------------

    async def _should_auto_resume(self, error) -> bool:
        """Auto-pause off: wait, then re-enter from the checkpoint (True)."""
        from src.config import RATE_LIMIT_AUTO_RESUME_DELAY
        if self.job.auto_pause_on_rate_limit:
            return False
        provider_name = error.provider or self.job.llm_provider
        retry_msg = f" Retry suggested after ~{error.retry_after}s." if error.retry_after else ""
        wait_seconds = error.retry_after or RATE_LIMIT_AUTO_RESUME_DELAY
        wait_msg = (f"⏳ Rate limited by {provider_name}.{retry_msg} "
                    f"Auto-resume in {wait_seconds}s (auto-pause disabled).")
        self.sinks.log("rate_limit_auto_resume", wait_msg)
        self.sinks.set_status('rate_limited', wait_msg)

        deadline = time.monotonic() + float(wait_seconds)
        while time.monotonic() < deadline:
            if self.sinks.is_interrupted():
                break
            await asyncio.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        if self.sinks.is_interrupted():
            self.sinks.log("rate_limit_auto_resume_cancelled",
                           "🛑 Auto-resume cancelled by user, pausing instead.")
            return False

        checkpoint = self._load_checkpoint()
        if not checkpoint:
            self.sinks.log("rate_limit_no_checkpoint",
                           "⚠️ Auto-resume requested but no checkpoint found, falling back to pause.")
            return False

        index = (checkpoint.get('refine_resume_index') if checkpoint.get('job_phase') == PHASE_REFINE
                 else checkpoint.get('resume_from_index'))
        if self._stuck['index'] == index:
            self._stuck['count'] += 1
        else:
            self._stuck = {'index': index, 'count': 1}
        if self._stuck['count'] >= _AUTO_RESUME_STUCK_WARNING:
            # Opted in, so keep going, but say the quota may be exhausted.
            self.sinks.log("rate_limit_auto_resume_stuck",
                           f"⚠️ Auto-resume has looped {self._stuck['count']} times without "
                           f"progress (still stuck at chunk {index}). Your daily quota may be "
                           f"exhausted or the provider is throttling this account; consider "
                           f"interrupting now and checking your provider's quota dashboard. "
                           f"Auto-resume will keep trying but may be wasting time.")
        self._mark('mark_running', terminal=False)
        self.sinks.set_status('running', f"▶️ Auto-resuming from chunk {index}...")
        self.is_resume = True
        return True

    def _pause_on_rate_limit(self, error) -> JobResult:
        provider_name = error.provider or self.job.llm_provider
        retry_msg = f" Retry suggested after ~{error.retry_after}s." if error.retry_after else ""
        message = (f"⏸️ Rate limited by {provider_name}.{retry_msg} Translation auto-paused, "
                   f"you can resume when ready.")
        self.sinks.log("rate_limit_auto_pause", message)
        self._mark('mark_interrupted')
        self.sinks.emit_stats({'elapsed_time': time.time() - self._started_at})
        self.sinks.job_rate_limited(self.job, message)
        return JobResult(status=JobStatus.RATE_LIMITED, output_path=self.job.output_path,
                         error=message, stats=self.sinks.current_stats())

    # -- failure -----------------------------------------------------------

    async def _fail(self, error: Exception) -> JobResult:
        from src.utils.notifier import EVENT_FAILURE
        message = f"Critical error during translation task ({self.translation_id}): {error}"
        self.sinks.log("critical_error_perform_task", message)
        self.sinks.log("critical_error_perform_task_traceback", traceback.format_exc())
        self._mark('mark_error')
        self.sinks.job_failed(self.job, message)
        try:
            await asyncio.to_thread(self.sinks.notify, EVENT_FAILURE,
                                    self._notice(time.time() - self._started_at, error=message))
        except Exception as notify_error:  # pragma: no cover - best effort
            _logger.warning("Failure notification failed: %s", notify_error)
        return JobResult.failed(message, output_path=self.job.output_path,
                                stats=self.sinks.current_stats())

    # -- helpers -----------------------------------------------------------

    def _engine(self, name: str):
        if name in self._engines:
            return self._engines[name]
        if name == 'term_fix_file':
            from src.core.adapters import term_fix_file
            return term_fix_file
        from src.core import adapters
        return getattr(adapters, name)

    def _context(self, resume_index: int, meta: Dict[str, Any],
                 checkpoint_phase: str = PHASE_TRANSLATE) -> JobContext:
        def stats_callback(stats):
            self.sinks.emit_stats({**dict(stats or {}), **meta})

        reported = []

        def check_interruption():
            if self.sinks.is_interrupted():
                if not reported:
                    reported.append(True)
                    self.sinks.log("interruption_check",
                                   f"Interruption signal detected for job {self.translation_id}. "
                                   f"Halting processing.")
                return True
            return False

        return JobContext(
            translation_id=self.translation_id,
            checkpoint_manager=self.checkpoint_manager,
            log_callback=self.sinks.log,
            stats_callback=stats_callback,
            check_interruption=check_interruption,
            resume_from_index=resume_index or 0,
            checkpoint_phase=checkpoint_phase,
        )

    def _auto_prep_progress(self, active: bool) -> None:
        self.sinks.emit_stats({'auto_prep_active': bool(active)})

    def _resolve_output_path(self, job: JobConfig) -> str:
        """Fresh jobs never overwrite (a " (1)" suffix); a resume rewrites
        the file the previous pass produced. Web names are confined to the
        output directory."""
        from src.utils.file_utils import get_unique_output_path
        if job.origin == 'web':
            from src.api.services.path_validator import PathValidator
            path = str(PathValidator.resolve_output_file(job.output_dir, job.output_filename))
        else:
            path = job.output_path
        return path if self.is_resume else get_unique_output_path(path)

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        try:
            return self.checkpoint_manager.load_checkpoint(self.translation_id)
        except Exception as e:
            _logger.warning("Could not load checkpoint %s: %s", self.translation_id, e)
            return None

    def _set_phase(self, phase: str) -> None:
        self._safe(lambda: self.checkpoint_manager.set_job_phase(self.translation_id, phase))

    def _mark(self, method: str, terminal: bool = True) -> None:
        def call():
            getattr(self.checkpoint_manager, method)(self.translation_id)
        self._safe(call)
        if terminal:
            self._terminal = True

    def _safe(self, fn) -> None:
        try:
            fn()
        except Exception as e:  # checkpoint bookkeeping never fails a job
            _logger.warning("Checkpoint update failed for %s: %s", self.translation_id, e)

    async def _heartbeat(self) -> None:
        from src.persistence.database import OWNER_HEARTBEAT_INTERVAL
        touch = getattr(self.checkpoint_manager, 'touch_owner', None)
        if touch is None:
            return
        while True:
            await asyncio.sleep(OWNER_HEARTBEAT_INTERVAL)
            self._safe(lambda: touch(self.translation_id))

    def _remove_temp_input(self) -> None:
        if not self._temp_input:
            return
        try:
            os.remove(self._temp_input)
        except FileNotFoundError:
            pass
        except OSError as exc:
            _logger.warning("Could not remove temporary input file %s: %s", self._temp_input, exc)
        self._temp_input = None

    def _cleanup_upload(self, interrupted: bool = False) -> None:
        """Delete the web upload once it is no longer the only copy.

        Only web jobs (a CLI input is the user's own file), only a file
        directly inside ``<output_dir>/uploads``. On an interruption the
        upload goes only when the checkpoint preserved a copy for the resume.
        """
        job = self.job
        if job.origin != 'web' or not job.input_path or job.input_path == self._temp_input:
            return
        upload_path = Path(job.input_path)
        preserved = (job.extras or {}).get('preserved_input_path')
        if interrupted:
            if not (preserved and Path(preserved).exists()):
                self.sinks.log("cleanup_skipped_no_preserve",
                               "ℹ️ Skipped cleanup - preserved file not found, keeping original for resume")
                return
            if not upload_path.exists() or upload_path == Path(preserved):
                self.sinks.log("cleanup_info", "ℹ️ Original upload file not found or already cleaned up")
                return
        elif not upload_path.exists():
            return
        try:
            uploads_dir = Path(self.temp_dir or job.output_dir) / 'uploads'
            if upload_path.resolve().parent == uploads_dir.resolve():
                upload_path.unlink()
                if interrupted:
                    self.sinks.log("cleanup_uploaded_file",
                                   f"🗑️ Cleaned up uploaded source file (preserved copy exists): "
                                   f"{upload_path.name}")
            elif interrupted:
                self.sinks.log("cleanup_skipped", "ℹ️ Skipped cleanup - file is not in uploads root directory")
        except Exception as e:
            self.sinks.log("cleanup_error", f"⚠️ Could not delete uploaded file {upload_path.name}: {e}")

    def _notice(self, elapsed: float, error: Optional[str] = None) -> Dict[str, Any]:
        job = self.job
        context = {
            'translation_id': self.translation_id,
            'file': job.input_filename or job.input_path,
            'output': job.output_filename,
            'duration_seconds': elapsed,
            'provider': job.llm_provider,
            'model': job.model,
            'source_lang': None if job.operation == OP_REFINE else job.source_language,
            'target_lang': job.target_language,
            'mode': {OP_REFINE: 'refine-only', OP_TERM_FIX: 'term-fix'}.get(job.operation, 'translate'),
        }
        if error:
            context['error'] = error
        return context
