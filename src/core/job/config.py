"""
The job configuration: everything one translation / refine / term-fix job
needs, built in exactly one place per source.

* :meth:`JobConfig.from_web_request` turns a ``POST /api/translate`` body into
  a config (validation, upload-path confinement, credential guard).
* :meth:`JobConfig.from_cli_args` turns the ``translate.py`` arguments into one.
* :meth:`JobConfig.to_persisted` / :meth:`JobConfig.from_checkpoint` are the
  persistence pair. The persisted form is the flat dict the jobs database has
  always held (so existing checkpoints stay resumable and the frontend keeps
  reading the same ``config`` fields) and it never contains a secret: keys
  are re-resolved at resume through ``src/core/llm/credentials.py`` and the
  ``credential_guard`` rules.

A ``JobConfig`` is frozen. The runner derives the variants it needs (resolved
output name, refine-pass input, snapshotted prompt options) with
:meth:`JobConfig.with_changes`.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from src.core.llm.credentials import (
    KEY_PROVIDERS,
    env_key,
    key_required,
    normalize_provider,
    provider_env_var,
    resolve_provider_key,
)
from src.core.llm.policy import RetryPolicy
from src.core.llm.provider_config import ProviderConfig

OP_TRANSLATE = 'translate'
OP_REFINE = 'refine'
OP_TERM_FIX = 'term_fix'
OPERATIONS = (OP_TRANSLATE, OP_REFINE, OP_TERM_FIX)

#: Formats with a refine pass. PDF output is a rebuilt, reflowed document and
#: PGN translates comments only; neither can be refined yet.
REFINE_FORMATS = ('txt', 'srt', 'epub', 'docx')

#: Providers that talk to a user-supplied endpoint; the others use a built-in one.
ENDPOINT_PROVIDERS = ('ollama', 'openai')

#: Keys of the persisted form that are owned by JobConfig fields. Any other
#: key found in a stored config is kept verbatim in ``extras`` and written
#: back, so a field added by another layer (``preserved_input_path`` from the
#: checkpoint manager, for instance) survives a round trip.
_OWNED_KEYS = frozenset({
    'source_language', 'target_language', 'model', 'llm_api_endpoint', 'llm_provider',
    'request_timeout', 'timeout', 'context_window', 'max_attempts', 'retry_delay',
    'parallel_workers', 'output_filename', 'prompt_options', 'auto_pause_on_rate_limit',
    'bilingual_output', 'refine_only', 'refine_after', 'term_fix', 'tts_enabled',
    'tts_config', 'file_type', 'file_path', 'text', 'input_filename',
    'auto_adjust_context', 'min_chunk_size', 'max_tokens_per_chunk', 'retry_token_aligned',
    'origin', 'output_dir', 'refine_model', 'refine_llm_provider', 'refine_llm_api_endpoint',
    # Runtime-only markers older handlers wrote into the in-memory config.
    'is_resume', 'resume_from_index', '_auto_resume_last_index', '_auto_resume_stuck_count',
    # Keys of the slim config CLI jobs stored before v1.6.0 (issue #341).
    'input_file_path', 'output_file_path', 'model_name',
})

_UPLOAD_PREFIX_RE = re.compile(r'^[0-9a-f]{16}_')


class JobConfigError(ValueError):
    """A request or argument set that cannot become a job.

    ``status`` is the HTTP status the web layer answers with and ``payload``
    the JSON body; the CLI prints ``str(error)``.
    """

    def __init__(self, message: str, status: int = 400, payload: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.payload = payload if payload is not None else {"error": message}


def _cfg():
    """``src.config`` read at call time, so ``reload_config()`` is honoured."""
    import src.config as config
    return config


def _as_int(value, default, *, name=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        if name is not None:
            raise JobConfigError(f"{name} must be an integer")
        return default


def _clamp_parallel_workers(value) -> int:
    """Clamp the requested worker count to [1, MAX_PARALLEL_TRANSLATIONS].

    Falls back to the PARALLEL_TRANSLATIONS default when absent or malformed.
    Local-provider gating happens later in resolve_parallel_workers().
    """
    cfg = _cfg()
    if value is None:
        return cfg.PARALLEL_TRANSLATIONS
    try:
        return max(1, min(cfg.MAX_PARALLEL_TRANSLATIONS, int(value)))
    except (TypeError, ValueError):
        return cfg.PARALLEL_TRANSLATIONS


def display_input_filename(requested: object, safe_path) -> Optional[str]:
    """Human-readable source filename (issue #271).

    Prefers the client-supplied ``input_filename`` (basename only, truncated);
    falls back to the validated upload path with its random upload prefix
    stripped. Returns None when neither is available.
    """
    if isinstance(requested, str) and requested.strip():
        return os.path.basename(requested.strip())[:200]
    if safe_path is not None:
        return _UPLOAD_PREFIX_RE.sub('', safe_path.name, count=1)
    return None


def parse_term_fix_request(raw, check_mode: bool = True):
    """Validate a ``term_fix`` request field: {rules_text, mode}.

    Returns (normalized dict, None) or (None, error message). The rules are
    parsed here so a malformed rule fails the request with its line number
    instead of failing the job later.
    """
    from src.core.term_fix import (
        MODE_LITERAL, MODE_SMART, MODES, RuleParseError, check_literal_rules, parse_rules_text,
    )

    if not isinstance(raw, dict):
        return None, "term_fix must be an object with 'rules_text' and 'mode'."
    rules_text = raw.get('rules_text')
    if not isinstance(rules_text, str) or not rules_text.strip():
        return None, "Term fix: no correction rule given."
    mode = raw.get('mode') or MODE_SMART
    if check_mode and mode not in MODES:
        return None, f"Term fix: unknown mode '{mode}'."
    try:
        rules = parse_rules_text(rules_text)
        if check_mode and mode == MODE_LITERAL:
            check_literal_rules(rules)
    except (RuleParseError, ValueError) as e:
        return None, f"Term fix: {e}"
    return {'rules_text': rules_text, 'mode': mode}, None


@dataclass(frozen=True)
class JobConfig:
    """Everything one job needs. Immutable; derive variants with ``with_changes``.

    Attributes:
        operation: ``translate``, ``refine`` (refine-only) or ``term_fix``.
        file_type: Detected/declared format (txt, srt, epub, docx, pdf, pgn).
        input_path: File to read. None for an inline-text web job (``text``).
        text: Inline text of a web TXT job without an upload.
        output_path: File the job writes. The web builder sets the requested
            name inside the output directory; the runner resolves collisions.
        input_filename: Display name of the source (never a server path).
        source_language / target_language: Language names.
        refine_after: Run a refine pass on the translated output.
        term_fix: ``{'mode', 'rules_text'}`` (web) or ``{'mode', 'rules_file'}`` (CLI).
        provider: The translation provider. Its ``api_key`` lives only in
            memory; ``to_persisted`` never writes it.
        refine_provider: Provider of the refine pass when it differs.
        context_window / auto_adjust_context / min_chunk_size /
            max_tokens_per_chunk: Context and chunking parameters.
        parallel_workers: Requested concurrent chunks (local providers are
            forced to 1 downstream).
        request_timeout / max_attempts / retry_delay: Per-job retry policy
            overrides (None keeps the ``.env`` default).
        prompt_options: Prompt switches and the snapshots taken at job start
            (glossary, custom instructions, auto style). Treat as read-only.
        bilingual_output: Interleave source and translation.
        tts: ``TTSConfig.to_dict()`` when audio generation follows the job.
        auto_pause_on_rate_limit: Pause on a rate limit (False = auto-resume).
        retry_token_aligned: EPUB explicit retry of approximately-tagged chunks.
        origin: ``web`` or ``cli``; decides output naming.
        extras: Persisted keys owned by other layers, kept verbatim.
        notices: Warnings produced while building (CLI), never persisted.
    """

    operation: str = OP_TRANSLATE
    file_type: str = 'txt'
    input_path: Optional[str] = None
    text: Optional[str] = None
    output_path: str = ''
    input_filename: Optional[str] = None
    source_language: str = ''
    target_language: str = ''
    refine_after: bool = False
    term_fix: Optional[Mapping[str, Any]] = None
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    refine_provider: Optional[ProviderConfig] = None
    context_window: Optional[int] = None
    auto_adjust_context: bool = True
    min_chunk_size: int = 5
    max_tokens_per_chunk: Optional[int] = None
    parallel_workers: int = 1
    request_timeout: Optional[float] = None
    max_attempts: Optional[int] = None
    retry_delay: Optional[float] = None
    prompt_options: Mapping[str, Any] = field(default_factory=dict)
    bilingual_output: bool = False
    tts: Optional[Mapping[str, Any]] = None
    auto_pause_on_rate_limit: bool = True
    retry_token_aligned: bool = False
    origin: str = 'cli'
    extras: Mapping[str, Any] = field(default_factory=dict)
    notices: Tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self):
        if self.operation not in OPERATIONS:
            raise JobConfigError(f"Unknown job operation '{self.operation}'")
        object.__setattr__(self, 'file_type', (self.file_type or 'txt').lower())
        if self.prompt_options is None:
            object.__setattr__(self, 'prompt_options', {})

    # -- derived views -----------------------------------------------------

    def with_changes(self, **changes: Any) -> "JobConfig":
        return replace(self, **changes)

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def llm_provider(self) -> str:
        return self.provider.provider or 'ollama'

    @property
    def endpoint(self) -> Optional[str]:
        return self.provider.endpoint

    @property
    def output_filename(self) -> str:
        return os.path.basename(self.output_path)

    @property
    def output_dir(self) -> str:
        return os.path.dirname(self.output_path)

    @property
    def is_refine_only(self) -> bool:
        return self.operation == OP_REFINE

    @property
    def is_term_fix(self) -> bool:
        return self.operation == OP_TERM_FIX

    @property
    def wants_refine_after(self) -> bool:
        """A translate job that should be followed by exactly one refine pass.

        ``prompt_options['refine']`` and ``refine_after`` mean the same thing
        (the web UI sets both, the CLI ``--refine`` flag sets the first); the
        runner honours either once, as a separate refine phase.
        """
        return self.operation == OP_TRANSLATE and bool(
            self.refine_after or (self.prompt_options or {}).get('refine'))

    @property
    def refine_supported(self) -> bool:
        return self.file_type in REFINE_FORMATS

    def refine_provider_config(self) -> ProviderConfig:
        return self.refine_provider or self.provider

    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy.from_job_config({
            'request_timeout': self.request_timeout,
            'max_attempts': self.max_attempts,
            'retry_delay': self.retry_delay,
        })

    def engine_prompt_options(self) -> Dict[str, Any]:
        """A private, mutable copy of the prompt options for one engine call."""
        return copy.deepcopy(dict(self.prompt_options or {}))

    def provider_for(self, *, context_window: Optional[int] = None, log_callback=None,
                     refine: bool = False) -> ProviderConfig:
        """The provider config an engine builds its client from."""
        base = self.refine_provider_config() if refine else self.provider
        changes: Dict[str, Any] = {}
        if context_window is not None:
            changes['context_window'] = context_window
        if log_callback is not None:
            changes['log_callback'] = log_callback
        return base.with_changes(**changes) if changes else base

    # -- persistence -------------------------------------------------------

    def to_persisted(self) -> Dict[str, Any]:
        """The flat, secret-free dict stored with the job.

        Same keys as the config the jobs database always held, so old rows
        and the frontend's ``config`` readers keep working.
        """
        data: Dict[str, Any] = dict(self.extras or {})
        data.update({
            'source_language': self.source_language,
            'target_language': self.target_language,
            'model': self.provider.model,
            'llm_api_endpoint': self.provider.endpoint,
            'llm_provider': self.llm_provider,
            'request_timeout': self.request_timeout,
            'context_window': self.context_window,
            'max_attempts': self.max_attempts,
            'retry_delay': self.retry_delay,
            'parallel_workers': self.parallel_workers,
            'output_filename': self.output_filename,
            'prompt_options': copy.deepcopy(dict(self.prompt_options or {})),
            'auto_pause_on_rate_limit': self.auto_pause_on_rate_limit,
            'bilingual_output': self.bilingual_output,
            'refine_only': self.is_refine_only,
            'refine_after': self.refine_after,
            'term_fix': dict(self.term_fix) if self.term_fix else None,
            'tts_enabled': bool(self.tts),
            'tts_config': dict(self.tts) if self.tts else None,
            'file_type': self.file_type,
            'auto_adjust_context': self.auto_adjust_context,
            'min_chunk_size': self.min_chunk_size,
            'max_tokens_per_chunk': self.max_tokens_per_chunk,
            'retry_token_aligned': self.retry_token_aligned,
            'origin': self.origin,
        })
        if self.origin == 'cli':
            # A CLI job writes next to the user's files, not into a server
            # output directory, so `translate.py --resume` needs the folder.
            data['output_dir'] = self.output_dir
        if self.input_path is not None:
            data['file_path'] = self.input_path
        if self.text is not None:
            data['text'] = self.text
        if self.input_filename is not None:
            data['input_filename'] = self.input_filename
        if self.refine_provider is not None:
            data['refine_model'] = self.refine_provider.model
            data['refine_llm_provider'] = self.refine_provider.provider
            data['refine_llm_api_endpoint'] = self.refine_provider.endpoint
        # Belt and braces: whatever extras carried, no key leaves this method.
        return {k: v for k, v in data.items()
                if not (k == 'api_key' or k.endswith('_api_key'))}

    @classmethod
    def from_persisted(cls, stored: Mapping[str, Any], *, output_dir: Optional[str] = None,
                       input_path: Optional[str] = None) -> "JobConfig":
        """Rebuild a config from its persisted form, keys from ``.env``.

        Used for an auto-resume inside the server and by
        :meth:`from_checkpoint`. ``output_dir`` joins the stored bare output
        name; ``input_path`` replaces the stored path (the preserved copy).
        """
        stored = dict(stored or {})
        provider_name = normalize_provider(stored.get('llm_provider'))
        if stored.get('term_fix'):
            operation = OP_TERM_FIX
        elif stored.get('refine_only'):
            operation = OP_REFINE
        else:
            operation = OP_TRANSLATE
        # CLI jobs created before v1.6.0 stored a slim config: a full
        # 'output_file_path', 'input_file_path' and 'model_name', and no
        # 'origin'. Rebuild those as CLI jobs writing to the stored path
        # (issue #341).
        legacy_output = stored.get('output_file_path')
        if legacy_output and not stored.get('output_filename'):
            stored.setdefault('origin', 'cli')
            if output_dir is None:
                output_dir, stored['output_filename'] = os.path.split(legacy_output)
            else:
                stored['output_filename'] = os.path.basename(legacy_output)
        if not stored.get('model') and stored.get('model_name'):
            stored['model'] = stored['model_name']
        if not stored.get('file_path') and stored.get('input_file_path'):
            stored['file_path'] = stored['input_file_path']
        output_name = stored.get('output_filename') or ''
        output_dir = output_dir if output_dir is not None else stored.get('output_dir')
        output_path = os.path.join(output_dir, output_name) if output_dir else output_name
        refine_provider = None
        if stored.get('refine_model') or stored.get('refine_llm_provider'):
            refine_name = normalize_provider(stored.get('refine_llm_provider') or provider_name)
            refine_provider = ProviderConfig(
                provider=refine_name,
                model=stored.get('refine_model') or stored.get('model') or '',
                endpoint=stored.get('refine_llm_api_endpoint') or stored.get('llm_api_endpoint'),
                api_key=env_key(refine_name) or None,
                context_window=stored.get('context_window'),
            )
        extras = {k: v for k, v in stored.items()
                  if k not in _OWNED_KEYS and not (k == 'api_key' or k.endswith('_api_key'))}
        return cls(
            operation=operation,
            file_type=stored.get('file_type') or 'txt',
            input_path=input_path if input_path is not None else stored.get('file_path'),
            text=stored.get('text'),
            output_path=output_path,
            input_filename=stored.get('input_filename'),
            source_language=stored.get('source_language') or '',
            target_language=stored.get('target_language') or '',
            refine_after=bool(stored.get('refine_after')),
            term_fix=stored.get('term_fix') or None,
            provider=ProviderConfig(
                provider=provider_name,
                model=stored.get('model') or '',
                endpoint=stored.get('llm_api_endpoint'),
                api_key=env_key(provider_name) or None,
                context_window=stored.get('context_window'),
            ),
            refine_provider=refine_provider,
            context_window=stored.get('context_window'),
            auto_adjust_context=stored.get('auto_adjust_context', True),
            min_chunk_size=stored.get('min_chunk_size') or 5,
            max_tokens_per_chunk=stored.get('max_tokens_per_chunk'),
            parallel_workers=stored.get('parallel_workers') or 1,
            request_timeout=stored.get('request_timeout', stored.get('timeout')),
            max_attempts=stored.get('max_attempts'),
            retry_delay=stored.get('retry_delay'),
            prompt_options=copy.deepcopy(stored.get('prompt_options') or {}),
            bilingual_output=bool(stored.get('bilingual_output')),
            tts=(stored.get('tts_config') or None) if stored.get('tts_enabled') else None,
            auto_pause_on_rate_limit=stored.get('auto_pause_on_rate_limit', True),
            retry_token_aligned=bool(stored.get('retry_token_aligned')),
            origin=stored.get('origin') or 'web',
            extras=extras,
        )

    @classmethod
    def from_checkpoint(cls, stored: Mapping[str, Any], *, output_dir: str,
                        input_path: Optional[str] = None,
                        overrides: Optional[Mapping[str, Any]] = None) -> "JobConfig":
        """Rebuild a stored job for a user-initiated resume.

        ``overrides`` is the resume request body: it may switch model,
        provider, endpoint and context window for the remaining work (issue
        #183) and carry an ``api_key`` for the selected provider. Credentials
        are validated even with an empty body, since checkpoints never hold
        keys (issue #213): every resume must find its key in ``.env`` or in
        the request, and a request-chosen host must bring its own key.

        ``retry_token_aligned`` is set from the request on every resume
        (False when absent), so a stored value can never widen a routine
        resume (issue #261, D3). ``auto_pause_on_rate_limit`` is refreshed
        from the live setting unless the request overrides it.

        Raises:
            JobConfigError: with the HTTP status and body to answer.
        """
        from src.api.services.credential_guard import guard_request_credentials, is_endpoint_override

        job = cls.from_persisted(stored, output_dir=output_dir, input_path=input_path)
        overrides = overrides if isinstance(overrides, Mapping) else {}

        retry_token_aligned = overrides.get('retry_token_aligned')
        if retry_token_aligned is not None and not isinstance(retry_token_aligned, bool):
            raise JobConfigError("retry_token_aligned must be a boolean")

        if overrides.get('auto_pause_on_rate_limit') is not None:
            auto_pause = bool(overrides['auto_pause_on_rate_limit'])
        else:
            # Read as already refreshed by /api/settings (which reloads .env
            # itself); never reload here, that would clobber keys set in the
            # real environment.
            auto_pause = not (str(_cfg().DISABLE_AUTO_PAUSE).strip().lower() == 'true')

        provider = job.provider
        context_window = job.context_window
        raw_key = None
        if overrides:
            changes: Dict[str, Any] = {}
            if overrides.get('model'):
                changes['model'] = overrides['model']
            if overrides.get('llm_provider'):
                changes['provider'] = str(overrides['llm_provider']).lower()
            if overrides.get('llm_api_endpoint'):
                changes['endpoint'] = overrides['llm_api_endpoint']
            if overrides.get('context_window') is not None:
                context_window = _as_int(overrides['context_window'], None, name='context_window')
                changes['context_window'] = context_window
            if changes:
                provider = provider.with_changes(**changes)
            raw_key = overrides.get('api_key')

        provider_name = provider.provider or 'ollama'
        requested_endpoint = provider.endpoint
        _key, rejection = guard_request_credentials(provider_name, requested_endpoint, raw_key)
        if rejection is not None:
            raise JobConfigError(rejection.message, 400, rejection.as_dict())

        override = is_endpoint_override(provider_name, requested_endpoint)
        api_key = resolve_provider_key(provider_name, raw_key, allow_env_fallback=not override)

        if provider_name in KEY_PROVIDERS and key_required(provider_name, requested_endpoint):
            if not (api_key or env_key(provider_name)):
                env_var = provider_env_var(provider_name)
                raise JobConfigError("Missing API key for provider", 400, {
                    "error": "Missing API key for provider",
                    "message": (f"Resuming with '{provider_name}' requires an API key. "
                                f"Set {env_var} in .env or include it in the request."),
                })
            api_key = api_key or env_key(provider_name)
        if provider_name in ENDPOINT_PROVIDERS and not requested_endpoint:
            raise JobConfigError("Missing API endpoint for provider", 400, {
                "error": "Missing API endpoint for provider",
                "message": f"Resuming with '{provider_name}' requires an API endpoint.",
            })

        return job.with_changes(
            provider=provider.with_changes(api_key=api_key or None),
            context_window=context_window,
            retry_token_aligned=bool(retry_token_aligned),
            auto_pause_on_rate_limit=auto_pause,
        )

    # -- builders ----------------------------------------------------------

    @classmethod
    def from_web_request(cls, data: Mapping[str, Any], *, output_dir: str,
                         uploads_dir) -> "JobConfig":
        """Validate a ``POST /api/translate`` body and build the job.

        Raises:
            JobConfigError: 400 for a missing/invalid field or a refused
                endpoint, 403 for an upload path outside ``uploads_dir``.
        """
        from src.api.services.credential_guard import guard_request_credentials, is_endpoint_override
        from src.api.services.path_validator import PathValidator
        from src.tts.tts_config import TTSConfig

        if not isinstance(data, Mapping):
            raise JobConfigError("Request body must be a JSON object")
        cfg = _cfg()

        if 'file_path' in data:
            required = ['file_path', 'source_language', 'target_language',
                        'model', 'llm_api_endpoint', 'output_filename', 'file_type']
        else:
            required = ['text', 'source_language', 'target_language',
                        'model', 'llm_api_endpoint', 'output_filename']
        for name in required:
            value = data.get(name)
            missing = (name not in data
                       or (isinstance(value, str) and not value.strip())
                       or (not isinstance(value, str) and value is None))
            if missing and not (name == 'text' and data.get('file_type') == 'txt'
                                and data.get('text') == ""):
                raise JobConfigError(f"Missing or empty field: {name}")

        output_filename = PathValidator.safe_output_filename(data['output_filename'])
        if output_filename is None:
            raise JobConfigError("Invalid output_filename")

        term_fix = None
        if data.get('term_fix') is not None:
            term_fix, error = parse_term_fix_request(data.get('term_fix'))
            if error is not None:
                raise JobConfigError(error)

        safe_path = None
        input_path = None
        text = None
        if 'file_path' in data:
            # Confined to the uploads directory, otherwise any server-readable
            # file could be "translated" into a downloadable output (issue #209).
            safe_path, path_error = PathValidator.validate_upload_path(data['file_path'], uploads_dir)
            if path_error is not None:
                raise JobConfigError(path_error, 403)
            input_path = str(safe_path)
            file_type = data['file_type']
        else:
            text = data['text']
            file_type = data.get('file_type', 'txt')

        provider_name = normalize_provider(data.get('llm_provider', 'ollama'))
        requested_endpoint = data.get('llm_api_endpoint')
        request_key = data.get(f'{provider_name}_api_key')
        # A request-chosen endpoint must be allowlisted and carry its own key:
        # the server's .env key never travels to a host the client picked.
        _key, rejection = guard_request_credentials(provider_name, requested_endpoint, request_key)
        if rejection is not None:
            raise JobConfigError(rejection.message, 400, rejection.as_dict())
        override = is_endpoint_override(provider_name, requested_endpoint)
        api_key = resolve_provider_key(provider_name, request_key, allow_env_fallback=not override)

        context_window = _as_int(data.get('context_window'), cfg.OLLAMA_NUM_CTX, name='context_window')
        operation = OP_TERM_FIX if term_fix else (OP_REFINE if data.get('refine_only') else OP_TRANSLATE)
        tts = TTSConfig.from_web_request(data).to_dict() if data.get('tts_enabled') else None

        return cls(
            operation=operation,
            file_type=file_type,
            input_path=input_path,
            text=text,
            output_path=os.path.join(str(output_dir), output_filename),
            input_filename=display_input_filename(data.get('input_filename'), safe_path),
            source_language=data['source_language'],
            target_language=data['target_language'],
            # Term fix is mutually exclusive with the refine paths.
            refine_after=bool(data.get('refine_after', False)) and operation == OP_TRANSLATE,
            term_fix=term_fix,
            provider=ProviderConfig(
                provider=provider_name,
                model=data['model'],
                endpoint=requested_endpoint,
                api_key=api_key or None,
                context_window=context_window,
            ),
            context_window=context_window,
            auto_adjust_context=True,
            min_chunk_size=5,
            # Snapshotted at job start so a resume chunks the same way.
            max_tokens_per_chunk=cfg.MAX_TOKENS_PER_CHUNK,
            parallel_workers=_clamp_parallel_workers(data.get('parallel_workers')),
            request_timeout=_as_int(data.get('timeout'), cfg.REQUEST_TIMEOUT, name='timeout'),
            max_attempts=_as_int(data.get('max_attempts'), 2, name='max_attempts'),
            retry_delay=_as_int(data.get('retry_delay'), 2, name='retry_delay'),
            prompt_options=copy.deepcopy(dict(data.get('prompt_options') or {})),
            bilingual_output=bool(data.get('bilingual_output', False)),
            tts=tts,
            auto_pause_on_rate_limit=bool(data.get('auto_pause_on_rate_limit',
                                                   cfg.AUTO_PAUSE_ON_RATE_LIMIT)),
            origin='web',
        )

    @classmethod
    def from_cli_args(cls, args) -> "JobConfig":
        """Build the job from ``translate.py``'s parsed arguments.

        Applies the provider-specific default model, the default output name,
        the key checks, the refine/term-fix compatibility rules and loads the
        ``--glossary`` file. Recoverable oddities become ``notices``.

        Raises:
            JobConfigError: an argument combination that cannot run.
        """
        cfg = _cfg()
        notices = []
        provider_name = normalize_provider(getattr(args, 'provider', None) or cfg.LLM_PROVIDER)
        input_path = args.input
        lower = input_path.lower()
        ext = os.path.splitext(input_path)[1]
        file_type = {
            '.epub': 'epub', '.srt': 'srt', '.pgn': 'pgn', '.docx': 'docx', '.pdf': 'pdf',
        }.get(ext.lower(), 'txt')

        model = args.model
        if model == cfg.DEFAULT_MODEL:
            provider_default = {
                'nim': cfg.NIM_MODEL, 'litellm': cfg.LITELLM_MODEL, 'mistral': cfg.MISTRAL_MODEL,
                'deepseek': cfg.DEEPSEEK_MODEL, 'poe': cfg.POE_MODEL,
                'openrouter': cfg.OPENROUTER_MODEL, 'gemini': cfg.GEMINI_MODEL,
            }.get(provider_name)
            if provider_default:
                model = provider_default

        fix_terms = getattr(args, 'fix_terms', None)
        refine_only = bool(getattr(args, 'refine_only', False))
        refine = bool(getattr(args, 'refine', False))
        if getattr(args, 'fix_terms_scan', False) and not fix_terms:
            raise JobConfigError("--fix-terms-scan requires --fix-terms RULES")
        if fix_terms and (refine or refine_only):
            raise JobConfigError("--fix-terms cannot be combined with --refine or --refine-only")

        term_fix = None
        if fix_terms:
            operation = OP_TERM_FIX
            term_fix = {'mode': getattr(args, 'fix_terms_mode', 'smart') or 'smart',
                        'rules_file': fix_terms}
        elif refine_only:
            operation = OP_REFINE
        else:
            operation = OP_TRANSLATE

        needs_llm = not (operation == OP_TERM_FIX and term_fix['mode'] == 'literal')
        api_key = getattr(args, f'{provider_name}_api_key', None) if provider_name in KEY_PROVIDERS else None
        if needs_llm:
            if provider_name in KEY_PROVIDERS and key_required(provider_name, args.api_endpoint) \
                    and not api_key:
                hint = {'poe': " Get your key at https://poe.com/api_key",
                        'nim': " Get your key at https://build.nvidia.com/"}.get(provider_name, '')
                raise JobConfigError(f"--{provider_name}_api_key is required when using "
                                     f"{provider_name} provider.{hint}")
            if provider_name == 'litellm' and model == cfg.DEFAULT_MODEL:
                raise JobConfigError("litellm provider requires a provider-prefixed model. "
                                     "Set LITELLM_MODEL in .env or pass -m, e.g. "
                                     "-m anthropic/claude-sonnet-4-6")

        if file_type in ('pdf', 'pgn'):
            label = file_type.upper()
            if refine_only:
                raise JobConfigError(f"--refine-only does not support {label} input yet.")
            if refine:
                notices.append(f"⚠️ --refine is ignored for {label} input (not supported yet).")
                refine = False

        if refine_only and args.source_lang != args.target_lang:
            notices.append(
                f"⚠️ --refine-only: source language ({args.source_lang}) differs from "
                f"target language ({args.target_lang}). Refinement is monolingual; "
                f"source_lang will be ignored and the file will be polished as "
                f"{args.target_lang}.")

        output_path = args.output
        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_ext = ext
            if lower.endswith('.epub'):
                output_ext = '.epub'
            elif lower.endswith('.srt'):
                output_ext = '.srt'
            elif lower.endswith('.pgn'):
                output_ext = '.pgn'
            if operation == OP_TERM_FIX:
                output_path = f"{base} (term-fix){output_ext}"
            elif operation == OP_REFINE:
                output_path = f"{base} (refined){output_ext}"
            else:
                output_path = f"{base} ({args.target_lang}){output_ext}"

        prompt_options: Dict[str, Any] = {
            'text_cleanup': bool(getattr(args, 'text_cleanup', False)),
        }
        if custom_instruction := getattr(args, 'custom_instruction', None):
            prompt_options['custom_instruction_file'] = custom_instruction
        glossary_file = getattr(args, 'glossary', None)
        if glossary_file and operation != OP_TERM_FIX:
            try:
                from src.core.glossary.cli_loader import load_glossary_from_file
                terms, metadata = load_glossary_from_file(glossary_file)
            except Exception as e:
                raise JobConfigError(f"Failed to load glossary {glossary_file}: {e}")
            if terms:
                prompt_options['glossary_terms'] = terms
                if metadata:
                    prompt_options['glossary_term_metadata'] = metadata
                notices.append(f"Glossary loaded: {len(terms)} terms from {glossary_file}")
            else:
                notices.append(f"Glossary file {glossary_file} contained no usable entries")

        want_glossary = bool(getattr(args, 'auto_glossary', False))
        if want_glossary and glossary_file:
            notices.append("⚠️ --auto-glossary ignored: --glossary was provided.")
            want_glossary = False
        if want_glossary and refine_only:
            notices.append("⚠️ --auto-glossary ignored in --refine-only mode.")
            want_glossary = False
        if want_glossary:
            prompt_options['glossary_auto'] = True
        if getattr(args, 'auto_style', False):
            prompt_options['style_auto'] = True

        tts = None
        if getattr(args, 'tts', False) and operation != OP_TERM_FIX:
            from src.tts.tts_config import TTSConfig
            tts = TTSConfig.from_cli_args(args).to_dict()

        return cls(
            operation=operation,
            file_type=file_type,
            input_path=input_path,
            output_path=output_path,
            input_filename=os.path.basename(input_path),
            source_language=args.source_lang,
            target_language=args.target_lang,
            refine_after=refine and operation == OP_TRANSLATE,
            term_fix=term_fix,
            provider=ProviderConfig(provider=provider_name, model=model,
                                    endpoint=args.api_endpoint, api_key=api_key or None),
            context_window=None,
            auto_adjust_context=True,
            min_chunk_size=5,
            max_tokens_per_chunk=cfg.MAX_TOKENS_PER_CHUNK,
            parallel_workers=getattr(args, 'parallel', None) or cfg.PARALLEL_TRANSLATIONS,
            prompt_options=prompt_options,
            tts=tts,
            # A rate limit pauses the job; `translate.py --resume <id>` continues it.
            auto_pause_on_rate_limit=True,
            origin='cli',
            notices=tuple(notices),
        )
