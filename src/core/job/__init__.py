"""The job contract shared by the CLI and the web server.

* :class:`JobConfig` (``config.py``): what to do, built once per source.
* :class:`JobContext` (``context.py``): the live hooks of one engine run.
* :class:`JobResult` / :class:`JobStatus` (``result.py``): what came out.
* :func:`classify_completion` (``verdict.py``): the single verdict.
* :class:`JobRunner` (``runner.py``): prepare, translate / term-fix, refine,
  finalize; the only orchestration of a job.
"""

from .config import (
    OP_REFINE,
    OP_TERM_FIX,
    OP_TRANSLATE,
    REFINE_FORMATS,
    JobConfig,
    JobConfigError,
)
from .context import JobContext
from .result import JobResult, JobStatus
from .verdict import CompletionVerdict, classify_completion

__all__ = [
    "OP_REFINE",
    "OP_TERM_FIX",
    "OP_TRANSLATE",
    "REFINE_FORMATS",
    "CompletionVerdict",
    "JobConfig",
    "JobConfigError",
    "JobContext",
    "JobResult",
    "JobStatus",
    "classify_completion",
]
