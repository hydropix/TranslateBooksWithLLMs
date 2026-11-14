"""Re-export of the completion verdict, which now lives in the job package.

The CLI and the web handler share one verdict (``src/core/job/verdict.py``);
this module keeps the historical import path working.
"""
from src.core.job.verdict import CompletionVerdict, classify_completion  # noqa: F401
