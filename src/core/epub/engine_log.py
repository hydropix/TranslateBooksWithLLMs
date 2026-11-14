"""Per-job log routing of the file-level EPUB engine.

Several translations can run concurrently in one process, so a message must
reach the job that produced it: the job's ``log_callback`` always wins. When
there is no callback at all (direct library use), messages go to this
package's standard ``logging`` logger, never to the process-wide unified
logger singleton (which would attribute them to whichever job registered it
last).
"""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def report_error(log_callback: Optional[Callable], event_name: str, message: str) -> None:
    """Report an error to the job that produced it (or to the module logger)."""
    if log_callback:
        log_callback(event_name, message)
        return
    logger.error(message)


def report_info(log_callback: Optional[Callable], event_name: str, message: str) -> None:
    """Report an informational message to the job (or to the module logger)."""
    if log_callback:
        log_callback(event_name, message)
        return
    logger.info(message)
