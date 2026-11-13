"""Per-job loggers from create_job_logger() are independent of each other and
of the shared module-level logger used by setup_web_logger()."""

import pytest

from src.utils import unified_logger
from src.utils.unified_logger import LogLevel, create_job_logger, setup_web_logger


@pytest.fixture
def fresh_global(monkeypatch):
    monkeypatch.setattr(unified_logger, '_global_logger', None)


def _sinks():
    web, stored = [], []
    return web, stored, web.append, stored.append


def test_two_job_loggers_keep_their_own_callbacks(fresh_global):
    web_a, store_a, web_cb_a, store_cb_a = _sinks()
    web_b, store_b, web_cb_b, store_cb_b = _sinks()

    logger_a = create_job_logger(web_cb_a, store_cb_a, console_output=False)
    logger_b = create_job_logger(web_cb_b, store_cb_b, console_output=False)

    assert logger_a is not logger_b
    logger_a.info('from job A')
    logger_b.info('from job B')
    logger_a.warning('job A again')

    assert [e['message'] for e in web_a] == ['from job A', 'job A again']
    assert [e['message'] for e in store_a] == ['from job A', 'job A again']
    assert [e['message'] for e in web_b] == ['from job B']
    assert [e['message'] for e in store_b] == ['from job B']


def test_job_logger_does_not_touch_the_shared_logger(fresh_global):
    create_job_logger(lambda e: None, lambda e: None, console_output=False)
    assert unified_logger._global_logger is None

    web_shared, _, web_cb_shared, store_cb_shared = _sinks()
    shared = setup_web_logger(web_cb_shared, store_cb_shared)
    web_job, _, web_cb_job, store_cb_job = _sinks()
    job = create_job_logger(web_cb_job, store_cb_job, console_output=False)

    # Creating a job logger after setup_web_logger leaves the shared one alone.
    assert shared.web_callback is web_cb_shared
    shared.console_output = False
    job.info('job line')
    assert [e['message'] for e in web_job] == ['job line']
    assert web_shared == []


def test_job_loggers_have_independent_state(fresh_global):
    logger_a = create_job_logger(console_output=False)
    logger_b = create_job_logger(console_output=False)

    logger_a.update_progress(3, 10)

    assert logger_a.translation_state['total_chunks'] == 10
    assert logger_b.translation_state['total_chunks'] == 0


def test_min_level_is_honoured(fresh_global):
    web, _, web_cb, _ = _sinks()
    logger = create_job_logger(web_cb, console_output=False, min_level=LogLevel.WARNING)

    logger.info('hidden')
    logger.error('shown')

    assert [e['message'] for e in web] == ['shown']
