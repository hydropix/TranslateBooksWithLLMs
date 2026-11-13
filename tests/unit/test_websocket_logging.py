"""src/api/websocket.py reports through its module logger, not print()."""

import logging

from src.api import websocket


class _State:
    def exists(self, translation_id):
        return True

    def set_translation_field(self, *args):
        pass


class _BrokenSocket:
    def emit(self, *args, **kwargs):
        raise RuntimeError('socket closed')


def test_emit_error_is_logged(caplog, capsys):
    with caplog.at_level(logging.ERROR, logger='src.api.websocket'):
        websocket.emit_update(_BrokenSocket(), 'job-1', {'log': 'x'}, _State())

    assert any(
        r.levelno == logging.ERROR and 'job-1' in r.getMessage() and 'socket closed' in r.getMessage()
        for r in caplog.records
    )
    assert capsys.readouterr().out == ''
