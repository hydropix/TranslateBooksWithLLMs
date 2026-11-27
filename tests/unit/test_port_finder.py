"""Tests for the web server's busy-port fallback (src/utils/port_finder.py)."""

import socket

import pytest

from src.utils.port_finder import find_available_port, is_port_available


@pytest.fixture
def busy_port():
    """A port with a live listener on 127.0.0.1, released after the test."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(1)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def test_busy_port_is_not_available(busy_port):
    assert not is_port_available('127.0.0.1', busy_port)


def test_wildcard_host_rejects_port_busy_on_loopback(busy_port):
    # A loopback listener would steal local requests from a 0.0.0.0 server.
    assert not is_port_available('0.0.0.0', busy_port)


def test_falls_back_to_a_later_port(busy_port):
    port = find_available_port('127.0.0.1', busy_port)
    assert port is not None
    assert busy_port < port < busy_port + 20


def test_returns_preferred_port_when_free(busy_port):
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    free = sock.getsockname()[1]
    sock.close()
    assert find_available_port('127.0.0.1', free) == free


def test_returns_none_when_every_port_is_busy(busy_port):
    assert find_available_port('127.0.0.1', busy_port, max_attempts=1) is None
