"""
Unit tests for src.utils.console.ensure_utf8_stdio (issue #184).

These exercise the decision logic against fake streams so they never mutate the
real test-runner stdout/stderr.
"""

import io
import sys

from src.utils.console import ensure_utf8_stdio


class ReconfigurableStream:
    """Stand-in for a TextIOWrapper that supports reconfigure()."""

    def __init__(self, encoding, raises=False, buffer=None):
        self.encoding = encoding
        self.calls = []
        self._raises = raises
        self.buffer = buffer

    def reconfigure(self, encoding=None, errors=None):
        self.calls.append((encoding, errors))
        if self._raises:
            raise OSError("reconfigure not supported")
        self.encoding = encoding


class LegacyStream:
    """Stand-in for a stream without reconfigure() but with a raw buffer."""

    def __init__(self, encoding, buffer):
        self.encoding = encoding
        self.buffer = buffer


def test_reconfigures_non_utf8_stream(monkeypatch):
    out = ReconfigurableStream("cp1252")
    err = ReconfigurableStream("cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    ensure_utf8_stdio()

    assert out.calls == [("utf-8", "replace")]
    assert err.calls == [("utf-8", "replace")]


def test_idempotent_on_utf8_stream(monkeypatch):
    for enc in ("utf-8", "UTF-8", "utf8"):
        out = ReconfigurableStream(enc)
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", ReconfigurableStream(enc))
        ensure_utf8_stdio()
        assert out.calls == [], f"reconfigure should be skipped for {enc}"


def test_falls_back_to_buffer_wrapping(monkeypatch):
    raw = io.BytesIO()
    legacy = LegacyStream("cp1252", buffer=raw)
    monkeypatch.setattr(sys, "stdout", legacy)
    monkeypatch.setattr(sys, "stderr", LegacyStream("cp1252", buffer=io.BytesIO()))

    ensure_utf8_stdio()

    # stdout was swapped for a UTF-8 writer over the same buffer; an emoji must
    # now encode without raising (the original cp1252 crash, '\U0001f4ac').
    assert sys.stdout is not legacy
    sys.stdout.write("\U0001f4ac ok")
    sys.stdout.flush()
    assert "\U0001f4ac ok".encode("utf-8") in raw.getvalue()


def test_falls_back_when_reconfigure_raises(monkeypatch):
    raw = io.BytesIO()
    stream = ReconfigurableStream("cp1252", raises=True, buffer=raw)
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", ReconfigurableStream("cp1252", raises=True, buffer=io.BytesIO()))

    ensure_utf8_stdio()

    # reconfigure was attempted, then we fell back to wrapping the buffer.
    assert stream.calls == [("utf-8", "replace")]
    assert sys.stdout is not stream


def test_never_raises_on_missing_streams(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    # Must not raise even when there is nothing to fix.
    ensure_utf8_stdio()
