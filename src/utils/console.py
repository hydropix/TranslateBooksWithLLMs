"""
Console output helpers.

Centralizes the one piece of setup every entrypoint needs: forcing stdout and
stderr to UTF-8 so emoji log lines (e.g. 💬, ✅, ❌, ⚠️) never crash on Windows
consoles using the cp1252 'charmap' codec.

Previously each entrypoint reimplemented this (with two divergent patterns), and
the main CLI forgot it entirely, so an emoji print inside a provider's request
loop raised UnicodeEncodeError, was caught by the broad `except`, and failed the
translation of that unit. See issue #184.

Call ensure_utf8_stdio() once at the top of every entrypoint.
"""

import sys


def _is_utf8(stream) -> bool:
    enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
    return enc == "utf8"


def ensure_utf8_stdio() -> None:
    """Force sys.stdout / sys.stderr to UTF-8 with errors='replace'.

    Idempotent and defensive: it skips streams already in UTF-8, prefers the
    modern TextIOWrapper.reconfigure(), falls back to wrapping the raw buffer
    for streams that lack it, and never raises (a failure here must not take
    down the process it is meant to protect).
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or _is_utf8(stream):
            continue

        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
                continue
            except Exception:
                pass  # Fall through to the buffer-wrapping fallback.

        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            try:
                import codecs
                setattr(sys, name, codecs.getwriter("utf-8")(buffer, "replace"))
            except Exception:
                pass
