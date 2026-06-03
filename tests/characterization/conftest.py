"""Shared fixtures for the characterization suite.

These tests pin a deterministic fingerprint of translated/refined output. The
output text and the EPUB/DOCX metadata both carry a per-install identifier
(derived from the machine MAC address via ``text_encoding._client_token``),
which makes the captured output machine-specific and therefore non-portable:
a golden generated on one machine fails on every other one.

The fixture below pins that identifier to a fixed, inert sentinel for the whole
characterization package, so the goldens capture only the translation behaviour
under test — not the host machine's fingerprint.
"""

import pytest

# Inert 16-char hex sentinel. Decodes to ``SID:0000000000000000`` in any
# embedded payload, making it obvious in goldens that no real install id leaked.
_FIXED_CLIENT_TOKEN = "0" * 16


@pytest.fixture(autouse=True)
def _pin_client_token(monkeypatch):
    """Freeze the install identifier so captured output is host-independent."""
    monkeypatch.setattr(
        "src.utils.text_encoding._client_token",
        lambda: _FIXED_CLIENT_TOKEN,
        raising=True,
    )
