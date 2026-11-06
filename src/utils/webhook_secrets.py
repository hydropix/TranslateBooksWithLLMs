"""Masking helpers for webhook settings returned by ``GET /api/config``.

The webhook URL, headers and payload routinely embed secrets (Discord/Slack
tokens in the path, ``?token=`` query strings, ``Authorization`` headers, a
Pushover ``token`` in the body, ntfy topics). They are masked before being
sent to the browser, and the matching ``unmask_*`` helpers restore the stored
value when the UI saves a field it did not edit.

The mask token is ``***``. Safety rules:

* Secrets are restored into a URL only when scheme, hostname and port are
  identical to the stored URL, so editing the host cannot redirect a stored
  token to another server. The same origin check gates headers and payload.
* Restoration never guesses: anything that cannot be matched stays masked and
  ``has_mask`` lets the caller reject the save instead of writing ``***``.
"""
import json
import re
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

MASK = "***"

# Path segments that are plain API vocabulary, not secrets. Everything else in
# the path is masked (ntfy topics, Discord ids/tokens, Slack ids, UUIDs, ...).
_SAFE_PATH_SEGMENTS = {
    "api", "webhook", "webhooks", "services", "hooks", "message", "messages",
    "send", "publish", "ping", "notify", "push", "v1", "v2", "v3",
}

# Payload keys whose static string values are masked. Values containing a
# ``{placeholder}`` are templates, not secrets, and stay visible.
_SECRET_KEY_RE = re.compile(r"token|secret|passw|pwd|key|auth|credential|user", re.IGNORECASE)

_QUERY_SEP_RE = re.compile(r"([&;])")


def has_mask(value: Optional[str]) -> bool:
    """True when ``value`` still contains an unresolved mask marker."""
    return bool(value) and MASK in value


def _is_placeholder(segment: str) -> bool:
    """``{event}``-style template placeholders carry no secret."""
    return segment.startswith("{") and segment.endswith("}")


def _split_netloc(netloc: str):
    userinfo, sep, hostport = netloc.rpartition("@")
    return (userinfo if sep else None), hostport


def _origin(url: str):
    """(scheme, hostname, port) or None when the URL cannot be parsed."""
    try:
        parts = urlsplit(url)
        return (parts.scheme.lower(), (parts.hostname or "").lower(), parts.port)
    except ValueError:
        return None


def same_origin(a: Optional[str], b: Optional[str]) -> bool:
    """True when both URLs parse and share scheme, hostname and port."""
    if not a or not b:
        return False
    oa, ob = _origin(a), _origin(b)
    return oa is not None and oa == ob and bool(oa[1])


def _query_tokens(query: str):
    """Split a query on ``&`` and ``;`` keeping the separators."""
    return _QUERY_SEP_RE.split(query) if query else []


def mask_webhook_url(url: Optional[str]) -> str:
    """Mask userinfo, query params, fragment and secret-looking path segments."""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return MASK
    if not parts.scheme or not parts.netloc:
        return MASK

    userinfo, hostport = _split_netloc(parts.netloc)
    netloc = f"{MASK}@{hostport}" if userinfo is not None else hostport

    path = "/".join(
        s if (not s or s.lower() in _SAFE_PATH_SEGMENTS or _is_placeholder(s)) else MASK
        for s in parts.path.split("/")
    )

    query = []
    for i, token in enumerate(_query_tokens(parts.query)):
        if i % 2:  # separator
            query.append(token)
            continue
        key, eq, _value = token.partition("=")
        if eq:
            query.append(f"{key}={MASK}")
        else:
            query.append(MASK if key else "")

    fragment = MASK if parts.fragment else ""
    return urlunsplit((parts.scheme, netloc, path, "".join(query), fragment))


def unmask_webhook_url(submitted: str, stored: Optional[str]) -> str:
    """Restore masked parts of ``submitted`` from ``stored`` (same origin only)."""
    stored = stored or ""
    if MASK not in submitted or not stored:
        return submitted
    if not same_origin(submitted, stored):
        return submitted
    try:
        sub, old = urlsplit(submitted), urlsplit(stored)
    except ValueError:
        return submitted

    sub_userinfo, sub_host = _split_netloc(sub.netloc)
    old_userinfo, _ = _split_netloc(old.netloc)
    netloc = sub.netloc
    if sub_userinfo == MASK and old_userinfo is not None:
        netloc = f"{old_userinfo}@{sub_host}"

    sub_segments = sub.path.split("/")
    old_segments = old.path.split("/")
    if len(sub_segments) == len(old_segments):
        path = "/".join(o if s == MASK else s for s, o in zip(sub_segments, old_segments))
    else:
        path = sub.path

    old_tokens = _query_tokens(old.query)
    old_by_key = {}
    for i in range(0, len(old_tokens), 2):
        key, eq, value = old_tokens[i].partition("=")
        if eq:
            old_by_key.setdefault(key, value)
    query = []
    for i, token in enumerate(_query_tokens(sub.query)):
        if i % 2:
            query.append(token)
            continue
        key, eq, value = token.partition("=")
        if eq and value == MASK and key in old_by_key:
            token = f"{key}={old_by_key[key]}"
        elif not eq and key == MASK and i < len(old_tokens) and "=" not in old_tokens[i]:
            token = old_tokens[i]
        query.append(token)

    fragment = old.fragment if sub.fragment == MASK else sub.fragment
    return urlunsplit((sub.scheme, netloc, path, "".join(query), fragment))


def mask_webhook_headers(raw: Optional[str]) -> str:
    """Keep header names, mask every value. Unparseable content is fully masked."""
    if not raw or not raw.strip():
        return ""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return MASK
    if not isinstance(parsed, dict):
        return MASK
    return json.dumps({str(k): MASK for k in parsed}, ensure_ascii=False)


def unmask_webhook_headers(submitted: str, stored: Optional[str]) -> str:
    """Restore stored header values for entries still holding the mask."""
    stored = stored or ""
    if MASK not in submitted or not stored.strip():
        return submitted
    if submitted.strip() == MASK:
        return stored
    try:
        sub = json.loads(submitted)
        old = json.loads(stored)
    except (TypeError, ValueError):
        return submitted
    if not isinstance(sub, dict) or not isinstance(old, dict):
        return submitted
    old_ci = {str(k).lower(): v for k, v in old.items()}
    merged = {}
    for key, value in sub.items():
        if value == MASK and str(key).lower() in old_ci:
            value = old_ci[str(key).lower()]
        merged[key] = value
    return json.dumps(merged, ensure_ascii=False)


def _mask_payload_node(node: Any):
    """Return (masked_node, changed) for a parsed JSON payload template."""
    if isinstance(node, dict):
        out, changed = {}, False
        for k, v in node.items():
            if isinstance(v, str) and v and "{" not in v and _SECRET_KEY_RE.search(str(k)):
                out[k], changed = MASK, True
            else:
                out[k], sub_changed = _mask_payload_node(v)
                changed = changed or sub_changed
        return out, changed
    if isinstance(node, list):
        items, changed = [], False
        for v in node:
            masked, sub_changed = _mask_payload_node(v)
            items.append(masked)
            changed = changed or sub_changed
        return items, changed
    return node, False


def mask_webhook_payload(raw: Optional[str]) -> str:
    """Mask static string values stored under secret-looking payload keys.

    The template stays readable: only values under keys such as ``token``,
    ``secret``, ``password``, ``key``, ``auth`` or ``user`` are replaced, and
    only when they contain no ``{placeholder}``. The text is returned untouched
    when nothing needs masking; an unparseable payload is masked entirely.
    """
    if not raw or not raw.strip():
        return ""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return MASK
    masked, changed = _mask_payload_node(parsed)
    return json.dumps(masked, ensure_ascii=False) if changed else raw


def _unmask_payload_node(sub: Any, old: Any):
    if sub == MASK:
        return old if isinstance(old, str) else sub
    if isinstance(sub, dict):
        old_dict = old if isinstance(old, dict) else {}
        return {k: _unmask_payload_node(v, old_dict.get(k)) for k, v in sub.items()}
    if isinstance(sub, list):
        old_list = old if isinstance(old, list) else []
        return [
            _unmask_payload_node(v, old_list[i] if i < len(old_list) else None)
            for i, v in enumerate(sub)
        ]
    return sub


def unmask_webhook_payload(submitted: str, stored: Optional[str]) -> str:
    """Restore stored payload string values for entries still holding the mask."""
    stored = stored or ""
    if MASK not in submitted or not stored.strip():
        return submitted
    if submitted.strip() == MASK:
        return stored
    try:
        sub = json.loads(submitted)
        old = json.loads(stored)
    except (TypeError, ValueError):
        return submitted
    return json.dumps(_unmask_payload_node(sub, old), ensure_ascii=False)
