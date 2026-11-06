"""Masking helpers for webhook settings returned by ``GET /api/config``.

The webhook URL and headers routinely embed secrets (Discord/Slack tokens in
the path, ``?token=`` query strings, ``Authorization`` headers, ntfy topics).
They are masked before being sent to the browser, and the matching ``unmask_*``
helpers restore the stored value when the UI saves a field it did not edit.

The mask token is ``***``. Anything that is not a mask token is treated as a
brand new value and replaces the stored one.
"""
import json
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

MASK = "***"

# Path segments that are plain API vocabulary, not secrets. Everything else in
# the path is masked (ntfy topics, Discord ids/tokens, Slack ids, UUIDs, ...).
_SAFE_PATH_SEGMENTS = {
    "api", "webhook", "webhooks", "services", "hooks", "message", "messages",
    "send", "publish", "ping", "notify", "push", "v1", "v2", "v3",
}


def _is_placeholder(segment: str) -> bool:
    """``{event}``-style template placeholders carry no secret."""
    return segment.startswith("{") and segment.endswith("}")


def _split_netloc(netloc: str):
    userinfo, sep, hostport = netloc.rpartition("@")
    return (userinfo if sep else None), hostport


def mask_webhook_url(url: Optional[str]) -> str:
    """Mask userinfo, query values and secret-looking path segments."""
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

    segments = parts.path.split("/")
    path = "/".join(
        s if (not s or s.lower() in _SAFE_PATH_SEGMENTS or _is_placeholder(s)) else MASK
        for s in segments
    )

    query_items = []
    for item in parts.query.split("&") if parts.query else []:
        key, eq, _value = item.partition("=")
        query_items.append(f"{key}={MASK}" if eq else key)

    return urlunsplit((parts.scheme, netloc, path, "&".join(query_items), parts.fragment))


def unmask_webhook_url(submitted: str, stored: Optional[str]) -> str:
    """Rebuild the real URL from a (possibly masked) submitted one."""
    stored = stored or ""
    if MASK not in submitted or not stored:
        return submitted
    if submitted == mask_webhook_url(stored):
        return stored
    try:
        sub, old = urlsplit(submitted), urlsplit(stored)
    except ValueError:
        return submitted

    sub_userinfo, sub_host = _split_netloc(sub.netloc)
    old_userinfo, _ = _split_netloc(old.netloc)
    netloc = sub.netloc
    if sub_userinfo == MASK and old_userinfo is not None:
        netloc = f"{old_userinfo}@{sub_host}"

    old_segments = old.path.split("/")
    new_segments = []
    for i, seg in enumerate(sub.path.split("/")):
        if seg == MASK and i < len(old_segments):
            seg = old_segments[i]
        new_segments.append(seg)

    old_query = {}
    for item in old.query.split("&") if old.query else []:
        key, eq, value = item.partition("=")
        if eq:
            old_query.setdefault(key, value)
    new_items = []
    for item in sub.query.split("&") if sub.query else []:
        key, eq, value = item.partition("=")
        if eq and value == MASK and key in old_query:
            value = old_query[key]
        new_items.append(f"{key}={value}" if eq else key)

    return urlunsplit((sub.scheme, netloc, "/".join(new_segments), "&".join(new_items), sub.fragment))


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
