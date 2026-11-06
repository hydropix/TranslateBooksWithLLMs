"""
Generic webhook notifier for end-of-translation events.

Configured via environment variables read at startup from .env (see src/config.py
for the NOTIFY_* block). Settings are not hot-reloaded — restart the app after
editing .env.

A single HTTP request is sent when a translation reaches a terminal state.
All request failures are caught and logged; the notifier never raises, so a
misconfigured or unreachable webhook cannot disrupt the translation pipeline.
"""
import json
import logging
from typing import Any, Dict, Iterable, Optional

import requests

from src import config
from src.utils.webhook_secrets import mask_webhook_url

logger = logging.getLogger(__name__)

EVENT_SUCCESS = "success"
EVENT_FAILURE = "failure"
EVENT_INTERRUPTION = "interruption"

_EVENT_FLAGS: Dict[str, str] = {
    EVENT_SUCCESS: "NOTIFY_ON_SUCCESS",
    EVENT_FAILURE: "NOTIFY_ON_FAILURE",
    EVENT_INTERRUPTION: "NOTIFY_ON_INTERRUPTION",
}

_EVENT_TITLES: Dict[str, str] = {
    EVENT_SUCCESS: "Translation completed",
    EVENT_FAILURE: "Translation failed",
    EVENT_INTERRUPTION: "Translation interrupted",
}

_DEFAULT_METHOD = "POST"
_DEFAULT_TIMEOUT_SECONDS = 5


def _safe_format(template: str, context: Dict[str, Any]) -> str:
    """Render a template with str.format, leaving the template unchanged if it
    references unknown keys or has an invalid format spec.

    Catches KeyError (unknown placeholder), IndexError (positional out of range),
    ValueError (bad format spec like `{x:Q}`), and TypeError (spec/value mismatch
    like `{x:d}` against a str).
    """
    try:
        return template.format(**context)
    except (KeyError, IndexError, ValueError, TypeError):
        return template


def _render_json(node: Any, context: Dict[str, Any]) -> Any:
    """Recursively apply _safe_format to every string in a JSON-like structure."""
    if isinstance(node, str):
        return _safe_format(node, context)
    if isinstance(node, list):
        return [_render_json(item, context) for item in node]
    if isinstance(node, dict):
        return {key: _render_json(value, context) for key, value in node.items()}
    return node


def _build_default_payload(event: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Build the gotify-compatible default payload when no template is configured."""
    title = _EVENT_TITLES.get(event, "Translation event")
    lines = [f"Event: {event}"]
    if context.get("file"):
        lines.append(f"File: {context['file']}")
    duration = context.get("duration_seconds")
    if duration is not None:
        lines.append(f"Duration: {duration:.1f}s")
    if context.get("error"):
        lines.append(f"Error: {context['error']}")
    return {
        "title": title,
        "message": "\n".join(lines),
        "event": event,
        "file": context.get("file"),
        "duration_seconds": duration,
        "error": context.get("error"),
    }


def _build_payload(
    event: str,
    payload_context: Dict[str, Any],
    format_context: Dict[str, Any],
) -> Any:
    """Build the JSON body to send.

    Args:
        event: The event constant being notified.
        payload_context: Raw context used by the default payload (preserves None
            values so JSON consumers get explicit nulls).
        format_context: Sanitized context used for str.format substitution in
            user-defined JSON templates (None becomes "" so '{error}' does not
            render the literal 'None').

    If NOTIFY_WEBHOOK_PAYLOAD is set and parses as JSON, every string value is
    rendered through str.format(**format_context). Otherwise the default
    gotify-compatible payload is returned.
    """
    template = getattr(config, "NOTIFY_WEBHOOK_PAYLOAD", "") or ""
    if not template.strip():
        return _build_default_payload(event, payload_context)
    try:
        parsed = json.loads(template)
    except json.JSONDecodeError:
        logger.warning(
            "NOTIFY_WEBHOOK_PAYLOAD is not valid JSON; falling back to default payload"
        )
        return _build_default_payload(event, payload_context)
    return _render_json(parsed, format_context)


def _parse_headers(context: Dict[str, Any]) -> Dict[str, str]:
    """Parse NOTIFY_WEBHOOK_HEADERS as JSON and render value templates."""
    raw = getattr(config, "NOTIFY_WEBHOOK_HEADERS", "") or ""
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("NOTIFY_WEBHOOK_HEADERS is not valid JSON; ignoring")
        return {}
    if not isinstance(parsed, dict):
        logger.warning("NOTIFY_WEBHOOK_HEADERS must be a JSON object; ignoring")
        return {}
    return {str(k): _safe_format(str(v), context) for k, v in parsed.items()}


def _is_event_enabled(event: str) -> bool:
    """Check the per-event NOTIFY_ON_* flag in config."""
    flag_name = _EVENT_FLAGS.get(event)
    if flag_name is None:
        return False
    return bool(getattr(config, flag_name, False))


def _sanitize_format_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Replace None values with empty strings so str.format never renders 'None'.

    str.format on a None value works but yields the literal 'None', which is
    almost never what a user wants in a notification body. Non-None values
    (including 0 and False) are preserved so numeric formatting still works.
    """
    return {key: ("" if value is None else value) for key, value in context.items()}


def known_events() -> Iterable[str]:
    """Return the tuple of event constants this module knows about."""
    return tuple(_EVENT_FLAGS.keys())


def notify(event: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """Send a webhook notification for a terminal translation event.

    Args:
        event: One of EVENT_SUCCESS, EVENT_FAILURE, EVENT_INTERRUPTION.
        context: Free-form key/value pairs available to the payload template.
            Typical keys: file, output, duration_seconds, error, provider, model,
            source_lang, target_lang, translation_id.

    Returns:
        True if the request returned a 2xx response. False if notifications are
        disabled, the event is unknown or disabled, or the request raised. Never
        raises.
    """
    if event not in _EVENT_FLAGS:
        logger.warning("notify() called with unknown event %r; ignoring", event)
        return False

    url = getattr(config, "NOTIFY_WEBHOOK_URL", "") or ""
    if not url:
        return False
    if not _is_event_enabled(event):
        return False

    ctx = dict(context or {})
    ctx.setdefault("event", event)
    format_ctx = _sanitize_format_context(ctx)

    method = (getattr(config, "NOTIFY_WEBHOOK_METHOD", _DEFAULT_METHOD) or _DEFAULT_METHOD).upper()
    timeout = getattr(config, "NOTIFY_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS)
    headers = _parse_headers(format_ctx)
    payload = _build_payload(event, ctx, format_ctx)
    rendered_url = _safe_format(url, format_ctx)

    try:
        if method == "GET":
            response = requests.get(rendered_url, headers=headers, timeout=timeout)
        else:
            response = requests.request(
                method,
                rendered_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        response.raise_for_status()
    except requests.RequestException as exc:
        # The exception text embeds the full URL (tokens included): log only
        # its type and the masked URL.
        logger.warning(
            "Webhook notification failed (%s): %s on %s",
            event, type(exc).__name__, mask_webhook_url(rendered_url),
        )
        return False

    logger.debug("Webhook notification sent (%s) to %s", event, mask_webhook_url(rendered_url))
    return True
