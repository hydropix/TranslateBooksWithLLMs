"""
Per-session API token authentication (issue #210).

The web server has no user accounts; its only legitimate client is the SPA it
serves to the same origin. Before this layer, wildcard CORS plus the absence of
any auth let any website the user visited issue cross-origin requests to the
local API and read/trigger file-read, file-management, settings-write and
self-update endpoints (drive-by attacks).

The fix has two halves:

1. CORS is no longer permissive (see translation_api.py): the wildcard
   ``Access-Control-Allow-Origin`` is gone and Socket.IO falls back to its
   same-origin default. A foreign page can therefore no longer *read* responses.

2. This module adds a secret, per-process token that gates every ``/api/`` route
   and the Socket.IO handshake. The token is minted at startup, handed to the
   SPA inside the HTML page (which is same-origin, so a foreign page cannot read
   it), and replayed by the client on every request. A cross-origin attacker can
   neither read the token nor forge the custom header it travels in, which also
   closes the CSRF vector for "simple" requests that CORS alone does not block.

The token is regenerated on each server start; it is never persisted.
"""
import secrets
from flask import request, jsonify

# Minted once per process. token_urlsafe(32) yields ~43 chars of 256-bit
# entropy, infeasible to guess and safe to embed in a URL query string.
API_TOKEN = secrets.token_urlsafe(32)

# Where the client may present the token. The header is the primary channel
# (used by fetch/XHR and the Socket.IO handshake). The query string exists only
# for the handful of GET endpoints reached by a top-level navigation or an
# anchor download, which cannot set custom headers.
TOKEN_HEADER = 'X-API-Token'
TOKEN_QUERY = 'token'

# Reachable without a token: the page itself (it delivers the token), Flask's
# static endpoint, and the unauthenticated liveness probe. CORS preflight is
# handled separately below. Everything else under /api/ is gated.
#
# /api/health is intentionally public: container HEALTHCHECK and CI smoke tests
# hit it with no credentials, and it exposes nothing sensitive (status, version,
# default Ollama endpoint, startup_time used by the client to detect restarts) —
# never the session token, masked keys, or webhook secrets.
_EXEMPT_ENDPOINTS = frozenset({'config.serve_interface', 'config.health_check', 'static'})


def _token_from_request():
    return request.headers.get(TOKEN_HEADER) or request.args.get(TOKEN_QUERY)


def is_authorized(token):
    """Constant-time comparison against the live token."""
    return bool(token) and secrets.compare_digest(str(token), API_TOKEN)


def register_auth(app):
    """Install the before_request gate that protects every /api/ route."""

    @app.before_request
    def _require_api_token():
        # Preflight carries no credentials by design; let it through. With CORS
        # locked down it will not yield a permissive response anyway.
        if request.method == 'OPTIONS':
            return None
        if request.endpoint in _EXEMPT_ENDPOINTS:
            return None
        # Only the API surface is gated. The page and static assets are public;
        # Socket.IO traffic is handled by its own middleware (and its handshake
        # is checked separately in the connect handler), not by Flask routing.
        if not request.path.startswith('/api/'):
            return None
        if is_authorized(_token_from_request()):
            return None
        return jsonify({"error": "Unauthorized", "code": "missing_or_invalid_token"}), 401
