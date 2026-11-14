"""
DNS-rebinding protection: validate the Host header of every incoming request.

The per-session token (src/api/auth.py) stops a foreign page from *reading*
responses, but a DNS-rebinding attack sidesteps the same-origin policy: the
attacker's domain is re-pointed at 127.0.0.1, so the victim's browser treats the
local server as same-origin with the attacker's page, which can then load the
SPA, read the embedded token and drive the API. The one thing the attacker
cannot control is the Host header, which still carries the attacker's domain.

Policy:

* Bound to a loopback address (HOST=127.0.0.1, localhost or ::1, the default):
  only loopback Host names (localhost, 127.0.0.1, [::1], any port) and the
  entries of ALLOWED_HOSTS are accepted.
* Bound to a non-loopback address (HOST=0.0.0.0, Docker / LAN): no check, unless
  ALLOWED_HOSTS is set, in which case it is enforced (loopback names stay allowed).

The guard is a WSGI middleware wrapped around ``app.wsgi_app``. Installed after
``SocketIO(app)`` it sits outside the Engine.IO middleware, so it covers every
Flask route *and* the ``/socket.io/`` handshake (polling and WebSocket upgrade).
"""
import logging

logger = logging.getLogger(__name__)

LOOPBACK_NAMES = frozenset({'localhost', '127.0.0.1', '::1'})

_FORBIDDEN_BODY = (
    b'Forbidden: the Host header of this request is not allowed. '
    b'Add the host name to ALLOWED_HOSTS in .env to accept it.\n'
)


def parse_host(value):
    """Split a Host header (or an ALLOWED_HOSTS entry) into ``(host, port)``.

    The host is lower-cased, stripped of a trailing dot and of IPv6 brackets;
    the port is a string or ``None``. Returns ``None`` for an empty or
    malformed value (bare IPv6 without brackets, non-numeric port, ...).
    """
    if value is None:
        return None
    value = value.strip().lower()
    if not value:
        return None
    if value.startswith('['):
        end = value.find(']')
        if end == -1:
            return None
        host, rest = value[1:end], value[end + 1:]
        if rest and not rest.startswith(':'):
            return None
        port = rest[1:] if rest else None
    else:
        if value.count(':') > 1:
            # An IPv6 literal must be bracketed in a Host header.
            return None
        host, sep, port = value.partition(':')
        if not sep:
            port = None
    if port is not None and not (port.isdigit() and len(port) <= 5):
        return None
    if host.endswith('.'):
        host = host[:-1]
    if not host or any(c.isspace() or c in '/\\@' for c in host):
        return None
    return host, port


def parse_allowed_hosts(raw):
    """Parse the comma-separated ALLOWED_HOSTS value into ``(host, port)`` pairs.

    Invalid entries are skipped with a warning.
    """
    entries = []
    for item in (raw or '').split(','):
        item = item.strip()
        if not item:
            continue
        parsed = parse_host(item)
        if parsed is None:
            logger.warning('Ignoring invalid ALLOWED_HOSTS entry: %r', item)
            continue
        entries.append(parsed)
    return entries


def is_loopback_bind(bind_host):
    """True when the server listens on a loopback address only."""
    # HOST is a bare address (no port), possibly an unbracketed IPv6 literal.
    host = (bind_host or '').strip().lower().strip('[]').rstrip('.')
    return host in LOOPBACK_NAMES


class HostGuardMiddleware:
    """WSGI middleware rejecting requests whose Host header is not allowed."""

    def __init__(self, wsgi_app, bind_host, allowed_hosts=''):
        self.wsgi_app = wsgi_app
        self.allowed = parse_allowed_hosts(allowed_hosts)
        self.enforce = is_loopback_bind(bind_host) or bool(self.allowed)

    def is_allowed(self, host_header):
        if not self.enforce:
            return True
        parsed = parse_host(host_header)
        if parsed is None:
            return False
        host, port = parsed
        if host in LOOPBACK_NAMES:
            return True
        return any(
            host == allowed_host and (allowed_port is None or port == allowed_port)
            for allowed_host, allowed_port in self.allowed
        )

    def __call__(self, environ, start_response):
        host_header = environ.get('HTTP_HOST')
        if self.is_allowed(host_header):
            return self.wsgi_app(environ, start_response)
        logger.warning(
            'Rejected request with disallowed Host header %r (%s %s). '
            'If this host is legitimate, add it to ALLOWED_HOSTS in .env.',
            host_header, environ.get('REQUEST_METHOD', '?'), environ.get('PATH_INFO', ''),
        )
        start_response('403 Forbidden', [
            ('Content-Type', 'text/plain; charset=utf-8'),
            ('Content-Length', str(len(_FORBIDDEN_BODY))),
        ])
        return [_FORBIDDEN_BODY]


def register_host_guard(app, bind_host, allowed_hosts=''):
    """Wrap ``app.wsgi_app`` with the Host check.

    Call it after ``SocketIO(app)`` so the guard also covers ``/socket.io/``.
    """
    guard = HostGuardMiddleware(app.wsgi_app, bind_host, allowed_hosts)
    app.wsgi_app = guard
    if guard.enforce:
        logger.info(
            'Host header check enabled (loopback names%s)',
            ' + ALLOWED_HOSTS' if guard.allowed else '',
        )
    return guard
