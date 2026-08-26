"""
Allowlist validation for client-supplied LLM endpoints.

Several HTTP entry points let the request choose the host the server will talk
to (Ollama/LM Studio users legitimately need this). Left unchecked, that turns
the server into a credential-exfiltration relay: point the endpoint at an
attacker host and the server happily attaches its ``.env`` API key.

This module is the first of the two guards that close that path. It is a pure
input validator with no Flask dependency, which is why it lives next to
``PathValidator``.

Deliberate trade-off — loopback and private ranges are permitted, and so
are the hostnames that can only name a machine on such a network. Self-hosted
backends (Ollama, LM Studio, llama.cpp, vLLM) run on ``localhost``, on a LAN
address, behind ``host.docker.internal``, or on a name the operator's own
resolver answers (a container or LXC hostname, ``nas.local``, a tailnet name,
or a subdomain of a domain they own pointing at a LAN box), so rejecting those
would break the project's primary use case. The consequence is that this validator
allows SSRF against the operator's own network. That is only acceptable
because of the second guard, enforced at the call sites: an endpoint that
differs from the server default is treated as an override and never paired
with a server-stored API key. No credential travels with a request-chosen
host, so the worst an attacker gets is an unauthenticated probe of a network
they must already be able to reach the server from.
"""
import ipaddress
import socket
import time
from typing import Optional, Set, Tuple
from urllib.parse import urlparse

import src.config as _config


# Public LLM API hosts the project ships support for. Always accepted, so a
# default install needs no LLM_ENDPOINT_ALLOWLIST entry at all.
DEFAULT_ALLOWED_HOSTS = frozenset({
    'api.openai.com',
    'generativelanguage.googleapis.com',
    'openrouter.ai',
    'api.mistral.ai',
    'api.deepseek.com',
    'api.poe.com',
    'integrate.api.nvidia.com',
    'api.anthropic.com',
    'api.x.ai',
    'opencode.ai',
    'ollama.com',
})

# Server defaults read from src.config. A user who points one of these at their
# own domain must not be locked out by the allowlist.
_CONFIGURED_ENDPOINT_ATTRS = (
    'API_ENDPOINT',
    'OLLAMA_API_ENDPOINT',
    'OPENAI_API_ENDPOINT',
    'OPENROUTER_API_ENDPOINT',
    'MISTRAL_API_ENDPOINT',
    'DEEPSEEK_API_ENDPOINT',
    'POE_API_ENDPOINT',
    'NIM_API_ENDPOINT',
    'ANTHROPIC_API_ENDPOINT',
    'XAI_API_ENDPOINT',
    'OPENCODE_API_ENDPOINT',
    'OPENCODE_GO_API_ENDPOINT',
    'OLLAMA_CLOUD_API_ENDPOINT',
)

# Non-IP hostnames that always denote the machine the server runs on.
_LOCAL_HOSTNAMES = frozenset({'localhost', 'host.docker.internal'})

# Suffixes that can only name a machine on the operator's own network, so a
# hostname ending in one is treated exactly like a private IP. Without this,
# every self-hosted backend reachable by name rather than by literal address
# was rejected — container and LXC hostnames, router- and mDNS-assigned
# names, Tailscale MagicDNS names (issue #263). None of them resolve on the
# public internet: '.local' is reserved for mDNS (RFC 6762), '.home.arpa' for
# residential networks (RFC 8375), '.internal' for private use, '.lan'/'.home'/
# '.corp'/'.intranet'/'.private' are squatted but never delegated, and a
# '.ts.net' name only resolves inside the tailnet it belongs to.
_LOCAL_SUFFIXES = (
    '.localhost',
    '.local',
    '.lan',
    '.home',
    '.home.arpa',
    '.internal',
    '.intranet',
    '.corp',
    '.private',
    '.ts.net',
)

# 100.64.0.0/10, the shared address space (RFC 6598) Tailscale allocates from.
# Python's ipaddress does not report it as private, but it is never publicly
# routable, so an endpoint on it is as local as a 10/8 address.
_SHARED_ADDRESS_SPACE = ipaddress.ip_network('100.64.0.0/10')

# Resolution verdicts are cached for a short while: the UI polls /api/models
# while it waits for Ollama, and a hostname needing a lookup would otherwise
# pay for one on every poll. The TTL is deliberately short so that re-pointing
# a DNS record takes effect without a restart.
_RESOLUTION_TTL_SECONDS = 60
_resolution_cache = {}


class EndpointValidator:
    """Validates client-supplied LLM endpoint URLs against an allowlist."""

    @staticmethod
    def allowed_hosts() -> Set[str]:
        """Return the accepted hostnames, recomputed on every call.

        Never cached: ``reload_config()`` can change the configured endpoints
        at runtime, and a cached set would silently keep the stale hosts.
        """
        hosts = set(DEFAULT_ALLOWED_HOSTS)

        for attr in _CONFIGURED_ENDPOINT_ATTRS:
            value = getattr(_config, attr, '') or ''
            if not value.strip():
                continue
            try:
                hostname = urlparse(value.strip()).hostname
            except ValueError:
                continue
            if hostname:
                hosts.add(hostname.lower())

        hosts.update(getattr(_config, 'LLM_ENDPOINT_ALLOWLIST', ()) or ())
        return hosts

    @staticmethod
    def is_local_host(hostname: str) -> bool:
        """Return True for the operator's own machine or private network.

        Covers 'localhost' and 'host.docker.internal'; any hostname carrying a
        private-network suffix (see ``_LOCAL_SUFFIXES``); any single-label
        hostname — 'ollama', 'nas', a container or LXC name — since only an
        internal resolver (/etc/hosts, Docker DNS, the LAN router) can answer
        one; and any literal address that is loopback, private, link-local or
        in the shared address space, i.e. 127.0.0.0/8, ::1, 10/8, 172.16/12,
        192.168/16, 169.254/16, fc00::/7 and 100.64.0.0/10.
        """
        if not hostname:
            return False

        # A trailing dot marks a fully-qualified name; it must not defeat the
        # suffix match.
        host = hostname.strip().lower().rstrip('.')
        if not host:
            return False
        if host in _LOCAL_HOSTNAMES or host.endswith(_LOCAL_SUFFIXES):
            return True

        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            # A name with no dot cannot come from public DNS, so it denotes a
            # machine on the operator's own network. An IPv6 literal always
            # contains ':' and parses above, so it never reaches this test.
            return '.' not in host
        return (address.is_loopback or address.is_private or address.is_link_local
                or (address.version == 4 and address in _SHARED_ADDRESS_SPACE))

    @staticmethod
    def resolves_to_private_network(hostname: str) -> bool:
        """Return True when every address ``hostname`` resolves to is local.

        The last resort for a host the syntactic rules and the allowlist both
        reject. An operator is free to name a LAN box under a domain they own
        ("ai-server.example.com" answering 192.168.1.50 from an internal
        resolver), and refusing that was the remaining half of issue #263.

        Only the operator's own network is opened up, which the module docstring
        already accepts: an attacker-controlled name that resolves to a private
        address buys nothing more than the private literal it points at, and the
        pairing rule at the call sites still keeps every stored credential away
        from a request-chosen host. A name that fails to resolve, or that
        answers with a single public address, stays rejected.
        """
        if not hostname:
            return False

        host = hostname.strip().lower().rstrip('.')
        cached = _resolution_cache.get(host)
        now = time.monotonic()
        if cached is not None and now - cached[0] < _RESOLUTION_TTL_SECONDS:
            return cached[1]

        try:
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except (socket.gaierror, UnicodeError, OSError, ValueError):
            # Unresolvable, so unusable as an endpoint anyway. Cached too: a
            # broken resolver must not be re-queried on every poll.
            _resolution_cache[host] = (now, False)
            return False

        addresses = {info[4][0] for info in infos}
        verdict = bool(addresses) and all(
            EndpointValidator.is_local_host(address.split('%')[0])
            for address in addresses
        )
        _resolution_cache[host] = (now, verdict)
        return verdict

    @staticmethod
    def validate(endpoint: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Check a request-supplied endpoint URL.

        Returns ``(True, None)`` when the endpoint may be used, or
        ``(False, message)`` with a message safe to return to the client.

        An absent endpoint is accepted: it means "use the server default",
        which is not an override and not this function's problem.
        """
        if not endpoint or not str(endpoint).strip():
            return True, None

        try:
            parsed = urlparse(str(endpoint).strip())
        except ValueError:
            return False, "Endpoint is not a valid URL"

        if parsed.scheme not in ('http', 'https'):
            return False, "Endpoint must use http or https"

        try:
            if parsed.username or parsed.password:
                return False, "Endpoint must not embed credentials"
            # urlparse.hostname is already lowercased, and strips port/userinfo.
            host = parsed.hostname
        except ValueError:
            return False, "Endpoint is not a valid URL"

        if not host:
            return False, "Endpoint has no host"

        if EndpointValidator.is_local_host(host):
            return True, None

        allowed = EndpointValidator.allowed_hosts()
        if host in allowed or any(host.endswith('.' + h) for h in allowed):
            return True, None

        # Last resort, and the only branch that performs I/O: a name under a
        # public suffix may still point at the operator's own machine. Reached
        # only for a host that is about to be rejected, so the common paths
        # never wait on a resolver.
        if EndpointValidator.resolves_to_private_network(host):
            return True, None

        return False, (
            f"Endpoint host '{host}' is not allowed. "
            "It does not resolve to your local network. "
            "Add it to LLM_ENDPOINT_ALLOWLIST in .env to permit it."
        )
