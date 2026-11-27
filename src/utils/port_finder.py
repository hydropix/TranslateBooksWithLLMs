"""Pick a free TCP port for the web server, falling back past busy ones.

The configured port (5000 by default) is often taken by another local program
(e.g. a background service listening on 127.0.0.1:5000). Instead of crashing
with "address already in use" / WinError 10013, the server tries the next
ports and reports the one it actually uses.
"""

import os
import socket
from typing import Optional

# Wildcard binds also receive loopback traffic, but on Windows a program bound
# to 127.0.0.1:<port> wins over a 0.0.0.0:<port> listener for local requests.
# So for wildcard hosts we also require the loopback address to be free.
_WILDCARD_HOSTS = {'', '0.0.0.0', '::'}


def _can_bind(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ':' in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        # Mirror the server: on POSIX werkzeug sets SO_REUSEADDR, so a port
        # left in TIME_WAIT by a just-restarted server must count as free.
        # On Windows SO_REUSEADDR would let us bind over a live listener, so
        # it must stay off there.
        if os.name != 'nt':
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host.strip('[]'), port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def is_port_available(host: str, port: int) -> bool:
    """Return True if the server could listen on ``host:port`` right now."""
    if not _can_bind(host, port):
        return False
    if host in _WILDCARD_HOSTS:
        loopback = '::1' if host == '::' else '127.0.0.1'
        return _can_bind(loopback, port)
    return True


def find_available_port(host: str, preferred: int, max_attempts: int = 20) -> Optional[int]:
    """Return ``preferred`` if free, else the first free port after it.

    Tries ``max_attempts`` consecutive ports starting at ``preferred`` and
    returns ``None`` if all of them are busy. There is an unavoidable race
    between this probe and the real bind; it only matters if another program
    grabs the port in that instant, in which case the server fails as before.
    """
    for port in range(preferred, min(preferred + max_attempts, 65536)):
        if is_port_available(host, port):
            return port
    return None
