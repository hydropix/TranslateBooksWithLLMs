"""Headless launcher for the web server, used only by the e2e fixtures.

Bypasses translation_api.start_server() so no browser tab is opened, and writes
the per-process API token to the path given as argv[2] so the test process can
authenticate against the gated /api/ routes.

Usage: python -m tests.e2e._server_runner <port> <token_file>
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv):
    if len(argv) < 3:
        print("usage: _server_runner.py <port> <token_file>", file=sys.stderr)
        return 2

    port = int(argv[1])
    token_file = Path(argv[2])

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    import translation_api
    from src.api.auth import API_TOKEN

    token_file.write_text(API_TOKEN, encoding="utf-8")
    print(f"[e2e-server] listening on 127.0.0.1:{port}", flush=True)

    translation_api.socketio.run(
        translation_api.app,
        debug=False,
        host="127.0.0.1",
        port=port,
        allow_unsafe_werkzeug=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
