"""ChatGPT account OAuth (Codex-compatible PKCE / device code).

Tokens are stored in ``chatgpt_oauth.json`` (``data/`` when that directory
exists, otherwise the working directory) and never logged. The OAuth client
id is the public Codex CLI id — OpenAI's
token endpoint only accepts that client. Using a ChatGPT subscription from a
third-party app is not an officially supported OpenAI product path; the UI
labels it as a ChatGPT sign-in, not a platform API key.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
DEVICE_USERCODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
DEVICE_VERIFY_URL = "https://auth.openai.com/codex/device"
# Codex registers this redirect; TBL's Flask port cannot be substituted.
PKCE_REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
# Catalog entries are filtered by this query param; too-old values return [].
CODEX_CLIENT_VERSION = "0.149.0"
# ChatGPT's Codex backend only accepts first-party originators.
ORIGINATOR = "codex_cli_rs"
USER_AGENT = f"codex_cli_rs/{CODEX_CLIENT_VERSION} (Windows; x86_64)"
REFRESH_SKEW_SECONDS = 60


def oauth_store_path() -> Path:
    """Durable token file next to .env, or in data/ when that directory exists."""
    cwd = Path.cwd()
    data_dir = cwd / "data"
    if data_dir.is_dir():
        return data_dir / "chatgpt_oauth.json"
    return cwd / "chatgpt_oauth.json"


def codex_auth_path() -> Path:
    return Path.home() / ".codex" / "auth.json"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_pkce() -> Dict[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return {"verifier": verifier, "challenge": challenge, "state": secrets.token_urlsafe(24)}


def build_authorize_url(pkce: Dict[str, str], redirect_uri: str = PKCE_REDIRECT_URI) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": SCOPE,
        "code_challenge": pkce["challenge"],
        "code_challenge_method": "S256",
        "state": pkce["state"],
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": ORIGINATOR,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def account_id_from_jwt(token: str) -> str:
    """Read ChatGPT account id from an unverified JWT payload."""
    if not token or token.count(".") < 2:
        return ""
    payload = token.split(".")[1]
    padded = payload + "=" * (-len(payload) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    auth = data.get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        for key in ("chatgpt_account_id", "account_id"):
            value = auth.get(key)
            if value:
                return str(value)
    for key in ("chatgpt_account_id", "account_id"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _record_from_token_response(data: Dict[str, Any], refresh_fallback: str = "") -> Dict[str, Any]:
    access = data.get("access_token") or ""
    refresh = data.get("refresh_token") or refresh_fallback
    if not access or not refresh:
        raise ValueError("token response missing access_token or refresh_token")
    account_id = account_id_from_jwt(access) or account_id_from_jwt(str(data.get("id_token") or ""))
    expires_in = data.get("expires_in")
    ttl = float(expires_in) if isinstance(expires_in, (int, float)) else 3600.0
    return {
        "access_token": access,
        "refresh_token": refresh,
        "id_token": data.get("id_token") or "",
        "account_id": account_id,
        "expires_at": time.time() + ttl,
    }


def load_tokens(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    store = path or oauth_store_path()
    if not store.is_file():
        return None
    try:
        data = json.loads(store.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("access_token"):
        return None
    return data


def save_tokens(tokens: Dict[str, Any], path: Optional[Path] = None) -> Path:
    store = path or oauth_store_path()
    store.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "id_token": tokens.get("id_token") or "",
        "account_id": tokens.get("account_id") or "",
        "expires_at": float(tokens.get("expires_at") or 0),
    }
    store.write_text(json.dumps(payload), encoding="utf-8")
    try:
        os.chmod(store, 0o600)
    except OSError:
        pass
    return store


def delete_tokens(path: Optional[Path] = None) -> None:
    store = path or oauth_store_path()
    try:
        store.unlink()
    except FileNotFoundError:
        pass


def import_codex_auth(codex_path: Optional[Path] = None, dest: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Reuse a local Codex CLI login when TBL has no tokens yet."""
    source = codex_path or codex_auth_path()
    if not source.is_file():
        return None
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    tokens = raw.get("tokens") if isinstance(raw, dict) else None
    if not isinstance(tokens, dict):
        tokens = raw if isinstance(raw, dict) else {}
    access = tokens.get("access_token") or ""
    refresh = tokens.get("refresh_token") or ""
    if not access or not refresh:
        return None
    record = {
        "access_token": access,
        "refresh_token": refresh,
        "id_token": tokens.get("id_token") or "",
        "account_id": account_id_from_jwt(access) or account_id_from_jwt(str(tokens.get("id_token") or "")),
        "expires_at": time.time() + 3600,
    }
    save_tokens(record, dest)
    return record


def status_payload(tokens: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    record = tokens if tokens is not None else load_tokens()
    if not record:
        imported = import_codex_auth()
        record = imported
    signed_in = bool(record and record.get("access_token") and record.get("refresh_token"))
    return {
        "signed_in": signed_in,
        "account_id": (record or {}).get("account_id") or "",
        "expires_at": (record or {}).get("expires_at") or 0,
    }


async def exchange_code(
    code: str,
    verifier: str,
    redirect_uri: str = PKCE_REDIRECT_URI,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    body = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": redirect_uri,
    }
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=30)
    try:
        response = await http.post(
            TOKEN_URL,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        return _record_from_token_response(response.json())
    finally:
        if own_client:
            await http.aclose()


async def refresh_tokens(
    refresh_token: str,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    body = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    }
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=30)
    try:
        response = await http.post(
            TOKEN_URL,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        return _record_from_token_response(response.json(), refresh_fallback=refresh_token)
    finally:
        if own_client:
            await http.aclose()


async def ensure_fresh_tokens(
    path: Optional[Path] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    record = load_tokens(path)
    if record is None:
        record = import_codex_auth(dest=path)
    if not record:
        raise ValueError("ChatGPT is not signed in. Use Sign in with ChatGPT first.")
    expires_at = float(record.get("expires_at") or 0)
    if expires_at and expires_at - time.time() > REFRESH_SKEW_SECONDS:
        return record
    refreshed = await refresh_tokens(record["refresh_token"], client=client)
    if not refreshed.get("account_id"):
        refreshed["account_id"] = record.get("account_id") or ""
    save_tokens(refreshed, path)
    return refreshed


async def start_device_login(client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=30)
    try:
        response = await http.post(
            DEVICE_USERCODE_URL,
            json={"client_id": CLIENT_ID},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
        user_code = body.get("user_code") or body.get("usercode") or ""
        return {
            "user_code": user_code,
            "device_auth_id": body.get("device_auth_id") or "",
            "interval": int(body.get("interval") or 5),
            "verification_url": DEVICE_VERIFY_URL,
        }
    finally:
        if own_client:
            await http.aclose()


async def poll_device_login(
    device_auth_id: str,
    user_code: str,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[Dict[str, Any]]:
    """Return tokens when the user finishes device login, else None (still pending)."""
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=30)
    try:
        response = await http.post(
            DEVICE_TOKEN_URL,
            json={"device_auth_id": device_auth_id, "user_code": user_code},
            headers={"Content-Type": "application/json"},
        )
        if response.status_code in (403, 404):
            return None
        response.raise_for_status()
        body = response.json()
        authorization_code = body.get("authorization_code")
        verifier = body.get("code_verifier")
        if authorization_code and verifier:
            tokens = await exchange_code(
                authorization_code,
                verifier,
                redirect_uri="https://auth.openai.com/deviceauth/callback",
                client=http,
            )
            save_tokens(tokens)
            return tokens
        if body.get("access_token"):
            tokens = _record_from_token_response(body)
            save_tokens(tokens)
            return tokens
        return None
    finally:
        if own_client:
            await http.aclose()


def request_headers(tokens: Dict[str, Any]) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {tokens['access_token']}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "responses=v1",
        "originator": ORIGINATOR,
        "User-Agent": USER_AGENT,
        "version": CODEX_CLIENT_VERSION,
    }
    account_id = tokens.get("account_id") or ""
    if account_id:
        headers["ChatGPT-Account-Id"] = str(account_id)
    return headers
