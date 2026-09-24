"""
Model context window size detection.

This module provides utilities for detecting the context window size of various
LLM models through API queries and heuristics.
"""

import re
from typing import Optional, Callable, Any


class ContextDetector:
    """
    Detects model context window sizes through multiple strategies.

    Strategies (tried in order):
        1. Provider-specific endpoints (e.g., /api/ps for Ollama)
        2. Model info endpoints
        3. Model list endpoints
        4. Model family heuristics (fallback)

    Example:
        >>> detector = ContextDetector()
        >>> context_size = await detector.detect(
        ...     client=http_client,
        ...     model="llama3",
        ...     endpoint="http://localhost:11434"
        ... )
    """

    async def detect(
        self,
        client,
        model: str,
        endpoint: str,
        api_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, str], None]] = None,
        extra_headers: Optional[dict] = None
    ) -> int:
        """
        Detect context size for a model.

        Args:
            client: HTTP client for API requests
            model: Model name/identifier
            endpoint: API endpoint URL
            api_key: Optional API key for authenticated endpoints
            log_callback: Optional callback for logging (log_type, message)
            extra_headers: Optional headers merged into every probe request
                (e.g. a provider-mandated session header)

        Returns:
            Context window size in tokens

        Process:
            1. Try provider-specific endpoints
            2. Try model info endpoints
            3. Try model list endpoints
            4. Fall back to model family defaults
        """
        # Provider-mandated extras first, then the base headers, so a caller
        # cannot clobber Content-Type or the Authorization derived from the key.
        headers = dict(extra_headers or {})
        headers["Content-Type"] = "application/json"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        base_url = endpoint.replace("/v1/chat/completions", "").replace("/chat/completions", "")

        # Try endpoints in order: /props -> /v1/models/{model} -> /v1/models -> fallback
        for strategy in [
            lambda: self._try_props_endpoint(client, base_url, headers, log_callback),
            lambda: self._try_model_info_endpoint(client, base_url, model, headers, log_callback),
            lambda: self._try_models_list_endpoint(client, base_url, model, headers, log_callback)
        ]:
            ctx = await strategy()
            if ctx:
                return ctx

        # Fallback to model family defaults
        return self._get_model_family_default(model, log_callback)

    async def detect_ollama(
        self,
        client,
        model: str,
        endpoint: str,
        log_callback: Optional[Callable[[str, str], None]] = None,
        fallback_context: int = 2048
    ) -> int:
        """
        Detect context size for Ollama models using /api/show endpoint.

        Args:
            client: HTTP client for API requests
            model: Model name/identifier
            endpoint: API endpoint URL (e.g., http://localhost:11434/api/chat)
            log_callback: Optional callback for logging (log_type, message)
            fallback_context: Fallback value if detection fails

        Returns:
            Context window size in tokens
        """
        try:
            # Build /api/show endpoint from chat endpoint
            show_endpoint = endpoint.replace('/api/chat', '/api/show').replace('/api/generate', '/api/show')

            response = await client.post(
                show_endpoint,
                json={"name": model},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

            # Parse modelfile and parameters to find num_ctx
            modelfile = data.get("modelfile", "")
            parameters = data.get("parameters", "")
            combined = modelfile + "\n" + parameters

            # Look for PARAMETER num_ctx or num_ctx in parameters
            match = re.search(r'num_ctx[\s"]+(\d+)', combined, re.IGNORECASE)
            if match:
                detected_ctx = int(match.group(1))
                if log_callback:
                    log_callback("info", f"Detected model context size: {detected_ctx} tokens")
                return detected_ctx

            # Fallback to model family defaults
            return self._get_model_family_default(model, log_callback)

        except Exception as e:
            if log_callback:
                log_callback("warning", f"Failed to query model context size: {e}. Using configured value.")
            return fallback_context

    async def _try_props_endpoint(
        self,
        client,
        base_url: str,
        headers: dict,
        log_callback: Optional[Callable[[str, str], None]] = None
    ) -> Optional[int]:
        """
        Try to get context from /props endpoint (llama.cpp-specific).

        Args:
            client: HTTP client
            base_url: Base URL of the API
            headers: Request headers
            log_callback: Optional logging callback

        Returns:
            Context size if found, None otherwise
        """
        try:
            response = await client.get(f"{base_url}/props", headers=headers, timeout=5.0)
            if response.status_code == 200:
                n_ctx = response.json().get("default_generation_settings", {}).get("n_ctx")
                if n_ctx and isinstance(n_ctx, int) and n_ctx > 0:
                    if log_callback:
                        log_callback("info", f"Detected context size from /props: {n_ctx}")
                    return n_ctx
        except Exception:
            pass
        return None

    async def _try_model_info_endpoint(
        self,
        client,
        base_url: str,
        model: str,
        headers: dict,
        log_callback: Optional[Callable[[str, str], None]] = None
    ) -> Optional[int]:
        """
        Try to get context from model info endpoint (OpenAI /v1/models/{model}).

        Args:
            client: HTTP client
            base_url: Base URL of the API
            model: Model name
            headers: Request headers
            log_callback: Optional logging callback

        Returns:
            Context size if found, None otherwise
        """
        try:
            response = await client.get(f"{base_url}/v1/models/{model}", headers=headers, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                for field in ["context_length", "max_model_len", "context_window", "max_context_length"]:
                    ctx = data.get(field)
                    if ctx and isinstance(ctx, int) and ctx > 0:
                        if log_callback:
                            log_callback("info", f"Detected context size: {ctx}")
                        return ctx
        except Exception:
            pass
        return None

    async def _try_models_list_endpoint(
        self,
        client,
        base_url: str,
        model: str,
        headers: dict,
        log_callback: Optional[Callable[[str, str], None]] = None
    ) -> Optional[int]:
        """
        Try to get context from models list endpoint (/v1/models).

        Args:
            client: HTTP client
            base_url: Base URL of the API
            model: Model name
            headers: Request headers
            log_callback: Optional logging callback

        Returns:
            Context size if found, None otherwise
        """
        try:
            response = await client.get(f"{base_url}/v1/models", headers=headers, timeout=5.0)
            if response.status_code == 200:
                for model_info in response.json().get("data", []):
                    if model_info.get("id", "") == model:
                        for field in ["context_length", "max_model_len", "context_window"]:
                            ctx = model_info.get(field)
                            if ctx and isinstance(ctx, int) and ctx > 0:
                                if log_callback:
                                    log_callback("info", f"Detected context size: {ctx}")
                                return ctx
        except Exception:
            pass
        return None

    def _get_model_family_default(
        self,
        model: str,
        log_callback: Optional[Callable[[str, str], None]] = None
    ) -> int:
        """
        Get default context size based on model family.

        This is a fallback when API detection fails.

        Args:
            model: Model name
            log_callback: Optional logging callback

        Returns:
            Default context size for the model family

        Defaults (from config.py):
            - gpt-4: 128000
            - gpt: 8192
            - claude: 100000
            - deepseek: 16384
            - mistral: 8192
            - gemma: 8192
            - qwen: 8192
            - llama: 4096
            - phi: 2048
            - other: 2048
        """
        from src.config import MODEL_FAMILY_CONTEXT_DEFAULTS, DEFAULT_CONTEXT_FALLBACK

        model_lower = model.lower()
        for family, default_ctx in MODEL_FAMILY_CONTEXT_DEFAULTS.items():
            if family in model_lower:
                if log_callback:
                    log_callback("info", f"Using default for {family}: {default_ctx}")
                return default_ctx

        if log_callback:
            log_callback("warning", f"Using fallback: {DEFAULT_CONTEXT_FALLBACK}")
        return DEFAULT_CONTEXT_FALLBACK
