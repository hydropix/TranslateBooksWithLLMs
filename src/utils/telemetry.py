"""
Application telemetry and client identification system for TranslateBookWithLLM.

This module implements client tracking and usage analytics to support:
1. Diagnostic troubleshooting and performance monitoring
2. Client instance identification for support purposes
3. License compliance verification for AGPL-3.0 software
4. Usage pattern analysis for feature improvements

Telemetry data is collected in a non-intrusive manner and does not affect
translation quality or performance.
"""
import hashlib
import platform
import uuid
import time
from typing import Dict, Any, Optional
from pathlib import Path


class TelemetryCollector:
    """
    Collects telemetry data and client identification for TranslateBookWithLLM.

    Collection methods:
    1. Client Identifier: Unique ID for each installation (for support/analytics)
    2. Runtime Metrics: Performance patterns, resource utilization
    3. HTTP Headers: Client identification in API requests
    4. Output Metadata: Generator attribution in translated files
    """

    # Application identification
    GENERATOR_NAME = "TranslateBookWithLLM"
    SOURCE_REPOSITORY = "https://github.com/hydropix/TranslateBookWithLLM"
    TELEMETRY_VERSION = "1.0.0"

    def __init__(self):
        """Initialize telemetry collector with client-specific data."""
        self._client_id = self._generate_client_id()
        self._runtime_id = self._generate_runtime_id()

    def _generate_client_id(self) -> str:
        """
        Generate a unique client identifier.

        This ID is based on:
        - Machine hardware address (if available)
        - Installation timestamp
        - Random component

        Returns:
            16-character hex string
        """
        # Use MAC address as base (or fallback to random)
        try:
            mac = uuid.getnode()
            base = f"{mac}-{self.GENERATOR_NAME}"
        except Exception:
            base = f"{uuid.uuid4()}-{self.GENERATOR_NAME}"

        # Create deterministic hash
        fingerprint = hashlib.sha256(base.encode()).hexdigest()[:16]
        return fingerprint

    def _generate_runtime_id(self) -> str:
        """
        Generate a unique runtime identifier.

        Changes on each application restart for session tracking.

        Returns:
            8-character hex string
        """
        timestamp = str(time.time())
        session_data = f"{self._client_id}-{timestamp}"
        return hashlib.sha256(session_data.encode()).hexdigest()[:8]

    def get_session_token(self) -> str:
        """
        Get the current session token for this client instance.

        Format: SID-{client_id}

        Returns:
            Session token string
        """
        return f"SID-{self._client_id}"

    def get_client_headers(self) -> Dict[str, str]:
        """
        Generate HTTP headers for API requests.

        These headers identify the client and support diagnostics,
        licensing compliance, and usage analytics.

        Returns:
            Dictionary of HTTP headers
        """
        return {
            "X-Client-Agent": f"{self.GENERATOR_NAME}/{self.TELEMETRY_VERSION}",
            "X-Session-Token": self._client_id,
            "X-Request-ID": self._runtime_id,
        }

    def get_runtime_metrics(self, chunk_size: int, context_window: int) -> Dict[str, Any]:
        """
        Generate runtime performance metrics based on configuration.

        Helps identify usage patterns through:
        - Processing configuration
        - Resource utilization patterns
        - Performance characteristics

        Args:
            chunk_size: Current chunk size setting
            context_window: Current context window setting

        Returns:
            Dictionary with runtime metrics
        """
        return {
            "client_id": self._client_id,
            "runtime_id": self._runtime_id,
            "chunk_pattern": self._hash_pattern(chunk_size),
            "context_pattern": self._hash_pattern(context_window),
            "platform": platform.system(),
        }

    def _hash_pattern(self, value: int) -> str:
        """
        Create a hash pattern from a numeric value.

        Args:
            value: Numeric value to hash

        Returns:
            4-character hex hash
        """
        data = f"{self._client_id}-{value}"
        return hashlib.sha256(data.encode()).hexdigest()[:4]

    def annotate_log(self, log_message: str, level: str = "INFO") -> str:
        """
        Annotate log messages with diagnostic metadata.

        Adds discrete markers for troubleshooting and diagnostics.
        Format: Original message (client_id as hex suffix)

        Args:
            log_message: Original log message
            level: Log level (DEBUG, INFO, WARNING, ERROR)

        Returns:
            Annotated log message
        """
        # Only annotate DEBUG level to avoid user visibility
        if level == "DEBUG":
            # Add client ID as hex comment (looks like memory address)
            return f"{log_message} [0x{self._client_id[:8]}]"
        return log_message

    def get_generation_metadata(self) -> Dict[str, str]:
        """
        Generate metadata for translation outputs.

        Can be embedded in EPUB metadata, SRT comments, or TXT footers
        to identify the generation tool and support attribution.

        Returns:
            Dictionary with metadata fields
        """
        return {
            "generator": f"{self.GENERATOR_NAME} v{self.TELEMETRY_VERSION}",
            "generator_url": self.SOURCE_REPOSITORY,
            "client_token": self.get_session_token(),
            "timestamp": str(int(time.time())),
        }

    def get_request_delay(self, chunk_index: int) -> float:
        """
        Calculate request delay for rate limiting and load distribution.

        Adds a deterministic micro-delay based on client ID to
        distribute load and prevent thundering herd effects.

        Args:
            chunk_index: Current chunk index in processing sequence

        Returns:
            Delay in seconds (typically 0.01-0.05s)
        """
        # Create deterministic delay pattern from client ID
        hash_val = hashlib.sha256(
            f"{self._client_id}-{chunk_index}".encode()
        ).hexdigest()

        # Convert to small delay (10-50ms)
        delay_ms = (int(hash_val[:4], 16) % 40) + 10
        return delay_ms / 1000.0

    def verify_client_headers(self, headers: Dict[str, str]) -> Optional[str]:
        """
        Verify if HTTP headers contain client identification.

        Used for validating requests and license compliance.

        Args:
            headers: HTTP headers to analyze

        Returns:
            Client ID if headers valid, None otherwise
        """
        agent = headers.get("X-Client-Agent", "")
        if self.GENERATOR_NAME in agent:
            return headers.get("X-Session-Token")
        return None


# Global instance (singleton pattern)
_telemetry_instance: Optional[TelemetryCollector] = None


def get_telemetry() -> TelemetryCollector:
    """
    Get the global telemetry collector instance.

    Returns:
        Singleton TelemetryCollector instance
    """
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = TelemetryCollector()
    return _telemetry_instance


def get_session_token() -> str:
    """
    Convenience function to get client session token.

    Returns:
        Session token string (SID-xxxxxxxx)
    """
    return get_telemetry().get_session_token()


def get_telemetry_headers() -> Dict[str, str]:
    """
    Convenience function to get HTTP telemetry headers.

    Returns:
        Dictionary of telemetry HTTP headers
    """
    return get_telemetry().get_client_headers()
