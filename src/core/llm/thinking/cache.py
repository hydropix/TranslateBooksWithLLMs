"""
Persistent cache for model thinking behavior.

This module manages a JSON cache that stores information about which models
use thinking tokens and how they behave, avoiding repeated detection.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from .behavior import ThinkingBehavior


class ThinkingCache:
    """
    Manages persistent cache of model thinking behavior.

    The cache stores:
        - Model thinking behavior (STANDARD, CONTROLLABLE, UNCONTROLLABLE)
        - Whether the model supports the "think" parameter
        - Endpoint-specific overrides
        - Timestamp when behavior was tested

    Cache format:
        {
            "model_name@endpoint": {
                "behavior": "controllable",
                "supports_think_param": true,
                "tested_at": 1234567890.0
            }
        }
    """

    def __init__(self, cache_file: Path):
        """
        Initialize the cache.

        Args:
            cache_file: Path to the JSON cache file
        """
        self._cache_file = cache_file
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def load(self) -> None:
        """
        Load the cache from disk.

        Creates an empty cache if file doesn't exist or is invalid.
        """
        if self._loaded:
            return

        try:
            if self._cache_file.exists():
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
        except Exception:
            # Silently fail - cache is just an optimization
            self._cache = {}

        self._loaded = True

    def save(self) -> None:
        """
        Save the cache to disk.

        Creates parent directories if they don't exist.
        Silently fails if write is not possible (cache is just an optimization).
        """
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception:
            # Silently fail - cache is just an optimization
            pass

    def get(
        self,
        model: str,
        endpoint: str = ""
    ) -> Optional[ThinkingBehavior]:
        """
        Get cached thinking behavior for a model.

        Args:
            model: Model name/identifier
            endpoint: Optional API endpoint URL

        Returns:
            ThinkingBehavior if cached, None otherwise
        """
        self.load()

        # Create cache key (model + endpoint hash for uniqueness)
        cache_key = f"{model}@{endpoint}" if endpoint else model

        if cache_key in self._cache:
            behavior_str = self._cache[cache_key].get("behavior")
            if behavior_str:
                try:
                    return ThinkingBehavior(behavior_str)
                except ValueError:
                    # Invalid behavior value in cache
                    pass

        return None

    def set(
        self,
        model: str,
        behavior: ThinkingBehavior,
        supports_think_param: bool = True,
        endpoint: str = ""
    ) -> None:
        """
        Cache thinking behavior for a model.

        Args:
            model: Model name/identifier
            behavior: The detected ThinkingBehavior
            supports_think_param: Whether model supports "think" parameter
            endpoint: Optional API endpoint URL
        """
        self.load()

        cache_key = f"{model}@{endpoint}" if endpoint else model

        # Wall-clock seconds: the cache is persisted to disk and read back in a
        # later process, where a monotonic clock (loop.time()) is meaningless.
        self._cache[cache_key] = {
            "behavior": behavior.value,
            "supports_think_param": supports_think_param,
            "tested_at": time.time()
        }

        self.save()

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self.save()


# Global cache instance
_global_cache: Optional[ThinkingCache] = None


def get_thinking_cache() -> ThinkingCache:
    """
    Get the global thinking cache instance.

    Returns:
        The global ThinkingCache instance
    """
    global _global_cache
    if _global_cache is None:
        cache_file = Path("data/thinking_cache.json")
        _global_cache = ThinkingCache(cache_file)
    return _global_cache
