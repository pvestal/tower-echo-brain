"""Fallback cache for degraded-mode operation."""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FallbackHandler:
    """Caches successful operation results so they can be served when services are down."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def cache_operation(self, service: str, operation: str, result: Any, ttl_seconds: int = 300):
        key = f"{service}:{operation}"
        self._cache[key] = {
            "operation": f"{service}_{operation}",
            "result": result,
            "created_at": datetime.now(),
            "expires_at": time.time() + ttl_seconds,
        }

    def get_fallback(self, service: str, operation: str) -> Optional[Any]:
        key = f"{service}:{operation}"
        entry = self._cache.get(key)
        if entry and time.time() < entry["expires_at"]:
            return entry["result"]
        return None

    def get_cached_operations(self) -> Dict[str, Dict[str, Any]]:
        # Prune expired entries
        now = time.time()
        self._cache = {k: v for k, v in self._cache.items() if now < v["expires_at"]}
        return dict(self._cache)

    def clear_cache(self, service: Optional[str] = None):
        if service is None:
            self._cache.clear()
        else:
            self._cache = {k: v for k, v in self._cache.items() if not k.startswith(f"{service}:")}


# singleton
_handler_instance: Optional[FallbackHandler] = None


def get_fallback_handler() -> FallbackHandler:
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = FallbackHandler()
    return _handler_instance
