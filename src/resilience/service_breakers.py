"""Service-level circuit breakers for Tower infrastructure."""
import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class CircuitBreakerOpenError(Exception):
    """Raised when a call is attempted on an open circuit breaker."""


class ServiceCircuitBreaker:
    """Circuit breaker for a single external service."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
        half_open_max_attempts: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.half_open_max_attempts = half_open_max_attempts

        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.total_calls = 0
        self._half_open_attempts = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None

    async def call(self, async_fn: Callable) -> Any:
        """Execute *async_fn* through the circuit breaker."""
        now = time.time()

        if self.state == "open":
            if self.opened_at and (now - self.opened_at.timestamp()) >= self.recovery_timeout_seconds:
                self.state = "half_open"
                self._half_open_attempts = 0
                logger.info(f"[CB:{self.name}] transitioning open → half_open")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker for {self.name} is open")

        if self.state == "half_open" and self._half_open_attempts >= self.half_open_max_attempts:
            self._trip()
            raise CircuitBreakerOpenError(f"Circuit breaker for {self.name} re-opened after half_open max attempts")

        self.total_calls += 1
        if self.state == "half_open":
            self._half_open_attempts += 1

        try:
            result = await async_fn()
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    # ── internal ──────────────────────────────────────────────────────

    def _on_success(self):
        self.success_count += 1
        self.last_success_time = datetime.now()
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            logger.info(f"[CB:{self.name}] half_open → closed (success)")

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.state == "closed" and self.failure_count >= self.failure_threshold:
            self._trip()
        elif self.state == "half_open":
            self._trip()

    def _trip(self):
        self.state = "open"
        self.opened_at = datetime.now()
        logger.warning(f"[CB:{self.name}] circuit OPEN (failures={self.failure_count})")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }

    def reset(self):
        self.state = "closed"
        self.failure_count = 0
        self._half_open_attempts = 0
        self.opened_at = None

    def force_open(self):
        self._trip()


# ── manager ──────────────────────────────────────────────────────────────

_SERVICE_CONFIGS = {
    "comfyui": {"failure_threshold": 3, "recovery_timeout_seconds": 120.0},
    "qdrant": {"failure_threshold": 3, "recovery_timeout_seconds": 30.0},
    "ollama": {"failure_threshold": 3, "recovery_timeout_seconds": 90.0},
    "postgresql": {"failure_threshold": 2, "recovery_timeout_seconds": 30.0},
}


class ServiceBreakerManager:
    """Manages circuit breakers for all monitored services."""

    def __init__(self):
        self._breakers: Dict[str, ServiceCircuitBreaker] = {}
        for name, cfg in _SERVICE_CONFIGS.items():
            self._breakers[name] = ServiceCircuitBreaker(name, **cfg)

    def get_breaker(self, name: str) -> Optional[ServiceCircuitBreaker]:
        return self._breakers.get(name)

    def get_all_breakers(self) -> Dict[str, ServiceCircuitBreaker]:
        return dict(self._breakers)

    def get_service_status(self) -> Dict[str, Any]:
        return {name: b.get_metrics() for name, b in self._breakers.items()}

    def get_health_summary(self) -> Dict[str, Any]:
        total = len(self._breakers)
        healthy = sum(1 for b in self._breakers.values() if b.state == "closed")
        degraded = sum(1 for b in self._breakers.values() if b.state == "half_open")
        down = sum(1 for b in self._breakers.values() if b.state == "open")
        return {"total_services": total, "healthy": healthy, "degraded": degraded, "down": down}

    def force_open_breaker(self, name: str) -> bool:
        b = self._breakers.get(name)
        if not b:
            return False
        b.force_open()
        return True

    def reset_all_breakers(self):
        for b in self._breakers.values():
            b.reset()


# singleton
_manager_instance: Optional[ServiceBreakerManager] = None


def get_service_breaker_manager() -> ServiceBreakerManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ServiceBreakerManager()
    return _manager_instance
