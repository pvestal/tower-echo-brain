"""
Circuit breaker pattern implementation for resilient external service calls.
"""

import asyncio
import time
from enum import Enum
from typing import Callable, Any, TypeVar, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Blocking calls due to failures
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening circuit
    reset_timeout: int = 60             # Seconds before trying half-open
    half_open_max_calls: int = 3        # Max calls to test in half-open state
    timeout: float = 30.0               # Call timeout in seconds


class CircuitBreakerError(Exception):
    """Raised when circuit breaker blocks a call."""
    pass


class CircuitBreaker:
    """
    Implements circuit breaker pattern for external service resilience.

    Protects against cascading failures by:
    - Tracking failure rates
    - Opening circuit when failure threshold exceeded
    - Periodically testing service recovery
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception from function
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.config.reset_timeout:
                    raise CircuitBreakerError("Circuit breaker is OPEN")
                else:
                    # Try half-open state
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN state")

            elif self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerError("Circuit breaker HALF_OPEN limit exceeded")

        try:
            # Execute function with timeout
            result = await asyncio.wait_for(
                self._execute_async(func, *args, **kwargs),
                timeout=self.config.timeout
            )

            # Success - update circuit state
            await self._record_success()
            return result

        except Exception as e:
            # Failure - update circuit state
            await self._record_failure()
            raise

    async def _execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function, converting sync to async if needed."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

    async def _record_success(self):
        """Record successful call and update circuit state."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker closing - service recovered")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    async def _record_failure(self):
        """Record failed call and update circuit state."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker opening - service still failing")
            elif (self.state == CircuitState.CLOSED and
                  self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opening - {self.failure_count} failures")

    @property
    def is_closed(self) -> bool:
        """True if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """True if circuit is open (blocking calls)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """True if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    def get_metrics(self) -> dict:
        """Get current circuit breaker metrics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "half_open_calls": self.half_open_calls,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "reset_timeout": self.config.reset_timeout,
                "half_open_max_calls": self.config.half_open_max_calls,
                "timeout": self.config.timeout
            }
        }

    async def reset(self):
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            logger.info("Circuit breaker manually reset")