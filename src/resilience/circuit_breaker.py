#!/usr/bin/env python3
"""
Circuit Breaker Implementation for Echo Brain
Provides resilience patterns for external service calls
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable, Awaitable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import json

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing - blocking requests
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Seconds before trying recovery
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: float = 30.0               # Request timeout in seconds

    # Exponential backoff parameters
    min_backoff: float = 1.0           # Minimum backoff delay
    max_backoff: float = 300.0         # Maximum backoff delay
    backoff_multiplier: float = 2.0    # Backoff multiplier


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    circuit_opens: int = 0
    circuit_closes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_consecutive_failures: int = 0
    current_consecutive_successes: int = 0
    average_response_time: float = 0.0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open"""
    def __init__(self, service_name: str, state: CircuitBreakerState):
        self.service_name = service_name
        self.state = state
        super().__init__(f"Circuit breaker for {service_name} is {state.value}")


class CircuitBreaker:
    """
    Asynchronous circuit breaker implementation

    Protects external service calls with automatic failure detection,
    exponential backoff, and graceful recovery.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()

        # Thread safety
        self._lock = threading.RLock()
        self._state_changed_at = time.time()
        self._next_attempt_time = 0.0

        # State change callbacks
        self._state_listeners: list[Callable[[str, CircuitBreakerState, CircuitBreakerState], None]] = []

        logger.info(f"Circuit breaker '{name}' initialized with config: {config}")

    def add_state_listener(self, callback: Callable[[str, CircuitBreakerState, CircuitBreakerState], None]):
        """Add callback for state changes"""
        self._state_listeners.append(callback)

    def _notify_state_change(self, old_state: CircuitBreakerState, new_state: CircuitBreakerState):
        """Notify listeners of state changes"""
        for callback in self._state_listeners:
            try:
                callback(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

    def _can_attempt_request(self) -> bool:
        """Check if request can be attempted based on current state"""
        current_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if enough time has passed for recovery attempt
            if current_time >= self._next_attempt_time:
                with self._lock:
                    # Double-check after acquiring lock
                    if current_time >= self._next_attempt_time and self.state == CircuitBreakerState.OPEN:
                        self._transition_to_half_open()
                return self.state == CircuitBreakerState.HALF_OPEN
            return False

        return False

    def _transition_to_open(self):
        """Transition to OPEN state"""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self._state_changed_at = time.time()
        self.metrics.circuit_opens += 1

        # Calculate next attempt time with exponential backoff
        backoff_delay = min(
            self.config.min_backoff * (self.config.backoff_multiplier ** self.metrics.circuit_opens),
            self.config.max_backoff
        )
        self._next_attempt_time = time.time() + backoff_delay

        logger.warning(f"Circuit breaker '{self.name}' opened. Next attempt in {backoff_delay:.1f}s")
        self._notify_state_change(old_state, self.state)

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self._state_changed_at = time.time()
        self.metrics.current_consecutive_successes = 0

        logger.info(f"Circuit breaker '{self.name}' entered half-open state for recovery testing")
        self._notify_state_change(old_state, self.state)

    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self._state_changed_at = time.time()
        self.metrics.circuit_closes += 1
        self.metrics.current_consecutive_failures = 0

        logger.info(f"Circuit breaker '{self.name}' closed - service recovered")
        self._notify_state_change(old_state, self.state)

    def _record_success(self, response_time: float):
        """Record successful request"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.utcnow()
            self.metrics.current_consecutive_failures = 0
            self.metrics.current_consecutive_successes += 1

            # Update average response time
            if self.metrics.total_requests == 1:
                self.metrics.average_response_time = response_time
            else:
                # Exponential moving average
                alpha = 0.1  # Smoothing factor
                self.metrics.average_response_time = (
                    alpha * response_time + (1 - alpha) * self.metrics.average_response_time
                )

            # State transitions based on success
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.current_consecutive_successes >= self.config.success_threshold:
                    self._transition_to_closed()

    def _record_failure(self, error: Exception):
        """Record failed request"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.utcnow()
            self.metrics.current_consecutive_failures += 1
            self.metrics.current_consecutive_successes = 0

            if isinstance(error, asyncio.TimeoutError):
                self.metrics.timeout_requests += 1

            # State transitions based on failure
            if self.state == CircuitBreakerState.CLOSED:
                if self.metrics.current_consecutive_failures >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open state goes back to open
                self._transition_to_open()

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        fallback: Optional[Callable[..., Awaitable[Any]]] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection

        Args:
            func: Async function to execute
            *args: Arguments for function
            fallback: Optional fallback function
            **kwargs: Keyword arguments for function

        Returns:
            Result of function execution or fallback

        Raises:
            CircuitBreakerException: When circuit is open and no fallback provided
        """

        # Check if request can be attempted
        if not self._can_attempt_request():
            logger.warning(f"Circuit breaker '{self.name}' is open, request blocked")
            if fallback:
                logger.info(f"Executing fallback for '{self.name}'")
                return await fallback(*args, **kwargs)
            else:
                raise CircuitBreakerException(self.name, self.state)

        # Execute the function with timeout
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )

            # Record success
            response_time = time.time() - start_time
            self._record_success(response_time)

            logger.debug(f"Circuit breaker '{self.name}' - successful call ({response_time:.3f}s)")
            return result

        except Exception as e:
            # Record failure
            self._record_failure(e)

            logger.warning(f"Circuit breaker '{self.name}' - failed call: {type(e).__name__}: {e}")

            # Try fallback if available
            if fallback:
                logger.info(f"Executing fallback for '{self.name}' after failure")
                try:
                    return await fallback(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for '{self.name}': {fallback_error}")
                    raise e  # Raise original error
            else:
                raise e

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics"""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "state_duration": time.time() - self._state_changed_at,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout
                },
                "metrics": {
                    "total_requests": self.metrics.total_requests,
                    "successful_requests": self.metrics.successful_requests,
                    "failed_requests": self.metrics.failed_requests,
                    "timeout_requests": self.metrics.timeout_requests,
                    "failure_rate": round(self.metrics.failure_rate, 2),
                    "success_rate": round(self.metrics.success_rate, 2),
                    "circuit_opens": self.metrics.circuit_opens,
                    "circuit_closes": self.metrics.circuit_closes,
                    "consecutive_failures": self.metrics.current_consecutive_failures,
                    "consecutive_successes": self.metrics.current_consecutive_successes,
                    "average_response_time": round(self.metrics.average_response_time, 3),
                    "last_failure": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                    "last_success": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None
                }
            }

    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            old_state = self.state
            self.state = CircuitBreakerState.CLOSED
            self._state_changed_at = time.time()
            self.metrics = CircuitBreakerMetrics()
            self._next_attempt_time = 0.0

            logger.info(f"Circuit breaker '{self.name}' reset to closed state")
            self._notify_state_change(old_state, self.state)

    def force_open(self):
        """Force circuit breaker to open state (for testing/maintenance)"""
        with self._lock:
            old_state = self.state
            self._transition_to_open()
            logger.warning(f"Circuit breaker '{self.name}' forced to open state")

    def __str__(self) -> str:
        """String representation"""
        return f"CircuitBreaker(name='{self.name}', state={self.state.value}, " \
               f"failures={self.metrics.current_consecutive_failures}/{self.config.failure_threshold})"

    def __repr__(self) -> str:
        return self.__str__()