"""
Resilience module for Echo Brain system
Provides circuit breaker patterns for external service protection
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerState
from .service_breakers import ServiceBreakerManager
from .fallback_handlers import FallbackHandler
from .monitoring import CircuitBreakerMonitor

__all__ = [
    'CircuitBreaker',
    'CircuitBreakerState',
    'ServiceBreakerManager',
    'FallbackHandler',
    'CircuitBreakerMonitor'
]