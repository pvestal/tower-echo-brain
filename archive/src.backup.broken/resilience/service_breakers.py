#!/usr/bin/env python3
"""
Service-specific circuit breakers for Echo Brain system
Manages circuit breakers for ComfyUI, Qdrant, Ollama, and PostgreSQL
"""

import logging
from typing import Dict, Any, Optional
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState

logger = logging.getLogger(__name__)


class ServiceBreakerManager:
    """
    Manages circuit breakers for all external services
    """

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._initialize_breakers()

    def _initialize_breakers(self):
        """Initialize circuit breakers for all external services"""

        # ComfyUI Configuration
        comfyui_config = CircuitBreakerConfig(
            failure_threshold=3,        # Lower threshold - image generation is critical
            recovery_timeout=30.0,      # Shorter recovery time
            success_threshold=2,        # Quick recovery confirmation
            timeout=120.0,              # Longer timeout for image generation
            min_backoff=5.0,           # Minimum 5 second backoff
            max_backoff=300.0          # Maximum 5 minute backoff
        )

        # Qdrant Configuration
        qdrant_config = CircuitBreakerConfig(
            failure_threshold=5,        # Higher threshold - can handle more failures
            recovery_timeout=60.0,      # Standard recovery time
            success_threshold=3,        # Standard recovery confirmation
            timeout=10.0,               # Quick database operations
            min_backoff=1.0,           # Quick retry for database
            max_backoff=120.0          # Maximum 2 minute backoff
        )

        # Ollama Configuration
        ollama_config = CircuitBreakerConfig(
            failure_threshold=3,        # Lower threshold - LLM is critical
            recovery_timeout=45.0,      # Medium recovery time
            success_threshold=2,        # Quick recovery confirmation
            timeout=60.0,               # Medium timeout for model inference
            min_backoff=2.0,           # Minimum 2 second backoff
            max_backoff=180.0          # Maximum 3 minute backoff
        )

        # PostgreSQL Configuration
        postgresql_config = CircuitBreakerConfig(
            failure_threshold=5,        # Higher threshold - database is robust
            recovery_timeout=30.0,      # Quick recovery for database
            success_threshold=3,        # Standard recovery confirmation
            timeout=15.0,               # Quick database operations
            min_backoff=0.5,           # Very quick retry for database
            max_backoff=60.0           # Maximum 1 minute backoff
        )

        # Redis Configuration (if used)
        redis_config = CircuitBreakerConfig(
            failure_threshold=5,        # Higher threshold - cache failures acceptable
            recovery_timeout=20.0,      # Quick recovery for cache
            success_threshold=2,        # Quick recovery confirmation
            timeout=5.0,                # Very quick cache operations
            min_backoff=0.5,           # Very quick retry for cache
            max_backoff=30.0           # Maximum 30 second backoff
        )

        # Create circuit breakers
        self._breakers = {
            "comfyui": CircuitBreaker("comfyui", comfyui_config),
            "qdrant": CircuitBreaker("qdrant", qdrant_config),
            "ollama": CircuitBreaker("ollama", ollama_config),
            "postgresql": CircuitBreaker("postgresql", postgresql_config),
            "redis": CircuitBreaker("redis", redis_config)
        }

        # Add state change logging for all breakers
        for name, breaker in self._breakers.items():
            breaker.add_state_listener(self._log_state_change)

        logger.info(f"Initialized {len(self._breakers)} service circuit breakers")

    def _log_state_change(self, service_name: str, old_state: CircuitBreakerState, new_state: CircuitBreakerState):
        """Log circuit breaker state changes"""
        if new_state == CircuitBreakerState.OPEN:
            logger.error(f"🚨 SERVICE FAILURE: {service_name} circuit breaker opened ({old_state.value} -> {new_state.value})")
        elif new_state == CircuitBreakerState.CLOSED:
            logger.info(f"✅ SERVICE RECOVERY: {service_name} circuit breaker closed ({old_state.value} -> {new_state.value})")
        elif new_state == CircuitBreakerState.HALF_OPEN:
            logger.warning(f"🔄 SERVICE TESTING: {service_name} circuit breaker testing recovery ({old_state.value} -> {new_state.value})")

    def get_breaker(self, service_name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for specific service"""
        return self._breakers.get(service_name.lower())

    def get_comfyui_breaker(self) -> CircuitBreaker:
        """Get ComfyUI circuit breaker"""
        return self._breakers["comfyui"]

    def get_qdrant_breaker(self) -> CircuitBreaker:
        """Get Qdrant circuit breaker"""
        return self._breakers["qdrant"]

    def get_ollama_breaker(self) -> CircuitBreaker:
        """Get Ollama circuit breaker"""
        return self._breakers["ollama"]

    def get_postgresql_breaker(self) -> CircuitBreaker:
        """Get PostgreSQL circuit breaker"""
        return self._breakers["postgresql"]

    def get_redis_breaker(self) -> CircuitBreaker:
        """Get Redis circuit breaker"""
        return self._breakers["redis"]

    def get_all_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers"""
        return self._breakers.copy()

    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services"""
        status = {}
        for name, breaker in self._breakers.items():
            metrics = breaker.get_metrics()
            status[name] = {
                "state": metrics["state"],
                "available": metrics["state"] != "open",
                "failure_rate": metrics["metrics"]["failure_rate"],
                "success_rate": metrics["metrics"]["success_rate"],
                "total_requests": metrics["metrics"]["total_requests"],
                "consecutive_failures": metrics["metrics"]["consecutive_failures"],
                "last_failure": metrics["metrics"]["last_failure"],
                "last_success": metrics["metrics"]["last_success"]
            }
        return status

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        status = self.get_service_status()

        total_services = len(status)
        available_services = sum(1 for s in status.values() if s["available"])
        degraded_services = [name for name, s in status.items() if s["state"] == "half_open"]
        failed_services = [name for name, s in status.items() if s["state"] == "open"]

        overall_health = "healthy"
        if failed_services:
            overall_health = "critical" if len(failed_services) > total_services // 2 else "degraded"
        elif degraded_services:
            overall_health = "degraded"

        return {
            "overall_health": overall_health,
            "available_services": available_services,
            "total_services": total_services,
            "availability_percentage": (available_services / total_services) * 100,
            "failed_services": failed_services,
            "degraded_services": degraded_services,
            "healthy_services": [name for name, s in status.items()
                               if s["state"] == "closed" and s["available"]]
        }

    def reset_all_breakers(self):
        """Reset all circuit breakers to closed state"""
        for name, breaker in self._breakers.items():
            breaker.reset()
            logger.info(f"Reset circuit breaker for {name}")

    def reset_breaker(self, service_name: str) -> bool:
        """Reset specific circuit breaker"""
        breaker = self.get_breaker(service_name)
        if breaker:
            breaker.reset()
            logger.info(f"Reset circuit breaker for {service_name}")
            return True
        return False

    def force_open_breaker(self, service_name: str) -> bool:
        """Force specific circuit breaker open (for testing/maintenance)"""
        breaker = self.get_breaker(service_name)
        if breaker:
            breaker.force_open()
            logger.warning(f"Forced circuit breaker open for {service_name}")
            return True
        return False


# Global service breaker manager instance
_service_breaker_manager = None


def get_service_breaker_manager() -> ServiceBreakerManager:
    """Get global service breaker manager instance"""
    global _service_breaker_manager
    if _service_breaker_manager is None:
        _service_breaker_manager = ServiceBreakerManager()
    return _service_breaker_manager


def get_comfyui_breaker() -> CircuitBreaker:
    """Get ComfyUI circuit breaker"""
    return get_service_breaker_manager().get_comfyui_breaker()


def get_qdrant_breaker() -> CircuitBreaker:
    """Get Qdrant circuit breaker"""
    return get_service_breaker_manager().get_qdrant_breaker()


def get_ollama_breaker() -> CircuitBreaker:
    """Get Ollama circuit breaker"""
    return get_service_breaker_manager().get_ollama_breaker()


def get_postgresql_breaker() -> CircuitBreaker:
    """Get PostgreSQL circuit breaker"""
    return get_service_breaker_manager().get_postgresql_breaker()


def get_redis_breaker() -> CircuitBreaker:
    """Get Redis circuit breaker"""
    return get_service_breaker_manager().get_redis_breaker()