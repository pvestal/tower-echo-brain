#!/usr/bin/env python3
"""
Bulletproof model management with fallbacks, retries, circuit breakers, and verification.

Design Principles:
1. Never fail silently - always return actionable error or valid result
2. Degrade gracefully - if best model fails, use next best
3. Fail fast when appropriate - don't waste time on known-broken services
4. Verify everything - health checks confirm models actually work
5. Observable - every decision and failure is logged
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, Any, TypeVar, Generic, Dict, List
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import random
import logging
import hashlib
import json
from pathlib import Path

import httpx

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tower.resilient_model_manager")

T = TypeVar('T')


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class ModelState(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    BUSY = "busy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class TaskUrgency(Enum):
    INTERACTIVE = "interactive"  # User waiting: <5s required
    BACKGROUND = "background"    # Can wait: 30s acceptable
    BATCH = "batch"              # Minutes acceptable


class ErrorSeverity(Enum):
    TRANSIENT = "transient"      # Retry likely to help
    DEGRADED = "degraded"        # Service struggling but working
    FATAL = "fatal"              # Don't retry, escalate


@dataclass
class ModelConfig:
    """Static configuration for a model."""
    name: str
    model_id: str                         # Ollama model identifier
    vram_mb: int                          # Memory requirement
    load_time_seconds: float              # Expected cold start time
    strengths: list[str]                  # Task types this model excels at
    quality_scores: dict[str, float]      # Task type -> quality (0-1)
    preferred_gpu: int = 0
    max_concurrent: int = 1               # Max simultaneous requests
    timeout_seconds: float = 120.0


@dataclass
class ModelHealth:
    """Current health status of a model."""
    model_name: str
    state: ModelState
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0               # Recent error rate (0-1)
    last_health_check: Optional[datetime] = None
    last_error: Optional[str] = None


@dataclass
class CircuitState:
    """Circuit breaker state for a model."""
    model_name: str
    is_open: bool = False                 # True = rejecting requests
    opened_at: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    half_open_attempts: int = 0


@dataclass
class ExecutionResult(Generic[T]):
    """Result wrapper with full execution context."""
    success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    error_severity: Optional[ErrorSeverity] = None
    model_used: Optional[str] = None
    fallback_used: bool = False
    attempts: int = 1
    total_latency_ms: float = 0.0
    selection_reason: str = ""


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True                   # Add randomness to prevent thundering herd

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay_seconds * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay_seconds)

        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay


@dataclass
class FallbackChain:
    """Ordered list of fallback models for a task type."""
    task_type: str
    models: list[str]                     # Ordered by preference
    min_quality: float = 0.5              # Minimum acceptable quality


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Prevents cascading failures by temporarily blocking requests to failing services.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Blocking requests, service is known broken
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
        half_open_max_attempts: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout_seconds)
        self.half_open_max_attempts = half_open_max_attempts
        self.circuits: dict[str, CircuitState] = {}

    def get_state(self, model_name: str) -> CircuitState:
        """Get or create circuit state for model."""
        if model_name not in self.circuits:
            self.circuits[model_name] = CircuitState(model_name=model_name)
        return self.circuits[model_name]

    def is_available(self, model_name: str) -> tuple[bool, str]:
        """
        Check if model is available for requests.

        Returns:
            Tuple of (is_available, reason)
        """
        state = self.get_state(model_name)

        if not state.is_open:
            return True, "circuit_closed"

        # Check if recovery timeout has passed
        if state.opened_at:
            elapsed = datetime.now() - state.opened_at
            if elapsed >= self.recovery_timeout:
                # Transition to half-open
                state.half_open_attempts = 0
                return True, "circuit_half_open"

        return False, f"circuit_open_since_{state.opened_at}"

    def record_success(self, model_name: str) -> None:
        """Record successful request."""
        state = self.get_state(model_name)
        state.success_count += 1
        state.failure_count = 0

        # Close circuit if it was half-open
        if state.is_open:
            state.is_open = False
            state.opened_at = None
            logger.info(f"Circuit CLOSED for {model_name} after recovery")

    def record_failure(self, model_name: str) -> None:
        """Record failed request."""
        state = self.get_state(model_name)
        state.failure_count += 1

        if state.is_open:
            state.half_open_attempts += 1
            if state.half_open_attempts >= self.half_open_max_attempts:
                # Reset timeout, stay open
                state.opened_at = datetime.now()
                logger.warning(f"Circuit remains OPEN for {model_name} after failed recovery")
        elif state.failure_count >= self.failure_threshold:
            state.is_open = True
            state.opened_at = datetime.now()
            logger.warning(f"Circuit OPENED for {model_name} after {state.failure_count} failures")


# =============================================================================
# HEALTH CHECKER
# =============================================================================

class ModelHealthChecker:
    """
    Verifies models are actually working, not just running.

    A model is healthy only if:
    1. Process is running
    2. It responds to requests
    3. Response is valid
    4. Latency is acceptable
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        health_check_timeout: float = 30.0,
        max_acceptable_latency_ms: float = 30000.0
    ):
        self.ollama_url = ollama_url
        self.timeout = health_check_timeout
        self.max_latency = max_acceptable_latency_ms
        self.http = httpx.AsyncClient(timeout=self.timeout)
        self.health_cache: dict[str, ModelHealth] = {}
        self.cache_ttl = timedelta(seconds=30)

    async def check_model(self, model_name: str, force: bool = False) -> ModelHealth:
        """
        Perform health check on model.

        Args:
            model_name: Model to check
            force: Bypass cache

        Returns:
            Current health status
        """
        # Check cache first
        if not force and model_name in self.health_cache:
            cached = self.health_cache[model_name]
            if cached.last_health_check:
                age = datetime.now() - cached.last_health_check
                if age < self.cache_ttl:
                    return cached

        health = ModelHealth(
            model_name=model_name,
            state=ModelState.UNKNOWN,
            last_health_check=datetime.now()
        )

        try:
            # Step 1: Check if model is loaded
            loaded = await self._get_loaded_models()
            if model_name not in loaded:
                health.state = ModelState.UNLOADED
                self.health_cache[model_name] = health
                return health

            # Step 2: Send test request and verify response
            start = datetime.now()

            response = await self.http.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Respond with exactly: HEALTH_CHECK_OK",
                    "stream": False,
                    "options": {"num_predict": 20}
                }
            )

            latency_ms = (datetime.now() - start).total_seconds() * 1000

            if response.status_code != 200:
                health.state = ModelState.UNHEALTHY
                health.last_error = f"HTTP {response.status_code}"
                health.consecutive_failures += 1
            else:
                data = response.json()
                response_text = data.get("response", "")

                # Step 3: Verify response is coherent
                if "HEALTH_CHECK_OK" in response_text or len(response_text) > 0:
                    # Step 4: Check latency is acceptable
                    if latency_ms <= self.max_latency:
                        health.state = ModelState.LOADED
                        health.last_success = datetime.now()
                        health.consecutive_failures = 0
                    else:
                        health.state = ModelState.UNHEALTHY
                        health.last_error = f"High latency: {latency_ms:.0f}ms"
                else:
                    health.state = ModelState.UNHEALTHY
                    health.last_error = "Empty or invalid response"
                    health.consecutive_failures += 1

                health.avg_latency_ms = latency_ms

        except httpx.TimeoutException:
            health.state = ModelState.UNHEALTHY
            health.last_error = "Timeout"
            health.consecutive_failures += 1
            health.last_failure = datetime.now()

        except Exception as e:
            health.state = ModelState.UNHEALTHY
            health.last_error = str(e)
            health.consecutive_failures += 1
            health.last_failure = datetime.now()

        self.health_cache[model_name] = health
        return health

    async def _get_loaded_models(self) -> list[str]:
        """Get list of currently loaded models."""
        try:
            response = await self.http.get(f"{self.ollama_url}/api/ps")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    async def check_all(self, model_names: list[str]) -> dict[str, ModelHealth]:
        """Check health of all specified models concurrently."""
        tasks = [self.check_model(name) for name in model_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            name: result if isinstance(result, ModelHealth)
                  else ModelHealth(model_name=name, state=ModelState.UNKNOWN, last_error=str(result))
            for name, result in zip(model_names, results)
        }


# =============================================================================
# RETRY EXECUTOR
# =============================================================================

class RetryExecutor:
    """
    Executes operations with intelligent retry logic.

    Features:
    - Exponential backoff with jitter
    - Classifies errors to decide if retry is worthwhile
    - Respects circuit breaker state
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker,
        default_config: RetryConfig = None
    ):
        self.circuit_breaker = circuit_breaker
        self.default_config = default_config or RetryConfig()

    def classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error to determine retry strategy."""
        error_str = str(error).lower()

        # Transient errors - retry likely to help
        transient_patterns = [
            "timeout", "connection", "temporary", "unavailable",
            "rate limit", "429", "503", "504"
        ]
        if any(p in error_str for p in transient_patterns):
            return ErrorSeverity.TRANSIENT

        # Fatal errors - don't retry
        fatal_patterns = [
            "not found", "404", "invalid model", "unauthorized",
            "out of memory", "oom", "cuda error"
        ]
        if any(p in error_str for p in fatal_patterns):
            return ErrorSeverity.FATAL

        # Default to degraded - limited retries
        return ErrorSeverity.DEGRADED

    async def execute(
        self,
        model_name: str,
        operation: Callable[[], Any],
        config: RetryConfig = None
    ) -> ExecutionResult:
        """
        Execute operation with retries.

        Args:
            model_name: Model being used (for circuit breaker)
            operation: Async callable to execute
            config: Retry configuration

        Returns:
            ExecutionResult with value or error details
        """
        config = config or self.default_config

        # Check circuit breaker first
        is_available, reason = self.circuit_breaker.is_available(model_name)
        if not is_available:
            return ExecutionResult(
                success=False,
                error=f"Circuit breaker open: {reason}",
                error_severity=ErrorSeverity.FATAL,
                model_used=model_name,
                attempts=0
            )

        last_error = None
        last_severity = None
        start_time = datetime.now()

        for attempt in range(config.max_attempts):
            try:
                result = await operation()

                # Success
                self.circuit_breaker.record_success(model_name)
                latency = (datetime.now() - start_time).total_seconds() * 1000

                return ExecutionResult(
                    success=True,
                    value=result,
                    model_used=model_name,
                    attempts=attempt + 1,
                    total_latency_ms=latency
                )

            except Exception as e:
                last_error = str(e)
                last_severity = self.classify_error(e)

                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_attempts} failed for {model_name}: {e}"
                )

                # Don't retry fatal errors
                if last_severity == ErrorSeverity.FATAL:
                    self.circuit_breaker.record_failure(model_name)
                    break

                # Wait before retry (except on last attempt)
                if attempt < config.max_attempts - 1:
                    delay = config.get_delay(attempt)
                    await asyncio.sleep(delay)

        # All retries exhausted
        self.circuit_breaker.record_failure(model_name)
        latency = (datetime.now() - start_time).total_seconds() * 1000

        return ExecutionResult(
            success=False,
            error=last_error,
            error_severity=last_severity,
            model_used=model_name,
            attempts=config.max_attempts,
            total_latency_ms=latency
        )


# =============================================================================
# RESILIENT MODEL MANAGER
# =============================================================================

class ResilientModelManager:
    """
    Production-grade model manager with full resilience patterns.

    Features:
    - Automatic fallback to alternative models
    - Circuit breaker prevents cascading failures
    - Health checks verify models actually work
    - Retry with exponential backoff
    - Preloading for predicted workloads
    - Observable: all decisions logged
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        state_file: Path = None
    ):
        self.ollama_url = ollama_url
        self.state_file = state_file or Path("/opt/tower-echo-brain/data/model_state.json")

        # Core components
        self.http = httpx.AsyncClient(timeout=120.0)
        self.circuit_breaker = CircuitBreaker()
        self.health_checker = ModelHealthChecker(ollama_url)
        self.retry_executor = RetryExecutor(self.circuit_breaker)

        # Configuration
        self.models = self._load_model_configs()
        self.fallback_chains = self._load_fallback_chains()

        # Runtime state
        self._load_state()

    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load model configurations for Tower's dual-GPU setup."""
        return {
            # AMD RX 9070 XT (16GB) - Primary for large models
            "qwen2.5-coder:32b": ModelConfig(
                name="Qwen 2.5 Coder 32B",
                model_id="qwen2.5-coder:32b",
                vram_mb=20000,
                load_time_seconds=35,
                strengths=["code_generation", "code_review", "code_refactor", "technical"],
                quality_scores={
                    "code_generation": 0.95,
                    "code_review": 0.90,
                    "code_refactor": 0.90,
                    "technical": 0.92,
                    "reasoning": 0.85,
                    "general": 0.70
                },
                preferred_gpu=0
            ),
            "deepseek-r1:32b": ModelConfig(
                name="DeepSeek R1 32B",
                model_id="deepseek-r1:32b",
                vram_mb=18000,
                load_time_seconds=30,
                strengths=["reasoning", "analysis", "code_review", "complex"],
                quality_scores={
                    "reasoning": 0.95,
                    "analysis": 0.90,
                    "code_review": 0.85,
                    "code_generation": 0.80,
                    "complex": 0.92,
                    "general": 0.85
                },
                preferred_gpu=0
            ),
            "deepseek-r1:14b": ModelConfig(
                name="DeepSeek R1 14B",
                model_id="deepseek-r1:14b",
                vram_mb=9000,
                load_time_seconds=15,
                strengths=["reasoning", "general", "fast_response"],
                quality_scores={
                    "reasoning": 0.80,
                    "analysis": 0.75,
                    "code_review": 0.70,
                    "code_generation": 0.65,
                    "general": 0.75,
                    "fast_response": 0.85
                },
                preferred_gpu=0
            ),
            "llama3.2:8b": ModelConfig(
                name="Llama 3.2 8B",
                model_id="llama3.2:8b",
                vram_mb=5000,
                load_time_seconds=8,
                strengths=["general", "fast_response", "creative"],
                quality_scores={
                    "general": 0.70,
                    "fast_response": 0.90,
                    "creative": 0.75,
                    "code_generation": 0.50,
                    "reasoning": 0.60
                },
                preferred_gpu=0
            ),
            "llama3.2:3b": ModelConfig(
                name="Llama 3.2 3B",
                model_id="llama3.2:3b",
                vram_mb=2500,
                load_time_seconds=4,
                strengths=["fast_response", "simple"],
                quality_scores={
                    "fast_response": 0.95,
                    "simple": 0.80,
                    "general": 0.60,
                    "code_generation": 0.40,
                    "reasoning": 0.45
                },
                preferred_gpu=0
            ),
            "tinyllama:latest": ModelConfig(
                name="TinyLlama 1B",
                model_id="tinyllama:latest",
                vram_mb=1000,
                load_time_seconds=2,
                strengths=["ultrafast", "simple"],
                quality_scores={
                    "ultrafast": 1.0,
                    "simple": 0.60,
                    "general": 0.40,
                    "code_generation": 0.25,
                    "reasoning": 0.30
                },
                preferred_gpu=0
            ),

            # Tower-specific: 70B model that requires most of AMD GPU
            "llama3.1:70b": ModelConfig(
                name="Llama 3.1 70B",
                model_id="llama3.1:70b",
                vram_mb=40000,
                load_time_seconds=90,
                strengths=["complex", "reasoning", "analysis"],
                quality_scores={
                    "complex": 0.98,
                    "reasoning": 0.96,
                    "analysis": 0.94,
                    "code_generation": 0.85,
                    "general": 0.90
                },
                preferred_gpu=0,
                max_concurrent=1  # Only one instance possible
            ),

            # Specialized models
            "deepseek-coder:latest": ModelConfig(
                name="DeepSeek Coder",
                model_id="deepseek-coder:latest",
                vram_mb=8500,
                load_time_seconds=12,
                strengths=["code_generation", "code_review"],
                quality_scores={
                    "code_generation": 0.88,
                    "code_review": 0.85,
                    "technical": 0.80,
                    "general": 0.60
                },
                preferred_gpu=0
            ),
            "mixtral:8x7b": ModelConfig(
                name="Mixtral 8x7B",
                model_id="mixtral:8x7b",
                vram_mb=25000,
                load_time_seconds=40,
                strengths=["general", "reasoning", "creative"],
                quality_scores={
                    "general": 0.88,
                    "reasoning": 0.85,
                    "creative": 0.82,
                    "code_generation": 0.75,
                    "analysis": 0.80
                },
                preferred_gpu=0
            )
        }

    def _load_fallback_chains(self) -> Dict[str, FallbackChain]:
        """Define fallback chains for each task type."""
        return {
            "code_generation": FallbackChain(
                task_type="code_generation",
                models=[
                    "qwen2.5-coder:32b",
                    "deepseek-coder:latest",
                    "deepseek-r1:32b",
                    "deepseek-r1:14b",
                    "llama3.2:8b"
                ],
                min_quality=0.5
            ),
            "code_review": FallbackChain(
                task_type="code_review",
                models=[
                    "qwen2.5-coder:32b",
                    "deepseek-r1:32b",
                    "deepseek-coder:latest",
                    "deepseek-r1:14b"
                ],
                min_quality=0.6
            ),
            "reasoning": FallbackChain(
                task_type="reasoning",
                models=[
                    "deepseek-r1:32b",
                    "llama3.1:70b",
                    "deepseek-r1:14b",
                    "mixtral:8x7b",
                    "qwen2.5-coder:32b"
                ],
                min_quality=0.6
            ),
            "complex": FallbackChain(
                task_type="complex",
                models=[
                    "llama3.1:70b",
                    "deepseek-r1:32b",
                    "qwen2.5-coder:32b",
                    "mixtral:8x7b"
                ],
                min_quality=0.8
            ),
            "analysis": FallbackChain(
                task_type="analysis",
                models=[
                    "deepseek-r1:32b",
                    "llama3.1:70b",
                    "mixtral:8x7b",
                    "deepseek-r1:14b"
                ],
                min_quality=0.7
            ),
            "general": FallbackChain(
                task_type="general",
                models=[
                    "deepseek-r1:14b",
                    "llama3.2:8b",
                    "mixtral:8x7b",
                    "deepseek-r1:32b"
                ],
                min_quality=0.5
            ),
            "fast_response": FallbackChain(
                task_type="fast_response",
                models=[
                    "llama3.2:3b",
                    "tinyllama:latest",
                    "llama3.2:8b",
                    "deepseek-r1:14b"
                ],
                min_quality=0.4
            ),
            "creative": FallbackChain(
                task_type="creative",
                models=[
                    "llama3.2:8b",
                    "mixtral:8x7b",
                    "deepseek-r1:14b"
                ],
                min_quality=0.6
            ),
            "technical": FallbackChain(
                task_type="technical",
                models=[
                    "qwen2.5-coder:32b",
                    "deepseek-coder:latest",
                    "deepseek-r1:32b"
                ],
                min_quality=0.7
            ),
            "simple": FallbackChain(
                task_type="simple",
                models=[
                    "tinyllama:latest",
                    "llama3.2:3b",
                    "llama3.2:8b"
                ],
                min_quality=0.3
            ),
            "ultrafast": FallbackChain(
                task_type="ultrafast",
                models=[
                    "tinyllama:latest",
                    "llama3.2:3b"
                ],
                min_quality=0.3
            )
        }

    def _load_state(self) -> None:
        """Load persisted state."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                # Restore circuit breaker state
                for name, state_data in data.get("circuits", {}).items():
                    self.circuit_breaker.circuits[name] = CircuitState(
                        model_name=name,
                        is_open=state_data.get("is_open", False),
                        failure_count=state_data.get("failure_count", 0)
                    )
                logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self) -> None:
        """Persist state for recovery."""
        try:
            data = {
                "circuits": {
                    name: {
                        "is_open": state.is_open,
                        "failure_count": state.failure_count
                    }
                    for name, state in self.circuit_breaker.circuits.items()
                },
                "saved_at": datetime.now().isoformat()
            }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    async def select_model(
        self,
        task_type: str,
        urgency: TaskUrgency = TaskUrgency.BACKGROUND,
        exclude: List[str] = None
    ) -> tuple[Optional[str], str]:
        """
        Select best available model for task.

        Considers:
        - Model capability for task type
        - Current load state (prefer loaded models)
        - Health status
        - Circuit breaker state
        - Urgency (affects willingness to wait for load)

        Returns:
            Tuple of (model_name, selection_reason) or (None, failure_reason)
        """
        exclude = exclude or []
        chain = self.fallback_chains.get(task_type)

        if not chain:
            # Try to find a general fallback
            chain = self.fallback_chains.get("general")
            if not chain:
                return None, f"No fallback chain defined for task: {task_type}"

        # Get current state of all candidate models
        health_status = await self.health_checker.check_all(chain.models)

        # Score each candidate
        candidates = []

        for model_name in chain.models:
            if model_name in exclude:
                continue

            config = self.models.get(model_name)
            if not config:
                continue

            health = health_status.get(model_name)
            if not health:
                continue

            # Check circuit breaker
            cb_available, cb_reason = self.circuit_breaker.is_available(model_name)
            if not cb_available:
                logger.debug(f"Skipping {model_name}: {cb_reason}")
                continue

            # Calculate score
            quality = config.quality_scores.get(task_type, 0.5)

            if quality < chain.min_quality:
                continue

            # Availability scoring
            if health.state == ModelState.LOADED:
                availability_score = 1.0
                estimated_wait = 0
            elif health.state == ModelState.UNLOADED:
                availability_score = 0.3
                estimated_wait = config.load_time_seconds
            elif health.state == ModelState.UNHEALTHY:
                availability_score = 0.1
                estimated_wait = config.load_time_seconds * 2  # Pessimistic
            else:
                availability_score = 0.2
                estimated_wait = config.load_time_seconds

            # Urgency affects scoring
            if urgency == TaskUrgency.INTERACTIVE:
                # Heavily penalize models that need loading
                wait_penalty = estimated_wait / 5.0
            elif urgency == TaskUrgency.BACKGROUND:
                wait_penalty = estimated_wait / 30.0
            else:
                wait_penalty = estimated_wait / 120.0

            final_score = (
                quality * 0.4 +
                availability_score * 0.4 -
                wait_penalty * 0.2
            )

            candidates.append({
                "name": model_name,
                "score": final_score,
                "quality": quality,
                "health": health.state.value,
                "estimated_wait": estimated_wait
            })

        if not candidates:
            return None, f"No available models for task: {task_type}"

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        reason = (
            f"Selected {best['name']} "
            f"(quality={best['quality']:.2f}, "
            f"state={best['health']}, "
            f"wait={best['estimated_wait']:.0f}s)"
        )

        logger.info(reason)
        return best["name"], reason

    async def complete_with_fallback(
        self,
        task_type: str,
        prompt: str,
        system: str = "",
        urgency: TaskUrgency = TaskUrgency.BACKGROUND
    ) -> ExecutionResult[str]:
        """
        Execute completion with automatic fallback on failure.

        Tries models in fallback chain order until one succeeds.
        """
        # MEMORY AUGMENTATION - Add stored memories to prompt
        try:
            import sys
            sys.path.insert(0, '/opt/tower-echo-brain')
            from src.middleware.memory_augmentation_middleware import augment_with_memories
            original_prompt = prompt
            prompt = augment_with_memories(prompt)
            if prompt != original_prompt:
                logger.info(f"📚 Resilient: Query augmented with stored memories")
        except Exception as e:
            logger.debug(f"Memory augmentation skipped: {e}")

        chain = self.fallback_chains.get(task_type)
        if not chain:
            # Try general fallback
            chain = self.fallback_chains.get("general")
            if not chain:
                return ExecutionResult(
                    success=False,
                    error=f"Unknown task type: {task_type}",
                    error_severity=ErrorSeverity.FATAL
                )

        tried_models = []
        last_error = None
        start_time = datetime.now()

        while len(tried_models) < len(chain.models):
            # Select next model to try
            model_name, reason = await self.select_model(
                task_type=task_type,
                urgency=urgency,
                exclude=tried_models
            )

            if not model_name:
                break

            tried_models.append(model_name)
            config = self.models[model_name]

            # Execute with retries
            async def do_completion():
                response = await self.http.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": config.model_id,
                        "prompt": prompt,
                        "system": system,
                        "stream": False
                    },
                    timeout=config.timeout_seconds
                )
                response.raise_for_status()
                return response.json()["response"]

            result = await self.retry_executor.execute(
                model_name=model_name,
                operation=do_completion,
                config=RetryConfig(max_attempts=2)  # Fewer retries per model, more fallbacks
            )

            if result.success:
                result.fallback_used = len(tried_models) > 1
                result.selection_reason = reason
                result.total_latency_ms = (datetime.now() - start_time).total_seconds() * 1000

                # Persist state after successful operation
                self._save_state()

                return result

            last_error = result.error
            logger.warning(f"Model {model_name} failed, trying fallback...")

        # All models failed
        total_latency = (datetime.now() - start_time).total_seconds() * 1000

        return ExecutionResult(
            success=False,
            error=f"All models failed. Last error: {last_error}. Tried: {tried_models}",
            error_severity=ErrorSeverity.FATAL,
            attempts=len(tried_models),
            total_latency_ms=total_latency
        )

    async def preload_model(self, model_name: str) -> ExecutionResult[bool]:
        """Preload a model into memory."""
        config = self.models.get(model_name)
        if not config:
            return ExecutionResult(
                success=False,
                error=f"Unknown model: {model_name}",
                error_severity=ErrorSeverity.FATAL
            )

        logger.info(f"Preloading {model_name} (expected: {config.load_time_seconds}s)")

        async def do_preload():
            response = await self.http.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": config.model_id,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=config.load_time_seconds + 30
            )
            response.raise_for_status()
            return True

        result = await self.retry_executor.execute(
            model_name=model_name,
            operation=do_preload
        )

        if result.success:
            logger.info(f"Preloaded {model_name} in {result.total_latency_ms:.0f}ms")

        return result

    async def get_system_status(self) -> dict:
        """Get comprehensive status of all models and circuits."""
        health = await self.health_checker.check_all(list(self.models.keys()))

        status = {
            "models": {},
            "circuits": {},
            "timestamp": datetime.now().isoformat()
        }

        for model_name, config in self.models.items():
            model_health = health.get(model_name, ModelHealth(
                model_name=model_name,
                state=ModelState.UNKNOWN
            ))
            cb_available, cb_reason = self.circuit_breaker.is_available(model_name)
            cb_state = self.circuit_breaker.get_state(model_name)

            status["models"][model_name] = {
                "state": model_health.state.value,
                "last_error": model_health.last_error,
                "avg_latency_ms": model_health.avg_latency_ms,
                "consecutive_failures": model_health.consecutive_failures,
                "vram_mb": config.vram_mb,
                "load_time_s": config.load_time_seconds
            }

            status["circuits"][model_name] = {
                "available": cb_available,
                "reason": cb_reason,
                "failure_count": cb_state.failure_count
            }

        return status

    async def close(self):
        """Clean up resources."""
        self._save_state()
        await self.http.aclose()
        await self.health_checker.http.aclose()


# Singleton instance for Echo Brain
_manager_instance: Optional[ResilientModelManager] = None

async def get_resilient_manager() -> ResilientModelManager:
    """Get or create the singleton resilient model manager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ResilientModelManager()
    return _manager_instance