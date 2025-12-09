#!/usr/bin/env python3
"""
Test suite for resilient model manager in Echo Brain.

This verifies the integration works correctly with Tower's infrastructure.
"""

import asyncio
import json
from datetime import datetime
import httpx

# Add src to path
import sys
sys.path.insert(0, '/opt/tower-echo-brain')

from src.managers.resilient_model_manager import (
    ResilientModelManager,
    CircuitBreaker,
    ModelHealthChecker,
    RetryExecutor,
    TaskUrgency,
    ErrorSeverity,
    RetryConfig
)


class TestResults:
    """Track test results."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def record(self, name: str, passed: bool, details: str = ""):
        """Record a test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append({
            "name": name,
            "passed": passed,
            "details": details
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

        print(f"{status}: {name}")
        if details and not passed:
            print(f"       {details}")

    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print("\n" + "="*60)
        print(f"RESULTS: {self.passed}/{total} tests passed")
        if self.failed > 0:
            print(f"Failed tests:")
            for result in self.results:
                if not result["passed"]:
                    print(f"  - {result['name']}: {result['details']}")
        print("="*60 + "\n")


async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    results = TestResults()

    # Test 1: Circuit opens after threshold
    cb = CircuitBreaker(failure_threshold=3)
    model = "test_model"

    for i in range(3):
        cb.record_failure(model)

    is_available, reason = cb.is_available(model)
    results.record(
        "Circuit opens after threshold",
        not is_available and "open" in reason.lower(),
        f"Available: {is_available}, Reason: {reason}"
    )

    # Test 2: Circuit closes after success
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.1)
    model = "test_model2"

    cb.record_failure(model)
    cb.record_failure(model)
    await asyncio.sleep(0.2)  # Wait for recovery timeout

    # Should be half-open now
    is_available, _ = cb.is_available(model)
    cb.record_success(model)

    state = cb.get_state(model)
    results.record(
        "Circuit closes after recovery",
        is_available and not state.is_open,
        f"Available: {is_available}, Open: {state.is_open}"
    )

    return results


async def test_retry_logic():
    """Test retry executor with backoff."""
    results = TestResults()

    cb = CircuitBreaker()
    executor = RetryExecutor(cb, RetryConfig(
        max_attempts=3,
        base_delay_seconds=0.1,
        exponential_base=2.0
    ))

    attempt_times = []

    async def failing_operation():
        attempt_times.append(datetime.now())
        if len(attempt_times) < 3:
            raise Exception("Transient connection error")
        return "success"

    result = await executor.execute("test_model", failing_operation)

    # Should succeed on third attempt
    results.record(
        "Retry eventually succeeds",
        result.success and result.attempts == 3,
        f"Success: {result.success}, Attempts: {result.attempts}"
    )

    # Verify exponential backoff
    if len(attempt_times) >= 2:
        delays = [
            (attempt_times[i+1] - attempt_times[i]).total_seconds()
            for i in range(len(attempt_times) - 1)
        ]
        increasing = all(delays[i+1] > delays[i] for i in range(len(delays) - 1))
        results.record(
            "Exponential backoff works",
            increasing or len(delays) < 2,
            f"Delays: {[f'{d:.2f}s' for d in delays]}"
        )

    return results


async def test_error_classification():
    """Test error classification for retry decisions."""
    results = TestResults()

    executor = RetryExecutor(CircuitBreaker())

    test_cases = [
        (Exception("Connection timeout"), ErrorSeverity.TRANSIENT),
        (Exception("503 Service Unavailable"), ErrorSeverity.TRANSIENT),
        (Exception("Model not found"), ErrorSeverity.FATAL),
        (Exception("CUDA out of memory"), ErrorSeverity.FATAL),
        (Exception("Some random error"), ErrorSeverity.DEGRADED),
    ]

    all_correct = True
    for error, expected in test_cases:
        actual = executor.classify_error(error)
        if actual != expected:
            all_correct = False
            results.record(
                f"Classify '{error}'",
                False,
                f"Expected {expected}, got {actual}"
            )

    if all_correct:
        results.record("Error classification", True)

    return results


async def test_model_manager():
    """Test the full resilient model manager."""
    results = TestResults()

    manager = ResilientModelManager()

    # Test 1: Model configuration loaded
    results.record(
        "Model configs loaded",
        len(manager.models) > 0,
        f"Loaded {len(manager.models)} models"
    )

    # Test 2: Fallback chains configured
    results.record(
        "Fallback chains configured",
        len(manager.fallback_chains) > 0,
        f"Configured {len(manager.fallback_chains)} chains"
    )

    # Test 3: Can select model for task
    model, reason = await manager.select_model(
        task_type="general",
        urgency=TaskUrgency.BACKGROUND
    )
    results.record(
        "Model selection works",
        model is not None,
        f"Selected: {model}, Reason: {reason}"
    )

    # Test 4: State persistence
    manager._save_state()
    results.record(
        "State persistence works",
        manager.state_file.exists(),
        f"State file: {manager.state_file}"
    )

    return results


async def test_ollama_integration():
    """Test actual Ollama integration if available."""
    results = TestResults()

    # Check if Ollama is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/ps")
            ollama_available = response.status_code == 200
    except:
        ollama_available = False

    if not ollama_available:
        print("⚠️  Ollama not available, skipping integration tests")
        return results

    manager = ResilientModelManager()

    # Test actual model query
    result = await manager.complete_with_fallback(
        task_type="simple",
        prompt="Reply with exactly: TEST_SUCCESS",
        urgency=TaskUrgency.INTERACTIVE
    )

    results.record(
        "Ollama completion works",
        result.success,
        f"Model: {result.model_used}, Latency: {result.total_latency_ms:.0f}ms"
    )

    # Test health check
    health_checker = ModelHealthChecker()
    loaded_models = await health_checker._get_loaded_models()

    results.record(
        "Can list loaded models",
        isinstance(loaded_models, list),
        f"Found {len(loaded_models)} loaded models"
    )

    return results


async def test_echo_api():
    """Test the Echo Brain API integration."""
    results = TestResults()

    # Check if Echo Brain is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8309/api/echo/health")
            echo_available = response.status_code == 200
    except:
        echo_available = False

    if not echo_available:
        print("⚠️  Echo Brain not running, skipping API tests")
        return results

    async with httpx.AsyncClient() as client:
        # Test resilient query endpoint
        response = await client.post(
            "http://localhost:8309/api/echo/resilient/query",
            json={
                "query": "What is 2 + 2?",
                "task_type": "simple",
                "urgency": "interactive"
            }
        )

        results.record(
            "Resilient query endpoint",
            response.status_code in [200, 503],  # 503 if models not available
            f"Status: {response.status_code}"
        )

        # Test status endpoint
        response = await client.get(
            "http://localhost:8309/api/echo/resilient/status"
        )

        results.record(
            "Status endpoint works",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )

        if response.status_code == 200:
            data = response.json()
            results.record(
                "Status contains model info",
                "models" in data and "circuits" in data,
                f"Keys: {list(data.keys())}"
            )

        # Test model list endpoint
        response = await client.get(
            "http://localhost:8309/api/echo/resilient/models"
        )

        results.record(
            "Models list endpoint",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )

        if response.status_code == 200:
            data = response.json()
            results.record(
                "Models list contains data",
                len(data.get("models", [])) > 0,
                f"Found {len(data.get('models', []))} models"
            )

    return results


async def main():
    """Run all test suites."""
    print("\n" + "="*60)
    print("RESILIENT MODEL MANAGER TEST SUITE")
    print("Tower Echo Brain Integration")
    print("="*60 + "\n")

    all_results = []

    # Unit tests
    print("\n📦 UNIT TESTS")
    print("-"*40)
    all_results.append(await test_circuit_breaker())
    all_results.append(await test_retry_logic())
    all_results.append(await test_error_classification())

    # Integration tests
    print("\n🔗 INTEGRATION TESTS")
    print("-"*40)
    all_results.append(await test_model_manager())
    all_results.append(await test_ollama_integration())

    # API tests
    print("\n🌐 API TESTS")
    print("-"*40)
    all_results.append(await test_echo_api())

    # Summary
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📊 Success Rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")

    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Resilient model manager ready for production.")
    else:
        print("\n⚠️  Some tests failed. Review output above for details.")

    return total_failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)