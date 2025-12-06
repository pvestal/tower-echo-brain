#!/usr/bin/env python3
"""
Test suite for verified execution system.

Tests the complete elimination of "execution theater" - ensures Echo Brain
only claims success when actions actually work and can be verified.
"""

import asyncio
import httpx
import json
from datetime import datetime
import sys

# Add src to path for imports
sys.path.insert(0, '/opt/tower-echo-brain')

from src.managers.verified_execution_manager import (
    get_verified_execution_manager,
    restart_service_verified,
    check_service_status_verified,
    kill_process_verified,
    check_port_verified,
    ExecutionStatus
)


class TestResults:
    """Track test results with detailed reporting."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def record(self, name: str, passed: bool, details: str = "", critical: bool = False):
        """Record a test result."""
        status = "✅ PASS" if passed else ("🚨 CRITICAL FAIL" if critical else "❌ FAIL")
        self.results.append({
            "name": name,
            "passed": passed,
            "details": details,
            "critical": critical
        })

        if passed:
            self.passed += 1
        else:
            self.failed += 1

        print(f"{status}: {name}")
        if details and not passed:
            print(f"       {details}")
        elif details and passed and critical:
            print(f"       {details}")

    def summary(self):
        """Print comprehensive test summary."""
        total = self.passed + self.failed
        critical_failures = [r for r in self.results if not r["passed"] and r["critical"]]

        print("\n" + "="*70)
        print("VERIFIED EXECUTION TEST RESULTS")
        print("="*70)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📊 Success Rate: {self.passed / total * 100:.1f}%" if total > 0 else "📊 No tests run")

        if critical_failures:
            print(f"\n🚨 CRITICAL FAILURES: {len(critical_failures)}")
            for failure in critical_failures:
                print(f"  - {failure['name']}: {failure['details']}")

        if self.failed == 0:
            print("\n🎉 ALL TESTS PASSED! Execution theater eliminated.")
        else:
            print(f"\n⚠️  {self.failed} test(s) failed. Review output above.")

        return total == 0 or self.failed == 0


async def test_verified_execution_manager():
    """Test the core verified execution manager functionality."""
    results = TestResults()

    # Test 1: Manager initialization
    try:
        manager = await get_verified_execution_manager()
        results.record(
            "Verified execution manager initialization",
            manager is not None,
            f"Manager: {type(manager).__name__}"
        )
    except Exception as e:
        results.record(
            "Verified execution manager initialization",
            False,
            f"Exception: {e}",
            critical=True
        )
        return results

    # Test 2: Action templates loaded
    results.record(
        "Action templates loaded",
        len(manager.action_templates) > 0,
        f"Loaded {len(manager.action_templates)} action templates"
    )

    # Test 3: Test a safe read-only action (git status)
    try:
        git_result = await manager.execute_verified_action(
            "git_status",
            {}
        )
        results.record(
            "Read-only action execution (git status)",
            git_result.status == ExecutionStatus.SUCCEEDED,
            f"Status: {git_result.status.value}, Output: {git_result.actual_outcome[:100]}"
        )
    except Exception as e:
        results.record(
            "Read-only action execution",
            False,
            f"Exception: {e}"
        )

    # Test 4: Test action with missing parameters (should fail gracefully)
    try:
        missing_param_result = await manager.execute_verified_action(
            "restart_service",
            {}  # Missing 'service' parameter
        )
        results.record(
            "Graceful handling of missing parameters",
            missing_param_result.status == ExecutionStatus.FAILED,
            f"Correctly failed with: {missing_param_result.actual_outcome}"
        )
    except Exception as e:
        results.record(
            "Graceful handling of missing parameters",
            False,
            f"Should handle missing params gracefully, got exception: {e}"
        )

    # Test 5: Test unknown action template
    try:
        unknown_result = await manager.execute_verified_action(
            "nonexistent_action",
            {"param": "value"}
        )
        results.record(
            "Unknown action template handling",
            unknown_result.status == ExecutionStatus.FAILED and "not found" in unknown_result.stderr.lower(),
            f"Error: {unknown_result.stderr}"
        )
    except Exception as e:
        results.record(
            "Unknown action template handling",
            False,
            f"Should handle unknown actions gracefully: {e}"
        )

    return results


async def test_service_operations():
    """Test verified service management operations."""
    results = TestResults()

    # Test 1: Check status of Echo Brain (should be running)
    try:
        status_result = await check_service_status_verified("tower-echo-brain")
        results.record(
            "Service status check (tower-echo-brain)",
            status_result.actually_worked,
            f"Service status: {status_result.actual_outcome[:200]}",
            critical=True
        )
    except Exception as e:
        results.record(
            "Service status check",
            False,
            f"Exception: {e}",
            critical=True
        )

    # Test 2: Check status of non-existent service
    try:
        fake_service_result = await check_service_status_verified("nonexistent-service-12345")
        results.record(
            "Non-existent service status check",
            not fake_service_result.actually_worked,  # Should fail
            f"Correctly identified non-existent service: {fake_service_result.actual_outcome[:100]}"
        )
    except Exception as e:
        results.record(
            "Non-existent service status check",
            False,
            f"Exception: {e}"
        )

    return results


async def test_port_verification():
    """Test port checking with actual verification."""
    results = TestResults()

    # Test 1: Check Echo Brain port (should be responsive)
    try:
        port_result = await check_port_verified(8309)
        results.record(
            "Port verification (Echo Brain 8309)",
            port_result.actually_worked,
            f"Port 8309 verification: {port_result.actual_outcome}",
            critical=True
        )
    except Exception as e:
        results.record(
            "Port verification",
            False,
            f"Exception: {e}",
            critical=True
        )

    # Test 2: Check definitely unused port
    try:
        unused_port_result = await check_port_verified(9999)
        results.record(
            "Unused port verification",
            not unused_port_result.actually_worked,  # Should fail
            f"Correctly identified unused port: {unused_port_result.actual_outcome}"
        )
    except Exception as e:
        results.record(
            "Unused port verification",
            False,
            f"Exception: {e}"
        )

    return results


async def test_api_endpoints():
    """Test the verified execution API endpoints."""
    results = TestResults()

    # Check if Echo Brain is responding
    try:
        async with httpx.AsyncClient() as client:
            health_response = await client.get("http://localhost:8309/api/echo/health")
            echo_available = health_response.status_code == 200
    except:
        echo_available = False

    if not echo_available:
        results.record(
            "Echo Brain API availability",
            False,
            "Echo Brain API not responding, skipping API tests",
            critical=True
        )
        return results

    async with httpx.AsyncClient() as client:
        # Test 1: Health endpoint
        try:
            response = await client.get("http://localhost:8309/api/echo/verified/health")
            results.record(
                "Verified execution health endpoint",
                response.status_code == 200,
                f"Status: {response.status_code}, Content: {response.text[:100]}"
            )
        except Exception as e:
            results.record(
                "Verified execution health endpoint",
                False,
                f"Exception: {e}"
            )

        # Test 2: List available actions
        try:
            response = await client.get("http://localhost:8309/api/echo/verified/actions")
            if response.status_code == 200:
                data = response.json()
                results.record(
                    "List available actions endpoint",
                    data.get("success", False) and len(data.get("actions", [])) > 0,
                    f"Found {len(data.get('actions', []))} actions"
                )
            else:
                results.record(
                    "List available actions endpoint",
                    False,
                    f"HTTP {response.status_code}: {response.text[:100]}"
                )
        except Exception as e:
            results.record(
                "List available actions endpoint",
                False,
                f"Exception: {e}"
            )

        # Test 3: Execution summary
        try:
            response = await client.get("http://localhost:8309/api/echo/verified/summary")
            if response.status_code == 200:
                data = response.json()
                results.record(
                    "Execution summary endpoint",
                    "total_executions" in data,
                    f"Summary: {json.dumps(data, indent=2)}"
                )
            else:
                results.record(
                    "Execution summary endpoint",
                    False,
                    f"HTTP {response.status_code}: {response.text[:100]}"
                )
        except Exception as e:
            results.record(
                "Execution summary endpoint",
                False,
                f"Exception: {e}"
            )

        # Test 4: Service status via API
        try:
            response = await client.get("http://localhost:8309/api/echo/verified/service/tower-echo-brain/status")
            if response.status_code == 200:
                data = response.json()
                results.record(
                    "Service status API endpoint",
                    data.get("success", False),
                    f"Service status via API: {data.get('verification', 'No verification data')[:100]}",
                    critical=True
                )
            else:
                results.record(
                    "Service status API endpoint",
                    False,
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    critical=True
                )
        except Exception as e:
            results.record(
                "Service status API endpoint",
                False,
                f"Exception: {e}",
                critical=True
            )

        # Test 5: Port check via API
        try:
            response = await client.post(
                "http://localhost:8309/api/echo/verified/port/check",
                json={"port": 8309}
            )
            if response.status_code == 200:
                data = response.json()
                results.record(
                    "Port check API endpoint",
                    data.get("success", False),
                    f"Port check via API: {data.get('verification', 'No verification data')[:100]}",
                    critical=True
                )
            else:
                results.record(
                    "Port check API endpoint",
                    False,
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    critical=True
                )
        except Exception as e:
            results.record(
                "Port check API endpoint",
                False,
                f"Exception: {e}",
                critical=True
            )

    return results


async def test_llm_analysis():
    """Test LLM-powered failure analysis."""
    results = TestResults()

    try:
        manager = await get_verified_execution_manager()

        # Create a mock failure for analysis
        from src.managers.verified_execution_manager import ExecutionResult, ExecutionStatus

        mock_failure = ExecutionResult(
            status=ExecutionStatus.VERIFICATION_FAILED,
            action_taken="sudo systemctl restart fake-service",
            expected_outcome="Service should be active and running",
            actual_outcome="Service failed to start - port already in use",
            verification_method="systemctl is-active fake-service",
            stderr="Job for fake-service.service failed. See systemctl status for details.",
            duration_ms=1500,
            timestamp=datetime.now()
        )

        analysis = await manager.analyze_failure_with_llm(mock_failure)

        results.record(
            "LLM failure analysis",
            "analysis" in analysis and len(analysis["analysis"]) > 50,
            f"Analysis quality: {analysis.get('confidence', 'unknown')}, Model: {analysis.get('model_used', 'unknown')}"
        )

    except Exception as e:
        results.record(
            "LLM failure analysis",
            False,
            f"Exception: {e}"
        )

    return results


async def test_cooldown_system():
    """Test the cooldown system prevents aggressive retries."""
    results = TestResults()

    try:
        manager = await get_verified_execution_manager()

        # Execute a failing action twice quickly
        result1 = await manager.execute_verified_action("restart_service", {"service": "nonexistent-service-xyz"})
        result2 = await manager.execute_verified_action("restart_service", {"service": "nonexistent-service-xyz"})

        # First should fail due to nonexistent service, second should fail due to cooldown
        cooldown_working = (
            result1.status == ExecutionStatus.FAILED and
            result2.status == ExecutionStatus.FAILED and
            "cooldown" in result2.stderr.lower()
        )

        results.record(
            "Cooldown system prevents aggressive retries",
            cooldown_working,
            f"First: {result1.actual_outcome[:50]}, Second: {result2.stderr[:50]}"
        )

    except Exception as e:
        results.record(
            "Cooldown system",
            False,
            f"Exception: {e}"
        )

    return results


async def run_execution_theater_elimination_test():
    """
    Comprehensive test to verify execution theater is completely eliminated.

    This test validates that Echo Brain never claims success without proof.
    """
    print("\n" + "="*70)
    print("EXECUTION THEATER ELIMINATION TEST SUITE")
    print("Validating that Echo Brain only claims success with verification")
    print("="*70 + "\n")

    all_results = []

    # Core functionality tests
    print("\n🔧 CORE FUNCTIONALITY TESTS")
    print("-" * 50)
    all_results.append(await test_verified_execution_manager())

    # Service operations tests
    print("\n🔄 SERVICE MANAGEMENT TESTS")
    print("-" * 50)
    all_results.append(await test_service_operations())

    # Port verification tests
    print("\n🌐 NETWORK VERIFICATION TESTS")
    print("-" * 50)
    all_results.append(await test_port_verification())

    # API endpoint tests
    print("\n📡 API ENDPOINT TESTS")
    print("-" * 50)
    all_results.append(await test_api_endpoints())

    # LLM analysis tests
    print("\n🧠 INTELLIGENT ANALYSIS TESTS")
    print("-" * 50)
    all_results.append(await test_llm_analysis())

    # Cooldown system tests
    print("\n⏰ COOLDOWN SYSTEM TESTS")
    print("-" * 50)
    all_results.append(await test_cooldown_system())

    # Aggregate results
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    critical_failures = []

    for result_set in all_results:
        critical_failures.extend([r for r in result_set.results if not r["passed"] and r["critical"]])

    print("\n" + "="*70)
    print("FINAL EXECUTION THEATER ELIMINATION RESULTS")
    print("="*70)
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    print(f"📊 Overall Success Rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")

    if critical_failures:
        print(f"\n🚨 CRITICAL SYSTEM FAILURES: {len(critical_failures)}")
        print("These failures indicate core system issues that must be addressed:")
        for failure in critical_failures:
            print(f"  - {failure['name']}: {failure['details']}")
        print("\n⚠️  System is NOT ready for production with critical failures!")
    else:
        print(f"\n🎉 NO CRITICAL FAILURES - Core system integrity maintained!")

    if total_failed == 0:
        print("\n🏆 EXECUTION THEATER COMPLETELY ELIMINATED!")
        print("   Echo Brain now only claims success when actions are verified.")
        print("   The system has bulletproof execution integrity.")
    elif len(critical_failures) == 0:
        print("\n✅ EXECUTION THEATER MOSTLY ELIMINATED")
        print("   Minor issues remain but core verification works.")
    else:
        print("\n💥 EXECUTION THEATER STILL PRESENT")
        print("   Critical failures indicate system still has execution theater issues.")

    return total_failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_execution_theater_elimination_test())
    exit(0 if success else 1)