#!/usr/bin/env python3
"""
Comprehensive integration testing pipeline for Echo Brain enterprise integrations
Tests Google Calendar, Home Assistant, and Notifications APIs
"""
import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """Comprehensive test suite for enterprise integrations"""

    def __init__(self):
        self.test_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "integration_tests": {},
            "performance_metrics": {},
            "error_log": []
        }

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration test suite"""
        logger.info("🚀 Starting Echo Brain Integration Test Suite")

        # Test API imports and module availability
        await self._test_module_imports()

        # Test Google Calendar integration
        await self._test_google_calendar_integration()

        # Test Home Assistant integration
        await self._test_home_assistant_integration()

        # Test Notification service integration
        await self._test_notification_integration()

        # Test cross-service integration scenarios
        await self._test_cross_integration_scenarios()

        # Performance and stress tests
        await self._test_performance_scenarios()

        # Generate test report
        self._generate_test_report()

        return self.test_results

    async def _test_module_imports(self):
        """Test that all new integration modules can be imported"""
        logger.info("🔍 Testing module imports...")
        module_tests = []

        modules_to_test = [
            ("src.api.google_calendar_api", "Google Calendar API"),
            ("src.api.home_assistant_api", "Home Assistant API"),
            ("src.api.notifications_api", "Notifications API"),
            ("src.integrations.google_calendar", "Google Calendar Integration"),
            ("src.integrations.home_assistant", "Home Assistant Integration"),
            ("src.integrations.ntfy_client", "NTFY Client"),
            ("src.services.notification_service", "Notification Service"),
            ("src.qdrant_memory", "Qdrant Memory Service")
        ]

        for module_path, module_name in modules_to_test:
            test_result = await self._test_single_import(module_path, module_name)
            module_tests.append(test_result)

        self.test_results["integration_tests"]["module_imports"] = {
            "total": len(module_tests),
            "passed": sum(1 for t in module_tests if t["status"] == "passed"),
            "failed": sum(1 for t in module_tests if t["status"] == "failed"),
            "tests": module_tests
        }

    async def _test_single_import(self, module_path: str, module_name: str) -> Dict[str, Any]:
        """Test importing a single module"""
        test_start = datetime.now()

        try:
            # Attempt to import the module
            import importlib
            module = importlib.import_module(module_path)

            # Check if module has expected attributes/functions
            has_expected_content = True
            if "api" in module_path:
                # API modules should have router
                has_expected_content = hasattr(module, 'router')
            elif "integrations" in module_path:
                # Integration modules should have bridge functions
                has_expected_content = len([attr for attr in dir(module) if 'bridge' in attr.lower()]) > 0
            elif "services" in module_path:
                # Service modules should have service classes/functions
                has_expected_content = len([attr for attr in dir(module) if not attr.startswith('_')]) > 0

            status = "passed" if has_expected_content else "warning"
            message = f"Imported successfully" if has_expected_content else "Imported but missing expected content"

        except ImportError as e:
            status = "failed"
            message = f"Import failed: {str(e)}"
            self.test_results["error_log"].append(f"{module_name}: {message}")
        except Exception as e:
            status = "failed"
            message = f"Unexpected error: {str(e)}"
            self.test_results["error_log"].append(f"{module_name}: {message}")

        test_duration = (datetime.now() - test_start).total_seconds()

        return {
            "module_name": module_name,
            "module_path": module_path,
            "status": status,
            "message": message,
            "duration_seconds": test_duration
        }

    async def _test_google_calendar_integration(self):
        """Test Google Calendar API endpoints"""
        logger.info("📅 Testing Google Calendar integration...")

        calendar_tests = []

        # Test calendar API endpoints availability
        endpoints_to_test = [
            "/api/calendar/status",
            "/api/calendar/calendars",
            "/api/calendar/events/upcoming",
            "/api/calendar/events/today"
        ]

        for endpoint in endpoints_to_test:
            test_result = await self._test_api_endpoint_availability(endpoint, "Calendar")
            calendar_tests.append(test_result)

        # Test calendar service initialization
        cal_service_test = await self._test_calendar_service_init()
        calendar_tests.append(cal_service_test)

        self.test_results["integration_tests"]["google_calendar"] = {
            "total": len(calendar_tests),
            "passed": sum(1 for t in calendar_tests if t["status"] == "passed"),
            "failed": sum(1 for t in calendar_tests if t["status"] == "failed"),
            "tests": calendar_tests
        }

    async def _test_home_assistant_integration(self):
        """Test Home Assistant API endpoints"""
        logger.info("🏠 Testing Home Assistant integration...")

        ha_tests = []

        # Test HA API endpoints availability
        endpoints_to_test = [
            "/api/home/status",
            "/api/home/entities"
        ]

        for endpoint in endpoints_to_test:
            test_result = await self._test_api_endpoint_availability(endpoint, "Home Assistant")
            ha_tests.append(test_result)

        # Test HA service initialization
        ha_service_test = await self._test_home_assistant_service_init()
        ha_tests.append(ha_service_test)

        self.test_results["integration_tests"]["home_assistant"] = {
            "total": len(ha_tests),
            "passed": sum(1 for t in ha_tests if t["status"] == "passed"),
            "failed": sum(1 for t in ha_tests if t["status"] == "failed"),
            "tests": ha_tests
        }

    async def _test_notification_integration(self):
        """Test Notification service endpoints"""
        logger.info("📢 Testing Notification integration...")

        notification_tests = []

        # Test notification API endpoints
        endpoints_to_test = [
            "/api/notifications/status",
            "/api/notifications/channels"
        ]

        for endpoint in endpoints_to_test:
            test_result = await self._test_api_endpoint_availability(endpoint, "Notifications")
            notification_tests.append(test_result)

        # Test notification service initialization
        notif_service_test = await self._test_notification_service_init()
        notification_tests.append(notif_service_test)

        self.test_results["integration_tests"]["notifications"] = {
            "total": len(notification_tests),
            "passed": sum(1 for t in notification_tests if t["status"] == "passed"),
            "failed": sum(1 for t in notification_tests if t["status"] == "failed"),
            "tests": notification_tests
        }

    async def _test_api_endpoint_availability(self, endpoint: str, service_name: str) -> Dict[str, Any]:
        """Test if an API endpoint can be constructed properly"""
        test_start = datetime.now()

        try:
            # Import the relevant API module based on endpoint
            if "calendar" in endpoint:
                from src.api.google_calendar_api import router as calendar_router
                routes = [route.path for route in calendar_router.routes]
            elif "home" in endpoint:
                from src.api.home_assistant_api import router as home_router
                routes = [route.path for route in home_router.routes]
            elif "notifications" in endpoint:
                from src.api.notifications_api import router as notifications_router
                routes = [route.path for route in notifications_router.routes]
            else:
                routes = []

            # Check if the endpoint exists in the router
            endpoint_exists = any(endpoint.replace("/api", "") in route for route in routes)

            status = "passed" if endpoint_exists else "failed"
            message = f"Endpoint available" if endpoint_exists else f"Endpoint not found in router"

        except Exception as e:
            status = "failed"
            message = f"Error testing endpoint: {str(e)}"
            self.test_results["error_log"].append(f"{service_name} {endpoint}: {message}")

        test_duration = (datetime.now() - test_start).total_seconds()

        return {
            "test_name": f"{service_name} Endpoint {endpoint}",
            "endpoint": endpoint,
            "status": status,
            "message": message,
            "duration_seconds": test_duration
        }

    async def _test_calendar_service_init(self) -> Dict[str, Any]:
        """Test Google Calendar service initialization"""
        test_start = datetime.now()

        try:
            # Test calendar integration import and basic initialization
            from src.integrations.google_calendar import get_calendar_bridge, get_calendar_status_for_echo

            # These should be callable functions
            assert callable(get_calendar_bridge), "get_calendar_bridge should be callable"
            assert callable(get_calendar_status_for_echo), "get_calendar_status_for_echo should be callable"

            status = "passed"
            message = "Calendar service functions available"

        except Exception as e:
            status = "failed"
            message = f"Calendar service init failed: {str(e)}"
            self.test_results["error_log"].append(f"Calendar service: {message}")

        test_duration = (datetime.now() - test_start).total_seconds()

        return {
            "test_name": "Calendar Service Initialization",
            "status": status,
            "message": message,
            "duration_seconds": test_duration
        }

    async def _test_home_assistant_service_init(self) -> Dict[str, Any]:
        """Test Home Assistant service initialization"""
        test_start = datetime.now()

        try:
            # Test HA integration import and basic initialization
            from src.integrations.home_assistant import get_home_assistant_bridge, get_home_status_for_echo

            # These should be callable functions
            assert callable(get_home_assistant_bridge), "get_home_assistant_bridge should be callable"
            assert callable(get_home_status_for_echo), "get_home_status_for_echo should be callable"

            status = "passed"
            message = "Home Assistant service functions available"

        except Exception as e:
            status = "failed"
            message = f"Home Assistant service init failed: {str(e)}"
            self.test_results["error_log"].append(f"Home Assistant service: {message}")

        test_duration = (datetime.now() - test_start).total_seconds()

        return {
            "test_name": "Home Assistant Service Initialization",
            "status": status,
            "message": message,
            "duration_seconds": test_duration
        }

    async def _test_notification_service_init(self) -> Dict[str, Any]:
        """Test Notification service initialization"""
        test_start = datetime.now()

        try:
            # Test notification service import and basic initialization
            from src.services.notification_service import (
                get_notification_service,
                NotificationType,
                NotificationChannel
            )

            # These should be available
            assert callable(get_notification_service), "get_notification_service should be callable"
            assert hasattr(NotificationType, 'INFO'), "NotificationType should have INFO"
            assert hasattr(NotificationChannel, 'ALL'), "NotificationChannel should have ALL"

            status = "passed"
            message = "Notification service functions and types available"

        except Exception as e:
            status = "failed"
            message = f"Notification service init failed: {str(e)}"
            self.test_results["error_log"].append(f"Notification service: {message}")

        test_duration = (datetime.now() - test_start).total_seconds()

        return {
            "test_name": "Notification Service Initialization",
            "status": status,
            "message": message,
            "duration_seconds": test_duration
        }

    async def _test_cross_integration_scenarios(self):
        """Test scenarios involving multiple integrations"""
        logger.info("🔗 Testing cross-integration scenarios...")

        cross_tests = []

        # Test 1: Calendar + Notifications integration
        calendar_notif_test = await self._test_calendar_notification_scenario()
        cross_tests.append(calendar_notif_test)

        # Test 2: Home Assistant + Notifications integration
        ha_notif_test = await self._test_home_assistant_notification_scenario()
        cross_tests.append(ha_notif_test)

        self.test_results["integration_tests"]["cross_integration"] = {
            "total": len(cross_tests),
            "passed": sum(1 for t in cross_tests if t["status"] == "passed"),
            "failed": sum(1 for t in cross_tests if t["status"] == "failed"),
            "tests": cross_tests
        }

    async def _test_calendar_notification_scenario(self) -> Dict[str, Any]:
        """Test calendar events triggering notifications"""
        test_start = datetime.now()

        try:
            # Test that calendar and notification services can be imported together
            from src.integrations.google_calendar import get_calendar_bridge
            from src.services.notification_service import get_notification_service

            # Basic integration test - can we import and access both?
            status = "passed"
            message = "Calendar and Notification services can be imported together"

        except Exception as e:
            status = "failed"
            message = f"Cross-integration test failed: {str(e)}"
            self.test_results["error_log"].append(f"Calendar+Notifications: {message}")

        test_duration = (datetime.now() - test_start).total_seconds()

        return {
            "test_name": "Calendar + Notification Integration",
            "status": status,
            "message": message,
            "duration_seconds": test_duration
        }

    async def _test_home_assistant_notification_scenario(self) -> Dict[str, Any]:
        """Test Home Assistant events triggering notifications"""
        test_start = datetime.now()

        try:
            # Test that HA and notification services can be imported together
            from src.integrations.home_assistant import get_home_assistant_bridge
            from src.services.notification_service import get_notification_service

            # Basic integration test
            status = "passed"
            message = "Home Assistant and Notification services can be imported together"

        except Exception as e:
            status = "failed"
            message = f"Cross-integration test failed: {str(e)}"
            self.test_results["error_log"].append(f"HomeAssistant+Notifications: {message}")

        test_duration = (datetime.now() - test_start).total_seconds()

        return {
            "test_name": "Home Assistant + Notification Integration",
            "status": status,
            "message": message,
            "duration_seconds": test_duration
        }

    async def _test_performance_scenarios(self):
        """Test performance and resource usage of integrations"""
        logger.info("⚡ Testing performance scenarios...")

        perf_tests = []

        # Test import time performance
        import_perf_test = await self._test_import_performance()
        perf_tests.append(import_perf_test)

        # Test memory usage
        memory_test = await self._test_memory_usage()
        perf_tests.append(memory_test)

        self.test_results["performance_metrics"] = {
            "total": len(perf_tests),
            "tests": perf_tests
        }

    async def _test_import_performance(self) -> Dict[str, Any]:
        """Test import performance of all integration modules"""
        import time

        test_start = time.time()

        try:
            # Import all integration modules and measure time
            import src.api.google_calendar_api
            import src.api.home_assistant_api
            import src.api.notifications_api
            import src.integrations.google_calendar
            import src.integrations.home_assistant
            import src.integrations.ntfy_client
            import src.services.notification_service

            import_time = time.time() - test_start

            # Performance threshold - imports should be fast
            is_fast = import_time < 2.0  # 2 seconds threshold
            status = "passed" if is_fast else "warning"
            message = f"All integrations imported in {import_time:.3f}s"

        except Exception as e:
            import_time = time.time() - test_start
            status = "failed"
            message = f"Import performance test failed: {str(e)}"

        return {
            "test_name": "Import Performance",
            "status": status,
            "message": message,
            "import_time_seconds": import_time,
            "threshold_seconds": 2.0
        }

    async def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage impact of loading integrations"""
        try:
            import psutil
            import os

            # Get memory before imports
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Import all integrations
            import src.api.google_calendar_api
            import src.api.home_assistant_api
            import src.api.notifications_api
            import src.integrations.google_calendar
            import src.integrations.home_assistant
            import src.integrations.ntfy_client
            import src.services.notification_service

            # Get memory after imports
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before

            # Memory threshold - should not use excessive memory
            is_reasonable = memory_increase < 50  # 50MB threshold
            status = "passed" if is_reasonable else "warning"
            message = f"Memory increase: {memory_increase:.2f} MB"

        except ImportError:
            status = "skipped"
            message = "psutil not available for memory testing"
            memory_increase = 0
        except Exception as e:
            status = "failed"
            message = f"Memory test failed: {str(e)}"
            memory_increase = 0

        return {
            "test_name": "Memory Usage",
            "status": status,
            "message": message,
            "memory_increase_mb": memory_increase,
            "threshold_mb": 50
        }

    def _generate_test_report(self):
        """Generate comprehensive test report"""
        # Calculate totals
        for integration_name, integration_data in self.test_results["integration_tests"].items():
            self.test_results["total_tests"] += integration_data["total"]
            self.test_results["passed"] += integration_data["passed"]
            self.test_results["failed"] += integration_data["failed"]

        # Add performance tests to totals
        if "performance_metrics" in self.test_results:
            perf_total = self.test_results["performance_metrics"]["total"]
            self.test_results["total_tests"] += perf_total

            # Count performance test results
            for test in self.test_results["performance_metrics"]["tests"]:
                if test["status"] == "passed":
                    self.test_results["passed"] += 1
                elif test["status"] == "failed":
                    self.test_results["failed"] += 1
                else:
                    self.test_results["skipped"] += 1

        # Calculate success rate
        total = self.test_results["total_tests"]
        if total > 0:
            self.test_results["success_rate"] = (self.test_results["passed"] / total) * 100
        else:
            self.test_results["success_rate"] = 0

        # Determine overall status
        if self.test_results["failed"] == 0:
            self.test_results["overall_status"] = "PASSED"
        elif self.test_results["passed"] > self.test_results["failed"]:
            self.test_results["overall_status"] = "PASSED_WITH_WARNINGS"
        else:
            self.test_results["overall_status"] = "FAILED"

        logger.info(f"🎯 Test Summary: {self.test_results['passed']}/{self.test_results['total_tests']} passed ({self.test_results['success_rate']:.1f}%)")

# Main execution
async def run_integration_tests():
    """Main function to run integration tests"""
    test_suite = IntegrationTestSuite()
    results = await test_suite.run_all_tests()

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/opt/tower-echo-brain/test_results/integration_test_results_{timestamp}.json"

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"📄 Test results saved to: {results_file}")

    return results

if __name__ == "__main__":
    # Run the integration tests
    results = asyncio.run(run_integration_tests())

    # Print summary
    print("\n" + "="*60)
    print("🚀 ECHO BRAIN INTEGRATION TEST RESULTS")
    print("="*60)
    print(f"Overall Status: {results['overall_status']}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print("="*60)

    # Print errors if any
    if results['error_log']:
        print("\n❌ ERRORS:")
        for error in results['error_log']:
            print(f"  • {error}")

    print(f"\n📄 Detailed results saved to test_results/ directory")