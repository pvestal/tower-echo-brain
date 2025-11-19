#!/usr/bin/env python3
"""
Test Runner for Echo Brain Anime Generation Testing Framework
Provides convenient command-line interface for running various test suites.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ensure tests directory is in Python path
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))


class TestRunner:
    """Main test runner for anime generation testing framework"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = None
        self.results = {
            "suites_run": [],
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "errors": [],
            "performance_data": {}
        }

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def run_pytest_command(self, args: List[str]) -> Dict:
        """Run pytest with given arguments and capture results"""
        cmd = ["pytest"] + args

        if self.verbose:
            self.log(f"Running command: {' '.join(cmd)}")

        try:
            # Ensure we're in the correct directory
            os.chdir(tests_dir.parent)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Test execution timed out after 30 minutes",
                "return_code": 124
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Test execution failed: {str(e)}",
                "return_code": 1
            }

    def run_unit_tests(self, fast: bool = False) -> bool:
        """Run unit tests"""
        self.log("Running unit tests...")

        args = [
            "tests/unit/",
            "-v",
            "--tb=short",
            "--junitxml=tests/junit/unit-results.xml"
        ]

        if fast:
            args.extend(["-m", "not slow"])

        result = self.run_pytest_command(args)
        self.results["suites_run"].append("unit")

        if result["success"]:
            self.log("Unit tests PASSED", "SUCCESS")
            return True
        else:
            self.log(f"Unit tests FAILED: {result['stderr']}", "ERROR")
            self.results["errors"].append(f"Unit tests: {result['stderr']}")
            return False

    def run_integration_tests(self, service: Optional[str] = None) -> bool:
        """Run integration tests"""
        self.log(f"Running integration tests{f' for {service}' if service else ''}...")

        args = [
            "tests/integration/",
            "-v",
            "--tb=short",
            "--junitxml=tests/junit/integration-results.xml"
        ]

        if service:
            if service == "comfyui":
                args = ["tests/integration/test_comfyui_integration.py"] + args[1:]
            elif service == "character":
                args = ["tests/integration/test_character_generation.py"] + args[1:]

        result = self.run_pytest_command(args)
        self.results["suites_run"].append("integration")

        if result["success"]:
            self.log("Integration tests PASSED", "SUCCESS")
            return True
        else:
            self.log(f"Integration tests FAILED: {result['stderr']}", "ERROR")
            self.results["errors"].append(f"Integration tests: {result['stderr']}")
            return False

    def run_visual_tests(self) -> bool:
        """Run visual validation tests"""
        self.log("Running visual validation tests...")

        # Check if LLaVA is available
        import httpx
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get("http://***REMOVED***:11434/api/tags")
                if response.status_code != 200:
                    self.log("LLaVA service not available, skipping visual tests", "WARNING")
                    return True
        except Exception:
            self.log("LLaVA service not available, skipping visual tests", "WARNING")
            return True

        args = [
            "tests/visual/",
            "-v",
            "--tb=short",
            "--junitxml=tests/junit/visual-results.xml"
        ]

        result = self.run_pytest_command(args)
        self.results["suites_run"].append("visual")

        if result["success"]:
            self.log("Visual tests PASSED", "SUCCESS")
            return True
        else:
            self.log(f"Visual tests FAILED: {result['stderr']}", "ERROR")
            self.results["errors"].append(f"Visual tests: {result['stderr']}")
            return False

    def run_performance_tests(self, quick: bool = False) -> bool:
        """Run performance benchmark tests"""
        self.log("Running performance tests...")

        args = [
            "tests/performance/",
            "-v",
            "--tb=short",
            "--junitxml=tests/junit/performance-results.xml"
        ]

        if quick:
            args.extend(["-m", "not slow"])
        else:
            args.extend(["--benchmark-only"])

        result = self.run_pytest_command(args)
        self.results["suites_run"].append("performance")

        if result["success"]:
            self.log("Performance tests PASSED", "SUCCESS")
            return True
        else:
            self.log(f"Performance tests FAILED: {result['stderr']}", "ERROR")
            self.results["errors"].append(f"Performance tests: {result['stderr']}")
            return False

    def check_services(self) -> Dict[str, bool]:
        """Check if required services are available"""
        self.log("Checking service availability...")

        services = {
            "ComfyUI": "http://***REMOVED***:8188",
            "Anime API": "http://***REMOVED***:8328/api/health",
            "Echo API": "http://***REMOVED***:8309/api/echo/health",
            "LLaVA": "http://***REMOVED***:11434/api/tags"
        }

        status = {}
        import httpx

        for service_name, url in services.items():
            try:
                with httpx.Client(timeout=10) as client:
                    response = client.get(url)
                    available = response.status_code == 200
                    status[service_name] = available

                    if available:
                        self.log(f"✅ {service_name} is available")
                    else:
                        self.log(f"❌ {service_name} is not available (HTTP {response.status_code})")

            except Exception as e:
                status[service_name] = False
                self.log(f"❌ {service_name} is not available ({str(e)})")

        return status

    def run_smoke_tests(self) -> bool:
        """Run basic smoke tests to verify system functionality"""
        self.log("Running smoke tests...")

        # Check services first
        services = self.check_services()
        required_services = ["ComfyUI", "Anime API", "Echo API"]

        missing_services = [name for name in required_services if not services.get(name, False)]
        if missing_services:
            self.log(f"Critical services unavailable: {missing_services}", "ERROR")
            return False

        # Run basic unit tests
        args = [
            "tests/unit/test_database_operations.py::TestDatabaseConsistency::test_acid_compliance",
            "tests/integration/test_comfyui_integration.py::TestComfyUIIntegration::test_comfyui_service_availability",
            "-v",
            "--tb=short",
            "--junitxml=tests/junit/smoke-results.xml"
        ]

        result = self.run_pytest_command(args)

        if result["success"]:
            self.log("Smoke tests PASSED", "SUCCESS")
            return True
        else:
            self.log(f"Smoke tests FAILED: {result['stderr']}", "ERROR")
            return False

    def run_comprehensive_suite(self) -> bool:
        """Run the complete test suite"""
        self.log("Starting comprehensive test suite...")
        self.start_time = time.time()

        success = True

        # 1. Smoke tests first
        if not self.run_smoke_tests():
            self.log("Smoke tests failed, aborting comprehensive suite", "ERROR")
            return False

        # 2. Unit tests
        if not self.run_unit_tests():
            success = False

        # 3. Integration tests
        if not self.run_integration_tests():
            success = False

        # 4. Visual tests (if available)
        if not self.run_visual_tests():
            success = False

        # 5. Performance tests (quick version)
        if not self.run_performance_tests(quick=True):
            success = False

        return success

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate test execution report"""
        if self.start_time:
            self.results["total_duration"] = time.time() - self.start_time
        else:
            self.results["total_duration"] = 0

        self.results["timestamp"] = datetime.now().isoformat()
        self.results["success"] = len(self.results["errors"]) == 0

        # Parse JUnit XML files for detailed results
        self._parse_junit_results()

        report = {
            "test_execution_summary": self.results,
            "recommendations": self._generate_recommendations()
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.log(f"Test report saved to {output_file}")

        return json.dumps(report, indent=2, default=str)

    def _parse_junit_results(self):
        """Parse JUnit XML files to extract test statistics"""
        import xml.etree.ElementTree as ET
        from pathlib import Path

        junit_dir = Path(tests_dir) / "junit"
        if not junit_dir.exists():
            return

        total_tests = 0
        total_failures = 0
        total_errors = 0

        for xml_file in junit_dir.glob("*-results.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                tests = int(root.get("tests", 0))
                failures = int(root.get("failures", 0))
                errors = int(root.get("errors", 0))

                total_tests += tests
                total_failures += failures
                total_errors += errors

            except Exception as e:
                self.log(f"Error parsing {xml_file}: {e}", "WARNING")

        self.results.update({
            "total_tests": total_tests,
            "failed_tests": total_failures + total_errors,
            "passed_tests": total_tests - total_failures - total_errors
        })

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        if self.results["failed_tests"] > 0:
            recommendations.append("Review failed tests and address underlying issues")

        if "performance" in self.results["suites_run"]:
            recommendations.append("Analyze performance metrics for optimization opportunities")

        if "visual" not in self.results["suites_run"]:
            recommendations.append("Set up LLaVA service for visual validation testing")

        if self.results["errors"]:
            recommendations.append("Address test infrastructure errors for reliable CI/CD")

        return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Test Runner for Echo Brain Anime Generation Testing Framework"
    )

    parser.add_argument(
        "suite",
        choices=["unit", "integration", "visual", "performance", "smoke", "comprehensive"],
        help="Test suite to run"
    )

    parser.add_argument(
        "--service",
        choices=["comfyui", "character"],
        help="Specific service for integration tests"
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file for test report"
    )

    parser.add_argument(
        "--check-services",
        action="store_true",
        help="Only check service availability"
    )

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose)

    if args.check_services:
        services = runner.check_services()
        all_available = all(services.values())
        sys.exit(0 if all_available else 1)

    # Run selected test suite
    success = False

    if args.suite == "unit":
        success = runner.run_unit_tests(fast=args.fast)
    elif args.suite == "integration":
        success = runner.run_integration_tests(service=args.service)
    elif args.suite == "visual":
        success = runner.run_visual_tests()
    elif args.suite == "performance":
        success = runner.run_performance_tests(quick=args.fast)
    elif args.suite == "smoke":
        success = runner.run_smoke_tests()
    elif args.suite == "comprehensive":
        success = runner.run_comprehensive_suite()

    # Generate report
    if args.output or not success:
        output_file = args.output or f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        runner.generate_report(output_file)

    # Print summary
    print("\n" + "="*60)
    print("TEST EXECUTION SUMMARY")
    print("="*60)
    print(f"Suites run: {', '.join(runner.results['suites_run'])}")
    print(f"Total tests: {runner.results['total_tests']}")
    print(f"Passed: {runner.results['passed_tests']}")
    print(f"Failed: {runner.results['failed_tests']}")
    print(f"Success rate: {(runner.results['passed_tests'] / max(runner.results['total_tests'], 1)) * 100:.1f}%")

    if runner.results["errors"]:
        print(f"Errors: {len(runner.results['errors'])}")
        for error in runner.results["errors"]:
            print(f"  - {error}")

    print("="*60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()