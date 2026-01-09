#!/usr/bin/env python3
"""
Comprehensive Test Runner for Echo Brain

Runs all comprehensive tests with proper reporting and categorization.
Supports running specific test categories or all tests.

Usage:
    python run_comprehensive_tests.py                    # Run all tests
    python run_comprehensive_tests.py --category api    # Run only API tests
    python run_comprehensive_tests.py --live            # Run with live server
    python run_comprehensive_tests.py --report          # Generate HTML report
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Test categories and their files
TEST_CATEGORIES = {
    "api": "test_api_endpoints.py",
    "routing": "test_model_routing.py",
    "memory": "test_memory_qdrant.py",
    "database": "test_database.py",
    "cognitive": "test_cognitive_reasoning.py",
    "workers": "test_autonomous_workers.py",
    "anime": "test_anime_integration.py",
    "auth": "test_auth_services.py",
}

# Test markers for pytest
MARKERS = {
    "unit": "Unit tests (fast, isolated)",
    "integration": "Integration tests (require services)",
    "live": "Live tests (require running Echo Brain)",
    "slow": "Slow running tests",
}


def get_test_dir():
    """Get the comprehensive tests directory"""
    return Path(__file__).parent


def run_pytest(args: list, capture_output: bool = False):
    """Run pytest with given arguments"""
    cmd = [sys.executable, "-m", "pytest"] + args

    print(f"Running: {' '.join(cmd)}")

    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result
    else:
        return subprocess.run(cmd)


def run_category(category: str, extra_args: list = None):
    """Run tests for a specific category"""
    if category not in TEST_CATEGORIES:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(TEST_CATEGORIES.keys())}")
        return 1

    test_file = get_test_dir() / TEST_CATEGORIES[category]
    args = [str(test_file), "-v"]

    if extra_args:
        args.extend(extra_args)

    result = run_pytest(args)
    return result.returncode


def run_all_tests(extra_args: list = None):
    """Run all comprehensive tests"""
    test_dir = get_test_dir()
    args = [str(test_dir), "-v", "--tb=short"]

    if extra_args:
        args.extend(extra_args)

    result = run_pytest(args)
    return result.returncode


def run_quick_smoke_tests():
    """Run quick smoke tests to verify basic functionality"""
    print("\n" + "=" * 60)
    print("RUNNING QUICK SMOKE TESTS")
    print("=" * 60 + "\n")

    test_dir = get_test_dir()
    args = [
        str(test_dir),
        "-v",
        "-x",  # Stop on first failure
        "--tb=short",
        "-k", "health or init or import",  # Run only basic tests
        "--timeout=30"
    ]

    result = run_pytest(args)
    return result.returncode


def run_unit_tests():
    """Run only unit tests (no external dependencies)"""
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60 + "\n")

    test_dir = get_test_dir()
    args = [
        str(test_dir),
        "-v",
        "-m", "not integration and not live",
        "--tb=short"
    ]

    result = run_pytest(args)
    return result.returncode


def run_integration_tests():
    """Run integration tests (require services)"""
    print("\n" + "=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60 + "\n")

    test_dir = get_test_dir()
    args = [
        str(test_dir),
        "-v",
        "-m", "integration",
        "--tb=short"
    ]

    result = run_pytest(args)
    return result.returncode


def run_live_tests(echo_brain_url: str = "http://localhost:8309"):
    """Run tests against live Echo Brain server"""
    print("\n" + "=" * 60)
    print(f"RUNNING LIVE TESTS against {echo_brain_url}")
    print("=" * 60 + "\n")

    # Set environment variable for test config
    os.environ["ECHO_BRAIN_URL"] = echo_brain_url

    test_dir = get_test_dir()
    args = [
        str(test_dir),
        "-v",
        "-m", "not skip_live",
        "--tb=short",
        "--timeout=60"
    ]

    result = run_pytest(args)
    return result.returncode


def generate_report(output_dir: str = "test_reports"):
    """Generate HTML test report"""
    print("\n" + "=" * 60)
    print("GENERATING TEST REPORT")
    print("=" * 60 + "\n")

    test_dir = get_test_dir()
    report_dir = Path(output_dir)
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"echo_brain_test_report_{timestamp}.html"

    args = [
        str(test_dir),
        "-v",
        f"--html={report_file}",
        "--self-contained-html",
        "--tb=short"
    ]

    result = run_pytest(args)

    if result.returncode == 0:
        print(f"\n✅ Report generated: {report_file}")
    else:
        print(f"\n⚠️ Tests had failures. Report: {report_file}")

    return result.returncode


def print_summary():
    """Print test suite summary"""
    print("\n" + "=" * 60)
    print("ECHO BRAIN COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print("\nTest Categories:")
    for category, file in TEST_CATEGORIES.items():
        print(f"  {category:12} -> {file}")

    print("\nUsage Examples:")
    print("  python run_comprehensive_tests.py                    # Run all tests")
    print("  python run_comprehensive_tests.py --smoke            # Quick smoke tests")
    print("  python run_comprehensive_tests.py --unit             # Unit tests only")
    print("  python run_comprehensive_tests.py --category api     # Run API tests")
    print("  python run_comprehensive_tests.py --live             # Test live server")
    print("  python run_comprehensive_tests.py --report           # Generate report")
    print()


def main():
    parser = argparse.ArgumentParser(description="Echo Brain Comprehensive Test Runner")

    parser.add_argument("--category", "-c",
                       choices=list(TEST_CATEGORIES.keys()),
                       help="Run specific test category")
    parser.add_argument("--smoke", action="store_true",
                       help="Run quick smoke tests")
    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests only")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests")
    parser.add_argument("--live", action="store_true",
                       help="Run tests against live server")
    parser.add_argument("--url", default="http://localhost:8309",
                       help="Echo Brain URL for live tests")
    parser.add_argument("--report", action="store_true",
                       help="Generate HTML report")
    parser.add_argument("--summary", action="store_true",
                       help="Print test suite summary")
    parser.add_argument("pytest_args", nargs="*",
                       help="Additional pytest arguments")

    args = parser.parse_args()

    if args.summary:
        print_summary()
        return 0

    # Determine which tests to run
    if args.category:
        return run_category(args.category, args.pytest_args)
    elif args.smoke:
        return run_quick_smoke_tests()
    elif args.unit:
        return run_unit_tests()
    elif args.integration:
        return run_integration_tests()
    elif args.live:
        return run_live_tests(args.url)
    elif args.report:
        return generate_report()
    else:
        return run_all_tests(args.pytest_args)


if __name__ == "__main__":
    sys.exit(main())
