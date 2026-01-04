#!/usr/bin/env python3
"""
Comprehensive Test Runner for Echo Brain API Testing Suite

Runs all test categories with proper configuration:
- Security Tests
- Performance Tests
- Integration Tests
- Resilience Tests
- CI/CD Tests

Features:
- Test category selection
- Parallel execution
- Coverage reporting
- Performance benchmarking
- Results aggregation
- CI/CD integration

Usage:
    python run_comprehensive_tests.py --all
    python run_comprehensive_tests.py --security --performance
    python run_comprehensive_tests.py --ci-mode
    python run_comprehensive_tests.py --benchmark

Author: Echo Brain Testing Framework
Created: 2026-01-02
"""

import argparse
import sys
import os
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import multiprocessing


class TestRunner:
    """Comprehensive test runner for Echo Brain API tests"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = None

    def run_tests(self, categories: List[str], parallel: bool = False,
                  coverage: bool = False, benchmark: bool = False) -> Dict[str, Any]:
        """Run specified test categories"""

        self.start_time = time.time()
        print(f"🚀 Starting Echo Brain Comprehensive Test Suite")
        print(f"📂 Project root: {self.project_root}")
        print(f"🎯 Test categories: {', '.join(categories)}")
        print(f"⚡ Parallel execution: {parallel}")
        print(f"📊 Coverage reporting: {coverage}")
        print(f"🏃 Benchmarking: {benchmark}")
        print("=" * 60)

        # Prepare environment
        self._setup_test_environment()

        # Run each test category
        for category in categories:
            print(f"\n🧪 Running {category} tests...")
            self.test_results[category] = self._run_category(
                category, parallel=parallel, coverage=coverage, benchmark=benchmark
            )

        # Generate comprehensive report
        total_time = time.time() - self.start_time
        report = self._generate_report(total_time)

        return report

    def _setup_test_environment(self):
        """Setup test environment and dependencies"""
        print("🔧 Setting up test environment...")

        # Set environment variables for testing
        os.environ.update({
            'ENVIRONMENT': 'test',
            'JWT_SECRET': 'test_secret_key_for_comprehensive_testing_suite_2026',
            'DB_HOST': 'localhost',
            'DB_NAME': 'test_echo_brain',
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_password',
            'REDIS_HOST': 'localhost',
            'REDIS_PORT': '6379',
            'REDIS_DB': '15'
        })

        # Ensure test dependencies are available
        self._check_dependencies()

    def _check_dependencies(self):
        """Check that required dependencies are installed"""
        required_packages = [
            'pytest',
            'pytest-asyncio',
            'httpx',
            'psutil'
        ]

        print("📦 Checking dependencies...")
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} - attempting to install...")
                self._install_package(package)

    def _install_package(self, package: str):
        """Install missing package"""
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
            print(f"  ✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  Failed to install {package}")

    def _run_category(self, category: str, parallel: bool = False,
                     coverage: bool = False, benchmark: bool = False) -> Dict[str, Any]:
        """Run specific test category"""

        category_configs = {
            'security': {
                'path': 'tests/api/test_security_comprehensive.py',
                'markers': 'security',
                'timeout': 300
            },
            'performance': {
                'path': 'tests/api/test_performance.py',
                'markers': 'performance',
                'timeout': 600
            },
            'integration': {
                'path': 'tests/integration/test_service_integration.py',
                'markers': 'integration',
                'timeout': 400
            },
            'resilience': {
                'path': 'tests/resilience/test_circuit_breakers.py',
                'markers': 'resilience',
                'timeout': 300
            },
            'ci_cd': {
                'path': 'tests/ci_cd/test_pipeline_integration.py',
                'markers': 'ci',
                'timeout': 600
            },
            'all': {
                'path': 'tests/',
                'markers': 'not slow',
                'timeout': 900
            }
        }

        config = category_configs.get(category)
        if not config:
            return {'status': 'error', 'message': f'Unknown category: {category}'}

        # Build pytest command
        cmd = [sys.executable, '-m', 'pytest']

        # Add test path
        test_path = self.project_root / config['path']
        cmd.append(str(test_path))

        # Add markers
        if config['markers']:
            cmd.extend(['-m', config['markers']])

        # Add parallel execution
        if parallel:
            workers = min(multiprocessing.cpu_count(), 4)
            cmd.extend(['-n', str(workers)])

        # Add coverage
        if coverage:
            cmd.extend([
                '--cov=src',
                '--cov-report=json',
                '--cov-report=html',
                '--cov-report=term'
            ])

        # Add verbose output
        cmd.extend(['-v', '--tb=short'])

        # Add JSON report
        json_report = self.project_root / f"test_report_{category}.json"
        cmd.extend(['--json-report', f'--json-report-file={json_report}'])

        print(f"  📝 Command: {' '.join(cmd)}")

        # Run tests
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=config['timeout']
            )

            execution_time = time.time() - start_time

            # Parse results
            test_result = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'return_code': result.returncode,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

            # Try to load JSON report for detailed metrics
            if json_report.exists():
                try:
                    with open(json_report) as f:
                        json_data = json.load(f)
                    test_result['detailed_results'] = json_data

                    # Extract summary statistics
                    summary = json_data.get('summary', {})
                    test_result['tests_collected'] = summary.get('total', 0)
                    test_result['tests_passed'] = summary.get('passed', 0)
                    test_result['tests_failed'] = summary.get('failed', 0)
                    test_result['tests_skipped'] = summary.get('skipped', 0)

                except Exception as e:
                    print(f"  ⚠️  Could not parse JSON report: {e}")

            # Display results
            status_icon = "✅" if test_result['status'] == 'passed' else "❌"
            print(f"  {status_icon} {category} tests: {test_result['status']} "
                  f"({execution_time:.1f}s)")

            if 'tests_passed' in test_result:
                print(f"     📊 {test_result['tests_passed']} passed, "
                      f"{test_result['tests_failed']} failed, "
                      f"{test_result['tests_skipped']} skipped")

            return test_result

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"  ⏰ {category} tests timed out after {execution_time:.1f}s")
            return {
                'status': 'timeout',
                'execution_time': execution_time,
                'timeout_limit': config['timeout']
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  💥 {category} tests failed with error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': execution_time
            }

    def _generate_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""

        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 60)

        # Overall statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        categories_passed = 0
        categories_failed = 0

        for category, results in self.test_results.items():
            if results['status'] == 'passed':
                categories_passed += 1
            elif results['status'] in ['failed', 'error', 'timeout']:
                categories_failed += 1

            # Aggregate test counts
            total_tests += results.get('tests_collected', 0)
            total_passed += results.get('tests_passed', 0)
            total_failed += results.get('tests_failed', 0)
            total_skipped += results.get('tests_skipped', 0)

        # Calculate success rates
        overall_success_rate = (categories_passed / len(self.test_results) * 100) if self.test_results else 0
        test_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        print(f"⏱️  Total execution time: {total_time:.1f}s")
        print(f"📂 Categories tested: {len(self.test_results)}")
        print(f"✅ Categories passed: {categories_passed}")
        print(f"❌ Categories failed: {categories_failed}")
        print(f"📊 Overall success rate: {overall_success_rate:.1f}%")
        print(f"🧪 Total individual tests: {total_tests}")
        print(f"✅ Tests passed: {total_passed}")
        print(f"❌ Tests failed: {total_failed}")
        print(f"⏭️  Tests skipped: {total_skipped}")
        print(f"📈 Test success rate: {test_success_rate:.1f}%")

        # Category breakdown
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for category, results in self.test_results.items():
            status_icon = {
                'passed': '✅',
                'failed': '❌',
                'error': '💥',
                'timeout': '⏰'
            }.get(results['status'], '❓')

            execution_time = results.get('execution_time', 0)
            test_count = results.get('tests_collected', 0)

            print(f"  {status_icon} {category}: {results['status']} "
                  f"({execution_time:.1f}s, {test_count} tests)")

        # Performance insights
        print(f"\n⚡ PERFORMANCE INSIGHTS:")
        fastest_category = min(self.test_results.items(),
                              key=lambda x: x[1].get('execution_time', float('inf')))
        slowest_category = max(self.test_results.items(),
                              key=lambda x: x[1].get('execution_time', 0))

        print(f"  🚀 Fastest category: {fastest_category[0]} "
              f"({fastest_category[1].get('execution_time', 0):.1f}s)")
        print(f"  🐌 Slowest category: {slowest_category[0]} "
              f"({slowest_category[1].get('execution_time', 0):.1f}s)")

        # Generate recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if categories_failed > 0:
            print(f"  🔧 Fix {categories_failed} failing test categories")
        if test_success_rate < 95:
            print(f"  📈 Improve test success rate (current: {test_success_rate:.1f}%)")
        if total_time > 600:  # 10 minutes
            print(f"  ⚡ Optimize test execution time (current: {total_time:.1f}s)")

        # Create comprehensive report object
        report = {
            'timestamp': time.time(),
            'total_execution_time': total_time,
            'categories': self.test_results,
            'summary': {
                'categories_tested': len(self.test_results),
                'categories_passed': categories_passed,
                'categories_failed': categories_failed,
                'overall_success_rate': overall_success_rate,
                'total_tests': total_tests,
                'tests_passed': total_passed,
                'tests_failed': total_failed,
                'tests_skipped': total_skipped,
                'test_success_rate': test_success_rate
            },
            'performance': {
                'fastest_category': fastest_category[0],
                'fastest_time': fastest_category[1].get('execution_time', 0),
                'slowest_category': slowest_category[0],
                'slowest_time': slowest_category[1].get('execution_time', 0)
            }
        }

        # Save report to file
        report_file = self.project_root / "comprehensive_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Full report saved to: {report_file}")
        print("=" * 60)

        return report


def main():
    """Main entry point for test runner"""

    parser = argparse.ArgumentParser(
        description="Echo Brain Comprehensive Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comprehensive_tests.py --all
  python run_comprehensive_tests.py --security --performance
  python run_comprehensive_tests.py --integration --resilience
  python run_comprehensive_tests.py --ci-mode
  python run_comprehensive_tests.py --benchmark --coverage
        """
    )

    # Test category selection
    parser.add_argument('--all', action='store_true',
                       help='Run all test categories')
    parser.add_argument('--security', action='store_true',
                       help='Run security tests')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--resilience', action='store_true',
                       help='Run resilience tests')
    parser.add_argument('--ci-cd', action='store_true',
                       help='Run CI/CD tests')

    # Execution options
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage reports')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')

    # Special modes
    parser.add_argument('--ci-mode', action='store_true',
                       help='Run in CI mode (security + integration + ci-cd)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test suite (security + integration)')

    args = parser.parse_args()

    # Determine test categories to run
    categories = []

    if args.all:
        categories = ['security', 'performance', 'integration', 'resilience', 'ci_cd']
    elif args.ci_mode:
        categories = ['security', 'integration', 'ci_cd']
    elif args.quick:
        categories = ['security', 'integration']
    else:
        if args.security:
            categories.append('security')
        if args.performance:
            categories.append('performance')
        if args.integration:
            categories.append('integration')
        if args.resilience:
            categories.append('resilience')
        if args.ci_cd:
            categories.append('ci_cd')

    if not categories:
        print("❌ No test categories specified. Use --all or specify individual categories.")
        parser.print_help()
        sys.exit(1)

    # Run tests
    runner = TestRunner()
    report = runner.run_tests(
        categories=categories,
        parallel=args.parallel,
        coverage=args.coverage,
        benchmark=args.benchmark
    )

    # Exit with appropriate code
    if report['summary']['categories_failed'] > 0:
        print(f"\n❌ Test suite failed ({report['summary']['categories_failed']} categories failed)")
        sys.exit(1)
    else:
        print(f"\n✅ Test suite passed (all {report['summary']['categories_passed']} categories passed)")
        sys.exit(0)


if __name__ == "__main__":
    main()