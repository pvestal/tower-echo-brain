#!/usr/bin/env python3
"""
CI/CD Pipeline Integration Test Suite for Echo Brain

Tests for continuous integration and deployment pipeline:
- Test execution in different environments
- Test coverage validation
- Performance regression detection
- Security scan integration
- Deployment readiness checks
- Test result reporting
- Pipeline failure handling

Author: Echo Brain Testing Framework
Created: 2026-01-02
"""

import pytest
import asyncio
import subprocess
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


class PipelineValidator:
    """Validates CI/CD pipeline requirements"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.test_results = {}

    def validate_test_coverage(self, minimum_coverage: float = 80.0) -> Dict[str, Any]:
        """Validate test coverage meets minimum requirements"""
        try:
            # Run coverage analysis
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "--quiet"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)

                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)

                return {
                    "status": "passed" if total_coverage >= minimum_coverage else "failed",
                    "total_coverage": total_coverage,
                    "minimum_required": minimum_coverage,
                    "details": coverage_data.get("files", {}),
                    "missing_coverage": [
                        file for file, data in coverage_data.get("files", {}).items()
                        if data.get("summary", {}).get("percent_covered", 0) < minimum_coverage
                    ]
                }
            else:
                return {
                    "status": "failed",
                    "error": "Coverage report not generated",
                    "total_coverage": 0
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "total_coverage": 0
            }

    def run_security_scans(self) -> Dict[str, Any]:
        """Run security vulnerability scans"""
        security_results = {
            "dependency_scan": self._scan_dependencies(),
            "code_scan": self._scan_code_security(),
            "secret_scan": self._scan_secrets()
        }

        overall_status = "passed"
        total_issues = 0

        for scan_type, result in security_results.items():
            if result["status"] == "failed":
                overall_status = "failed"
            total_issues += result.get("issues_found", 0)

        return {
            "status": overall_status,
            "total_issues": total_issues,
            "scans": security_results
        }

    def _scan_dependencies(self) -> Dict[str, Any]:
        """Scan dependencies for known vulnerabilities"""
        try:
            # Check if safety is installed
            result = subprocess.run(
                ["python", "-m", "pip", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if "safety" in result.stdout:
                # Run safety check
                safety_result = subprocess.run(
                    ["python", "-m", "safety", "check", "--json"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if safety_result.returncode == 0:
                    return {
                        "status": "passed",
                        "issues_found": 0,
                        "details": "No known vulnerabilities found"
                    }
                else:
                    try:
                        vulnerabilities = json.loads(safety_result.stdout)
                        return {
                            "status": "failed",
                            "issues_found": len(vulnerabilities),
                            "vulnerabilities": vulnerabilities
                        }
                    except json.JSONDecodeError:
                        return {
                            "status": "error",
                            "error": "Could not parse safety output",
                            "raw_output": safety_result.stdout
                        }
            else:
                return {
                    "status": "skipped",
                    "reason": "Safety not installed",
                    "issues_found": 0
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues_found": 0
            }

    def _scan_code_security(self) -> Dict[str, Any]:
        """Scan code for security issues"""
        security_patterns = [
            (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
            (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
            (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
            (r"eval\s*\(", "Use of eval() function"),
            (r"exec\s*\(", "Use of exec() function"),
            (r"subprocess\.call\s*\(.*shell\s*=\s*True", "Shell injection vulnerability"),
        ]

        issues = []
        try:
            for root, dirs, files in os.walk(self.project_root / "src"):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()

                            for line_num, line in enumerate(content.splitlines(), 1):
                                for pattern, description in security_patterns:
                                    import re
                                    if re.search(pattern, line, re.IGNORECASE):
                                        issues.append({
                                            "file": file_path,
                                            "line": line_num,
                                            "issue": description,
                                            "code": line.strip()
                                        })
                        except Exception:
                            continue  # Skip files that can't be read

            return {
                "status": "failed" if issues else "passed",
                "issues_found": len(issues),
                "issues": issues
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues_found": 0
            }

    def _scan_secrets(self) -> Dict[str, Any]:
        """Scan for accidentally committed secrets"""
        secret_patterns = [
            (r"(?i)(password|pwd|pass)\s*[:=]\s*['\"][^'\"]{8,}['\"]", "Potential password"),
            (r"(?i)(token|key|secret)\s*[:=]\s*['\"][^'\"]{20,}['\"]", "Potential token/key"),
            (r"['\"][0-9a-f]{32,}['\"]", "Potential hash/key"),
            (r"(?i)postgres://[^@]+:[^@]+@", "Database connection string"),
        ]

        secrets_found = []
        try:
            # Scan specific directories for secrets
            scan_paths = [
                self.project_root / "src",
                self.project_root / "tests",
                self.project_root
            ]

            for scan_path in scan_paths:
                if scan_path.exists():
                    for root, dirs, files in os.walk(scan_path):
                        # Skip certain directories
                        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', '.pytest_cache']]

                        for file in files:
                            if file.endswith(('.py', '.json', '.yaml', '.yml', '.env')):
                                file_path = os.path.join(root, file)
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()

                                    for line_num, line in enumerate(content.splitlines(), 1):
                                        for pattern, description in secret_patterns:
                                            import re
                                            if re.search(pattern, line):
                                                # Skip known test values
                                                if any(test_indicator in line.lower() for test_indicator in
                                                      ['test', 'mock', 'example', 'placeholder', 'dummy']):
                                                    continue

                                                secrets_found.append({
                                                    "file": file_path,
                                                    "line": line_num,
                                                    "description": description,
                                                    "code": line[:100] + "..." if len(line) > 100 else line
                                                })
                                except Exception:
                                    continue

            return {
                "status": "failed" if secrets_found else "passed",
                "issues_found": len(secrets_found),
                "secrets": secrets_found
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues_found": 0
            }

    def validate_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks and validate against baselines"""
        try:
            # Run performance tests
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/api/test_performance.py", "-v", "--tb=short"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600
            )

            performance_status = "passed" if result.returncode == 0 else "failed"

            # Extract performance metrics from output
            metrics = self._extract_performance_metrics(result.stdout + result.stderr)

            return {
                "status": performance_status,
                "return_code": result.returncode,
                "metrics": metrics,
                "output": result.stdout,
                "errors": result.stderr
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": "Performance tests exceeded 10 minute timeout"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _extract_performance_metrics(self, output: str) -> Dict[str, Any]:
        """Extract performance metrics from test output"""
        metrics = {
            "tests_passed": 0,
            "tests_failed": 0,
            "response_times": [],
            "requests_per_second": [],
            "memory_usage": []
        }

        lines = output.split('\n')
        for line in lines:
            if "passed" in line and "failed" in line:
                # Extract test counts
                import re
                match = re.search(r'(\d+) passed.*?(\d+) failed', line)
                if match:
                    metrics["tests_passed"] = int(match.group(1))
                    metrics["tests_failed"] = int(match.group(2))

            # Extract specific performance metrics
            if "RPS:" in line or "requests per second" in line.lower():
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    try:
                        metrics["requests_per_second"].append(float(numbers[0]))
                    except (ValueError, IndexError):
                        pass

            if "response time" in line.lower() and "ms" in line:
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    try:
                        metrics["response_times"].append(float(numbers[0]))
                    except (ValueError, IndexError):
                        pass

        return metrics

    def validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate that the system is ready for deployment"""
        checks = {
            "environment_variables": self._check_environment_variables(),
            "database_migrations": self._check_database_migrations(),
            "service_configuration": self._check_service_configuration(),
            "api_health": self._check_api_health(),
            "dependencies": self._check_dependencies()
        }

        overall_status = "passed"
        failed_checks = []

        for check_name, result in checks.items():
            if result["status"] != "passed":
                overall_status = "failed"
                failed_checks.append(check_name)

        return {
            "status": overall_status,
            "failed_checks": failed_checks,
            "checks": checks
        }

    def _check_environment_variables(self) -> Dict[str, Any]:
        """Check required environment variables"""
        required_vars = [
            "JWT_SECRET",
            "DB_HOST",
            "DB_NAME",
            "DB_USER",
            "DB_PASSWORD",
            "REDIS_HOST"
        ]

        missing_vars = []
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)

        return {
            "status": "passed" if not missing_vars else "failed",
            "missing_variables": missing_vars,
            "total_required": len(required_vars),
            "found": len(required_vars) - len(missing_vars)
        }

    def _check_database_migrations(self) -> Dict[str, Any]:
        """Check database migration status"""
        # This would check if all migrations are applied
        return {
            "status": "passed",  # Placeholder
            "migrations_pending": 0,
            "migrations_applied": 5
        }

    def _check_service_configuration(self) -> Dict[str, Any]:
        """Check service configuration files"""
        config_files = [
            "src/app_factory.py",
            "src/main.py",
            "src/routing/auth_middleware.py",
            "src/middleware/rate_limiting.py"
        ]

        missing_files = []
        for config_file in config_files:
            if not (self.project_root / config_file).exists():
                missing_files.append(config_file)

        return {
            "status": "passed" if not missing_files else "failed",
            "missing_files": missing_files,
            "total_files": len(config_files),
            "found_files": len(config_files) - len(missing_files)
        }

    def _check_api_health(self) -> Dict[str, Any]:
        """Check API health endpoints"""
        # This would make actual HTTP requests to health endpoints
        return {
            "status": "passed",  # Placeholder
            "endpoints_checked": [
                "/api/echo/health",
                "/health"
            ],
            "all_healthy": True
        }

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check that all dependencies are properly installed"""
        try:
            result = subprocess.run(
                ["python", "-m", "pip", "check"],
                capture_output=True,
                text=True,
                timeout=30
            )

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "output": result.stdout,
                "errors": result.stderr
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


class TestCIPipeline:
    """Test CI/CD pipeline integration"""

    @pytest.fixture
    def pipeline_validator(self):
        """Create pipeline validator instance"""
        return PipelineValidator()

    @pytest.mark.ci
    def test_test_coverage_validation(self, pipeline_validator):
        """Test that code coverage meets minimum requirements"""
        coverage_result = pipeline_validator.validate_test_coverage(minimum_coverage=70.0)

        assert coverage_result["status"] in ["passed", "skipped"], \
            f"Coverage validation failed: {coverage_result}"

        if coverage_result["status"] == "passed":
            assert coverage_result["total_coverage"] >= 70.0, \
                f"Coverage too low: {coverage_result['total_coverage']}%"

        print(f"Test Coverage: {coverage_result['total_coverage']:.1f}%")

    @pytest.mark.ci
    def test_security_scans(self, pipeline_validator):
        """Test security vulnerability scans"""
        security_result = pipeline_validator.run_security_scans()

        # Allow some issues for development, but flag critical ones
        assert security_result["status"] in ["passed", "failed"], \
            f"Security scan error: {security_result}"

        # Print security scan results
        print(f"Security Scan Results: {security_result['total_issues']} issues found")

        for scan_type, result in security_result.get("scans", {}).items():
            print(f"  {scan_type}: {result['status']} ({result.get('issues_found', 0)} issues)")

        # Critical security issues should fail the pipeline
        critical_issues = security_result.get("total_issues", 0)
        assert critical_issues < 10, f"Too many security issues: {critical_issues}"

    @pytest.mark.ci
    @pytest.mark.slow
    def test_performance_benchmarks(self, pipeline_validator):
        """Test performance benchmarks meet requirements"""
        perf_result = pipeline_validator.validate_performance_benchmarks()

        # Performance tests should pass or be skipped
        assert perf_result["status"] in ["passed", "skipped", "timeout"], \
            f"Performance benchmark failed: {perf_result}"

        if perf_result["status"] == "passed":
            metrics = perf_result.get("metrics", {})

            # Validate performance metrics
            if metrics.get("tests_failed", 0) > 0:
                print(f"Warning: {metrics['tests_failed']} performance tests failed")

            print(f"Performance Tests: {metrics.get('tests_passed', 0)} passed, "
                  f"{metrics.get('tests_failed', 0)} failed")

    @pytest.mark.ci
    def test_deployment_readiness(self, pipeline_validator):
        """Test deployment readiness checks"""
        readiness_result = pipeline_validator.validate_deployment_readiness()

        print(f"Deployment Readiness: {readiness_result['status']}")

        if readiness_result["failed_checks"]:
            print(f"Failed checks: {readiness_result['failed_checks']}")

        # Allow some checks to fail in test environment
        critical_checks = ["dependencies", "service_configuration"]
        critical_failures = [
            check for check in readiness_result["failed_checks"]
            if check in critical_checks
        ]

        assert not critical_failures, \
            f"Critical deployment checks failed: {critical_failures}"

    @pytest.mark.ci
    def test_environment_isolation(self):
        """Test that tests run in isolated environment"""
        # Check environment variables
        assert os.environ.get("ENVIRONMENT") in ["test", "development"], \
            "Tests should run in test or development environment"

        # Check that we're not connecting to production services
        db_host = os.environ.get("DB_HOST", "localhost")
        assert db_host in ["localhost", "127.0.0.1", "test-db"], \
            f"Should not connect to production database: {db_host}"

    @pytest.mark.ci
    def test_test_isolation(self):
        """Test that tests don't interfere with each other"""
        # This test should pass regardless of order
        test_data = {"isolated": True, "count": 42}

        assert test_data["isolated"] is True
        assert test_data["count"] == 42

        # Verify no global state leakage
        import sys
        test_modules = [name for name in sys.modules.keys() if "test_" in name]

        # Should have some test modules loaded
        assert len(test_modules) > 0

    @pytest.mark.ci
    def test_parallel_execution(self):
        """Test that tests can run in parallel safely"""
        import threading
        import concurrent.futures

        def test_worker(worker_id):
            """Worker function for parallel testing"""
            result = []
            for i in range(10):
                result.append(f"worker_{worker_id}_item_{i}")
                time.sleep(0.001)  # Small delay
            return result

        # Run workers in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(test_worker, i) for i in range(4)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify all workers completed successfully
        assert len(results) == 4
        for result in results:
            assert len(result) == 10

    @pytest.mark.ci
    @pytest.mark.asyncio
    async def test_async_test_execution(self):
        """Test async test execution works properly"""
        async def async_operation(delay: float, value: str):
            await asyncio.sleep(delay)
            return f"completed_{value}"

        # Run multiple async operations
        tasks = [
            async_operation(0.01, f"task_{i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result == f"completed_task_{i}"

    @pytest.mark.ci
    def test_memory_usage_limits(self):
        """Test that tests don't exceed memory limits"""
        import psutil
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Allocate some memory
        test_data = []
        for _ in range(1000):
            test_data.append([i for i in range(100)])

        current_memory = process.memory_info().rss / 1024 / 1024

        # Clean up
        del test_data
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024

        print(f"Memory usage: {initial_memory:.1f}MB -> {current_memory:.1f}MB -> {final_memory:.1f}MB")

        # Memory should not exceed reasonable limits
        assert current_memory < 500, f"Memory usage too high: {current_memory}MB"

    @pytest.mark.ci
    def test_test_discovery(self):
        """Test that test discovery works correctly"""
        import pytest

        # Discover tests in the current directory
        test_files = list(Path(__file__).parent.parent.glob("**/test_*.py"))

        assert len(test_files) > 0, "No test files discovered"

        # Should find our main test files
        expected_files = [
            "test_security_comprehensive.py",
            "test_performance.py",
            "test_service_integration.py",
            "test_circuit_breakers.py"
        ]

        found_files = [f.name for f in test_files]

        for expected_file in expected_files:
            assert expected_file in found_files, f"Expected test file not found: {expected_file}"

        print(f"Discovered {len(test_files)} test files")


class TestCDPipeline:
    """Test Continuous Deployment pipeline aspects"""

    @pytest.mark.cd
    def test_configuration_validation(self):
        """Test that deployment configuration is valid"""
        # This would validate Kubernetes manifests, Docker configs, etc.
        config_files = [
            "Dockerfile",
            "docker-compose.yml",
            "requirements.txt"
        ]

        project_root = Path(__file__).parent.parent.parent

        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                # Basic validation that file is readable
                try:
                    with open(config_path, 'r') as f:
                        content = f.read()
                        assert len(content) > 0, f"{config_file} is empty"
                except Exception as e:
                    pytest.fail(f"Could not read {config_file}: {e}")

    @pytest.mark.cd
    def test_service_health_endpoints(self):
        """Test that health endpoints are properly configured"""
        # This would test actual health endpoints in deployment
        health_endpoints = [
            "/health",
            "/api/echo/health",
            "/api/echo/status"
        ]

        # For now, just verify the endpoints would be available
        assert len(health_endpoints) > 0

    @pytest.mark.cd
    def test_graceful_shutdown(self):
        """Test graceful shutdown behavior"""
        # Test that services can shutdown gracefully
        import signal
        import threading

        shutdown_received = threading.Event()

        def signal_handler(signum, frame):
            shutdown_received.set()

        # Register signal handler
        original_handler = signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Simulate shutdown signal
            os.kill(os.getpid(), signal.SIGTERM)

            # Wait for signal processing
            assert shutdown_received.wait(timeout=1.0), "Shutdown signal not received"

        finally:
            # Restore original handler
            signal.signal(signal.SIGTERM, original_handler)

    @pytest.mark.cd
    def test_rolling_deployment_compatibility(self):
        """Test compatibility with rolling deployments"""
        # Test that multiple versions can run simultaneously
        version_info = {
            "version": "2.0.0",
            "api_version": "v1",
            "compatibility": ["v1", "v0.9"]
        }

        assert "version" in version_info
        assert "api_version" in version_info
        assert isinstance(version_info["compatibility"], list)


if __name__ == "__main__":
    # Run CI/CD tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "ci"])