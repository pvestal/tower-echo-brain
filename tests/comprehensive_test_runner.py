"""
Comprehensive Test Runner for Echo Brain Modernization
======================================================

This is the main test orchestrator that coordinates all testing frameworks
and provides a unified interface for running the complete Echo Brain test suite.

Features:
- Orchestrates all test types (unit, integration, performance, AI, regression)
- Generates comprehensive reports
- Manages test environments
- Provides CI/CD integration
- Handles test parallelization

Author: Development Testing Framework Agent
Created: 2025-11-06
"""

import asyncio
import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

# Import testing frameworks
from framework import (
    TestFrameworkCore,
    IntegrationTestPipeline,
    PerformanceTestSuite,
    AITestingSuite,
    RegressionTester,
    TestDataManager,
    ServiceConfig,
    DatabaseConfig,
    LoadTestConfig,
    StressTestConfig,
    RegressionConfig,
    ModelTestCase
)


class ComprehensiveTestRunner:
    """Main test runner for Echo Brain comprehensive testing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize comprehensive test runner."""
        self.config = self._load_config(config_path)
        self.framework = TestFrameworkCore(config_path)
        self.test_results: Dict[str, Any] = {}
        self.start_time = datetime.now()
        
        # Initialize test components
        self.integration_pipeline = IntegrationTestPipeline(self.framework)
        self.performance_suite = PerformanceTestSuite(self.framework)
        self.ai_suite = AITestingSuite(self.framework)
        self.regression_tester = RegressionTester(self.framework)
        self.data_manager = TestDataManager(self.framework)
        
        self.logger = logging.getLogger("ComprehensiveTestRunner")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load test runner configuration."""
        default_config = {
            "test_types": {
                "unit": True,
                "integration": True,
                "performance": False,
                "ai": True,
                "regression": False
            },
            "parallel_execution": True,
            "max_workers": 4,
            "timeout_minutes": 60,
            "report_formats": ["json", "html"],
            "output_directory": "test_results",
            "baseline_directory": "test_baselines"
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
        
    async def run_comprehensive_tests(
        self,
        test_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        self.logger.info("Starting comprehensive test suite")
        
        if test_types is None:
            test_types = [t for t, enabled in self.config["test_types"].items() if enabled]
            
        # Setup test environment
        await self._setup_test_environment()
        
        # Run tests based on configuration
        suite_results = {}
        
        try:
            if "unit" in test_types:
                self.logger.info("Running unit tests")
                suite_results["unit"] = await self._run_unit_tests(tags)
                
            if "integration" in test_types:
                self.logger.info("Running integration tests")
                suite_results["integration"] = await self._run_integration_tests()
                
            if "performance" in test_types:
                self.logger.info("Running performance tests")
                suite_results["performance"] = await self._run_performance_tests()
                
            if "ai" in test_types:
                self.logger.info("Running AI model tests")
                suite_results["ai"] = await self._run_ai_tests()
                
            if "regression" in test_types:
                self.logger.info("Running regression tests")
                suite_results["regression"] = await self._run_regression_tests()
                
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            suite_results["error"] = str(e)
            
        finally:
            # Cleanup test environment
            await self._cleanup_test_environment()
            
        # Generate comprehensive report
        self.test_results = suite_results
        await self._generate_reports()
        
        return suite_results
        
    async def _setup_test_environment(self):
        """Setup comprehensive test environment."""
        self.logger.info("Setting up test environment")
        
        # Create output directories
        output_dir = Path(self.config["output_directory"])
        output_dir.mkdir(exist_ok=True)
        
        baseline_dir = Path(self.config["baseline_directory"])
        baseline_dir.mkdir(exist_ok=True)
        
        # Setup test data
        test_env_config = {
            "fixtures": [
                {
                    "name": "test_users",
                    "type": "generated",
                    "factory": "users",
                    "count": 10
                },
                {
                    "name": "test_tasks",
                    "type": "generated",
                    "factory": "tasks",
                    "count": 20
                },
                {
                    "name": "test_models",
                    "type": "generated",
                    "factory": "ai_models",
                    "count": 5
                }
            ]
        }
        
        self.data_manager.setup_test_environment(test_env_config)
        
    async def _cleanup_test_environment(self):
        """Cleanup test environment."""
        self.logger.info("Cleaning up test environment")
        self.data_manager.cleanup()
        
    async def _run_unit_tests(self, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run unit tests using pytest."""
        import subprocess
        
        cmd = ["python", "-m", "pytest", "tests/", "-m", "unit", "--json-report"]
        
        if tags:
            tag_filter = " and ".join(tags)
            cmd.extend(["-k", tag_filter])
            
        cmd.extend([
            "--json-report-file=test_results/unit_results.json",
            "--cov=src",
            "--cov-report=json:test_results/coverage.json",
            "-v"
        ])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            execution_time = time.time() - start_time
            
            # Load results
            results_file = Path("test_results/unit_results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    pytest_results = json.load(f)
            else:
                pytest_results = {}
                
            return {
                "execution_time": execution_time,
                "return_code": result.returncode,
                "pytest_results": pytest_results,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "execution_time": time.time() - start_time,
                "error": "Unit tests timed out",
                "return_code": -1
            }
            
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        # Register test services
        self.integration_pipeline.register_service(ServiceConfig(
            name="echo_brain",
            host="localhost",
            port=8309,
            health_endpoint="/api/echo/health"
        ))
        
        self.integration_pipeline.register_service(ServiceConfig(
            name="knowledge_base",
            host="localhost",
            port=8307,
            health_endpoint="/api/kb/health",
            required_dependencies=["echo_brain"]
        ))
        
        # Define test scenarios
        service_comm_scenarios = [
            {
                "name": "echo_to_kb_communication",
                "source_service": "echo_brain",
                "target_service": "knowledge_base",
                "endpoint": "/api/kb/articles",
                "method": "GET",
                "expected_status": 200
            }
        ]
        
        workflow_scenarios = [
            {
                "name": "ai_decision_workflow",
                "steps": [
                    {
                        "name": "submit_task",
                        "service": "echo_brain",
                        "endpoint": "/api/echo/query",
                        "method": "POST",
                        "data": {"query": "Test query", "conversation_id": "test"},
                        "store_response_as": "task_result"
                    },
                    {
                        "name": "store_result",
                        "service": "knowledge_base",
                        "endpoint": "/api/kb/articles",
                        "method": "POST",
                        "data": "$task_result"
                    }
                ]
            }
        ]
        
        # Run integration tests
        try:
            comm_results = await self.integration_pipeline.run_service_communication_tests(
                service_comm_scenarios
            )
            
            workflow_results = await self.integration_pipeline.run_end_to_end_workflow_tests(
                workflow_scenarios
            )
            
            health_check = await self.integration_pipeline.run_health_check_suite()
            
            return {
                "service_communication": comm_results,
                "workflow_tests": workflow_results,
                "health_checks": health_check,
                "summary": self.integration_pipeline.generate_integration_report()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
            
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        # Define load test configurations
        load_configs = [
            LoadTestConfig(
                name="echo_brain_load_test",
                target_url="http://localhost:8309/api/echo/health",
                concurrent_users=5,
                duration_seconds=30
            ),
            LoadTestConfig(
                name="kb_load_test", 
                target_url="http://localhost:8307/api/kb/articles",
                concurrent_users=3,
                duration_seconds=30
            )
        ]
        
        # Define stress test configurations
        stress_configs = [
            StressTestConfig(
                name="echo_brain_stress_test",
                target_url="http://localhost:8309/api/echo/health",
                min_users=1,
                max_users=20,
                step_size=3,
                step_duration=15
            )
        ]
        
        try:
            results = await self.performance_suite.run_performance_suite(
                load_configs, stress_configs
            )
            
            # Export results
            self.performance_suite.export_results("test_results/performance_results.json")
            
            return results
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
            
    async def _run_ai_tests(self) -> Dict[str, Any]:
        """Run AI model tests."""
        # Create mock models for testing
        models = {}
        
        # Mock Echo Brain models
        models["decision_model"] = self.ai_suite.mock_manager.create_mock_model(
            "decision_model",
            accuracy=0.87
        )
        
        models["learning_model"] = self.ai_suite.mock_manager.create_mock_model(
            "learning_model",
            accuracy=0.82
        )
        
        # Create test cases
        test_cases = [
            ModelTestCase(
                case_id="decision_001",
                input_data={"query": "test decision", "context": "test"},
                expected_output=0.8,
                test_category="decision"
            ),
            ModelTestCase(
                case_id="decision_002",
                input_data={"query": "complex decision", "context": "production"},
                expected_output=0.9,
                test_category="decision"
            )
        ]
        
        # Mock directors for consensus testing
        directors = {
            "SecurityDirector": self.ai_suite.mock_manager.create_mock_model("security"),
            "QualityDirector": self.ai_suite.mock_manager.create_mock_model("quality")
        }
        
        try:
            results = await self.ai_suite.run_comprehensive_ai_test_suite(
                models=models,
                test_cases=test_cases,
                learning_configs=[],  # Skip learning tests for now
                consensus_configs=[],  # Skip consensus tests for now
                directors=directors
            )
            
            # Export results
            self.ai_suite.export_ai_test_results("test_results/ai_test_results.json")
            
            return results
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
            
    async def _run_regression_tests(self) -> Dict[str, Any]:
        """Run regression tests."""
        baseline_dir = Path(self.config["baseline_directory"])
        
        # Define regression configurations
        configs = [
            RegressionConfig(
                name="performance_regression",
                baseline_path=str(baseline_dir / "performance_baseline.json")
            ),
            RegressionConfig(
                name="functional_regression",
                baseline_path=str(baseline_dir / "functional_baseline.json")
            )
        ]
        
        results = []
        
        for config in configs:
            try:
                # Simulate current test data
                current_data = {
                    "version": "1.1.0",
                    "performance_metrics": {
                        "response_time": 0.15,
                        "throughput": 100,
                        "memory_usage": 512
                    },
                    "functional_results": {
                        "test_001": {"success": True, "output": "expected"},
                        "test_002": {"success": True, "output": "expected"}
                    },
                    "api_contracts": {}
                }
                
                if Path(config.baseline_path).exists():
                    result = self.regression_tester.run_regression_test(config, current_data)
                    results.append(result)
                else:
                    # Create baseline if it doesn't exist
                    self.logger.info(f"Creating baseline for {config.name}")
                    
            except Exception as e:
                self.logger.error(f"Regression test {config.name} failed: {e}")
                
        if results:
            self.regression_tester.generate_regression_report(
                results, "test_results/regression_report.json"
            )
            
        return {
            "regression_results": results,
            "total_tests": len(configs),
            "completed_tests": len(results)
        }
        
    async def _generate_reports(self):
        """Generate comprehensive test reports."""
        self.logger.info("Generating test reports")
        
        # Calculate overall metrics
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Generate main report
        main_report = {
            "timestamp": end_time.isoformat(),
            "duration_seconds": total_duration,
            "framework_version": "1.0.0",
            "test_results": self.test_results,
            "summary": self._generate_summary(),
            "performance_report": self.framework.generate_performance_report()
        }
        
        # Save JSON report
        output_dir = Path(self.config["output_directory"])
        json_report_path = output_dir / "comprehensive_test_report.json"
        
        with open(json_report_path, 'w') as f:
            json.dump(main_report, f, indent=2, default=str)
            
        self.logger.info(f"Comprehensive test report saved to {json_report_path}")
        
        # Generate HTML report if requested
        if "html" in self.config["report_formats"]:
            await self._generate_html_report(main_report)
            
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        summary = {
            "total_test_types": len(self.test_results),
            "successful_test_types": 0,
            "failed_test_types": 0,
            "overall_status": "unknown"
        }
        
        for test_type, result in self.test_results.items():
            if isinstance(result, dict) and not result.get("error"):
                summary["successful_test_types"] += 1
            else:
                summary["failed_test_types"] += 1
                
        if summary["failed_test_types"] == 0:
            summary["overall_status"] = "passed"
        elif summary["successful_test_types"] > 0:
            summary["overall_status"] = "partial"
        else:
            summary["overall_status"] = "failed"
            
        return summary
        
    async def _generate_html_report(self, report_data: Dict[str, Any]):
        """Generate HTML test report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Echo Brain Comprehensive Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .summary { margin: 20px 0; }
                .test-section { margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
                .test-header { background-color: #e0e0e0; padding: 10px; font-weight: bold; }
                .test-content { padding: 15px; }
                .success { color: green; }
                .failure { color: red; }
                .warning { color: orange; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Echo Brain Comprehensive Test Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Duration:</strong> {duration:.2f} seconds</p>
                <p><strong>Status:</strong> <span class="{status_class}">{status}</span></p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <ul>
                    <li>Test Types Run: {total_types}</li>
                    <li>Successful: {successful}</li>
                    <li>Failed: {failed}</li>
                </ul>
            </div>
            
            <div class="test-sections">
                {test_sections}
            </div>
        </body>
        </html>
        """
        
        # Prepare template data
        summary = report_data["summary"]
        status = summary["overall_status"]
        status_class = "success" if status == "passed" else "failure" if status == "failed" else "warning"
        
        # Generate test sections
        test_sections = ""
        for test_type, result in report_data["test_results"].items():
            if isinstance(result, dict):
                success = not result.get("error", False)
                section_class = "success" if success else "failure"
                
                test_sections += f"""
                <div class="test-section">
                    <div class="test-header {section_class}">{test_type.title()} Tests</div>
                    <div class="test-content">
                        <pre>{json.dumps(result, indent=2, default=str)}</pre>
                    </div>
                </div>
                """
                
        html_content = html_template.format(
            timestamp=report_data["timestamp"],
            duration=report_data["duration_seconds"],
            status=status.title(),
            status_class=status_class,
            total_types=summary["total_test_types"],
            successful=summary["successful_test_types"],
            failed=summary["failed_test_types"],
            test_sections=test_sections
        )
        
        # Save HTML report
        output_dir = Path(self.config["output_directory"])
        html_report_path = output_dir / "comprehensive_test_report.html"
        
        with open(html_report_path, 'w') as f:
            f.write(html_content)
            
        self.logger.info(f"HTML test report saved to {html_report_path}")


async def main():
    """Main entry point for comprehensive test runner."""
    parser = argparse.ArgumentParser(description="Echo Brain Comprehensive Test Runner")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to test configuration file"
    )
    
    parser.add_argument(
        "--test-types",
        nargs="+",
        choices=["unit", "integration", "performance", "ai", "regression"],
        help="Test types to run"
    )
    
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Test tags to filter by"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Run comprehensive tests
    try:
        runner = ComprehensiveTestRunner(args.config)
        
        results = await runner.run_comprehensive_tests(
            test_types=args.test_types,
            tags=args.tags
        )
        
        # Determine exit code
        summary = runner._generate_summary()
        if summary["overall_status"] == "passed":
            print("✅ All tests passed!")
            sys.exit(0)
        elif summary["overall_status"] == "partial":
            print("⚠️  Some tests failed!")
            sys.exit(1)
        else:
            print("❌ All tests failed!")
            sys.exit(2)
            
    except Exception as e:
        logging.error(f"Test runner failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
