#!/usr/bin/env python3
"""
Test Framework for Model Decision Engine
Part of CI/CD pipeline - validates decision quality
"""

import os
import asyncio
import json
import time
from typing import Dict, List, Tuple
import psycopg2
import aiohttp

class ModelDecisionTester:
    """Test suite for model decision quality"""

    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.test_cases = self.load_test_cases()

    def load_test_cases(self) -> List[Dict]:
        """Load test cases with expected complexity and model tiers"""
        return [
            # Simple queries - should use tiny/small models
            {
                "query": "What is a variable in Python?",
                "expected_tier": "tiny",
                "max_tier": "small",
                "context": {},
                "description": "Basic question"
            },
            {
                "query": "Write a hello world function",
                "expected_tier": "small",
                "max_tier": "small",
                "context": {},
                "description": "Simple code generation"
            },

            # Medium complexity - should use small/medium models
            {
                "query": "Debug this async function that's causing memory leaks",
                "expected_tier": "medium",
                "max_tier": "medium",
                "context": {"code_lines": 100},
                "description": "Debugging task"
            },
            {
                "query": "Implement a REST API endpoint with authentication",
                "expected_tier": "medium",
                "max_tier": "medium",
                "context": {},
                "description": "API implementation"
            },

            # High complexity - should use large models
            {
                "query": "Refactor this microservice architecture to improve scalability",
                "expected_tier": "large",
                "max_tier": "large",
                "context": {"files": 20},
                "description": "Architecture refactoring"
            },
            {
                "query": "Design a distributed caching system with eventual consistency",
                "expected_tier": "large",
                "max_tier": "cloud",
                "context": {},
                "description": "System design"
            },

            # Should trigger API
            {
                "query": "Analyze this entire codebase and suggest a complete restructuring with modern design patterns",
                "expected_tier": "cloud",
                "max_tier": "cloud",
                "context": {"files": 100, "complexity": "extreme"},
                "description": "Massive refactoring"
            }
        ]

    async def run_tests(self, decision_engine) -> Dict:
        """Run all test cases and evaluate decisions"""
        results = {
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "details": []
        }

        for test in self.test_cases:
            decision = await decision_engine.decide_model(
                test["query"],
                test["context"]
            )

            # Evaluate decision
            tier = decision["tier"].value if hasattr(decision["tier"], 'value') else decision["tier"]

            test_result = {
                "description": test["description"],
                "query": test["query"][:50] + "...",
                "expected": test["expected_tier"],
                "actual": tier,
                "complexity_score": decision["complexity_score"],
                "model": decision["model"]
            }

            # Check if decision is appropriate
            tier_order = ["tiny", "small", "medium", "large", "cloud"]
            expected_idx = tier_order.index(test["expected_tier"])
            max_idx = tier_order.index(test["max_tier"])
            actual_idx = tier_order.index(tier)

            if actual_idx == expected_idx:
                test_result["status"] = "PASSED"
                results["passed"] += 1
            elif actual_idx <= max_idx:
                test_result["status"] = "WARNING"
                test_result["message"] = f"Used {tier} instead of {test['expected_tier']}"
                results["warnings"] += 1
            else:
                test_result["status"] = "FAILED"
                test_result["message"] = f"Over-provisioned: used {tier} for {test['expected_tier']} task"
                results["failed"] += 1

            results["details"].append(test_result)

        # Calculate score
        total = results["passed"] + results["failed"] + results["warnings"]
        results["score"] = (results["passed"] + 0.5 * results["warnings"]) / total * 100

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate test report for CI/CD"""
        report = []
        report.append("=" * 60)
        report.append("MODEL DECISION ENGINE TEST REPORT")
        report.append("=" * 60)
        report.append(f"Score: {results['score']:.1f}%")
        report.append(f"Passed: {results['passed']}")
        report.append(f"Warnings: {results['warnings']}")
        report.append(f"Failed: {results['failed']}")
        report.append("")
        report.append("DETAILS:")
        report.append("-" * 60)

        for detail in results["details"]:
            status_icon = "✅" if detail["status"] == "PASSED" else "⚠️" if detail["status"] == "WARNING" else "❌"
            report.append(f"{status_icon} {detail['description']}")
            report.append(f"   Expected: {detail['expected']} -> Actual: {detail['actual']}")
            report.append(f"   Complexity: {detail['complexity_score']:.1f}, Model: {detail['model']}")
            if "message" in detail:
                report.append(f"   {detail['message']}")
            report.append("")

        return "\n".join(report)

class PerformanceBenchmark:
    """Benchmark model performance for CI/CD"""

    def __init__(self, db_config: Dict):
        self.db_config = db_config

    async def benchmark_models(self) -> Dict:
        """Benchmark response times for each model tier"""

        # Test queries for each tier
        benchmarks = {
            "tinyllama:latest": "What is 2+2?",
            "deepseek-coder:latest": "Write a Python function to sort a list",
            "mistral:7b": "Explain how async/await works",
            "codellama:13b": "Implement a binary search tree",
            "deepseek-coder-v2:16b": "Write a REST API with authentication",
            "qwen2.5-coder:32b": "Design a microservice architecture"
        }

        results = {}

        for model, query in benchmarks.items():
            try:
                start = time.time()

                # Make actual query to model
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": model,
                        "prompt": query,
                        "stream": False,
                        "options": {"num_predict": 100}
                    }

                    async with session.post(
                        "http://localhost:11434/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            elapsed = time.time() - start

                            results[model] = {
                                "response_time": elapsed,
                                "tokens": len(result.get("response", "").split()),
                                "tokens_per_second": len(result.get("response", "").split()) / elapsed,
                                "status": "success"
                            }
                        else:
                            results[model] = {
                                "status": "failed",
                                "error": f"HTTP {response.status}"
                            }

            except Exception as e:
                results[model] = {
                    "status": "error",
                    "error": str(e)
                }

        return results

    def check_thresholds(self, benchmarks: Dict) -> Tuple[bool, List[str]]:
        """Check if performance meets thresholds"""

        # Define acceptable response times (seconds)
        thresholds = {
            "tinyllama:latest": 2.0,
            "deepseek-coder:latest": 3.0,
            "mistral:7b": 5.0,
            "codellama:13b": 10.0,
            "deepseek-coder-v2:16b": 15.0,
            "qwen2.5-coder:32b": 30.0
        }

        issues = []
        all_passed = True

        for model, threshold in thresholds.items():
            if model in benchmarks:
                bench = benchmarks[model]
                if bench["status"] == "success":
                    if bench["response_time"] > threshold:
                        issues.append(f"{model}: {bench['response_time']:.1f}s > {threshold}s threshold")
                        all_passed = False
                else:
                    issues.append(f"{model}: {bench.get('error', 'Failed')}")
                    all_passed = False

        return all_passed, issues

async def run_ci_tests():
    """Main CI/CD test runner"""

    db_config = {
        "host": "localhost",
        "database": "echo_brain",
        "user": os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick"))
    }

    print("Starting Model Decision Engine Tests...")
    print("=" * 60)

    # Import decision engine
    from model_decision_engine import get_decision_engine

    engine = get_decision_engine(db_config)

    # Run decision quality tests
    tester = ModelDecisionTester(db_config)
    test_results = await tester.run_tests(engine)

    print(tester.generate_report(test_results))

    # Run performance benchmarks
    print("\nRunning Performance Benchmarks...")
    print("=" * 60)

    benchmark = PerformanceBenchmark(db_config)
    bench_results = await benchmark.benchmark_models()

    for model, result in bench_results.items():
        if result["status"] == "success":
            print(f"{model}: {result['response_time']:.2f}s ({result['tokens_per_second']:.1f} tok/s)")
        else:
            print(f"{model}: {result.get('error', 'Failed')}")

    # Check thresholds
    passed, issues = benchmark.check_thresholds(bench_results)

    if issues:
        print("\n⚠️  Performance Issues:")
        for issue in issues:
            print(f"  - {issue}")

    # Overall result
    print("\n" + "=" * 60)
    if test_results["score"] >= 80 and passed:
        print("✅ CI/CD TESTS PASSED")
        return 0
    else:
        print("❌ CI/CD TESTS FAILED")
        if test_results["score"] < 80:
            print(f"  Decision quality score: {test_results['score']:.1f}% (minimum 80%)")
        if not passed:
            print(f"  Performance benchmarks failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(run_ci_tests()))