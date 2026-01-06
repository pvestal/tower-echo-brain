#!/usr/bin/env python3
"""
Echo Brain Comprehensive Test Suite
Tests all major components and features
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any

class EchoBrainTester:
    def __init__(self):
        self.base_url = "http://localhost:8309"
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "passed": 0,
            "failed": 0,
            "tests": []
        }

    def run_test(self, name: str, test_func):
        """Run a single test and record results"""
        print(f"\n🧪 Testing: {name}...")
        start_time = time.time()

        try:
            result = test_func()
            duration = time.time() - start_time

            if result.get("success", False):
                print(f"  ✅ PASSED ({duration:.2f}s)")
                self.test_results["passed"] += 1
                status = "passed"
            else:
                print(f"  ❌ FAILED: {result.get('error', 'Unknown error')}")
                self.test_results["failed"] += 1
                status = "failed"

        except Exception as e:
            duration = time.time() - start_time
            print(f"  ❌ ERROR: {str(e)}")
            self.test_results["failed"] += 1
            status = "error"
            result = {"error": str(e)}

        self.test_results["tests"].append({
            "name": name,
            "status": status,
            "duration": duration,
            "details": result
        })

        return result

    def test_health_check(self):
        """Test health endpoint"""
        response = requests.get(f"{self.base_url}/api/echo/health", timeout=5)
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response": response.json()
        }

    def test_basic_query(self):
        """Test basic query processing"""
        payload = {
            "query": "What is the capital of France?",
            "conversation_id": "test_basic"
        }
        response = requests.post(
            f"{self.base_url}/api/echo/query",
            json=payload,
            timeout=30
        )
        return {
            "success": response.status_code == 200 and "response" in response.json(),
            "status_code": response.status_code,
            "response_length": len(response.json().get("response", ""))
        }

    def test_model_listing(self):
        """Test model listing endpoint"""
        response = requests.get(f"{self.base_url}/api/echo/models/list", timeout=10)
        data = response.json()
        return {
            "success": response.status_code == 200 and "models" in data,
            "model_count": len(data.get("models", [])) if "models" in data else 0,
            "status_code": response.status_code
        }

    def test_conversation_history(self):
        """Test conversation history retrieval"""
        response = requests.get(f"{self.base_url}/api/echo/conversations", timeout=10)
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "has_conversations": len(response.json()) > 0 if response.status_code == 200 else False
        }

    def test_complex_query(self):
        """Test complex query with context"""
        payload = {
            "query": "Analyze the temporal implications of quantum entanglement on consciousness emergence patterns in distributed AI systems",
            "conversation_id": "test_complex",
            "context": "Consider both epistemological and ontological perspectives"
        }
        response = requests.post(
            f"{self.base_url}/api/echo/query",
            json=payload,
            timeout=60
        )
        data = response.json() if response.status_code == 200 else {}
        return {
            "success": response.status_code == 200 and len(data.get("response", "")) > 100,
            "model_used": data.get("model_used", "unknown"),
            "intelligence_level": data.get("intelligence_level", "unknown"),
            "processing_time": data.get("processing_time", 0)
        }

    def test_memory_search(self):
        """Test semantic memory search"""
        response = requests.get(
            f"{self.base_url}/api/echo/memory/search?query=tower",
            timeout=10
        )
        return {
            "success": response.status_code in [200, 404],  # 404 is ok if no memories
            "status_code": response.status_code,
            "memories_found": len(response.json()) if response.status_code == 200 else 0
        }

    def test_brain_visualization(self):
        """Test brain activity endpoint"""
        response = requests.get(f"{self.base_url}/api/echo/brain", timeout=10)
        return {
            "success": response.status_code == 200,
            "has_neural_data": "neural_activity" in response.json() if response.status_code == 200 else False
        }

    def test_task_creation(self):
        """Test task creation endpoint"""
        payload = {
            "task_type": "test_task",
            "task_data": {"action": "verify_system"},
            "priority": 1
        }
        response = requests.post(
            f"{self.base_url}/api/echo/tasks",
            json=payload,
            timeout=10
        )
        return {
            "success": response.status_code in [200, 201],
            "status_code": response.status_code
        }

    def test_code_execution(self):
        """Test code execution endpoint"""
        payload = {
            "code": "print('Hello from Echo Brain')",
            "language": "python"
        }
        response = requests.post(
            f"{self.base_url}/api/echo/code",
            json=payload,
            timeout=10
        )
        return {
            "success": response.status_code == 200,
            "output_received": "output" in response.json() if response.status_code == 200 else False
        }

    def test_multimedia_status(self):
        """Test multimedia services status"""
        response = requests.get(
            f"{self.base_url}/api/echo/multimedia/services/status",
            timeout=10
        )
        return {
            "success": response.status_code == 200,
            "services_available": response.json() if response.status_code == 200 else {}
        }

    def test_user_preferences(self):
        """Test user preferences endpoint"""
        response = requests.get(
            f"{self.base_url}/api/echo/users/patrick",
            timeout=10
        )
        return {
            "success": response.status_code in [200, 404],  # 404 is ok if user not found
            "status_code": response.status_code
        }

    def test_parallel_queries(self):
        """Test system under parallel load"""
        import concurrent.futures

        def single_query(i):
            payload = {
                "query": f"Test query {i}: What is {i} + {i}?",
                "conversation_id": f"test_parallel_{i}"
            }
            try:
                response = requests.post(
                    f"{self.base_url}/api/echo/query",
                    json=payload,
                    timeout=30
                )
                return response.status_code == 200
            except:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(single_query, i) for i in range(5)]
            results = [f.result() for f in futures]

        return {
            "success": all(results),
            "successful_queries": sum(results),
            "total_queries": len(results)
        }

    def generate_report(self) -> str:
        """Generate test report"""
        report = []
        report.append("=" * 80)
        report.append("ECHO BRAIN COMPREHENSIVE TEST REPORT")
        report.append(f"Timestamp: {self.test_results['timestamp']}")
        report.append("=" * 80)

        # Summary
        total = self.test_results["passed"] + self.test_results["failed"]
        pass_rate = (self.test_results["passed"] / total * 100) if total > 0 else 0

        report.append(f"\n📊 SUMMARY")
        report.append(f"Total Tests: {total}")
        report.append(f"✅ Passed: {self.test_results['passed']}")
        report.append(f"❌ Failed: {self.test_results['failed']}")
        report.append(f"Pass Rate: {pass_rate:.1f}%")

        # Individual test results
        report.append(f"\n📋 TEST RESULTS")
        report.append("-" * 40)

        for test in self.test_results["tests"]:
            icon = "✅" if test["status"] == "passed" else "❌"
            report.append(f"{icon} {test['name']}: {test['status']} ({test['duration']:.2f}s)")

            if test["status"] != "passed" and test["details"].get("error"):
                report.append(f"   Error: {test['details']['error']}")

        # Critical issues
        critical_failures = [
            t for t in self.test_results["tests"]
            if t["status"] != "passed" and t["name"] in [
                "Health Check", "Basic Query", "Model Listing"
            ]
        ]

        if critical_failures:
            report.append(f"\n🚨 CRITICAL ISSUES")
            report.append("-" * 40)
            for failure in critical_failures:
                report.append(f"- {failure['name']}: {failure['details'].get('error', 'Failed')}")

        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS")
        report.append("-" * 40)

        if pass_rate < 50:
            report.append("⚠️ System has major issues - immediate attention required")
        elif pass_rate < 80:
            report.append("⚠️ System has some issues - review failed tests")
        else:
            report.append("✅ System is healthy - minor issues only")

        report.append("\n" + "=" * 80)
        return "\n".join(report)

    def run_all_tests(self):
        """Run all tests"""
        print("🏥 Starting Echo Brain Comprehensive Test Suite...")

        # Core functionality tests
        self.run_test("Health Check", self.test_health_check)
        self.run_test("Basic Query", self.test_basic_query)
        self.run_test("Model Listing", self.test_model_listing)
        self.run_test("Conversation History", self.test_conversation_history)

        # Advanced features
        self.run_test("Complex Query", self.test_complex_query)
        self.run_test("Memory Search", self.test_memory_search)
        self.run_test("Brain Visualization", self.test_brain_visualization)

        # Task and execution
        self.run_test("Task Creation", self.test_task_creation)
        self.run_test("Code Execution", self.test_code_execution)

        # Integration tests
        self.run_test("Multimedia Status", self.test_multimedia_status)
        self.run_test("User Preferences", self.test_user_preferences)

        # Load testing
        self.run_test("Parallel Queries", self.test_parallel_queries)

        # Generate report
        report = self.generate_report()
        print(report)

        # Save results
        with open("/opt/tower-echo-brain/test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)

        with open("/opt/tower-echo-brain/test_report.txt", "w") as f:
            f.write(report)

        print(f"\n📄 Results saved to:")
        print(f"  - /opt/tower-echo-brain/test_results.json")
        print(f"  - /opt/tower-echo-brain/test_report.txt")

        return self.test_results

if __name__ == "__main__":
    tester = EchoBrainTester()
    tester.run_all_tests()