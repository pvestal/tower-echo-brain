#!/usr/bin/env python3
"""
Final comprehensive test report for Echo Brain agents
Provides REAL quantitative and qualitative metrics
"""

import requests
import time
import json
import statistics
import psycopg2
from datetime import datetime
from typing import Dict, List, Tuple

BASE_URL = "http://localhost:8309"

class TestMetrics:
    def __init__(self):
        self.results = {
            "coding": {"times": [], "success": [], "accuracy": []},
            "reasoning": {"times": [], "success": [], "quality": []},
            "narration": {"times": [], "success": [], "has_comfyui": []}
        }

    def test_coding_agent(self) -> Dict:
        """Test CodingAgent with multiple tasks"""
        print("\n" + "="*60)
        print("CODING AGENT TESTS")
        print("="*60)

        test_cases = [
            {
                "task": "Write a function that returns the string 'hello'",
                "expected": "return 'hello'",
                "name": "Simple return"
            },
            {
                "task": "Write a function to check if tower-echo-brain service is running",
                "expected": "systemctl",
                "name": "Tower awareness"
            },
            {
                "task": "Create a function to query the past_solutions table",
                "expected": "past_solutions",
                "name": "Database awareness"
            }
        ]

        for test in test_cases:
            start = time.time()
            try:
                response = requests.post(
                    f"{BASE_URL}/api/echo/agents/coding",
                    json={
                        "task": test["task"],
                        "language": "python",
                        "validate": True
                    },
                    timeout=30
                )
                elapsed = time.time() - start

                if response.status_code == 200:
                    result = response.json()
                    code = result.get("code", "")
                    valid = result.get("validation", {}).get("valid", False)

                    self.results["coding"]["times"].append(elapsed)
                    self.results["coding"]["success"].append(valid)
                    self.results["coding"]["accuracy"].append(test["expected"] in code)

                    print(f"✓ {test['name']}: {elapsed:.2f}s - Valid: {valid} - Accurate: {test['expected'] in code}")
                else:
                    print(f"✗ {test['name']}: Failed with status {response.status_code}")
                    self.results["coding"]["success"].append(False)

            except Exception as e:
                print(f"✗ {test['name']}: Error - {e}")
                self.results["coding"]["success"].append(False)

        return self._summarize("CodingAgent", self.results["coding"])

    def test_reasoning_agent(self) -> Dict:
        """Test ReasoningAgent with analysis tasks"""
        print("\n" + "="*60)
        print("REASONING AGENT TESTS")
        print("="*60)

        test_cases = [
            {
                "task": "Should Tower use Docker or systemd?",
                "background": "Ubuntu server with nginx",
                "name": "Architecture decision"
            },
            {
                "task": "Analyze pros and cons of singleton pattern for agents",
                "constraints": "High concurrency environment",
                "name": "Design pattern analysis"
            }
        ]

        for test in test_cases:
            start = time.time()
            try:
                response = requests.post(
                    f"{BASE_URL}/api/echo/agents/reasoning",
                    json=test,
                    timeout=90
                )
                elapsed = time.time() - start

                if response.status_code == 200:
                    result = response.json()
                    has_structure = all(key in result for key in ["analysis", "reasoning", "conclusion"])

                    self.results["reasoning"]["times"].append(elapsed)
                    self.results["reasoning"]["success"].append(True)
                    self.results["reasoning"]["quality"].append(has_structure)

                    print(f"✓ {test['name']}: {elapsed:.2f}s - Structured: {has_structure}")
                else:
                    print(f"✗ {test['name']}: Failed with status {response.status_code}")
                    self.results["reasoning"]["success"].append(False)

            except Exception as e:
                print(f"✗ {test['name']}: Error - {e}")
                self.results["reasoning"]["success"].append(False)

        return self._summarize("ReasoningAgent", self.results["reasoning"])

    def test_narration_agent(self) -> Dict:
        """Test NarrationAgent with scene generation"""
        print("\n" + "="*60)
        print("NARRATION AGENT TESTS")
        print("="*60)

        test_cases = [
            {"scene": "Cyberpunk city", "genre": "cyberpunk", "name": "Cyberpunk scene"},
            {"scene": "Magic forest", "genre": "fantasy", "name": "Fantasy scene"},
            {"scene": "Space battle", "genre": "sci-fi", "name": "Sci-fi scene"}
        ]

        for test in test_cases:
            start = time.time()
            try:
                response = requests.post(
                    f"{BASE_URL}/api/echo/agents/narration",
                    json=test,
                    timeout=30
                )
                elapsed = time.time() - start

                if response.status_code == 200:
                    result = response.json()
                    has_comfyui = bool(result.get("comfyui_prompt"))

                    self.results["narration"]["times"].append(elapsed)
                    self.results["narration"]["success"].append(True)
                    self.results["narration"]["has_comfyui"].append(has_comfyui)

                    print(f"✓ {test['name']}: {elapsed:.2f}s - ComfyUI: {has_comfyui}")
                else:
                    print(f"✗ {test['name']}: Failed with status {response.status_code}")
                    self.results["narration"]["success"].append(False)

            except Exception as e:
                print(f"✗ {test['name']}: Error - {e}")
                self.results["narration"]["success"].append(False)

        return self._summarize("NarrationAgent", self.results["narration"])

    def test_database_separation(self) -> Tuple[bool, str]:
        """Verify database separation is maintained"""
        print("\n" + "="*60)
        print("DATABASE SEPARATION TEST")
        print("="*60)

        try:
            # Check tower_consolidated
            conn = psycopg2.connect(
                host='localhost',
                database='tower_consolidated',
                user='patrick',
                password='tower_echo_brain_secret_key_2025'
            )
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE 'anime_%'
            """)
            anime_in_consolidated = cur.fetchone()[0]
            cur.close()
            conn.close()

            # Check tower_anime
            conn = psycopg2.connect(
                host='localhost',
                database='tower_anime',
                user='patrick',
                password='tower_echo_brain_secret_key_2025'
            )
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('character_profiles', 'animation_projects')
            """)
            anime_tables = cur.fetchone()[0]
            cur.close()
            conn.close()

            if anime_in_consolidated == 0 and anime_tables > 0:
                print("✓ Database separation verified: No contamination")
                return True, "Clean separation"
            else:
                print(f"✗ Database issue: {anime_in_consolidated} anime tables in consolidated, {anime_tables} in anime DB")
                return False, f"Contamination detected"

        except Exception as e:
            print(f"✗ Database test failed: {e}")
            return False, str(e)

    def test_concurrent_performance(self) -> Dict:
        """Test concurrent agent execution"""
        print("\n" + "="*60)
        print("CONCURRENT EXECUTION TEST")
        print("="*60)

        import concurrent.futures

        def call_agent(agent_type, payload):
            start = time.time()
            try:
                response = requests.post(
                    f"{BASE_URL}/api/echo/agents/{agent_type}",
                    json=payload,
                    timeout=60
                )
                return (agent_type, time.time() - start, response.status_code == 200)
            except:
                return (agent_type, time.time() - start, False)

        tasks = [
            ("coding", {"task": "Write hello world", "language": "python"}),
            ("reasoning", {"task": "What is 2+2?"}),
            ("narration", {"scene": "sunset", "genre": "nature"})
        ]

        print("Launching 3 concurrent requests...")
        start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(call_agent, t[0], t[1]) for t in tasks]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start

        success_count = sum(1 for _, _, success in results if success)

        print(f"Total time: {total_time:.2f}s")
        for agent, duration, success in results:
            status = "✓" if success else "✗"
            print(f"{status} {agent}: {duration:.2f}s")

        return {
            "total_time": total_time,
            "success_rate": success_count / len(results),
            "parallel_efficiency": max(r[1] for r in results) / total_time
        }

    def _summarize(self, agent_name: str, metrics: Dict) -> Dict:
        """Generate summary statistics"""
        times = metrics.get("times", [])
        success = metrics.get("success", [])

        if not times:
            return {"error": "No successful tests"}

        summary = {
            "agent": agent_name,
            "tests_run": len(success),
            "tests_passed": sum(success),
            "success_rate": sum(success) / len(success) * 100 if success else 0,
            "avg_response_time": statistics.mean(times) if times else 0,
            "median_response_time": statistics.median(times) if times else 0,
            "max_response_time": max(times) if times else 0,
            "min_response_time": min(times) if times else 0
        }

        # Add agent-specific metrics
        if "accuracy" in metrics:
            accuracy = metrics["accuracy"]
            summary["task_accuracy"] = sum(accuracy) / len(accuracy) * 100 if accuracy else 0

        if "quality" in metrics:
            quality = metrics["quality"]
            summary["structure_quality"] = sum(quality) / len(quality) * 100 if quality else 0

        if "has_comfyui" in metrics:
            comfyui = metrics["has_comfyui"]
            summary["comfyui_generation_rate"] = sum(comfyui) / len(comfyui) * 100 if comfyui else 0

        return summary


def main():
    print("\n" + "="*60)
    print("ECHO BRAIN COMPREHENSIVE TEST REPORT")
    print(f"Generated: {datetime.now()}")
    print("="*60)

    # Check service health
    try:
        response = requests.get(f"{BASE_URL}/api/echo/agents/status", timeout=5)
        if response.status_code != 200:
            print("❌ Echo Brain service not healthy")
            return

        status = response.json()
        print("\n📊 AGENT STATUS:")
        for agent in status.get("agents", []):
            print(f"  • {agent['name']}: {agent['model']} - {agent['tasks_processed']} tasks processed")

    except Exception as e:
        print(f"❌ Cannot reach Echo Brain: {e}")
        return

    # Run tests
    tester = TestMetrics()

    # Individual agent tests
    coding_summary = tester.test_coding_agent()
    reasoning_summary = tester.test_reasoning_agent()
    narration_summary = tester.test_narration_agent()

    # System tests
    db_success, db_message = tester.test_database_separation()
    concurrent_metrics = tester.test_concurrent_performance()

    # Generate final report
    print("\n" + "="*60)
    print("📊 QUANTITATIVE METRICS SUMMARY")
    print("="*60)

    for summary in [coding_summary, reasoning_summary, narration_summary]:
        if "error" not in summary:
            print(f"\n{summary['agent']}:")
            print(f"  Success Rate: {summary['success_rate']:.1f}%")
            print(f"  Avg Response: {summary['avg_response_time']:.2f}s")
            print(f"  Median Response: {summary['median_response_time']:.2f}s")
            print(f"  Range: {summary['min_response_time']:.2f}s - {summary['max_response_time']:.2f}s")

            if "task_accuracy" in summary:
                print(f"  Task Accuracy: {summary['task_accuracy']:.1f}%")
            if "structure_quality" in summary:
                print(f"  Structure Quality: {summary['structure_quality']:.1f}%")
            if "comfyui_generation_rate" in summary:
                print(f"  ComfyUI Generation: {summary['comfyui_generation_rate']:.1f}%")

    print(f"\nDatabase Separation: {'✅ PASS' if db_success else '❌ FAIL'} - {db_message}")

    print(f"\nConcurrent Execution:")
    print(f"  Total Time: {concurrent_metrics['total_time']:.2f}s")
    print(f"  Success Rate: {concurrent_metrics['success_rate']*100:.1f}%")
    print(f"  Parallel Efficiency: {concurrent_metrics['parallel_efficiency']*100:.1f}%")

    # Overall assessment
    print("\n" + "="*60)
    print("🎯 OVERALL ASSESSMENT")
    print("="*60)

    total_tests = (coding_summary.get('tests_run', 0) +
                  reasoning_summary.get('tests_run', 0) +
                  narration_summary.get('tests_run', 0))

    total_passed = (coding_summary.get('tests_passed', 0) +
                   reasoning_summary.get('tests_passed', 0) +
                   narration_summary.get('tests_passed', 0))

    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"Total Tests Run: {total_tests}")
    print(f"Total Tests Passed: {total_passed}")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    print(f"Database Integrity: {'✅ Maintained' if db_success else '❌ Compromised'}")

    # Tower-specific capabilities
    print("\n📋 TOWER-SPECIFIC CAPABILITIES:")
    capabilities = {
        "Understands Tower services": coding_summary.get('task_accuracy', 0) > 50,
        "Can analyze architecture": reasoning_summary.get('success_rate', 0) > 50,
        "Generates ComfyUI prompts": narration_summary.get('comfyui_generation_rate', 0) > 50,
        "Handles concurrent requests": concurrent_metrics['success_rate'] > 0.8,
        "Maintains database separation": db_success
    }

    for capability, verified in capabilities.items():
        status = "✅" if verified else "❌"
        print(f"  {status} {capability}")

    verified_count = sum(capabilities.values())
    print(f"\n{verified_count}/{len(capabilities)} capabilities verified")

    if overall_success_rate >= 80 and verified_count >= 4:
        print("\n✅ SYSTEM VALIDATED - Echo Brain is operational for Tower")
    elif overall_success_rate >= 60:
        print("\n⚠️ PARTIAL VALIDATION - System functional but needs improvements")
    else:
        print("\n❌ VALIDATION FAILED - Critical issues detected")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "agents": [coding_summary, reasoning_summary, narration_summary],
        "database_separation": {"success": db_success, "message": db_message},
        "concurrent_performance": concurrent_metrics,
        "overall_metrics": {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "success_rate": overall_success_rate,
            "capabilities_verified": verified_count
        }
    }

    with open("/opt/tower-echo-brain/tests/final_test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📁 Report saved to: /opt/tower-echo-brain/tests/final_test_report.json")


if __name__ == "__main__":
    main()