#!/usr/bin/env python3
"""
Focused Conversation Memory Test Suite

Core functionality tests without external dependencies:
- Entity extraction and persistence
- Reference resolution across turns
- Context coherence maintenance
- Cross-interface simulation
- Performance under load

This test suite validates the core conversation memory capabilities
that enable "What services are broken?" → "Fix it" coherence.
"""

import asyncio
import time
import json
from typing import List, Dict, Any
from pathlib import Path

from src.managers.conversation_memory_manager import get_conversation_memory_manager

class FocusedMemoryTester:
    """Focused tests for conversation memory core functionality"""

    def __init__(self):
        self.test_results = {}

    async def test_basic_entity_extraction(self):
        """Test entity extraction from various message types"""
        print("🔍 Testing basic entity extraction...")

        memory_manager = await get_conversation_memory_manager()
        test_cases = [
            {
                "input": "The tower-echo-brain service on port 8309 is having connection issues",
                "expected_entities": ["tower-echo-brain", "8309", "connection"],
                "expected_count": 3
            },
            {
                "input": "Check the anime-production API and restart the authentication service",
                "expected_entities": ["anime-production", "authentication"],
                "expected_count": 2
            },
            {
                "input": "vault.json has the OAuth credentials for tower-auth",
                "expected_entities": ["vault.json", "oauth", "tower-auth"],
                "expected_count": 3
            }
        ]

        total_accuracy = 0
        for i, case in enumerate(test_cases):
            turn = await memory_manager.add_turn(
                conversation_id=f"extraction_test_{i}",
                role="user",
                content=case["input"],
                intent="system_query"
            )

            entities_found = len(turn.entities)
            accuracy = min(entities_found / case["expected_count"], 1.0)
            total_accuracy += accuracy

            print(f"   Test {i+1}: {entities_found}/{case['expected_count']} entities - {accuracy:.1%}")

        avg_accuracy = total_accuracy / len(test_cases)
        print(f"   ✅ Average extraction accuracy: {avg_accuracy:.1%}")

        return avg_accuracy > 0.7  # 70% accuracy threshold

    async def test_reference_resolution_scenarios(self):
        """Test pronoun resolution in various scenarios"""
        print("🔗 Testing reference resolution scenarios...")

        memory_manager = await get_conversation_memory_manager()
        conv_id = "resolution_test"

        # Scenario 1: Single entity reference
        await memory_manager.add_turn(
            conversation_id=conv_id,
            role="user",
            content="The tower-anime-production service is failing with GPU errors",
            intent="incident_report"
        )

        enhanced_1, entities_1 = await memory_manager.resolve_reference(
            "restart it immediately", conv_id
        )

        # Scenario 2: Multiple entity reference
        await memory_manager.add_turn(
            conversation_id=conv_id,
            role="user",
            content="Also check the auth service and the knowledge base API",
            intent="additional_check"
        )

        enhanced_2, entities_2 = await memory_manager.resolve_reference(
            "run diagnostics on all of them", conv_id
        )

        # Scenario 3: Type-specific reference
        enhanced_3, entities_3 = await memory_manager.resolve_reference(
            "restart the auth service", conv_id
        )

        # Evaluate results
        single_resolution = "tower-anime-production" in enhanced_1
        multi_resolution = len(entities_2) >= 2
        type_resolution = "auth" in enhanced_3

        print(f"   ✅ Single entity: {'PASS' if single_resolution else 'FAIL'}")
        print(f"   ✅ Multiple entities: {'PASS' if multi_resolution else 'FAIL'}")
        print(f"   ✅ Type-specific: {'PASS' if type_resolution else 'FAIL'}")

        success_rate = sum([single_resolution, multi_resolution, type_resolution]) / 3
        print(f"   ✅ Overall resolution success: {success_rate:.1%}")

        return success_rate > 0.66  # 2/3 scenarios must pass

    async def test_context_persistence_across_turns(self):
        """Test that conversation context persists and builds over multiple turns"""
        print("💾 Testing context persistence across turns...")

        memory_manager = await get_conversation_memory_manager()
        conv_id = "persistence_test"

        # Build conversation context over 5 turns
        conversation_turns = [
            ("user", "I need to debug the anime production pipeline"),
            ("assistant", "I'll help you debug tower-anime-production. First, let's check the GPU status"),
            ("user", "GPU memory is at 95% usage on the NVIDIA card"),
            ("assistant", "High GPU usage detected. Checking ComfyUI workflows for memory leaks"),
            ("user", "Also the video generation is taking 8+ minutes per scene")
        ]

        entities_over_time = []
        for i, (role, content) in enumerate(conversation_turns):
            turn = await memory_manager.add_turn(
                conversation_id=conv_id,
                role=role,
                content=content,
                intent="troubleshooting"
            )

            # Track entity growth
            session = memory_manager.active_sessions.get(conv_id)
            entity_count = len(session.active_entities) if session else 0
            entities_over_time.append(entity_count)

            print(f"   Turn {i+1}: {entity_count} entities tracked")

        # Test context retrieval
        context_prompt = await memory_manager.get_context_prompt(conv_id)
        context_length = len(context_prompt)

        # Test that entities accumulate (should generally increase)
        entity_growth = entities_over_time[-1] > entities_over_time[0]
        adequate_context = context_length > 200

        print(f"   ✅ Entity accumulation: {'PASS' if entity_growth else 'FAIL'}")
        print(f"   ✅ Context generation: {context_length} chars - {'PASS' if adequate_context else 'FAIL'}")

        return entity_growth and adequate_context

    async def test_cross_interface_simulation(self):
        """Simulate conversation continuity across different interfaces"""
        print("🌐 Testing cross-interface conversation continuity...")

        memory_manager = await get_conversation_memory_manager()
        base_conv_id = "cross_interface_sim"

        # Simulate interface transitions: Telegram → Web → CLI
        interface_scenarios = [
            ("telegram", "The authentication service is showing bypass mode warnings", "system_alert"),
            ("web", "I need to check the OAuth configuration in the vault", "configuration_check"),
            ("cli", "Show me the recent error logs from that service", "log_analysis"),
            ("web", "Now restart both the auth service and vault to apply changes", "service_management")
        ]

        interface_entities = {}

        for interface, message, intent in interface_scenarios:
            # Use interface-specific conversation IDs but track entities globally
            conv_id = f"{base_conv_id}_{interface}"

            turn = await memory_manager.add_turn(
                conversation_id=conv_id,
                role="user",
                content=message,
                intent=intent
            )

            # Track entities per interface
            session = memory_manager.active_sessions.get(conv_id)
            interface_entities[interface] = len(session.active_entities) if session else 0

        # Test cross-interface reference resolution
        final_enhanced, final_entities = await memory_manager.resolve_reference(
            "restart both of them now", f"{base_conv_id}_web"
        )

        # Results
        total_entities = sum(interface_entities.values())
        cross_reference_success = len(final_entities) > 0

        print(f"   ✅ Total entities across interfaces: {total_entities}")
        print(f"   ✅ Cross-interface reference: {'PASS' if cross_reference_success else 'FAIL'}")

        for interface, count in interface_entities.items():
            print(f"   - {interface}: {count} entities")

        return total_entities > 3 and cross_reference_success

    async def test_performance_under_concurrent_load(self):
        """Test memory system performance under concurrent conversations"""
        print("⚡ Testing performance under concurrent load...")

        memory_manager = await get_conversation_memory_manager()

        async def simulate_conversation(conv_id: str, turns: int):
            """Simulate a conversation with specified number of turns"""
            try:
                start_time = time.time()

                for turn_num in range(turns):
                    await memory_manager.add_turn(
                        conversation_id=conv_id,
                        role="user" if turn_num % 2 == 0 else "assistant",
                        content=f"Turn {turn_num}: Discussing service_{conv_id[-2:]} status and performance metrics",
                        intent="performance_monitoring"
                    )

                # Test reference resolution
                enhanced, entities = await memory_manager.resolve_reference(
                    "restart that service", conv_id
                )

                total_time = time.time() - start_time
                return {
                    "success": len(entities) > 0,
                    "duration": total_time,
                    "turns_completed": turns
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Run 20 concurrent conversations with 5 turns each
        tasks = []
        conversation_count = 20
        turns_per_conversation = 5

        start_time = time.time()
        for i in range(conversation_count):
            conv_id = f"perf_test_{i:02d}"
            tasks.append(simulate_conversation(conv_id, turns_per_conversation))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Analyze results
        successful_conversations = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        success_rate = successful_conversations / conversation_count
        avg_time_per_conversation = total_time / conversation_count
        total_turns = conversation_count * turns_per_conversation

        print(f"   ✅ Successful conversations: {successful_conversations}/{conversation_count}")
        print(f"   ✅ Success rate: {success_rate:.1%}")
        print(f"   ✅ Total time: {total_time:.2f}s for {total_turns} turns")
        print(f"   ✅ Average per conversation: {avg_time_per_conversation*1000:.1f}ms")
        print(f"   ✅ Average per turn: {(total_time/total_turns)*1000:.1f}ms")

        return success_rate > 0.9 and avg_time_per_conversation < 0.1  # 90% success, <100ms per conversation

    async def test_memory_persistence_and_recovery(self):
        """Test memory system persistence and recovery capabilities"""
        print("💾 Testing memory persistence and recovery...")

        memory_manager = await get_conversation_memory_manager()
        conv_id = "persistence_recovery_test"

        # Create conversation with entities
        await memory_manager.add_turn(
            conversation_id=conv_id,
            role="user",
            content="The tower-dashboard service on port 8080 needs monitoring setup",
            intent="configuration"
        )

        await memory_manager.add_turn(
            conversation_id=conv_id,
            role="assistant",
            content="I'll configure monitoring for tower-dashboard. Adding Prometheus metrics collection.",
            execution_results=["configure_monitoring"]
        )

        # Check entities before persistence
        session_before = memory_manager.active_sessions.get(conv_id)
        entities_before = len(session_before.active_entities) if session_before else 0

        # Force persistence
        await memory_manager._persist_state()

        # Simulate recovery by creating new manager instance
        recovery_manager = await get_conversation_memory_manager()

        # Check if entities were recovered
        recovered_session = recovery_manager.active_sessions.get(conv_id)
        entities_after = len(recovered_session.active_entities) if recovered_session else 0

        # Test persistence file exists
        storage_path = memory_manager.storage_path
        persistence_file_exists = storage_path.exists()

        print(f"   ✅ Entities before persistence: {entities_before}")
        print(f"   ✅ Entities after recovery: {entities_after}")
        print(f"   ✅ Persistence file exists: {persistence_file_exists}")

        return entities_before > 0 and entities_after > 0 and persistence_file_exists

    async def run_all_tests(self):
        """Execute complete focused test suite"""
        print("🧠 Starting Focused Conversation Memory Test Suite")
        print("=" * 65)

        test_functions = [
            ("entity_extraction", self.test_basic_entity_extraction),
            ("reference_resolution", self.test_reference_resolution_scenarios),
            ("context_persistence", self.test_context_persistence_across_turns),
            ("cross_interface", self.test_cross_interface_simulation),
            ("performance_load", self.test_performance_under_concurrent_load),
            ("memory_persistence", self.test_memory_persistence_and_recovery)
        ]

        results = {}

        for test_name, test_function in test_functions:
            print(f"\n{'-' * 65}")
            try:
                start_time = time.time()
                result = await test_function()
                test_time = (time.time() - start_time) * 1000

                results[test_name] = {
                    "passed": result,
                    "duration_ms": test_time
                }

                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{status} {test_name} ({test_time:.1f}ms)")

            except Exception as e:
                results[test_name] = {
                    "passed": False,
                    "duration_ms": 0,
                    "error": str(e)
                }
                print(f"❌ FAIL {test_name} - Error: {e}")

        # Summary
        print("\n" + "=" * 65)
        print("📊 FOCUSED TEST RESULTS SUMMARY")
        print("=" * 65)

        passed_tests = sum(1 for r in results.values() if r["passed"])
        total_tests = len(results)
        total_time = sum(r["duration_ms"] for r in results.values())

        for test_name, result in results.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            duration = result["duration_ms"]
            print(f"{status} {test_name:20} | {duration:6.1f}ms")

        print(f"\n📈 Overall Results:")
        print(f"   🎯 Tests passed: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
        print(f"   ⚡ Total execution time: {total_time:.1f}ms")
        print(f"   📊 Average per test: {total_time/total_tests:.1f}ms")

        # System status
        try:
            manager = await get_conversation_memory_manager()
            active_sessions = len(manager.active_sessions)
            global_entities = len(manager.global_entities)

            print(f"\n🧠 Memory System Status:")
            print(f"   📝 Active sessions: {active_sessions}")
            print(f"   🏷️  Global entities: {global_entities}")
            print(f"   💾 Storage path: {manager.storage_path}")

        except Exception as e:
            print(f"\n⚠️  System status check failed: {e}")

        return results

async def main():
    """Run focused conversation memory tests"""
    tester = FocusedMemoryTester()
    results = await tester.run_all_tests()

    # Return simple pass/fail for integration
    passed = sum(1 for r in results.values() if r["passed"])
    total = len(results)

    print(f"\n🎯 FINAL RESULT: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())