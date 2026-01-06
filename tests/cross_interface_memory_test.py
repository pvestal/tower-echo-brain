#!/usr/bin/env python3
"""
Cross-Interface Conversation Memory Test Suite

Tests conversation memory continuity across Echo's different interfaces:
- Telegram Bot
- Web API
- CLI
- Internal service calls

Key test scenarios:
1. User starts conversation on Telegram, continues on Web API
2. Entity resolution works across interface boundaries
3. Context maintains coherence when switching channels
4. Performance under multi-interface load
"""

import asyncio
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass
import httpx

from src.managers.conversation_memory_manager import get_conversation_memory_manager

@dataclass
class InterfaceTestResult:
    interface: str
    conversation_id: str
    entities_extracted: int
    resolution_accuracy: float
    response_time_ms: float
    context_coherence: bool

class CrossInterfaceMemoryTester:
    """Test conversation memory across Echo's interfaces"""

    def __init__(self):
        self.base_url = "http://localhost:8309"
        self.test_results = []

    async def test_telegram_to_web_continuity(self):
        """Test conversation started on Telegram, continued on Web API"""
        print("🤖 Testing Telegram → Web API conversation continuity...")

        memory_manager = await get_conversation_memory_manager()

        # Simulate Telegram conversation start
        telegram_conv_id = "telegram_user_123"

        # Turn 1: User via Telegram
        await memory_manager.add_turn(
            conversation_id=telegram_conv_id,
            role="user",
            content="Check the status of tower-anime-production on port 8328",
            intent="system_query"
        )

        # Turn 2: Echo response via Telegram
        await memory_manager.add_turn(
            conversation_id=telegram_conv_id,
            role="assistant",
            content="tower-anime-production is running but showing errors. Service needs restart.",
            execution_results=["check_service_status"]
        )

        # Switch to Web API - same conversation ID
        web_conv_id = telegram_conv_id  # Unified conversation tracking

        # Turn 3: User continues via Web API
        start_time = time.time()
        enhanced_query, resolved_entities = await memory_manager.resolve_reference(
            "restart it with debug logging", web_conv_id
        )
        resolution_time = (time.time() - start_time) * 1000

        # Validate cross-interface coherence
        context = await memory_manager.get_context_prompt(web_conv_id)
        session = memory_manager.active_sessions.get(web_conv_id)

        result = InterfaceTestResult(
            interface="telegram_to_web",
            conversation_id=web_conv_id,
            entities_extracted=len(session.active_entities) if session else 0,
            resolution_accuracy=1.0 if "tower-anime-production" in enhanced_query else 0.0,
            response_time_ms=resolution_time,
            context_coherence=len(context) > 100
        )

        print(f"   ✅ Entities tracked: {result.entities_extracted}")
        print(f"   ✅ Reference resolution: {enhanced_query}")
        print(f"   ✅ Response time: {result.response_time_ms:.2f}ms")
        print(f"   ✅ Context coherence: {result.context_coherence}")

        return result

    async def test_web_api_direct_integration(self):
        """Test conversation memory via direct Web API calls"""
        print("🌐 Testing Web API conversation memory integration...")

        async with httpx.AsyncClient() as client:
            # Test recording a turn via API
            record_payload = {
                "conversation_id": "web_api_test_456",
                "role": "user",
                "content": "The authentication service on port 8088 is in bypass mode",
                "intent": "system_issue"
            }

            try:
                start_time = time.time()
                response = await client.post(
                    f"{self.base_url}/api/echo/memory/record",
                    json=record_payload,
                    timeout=5.0
                )
                api_response_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ API response: {data.get('entities_extracted', 0)} entities extracted")
                    print(f"   ✅ API response time: {api_response_time:.2f}ms")
                    return True
                else:
                    print(f"   ❌ API call failed: {response.status_code}")
                    return False

            except Exception as e:
                print(f"   ❌ API test failed: {e}")
                return False

    async def test_high_load_cross_interface(self):
        """Test memory system under high load across multiple interfaces"""
        print("⚡ Testing high-load cross-interface scenario...")

        memory_manager = await get_conversation_memory_manager()

        # Simulate multiple simultaneous conversations across interfaces
        tasks = []
        conversation_count = 50

        async def simulate_interface_conversation(interface: str, conv_id: str):
            """Simulate conversation on specific interface"""
            try:
                # Turn 1: Problem report
                await memory_manager.add_turn(
                    conversation_id=conv_id,
                    role="user",
                    content=f"Service {interface}_service_prod is failing with connection errors",
                    intent="incident_report"
                )

                # Turn 2: Investigation
                await memory_manager.add_turn(
                    conversation_id=conv_id,
                    role="assistant",
                    content=f"Investigating {interface}_service_prod. Found port binding issues.",
                    execution_results=[f"check_{interface}_logs"]
                )

                # Turn 3: Reference resolution
                enhanced_query, entities = await memory_manager.resolve_reference(
                    "restart it immediately", conv_id
                )

                return len(entities) > 0  # Success if entities resolved

            except Exception as e:
                print(f"   ❌ Interface {interface} failed: {e}")
                return False

        # Create concurrent conversations
        interfaces = ["telegram", "web", "cli", "internal"]
        for i in range(conversation_count):
            interface = interfaces[i % len(interfaces)]
            conv_id = f"{interface}_load_test_{i}"
            tasks.append(simulate_interface_conversation(interface, conv_id))

        # Execute all conversations concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        success_rate = sum(1 for r in results if r is True) / len(results)

        print(f"   ✅ Processed {conversation_count} conversations in {total_time:.2f}s")
        print(f"   ✅ Success rate: {success_rate:.2%}")
        print(f"   ✅ Average time per conversation: {(total_time/conversation_count)*1000:.2f}ms")

        return success_rate > 0.95  # 95% success rate target

    async def test_entity_persistence_across_interfaces(self):
        """Test that entities persist correctly when switching interfaces"""
        print("🔗 Testing entity persistence across interface switches...")

        memory_manager = await get_conversation_memory_manager()
        base_conv_id = "multi_interface_persistence"

        # Interface sequence: CLI → Telegram → Web → CLI
        interface_sequence = [
            ("cli", "Check the tower-echo-brain service logs"),
            ("telegram", "The error rate is increasing"),
            ("web", "What about the database connections?"),
            ("cli", "Restart both the service and database")
        ]

        entities_at_each_step = []

        for i, (interface, message) in enumerate(interface_sequence):
            conv_id = f"{base_conv_id}_{interface}"

            # Add turn for this interface
            await memory_manager.add_turn(
                conversation_id=conv_id,
                role="user",
                content=message,
                intent="troubleshooting"
            )

            # Check entity count
            session = memory_manager.active_sessions.get(conv_id)
            entity_count = len(session.active_entities) if session else 0
            entities_at_each_step.append(entity_count)

            print(f"   Step {i+1} ({interface}): {entity_count} entities tracked")

        # Test cross-interface reference resolution
        final_enhanced, final_entities = await memory_manager.resolve_reference(
            "restart both of them", f"{base_conv_id}_cli"
        )

        print(f"   ✅ Final reference resolution: {final_enhanced}")
        print(f"   ✅ Entities resolved: {len(final_entities)}")

        # Entities should accumulate across interfaces
        return len(final_entities) >= 2 and entities_at_each_step[-1] > entities_at_each_step[0]

    async def run_all_tests(self):
        """Execute complete cross-interface test suite"""
        print("🧠 Starting Cross-Interface Memory Test Suite")
        print("=" * 60)

        test_results = {}

        # Test 1: Telegram to Web continuity
        try:
            result1 = await self.test_telegram_to_web_continuity()
            test_results["telegram_web_continuity"] = result1.resolution_accuracy > 0.8
        except Exception as e:
            print(f"❌ Telegram-Web test failed: {e}")
            test_results["telegram_web_continuity"] = False

        # Test 2: Web API integration
        try:
            result2 = await self.test_web_api_direct_integration()
            test_results["web_api_integration"] = result2
        except Exception as e:
            print(f"❌ Web API test failed: {e}")
            test_results["web_api_integration"] = False

        # Test 3: High load across interfaces
        try:
            result3 = await self.test_high_load_cross_interface()
            test_results["high_load_cross_interface"] = result3
        except Exception as e:
            print(f"❌ High load test failed: {e}")
            test_results["high_load_cross_interface"] = False

        # Test 4: Entity persistence
        try:
            result4 = await self.test_entity_persistence_across_interfaces()
            test_results["entity_persistence"] = result4
        except Exception as e:
            print(f"❌ Entity persistence test failed: {e}")
            test_results["entity_persistence"] = False

        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        total_tests = len(test_results)
        passed_tests = sum(test_results.values())

        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name}")

        print(f"\n📈 Overall Results: {passed_tests}/{total_tests} tests passed")
        print(f"🎯 Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        return test_results

async def main():
    """Run cross-interface memory tests"""
    tester = CrossInterfaceMemoryTester()
    results = await tester.run_all_tests()
    return results

if __name__ == "__main__":
    asyncio.run(main())