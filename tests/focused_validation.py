#!/usr/bin/env python3
"""
Focused validation of core conversation memory functionality
"""
import asyncio
import time
import sys
sys.path.append('/opt/tower-echo-brain')

from src.managers.conversation_memory_manager import get_conversation_memory_manager

async def run_focused_validation():
    print("🧠 FOCUSED CONVERSATION MEMORY VALIDATION")
    print("=" * 50)

    memory_manager = await get_conversation_memory_manager()

    # Test 1: Core User Scenario
    print("\n1️⃣ Testing Core User Scenario: 'What services are broken?' → 'Fix it'")

    # Turn 1: User asks about broken services
    turn1 = await memory_manager.add_turn(
        "validation_core", "user",
        "What services are broken on the Tower system?",
        "system_query"
    )

    # Turn 2: Assistant responds with specific broken services
    turn2 = await memory_manager.add_turn(
        "validation_core", "assistant",
        "The anime-production service on port 8328 is failing with 8+ minute generation times, and the auth service is in bypass mode",
        "system_response"
    )

    print(f"   ✅ Entities extracted: {len(turn2.entities)}")
    for entity in turn2.entities[:3]:  # Show first 3
        print(f"      - {entity.name} ({entity.entity_type.value})")

    # Turn 3: Critical test - "Fix it"
    enhanced, entities = await memory_manager.resolve_reference("Fix it", "validation_core")

    print(f"\n   🎯 CRITICAL TEST:")
    print(f"      Input: 'Fix it'")
    print(f"      Enhanced: '{enhanced}'")
    print(f"      Resolved entities: {len(entities)}")
    for entity in entities:
        print(f"        → {entity.name} ({entity.entity_type.value})")

    core_success = len(entities) > 0 and any("anime" in str(e.name).lower() or "auth" in str(e.name).lower() for e in entities)
    print(f"   ✅ Core scenario success: {core_success}")

    # Test 2: Multi-service Context
    print("\n2️⃣ Testing Multi-service Context Resolution")

    turn3 = await memory_manager.add_turn(
        "validation_multi", "user",
        "The tower-kb service and ComfyUI are both having issues",
        "multi_service_issue"
    )

    turn4 = await memory_manager.add_turn(
        "validation_multi", "assistant",
        "Found connection pool exhaustion in tower-kb and GPU memory leak in ComfyUI",
        "analysis"
    )

    enhanced_multi, entities_multi = await memory_manager.resolve_reference("Restart them both", "validation_multi")

    print(f"   Input: 'Restart them both'")
    print(f"   Enhanced: '{enhanced_multi}'")
    print(f"   Resolved: {len(entities_multi)} entities")

    multi_success = len(entities_multi) > 1
    print(f"   ✅ Multi-service success: {multi_success}")

    # Test 3: Performance Check
    print("\n3️⃣ Testing Performance")

    start_time = time.perf_counter()

    perf_turn = await memory_manager.add_turn(
        "validation_perf", "user",
        "The tower-anime-production service needs emergency restart with memory optimization",
        "emergency"
    )

    extraction_time = (time.perf_counter() - start_time) * 1000

    start_time = time.perf_counter()

    enhanced_perf, entities_perf = await memory_manager.resolve_reference("Do it now", "validation_perf")

    resolution_time = (time.perf_counter() - start_time) * 1000

    print(f"   Entity extraction: {extraction_time:.1f}ms")
    print(f"   Reference resolution: {resolution_time:.1f}ms")

    perf_success = extraction_time < 5000 and resolution_time < 1000  # 5s extraction, 1s resolution
    print(f"   ✅ Performance success: {perf_success}")

    # Test 4: Edge Case Handling
    print("\n4️⃣ Testing Edge Case Handling")

    edge_cases = [
        "",  # Empty
        "   ",  # Whitespace
        "Fix it immediately with restart",  # No context
        "The 🚀 service needs 🛠️ fixing",  # Emojis
    ]

    edge_successes = 0
    for i, case in enumerate(edge_cases):
        try:
            turn = await memory_manager.add_turn(f"edge_{i}", "user", case, "edge_test")
            edge_successes += 1
        except Exception as e:
            print(f"   ⚠️ Edge case failed: {case[:20]}")

    edge_success = edge_successes == len(edge_cases)
    print(f"   ✅ Edge cases handled: {edge_successes}/{len(edge_cases)}")

    # Test 5: Session Summary
    print("\n5️⃣ Testing Session Summary Generation")

    summary = await memory_manager.get_session_summary("validation_core")
    summary_success = len(summary) > 20 and "anime" in summary.lower()

    print(f"   Summary length: {len(summary)} chars")
    print(f"   Contains context: {'anime' in summary.lower()}")
    print(f"   ✅ Summary success: {summary_success}")

    # Final Results
    print("\n" + "=" * 50)
    print("📊 VALIDATION RESULTS")
    print("=" * 50)

    tests = [
        ("Core Scenario", core_success),
        ("Multi-service Context", multi_success),
        ("Performance", perf_success),
        ("Edge Cases", edge_success),
        ("Session Summary", summary_success),
    ]

    passed = sum(1 for _, success in tests if success)
    total = len(tests)

    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")

    print(f"\n  🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed >= 4:  # 4/5 tests must pass
        print(f"  🎉 CONVERSATION MEMORY: PRODUCTION READY!")
        return True
    else:
        print(f"  ⚠️ CONVERSATION MEMORY: NEEDS IMPROVEMENT")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_focused_validation())
    exit(0 if success else 1)