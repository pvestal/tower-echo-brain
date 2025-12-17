#!/usr/bin/env python3
"""
Quick diagnostic tests for conversation memory system
"""
import asyncio
import sys
sys.path.append('/opt/tower-echo-brain')

async def quick_diagnostic():
    from src.managers.conversation_memory_manager import get_conversation_memory_manager

    print("🚀 Running Quick Conversation Memory Diagnostics")
    print("-" * 50)

    memory_manager = await get_conversation_memory_manager()

    # Test 1: Basic functionality
    print("\n1️⃣ Testing basic entity extraction...")
    turn = await memory_manager.add_turn("diag_1", "user", "The anime service on port 8328 is broken", "system_issue")
    print(f"   Extracted {len(turn.entities)} entities:")
    for e in turn.entities:
        print(f"   - {e.name} ({e.entity_type.value}): {e.value}")

    # Test 2: Pronoun resolution
    print("\n2️⃣ Testing pronoun resolution...")
    await memory_manager.add_turn("diag_1", "assistant", "Found memory leak in anime service", "analysis")
    enhanced, entities = await memory_manager.resolve_reference("Restart it immediately", "diag_1")
    print(f"   Original: 'Restart it immediately'")
    print(f"   Enhanced: '{enhanced}'")
    print(f"   Resolved entities: {len(entities)}")

    # Test 3: Session management
    print("\n3️⃣ Testing session management...")
    summary = await memory_manager.get_session_summary("diag_1")
    print(f"   Session summary: {summary}")

    # Test 4: Multiple sessions
    print("\n4️⃣ Testing multiple sessions...")
    for i in range(3):
        await memory_manager.add_turn(f"diag_{i}", "user", f"Testing service {i}", "test")

    print(f"   Active sessions: {len(memory_manager.active_sessions)}")
    print(f"   Global entities: {len(memory_manager.global_entities)}")

    # Test 5: Persistence check
    print("\n5️⃣ Testing persistence...")
    persistence_file = memory_manager.storage_path
    import os
    if persistence_file.exists():
        file_size = persistence_file.stat().st_size
        print(f"   Persistence file exists: {file_size} bytes")
    else:
        print("   ⚠️ Persistence file not found")

    print("\n" + "=" * 50)
    print("✅ Quick diagnostics completed!")

if __name__ == "__main__":
    asyncio.run(quick_diagnostic())