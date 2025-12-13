#!/usr/bin/env python3
"""
Test script for user context and identity systems
"""

import asyncio
import sys
sys.path.insert(0, '/opt/tower-echo-brain')

from src.core.echo_identity import get_echo_identity
from src.core.user_context_manager import get_user_context_manager
from src.integrations.vault_manager import get_vault_manager

async def test_user_contexts():
    """Test user context system"""
    print("=" * 50)
    print("TESTING ECHO USER CONTEXT SYSTEM")
    print("=" * 50)

    # Get managers
    echo_identity = get_echo_identity()
    user_manager = await get_user_context_manager()
    vault_manager = await get_vault_manager()

    # Test 1: Creator recognition
    print("\n1. Testing Creator Recognition:")
    print("-" * 30)
    patrick_recognition = echo_identity.recognize_user("patrick")
    print(f"Patrick recognized: {patrick_recognition['recognized']}")
    print(f"Identity: {patrick_recognition['identity']}")
    print(f"Access level: {patrick_recognition['access_level']}")
    print(f"Greeting: {patrick_recognition['greeting']}")

    # Test 2: External user recognition
    print("\n2. Testing External User Recognition:")
    print("-" * 30)
    external_recognition = echo_identity.recognize_user("external")
    print(f"External recognized: {external_recognition['recognized']}")
    print(f"Identity: {external_recognition['identity']}")
    print(f"Access level: {external_recognition['access_level']}")
    print(f"Permissions: {external_recognition['permissions']}")

    # Test 3: Creator context
    print("\n3. Testing Creator Context:")
    print("-" * 30)
    patrick_context = await user_manager.get_or_create_context("patrick")
    print(f"Username: {patrick_context.username}")
    print(f"User ID: {patrick_context.user_id}")
    print(f"Permissions: {patrick_context.permissions}")

    # Test 4: External user context
    print("\n4. Testing External User Context:")
    print("-" * 30)
    external_context = await user_manager.get_or_create_context("test_user")
    print(f"Username: {external_context.username}")
    print(f"User ID: {external_context.user_id}")
    print(f"Permissions: {external_context.permissions}")

    # Test 5: Permission checks
    print("\n5. Testing Permission Checks:")
    print("-" * 30)
    patrick_system_cmd = await user_manager.check_permission("patrick", "system_commands")
    external_system_cmd = await user_manager.check_permission("test_user", "system_commands")
    external_image_gen = await user_manager.check_permission("test_user", "image_generation")

    print(f"Patrick can execute system commands: {patrick_system_cmd}")
    print(f"External user can execute system commands: {external_system_cmd}")
    print(f"External user can generate images: {external_image_gen}")

    # Test 6: Task authorization
    print("\n6. Testing Task Authorization:")
    print("-" * 30)
    patrick_task = echo_identity.should_execute_task("run system command", "patrick")
    external_task = echo_identity.should_execute_task("run system command", "external")
    external_image = echo_identity.should_execute_task("generate an image", "external")

    print(f"Patrick system command: {patrick_task[0]} - {patrick_task[1]}")
    print(f"External system command: {external_task[0]} - {external_task[1]}")
    print(f"External image generation: {external_image[0]} - {external_image[1]}")

    # Test 7: Vault status
    print("\n7. Testing Vault Integration:")
    print("-" * 30)
    print(f"Vault initialized: {vault_manager.is_initialized}")
    if vault_manager.is_initialized:
        # Try to get credentials (creator only)
        creds = vault_manager.get_all_credentials("patrick")
        if isinstance(creds, dict) and "error" not in creds:
            print(f"Available services: {list(creds.keys())}")
        else:
            print("Could not retrieve credentials")

    # Test 8: Memory persistence
    print("\n8. Testing Memory Persistence:")
    print("-" * 30)
    await user_manager.learn_about_user("patrick", "favorite_model", "llama3.1:70b")
    await user_manager.add_conversation("patrick", "user", "Test message")

    memory = await user_manager.get_user_memory("patrick")
    print(f"Stored memories: {len(memory)} items")
    if "favorite_model" in memory:
        print(f"Favorite model: {memory['favorite_model']['value']}")

    # Test 9: User preferences
    print("\n9. Testing User Preferences:")
    print("-" * 30)
    await user_manager.update_preference("patrick", "response_style", "technical")
    await user_manager.update_style("patrick", "technical_level", "expert")

    updated_context = await user_manager.get_or_create_context("patrick")
    print(f"Response style: {updated_context.preferences['response_style']}")
    print(f"Technical level: {updated_context.communication_style['technical_level']}")

    # Test 10: Creator dashboard
    print("\n10. Testing Creator Dashboard:")
    print("-" * 30)
    dashboard = echo_identity.get_creator_dashboard()
    print(f"Echo status: {dashboard['echo_status']['status']}")
    print(f"Active services: {len(dashboard['active_services'])} services")
    print(f"Learning metrics: {dashboard['learning_metrics']['conversations']}")

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_user_contexts())