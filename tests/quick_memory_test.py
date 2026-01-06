#!/usr/bin/env python3
"""
Quick memory test for Echo Brain
Tests basic conversation memory functionality
"""

import asyncio
import httpx
import json
from datetime import datetime
from colorama import Fore, init

init(autoreset=True)

async def test_echo_memory():
    """Test Echo Brain's memory capabilities"""
    base_url = "http://localhost:8309"
    test_id = f"test_{datetime.now().timestamp()}"

    print(f"\n{Fore.CYAN}Testing Echo Brain Memory System")
    print(f"{Fore.CYAN}{'='*40}\n")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: Send initial message
        print(f"{Fore.YELLOW}Test 1: Sending initial message...")
        response1 = await client.post(
            f"{base_url}/api/echo/query",
            json={
                "query": f"Remember this test ID: {test_id} and that I like pizza",
                "conversation_id": f"memory_test_{test_id}",
                "username": "test_user"
            }
        )

        if response1.status_code == 200:
            print(f"{Fore.GREEN}✓ Initial message sent successfully")
            data = response1.json()
            print(f"  Response: {data.get('response', '')[:100]}...")
        else:
            print(f"{Fore.RED}✗ Failed to send message: {response1.status_code}")
            return False

        # Wait for processing
        await asyncio.sleep(2)

        # Test 2: Check memory recall
        print(f"\n{Fore.YELLOW}Test 2: Testing memory recall...")
        response2 = await client.post(
            f"{base_url}/api/echo/query",
            json={
                "query": "What was the test ID I just told you?",
                "conversation_id": f"memory_test_{test_id}",
                "username": "test_user"
            }
        )

        if response2.status_code == 200:
            data = response2.json()
            response_text = data.get("response", "")

            if test_id in response_text:
                print(f"{Fore.GREEN}✓ Memory recall successful - Echo remembered the test ID!")
                print(f"  Response: {response_text[:100]}...")
                return True
            else:
                print(f"{Fore.YELLOW}⚠ Echo responded but didn't recall the test ID")
                print(f"  Response: {response_text[:100]}...")
                return False
        else:
            print(f"{Fore.RED}✗ Failed to test recall: {response2.status_code}")
            return False


async def main():
    """Run quick memory test"""
    try:
        success = await test_echo_memory()

        if success:
            print(f"\n{Fore.GREEN}✅ Echo Brain memory system is working!")
        else:
            print(f"\n{Fore.RED}❌ Echo Brain memory system needs attention")

    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())