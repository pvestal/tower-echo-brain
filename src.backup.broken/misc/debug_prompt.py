#!/usr/bin/env python3
"""
Debug the actual prompt construction and LLM interaction
"""

import asyncio
import aiohttp
from src.core.echo_system_prompt import get_echo_system_prompt
from src.core.resilient_context import get_resilient_omniscient_context

async def debug_prompt_construction():
    """Debug what's actually being sent to the LLM"""

    # 1. Get the components
    system_prompt = get_echo_system_prompt()

    context_manager = get_resilient_omniscient_context()
    await context_manager.connect()
    omniscient_context = await context_manager.build_context_summary_resilient("List all running services")

    query_text = "List all running services"

    # 2. Build the full prompt exactly like intelligence.py does
    conversation_context = ""  # No conversation history for this test
    user_prefix = "\n\nUser: "
    echo_suffix = "\n\nEcho:"
    full_prompt = system_prompt + conversation_context + omniscient_context + user_prefix + query_text + echo_suffix

    # 3. Debug the prompt
    print("=== PROMPT DEBUGGING ===")
    print(f"System prompt length: {len(system_prompt)} chars")
    print(f"Omniscient context length: {len(omniscient_context)} chars")
    print(f"Full prompt length: {len(full_prompt)} chars")

    # 4. Check for corruption
    print("\n=== CORRUPTION CHECK ===")
    if '�' in full_prompt or '\x00' in full_prompt:
        print("❌ ENCODING CORRUPTION DETECTED")
    else:
        print("✅ No encoding corruption")

    # 5. Show prompt structure
    print("\n=== PROMPT STRUCTURE ===")
    print("STARTS WITH:")
    print(full_prompt[:300])
    print("\n" + "="*50)
    print("ENDS WITH:")
    print(full_prompt[-300:])

    # 6. Test with bare minimum
    print("\n=== TESTING BARE MINIMUM ===")
    simple_prompt = "You are Echo. User asks: List all running services. You respond:"

    async with aiohttp.ClientSession() as session:
        # Test complex prompt
        payload = {
            "model": "llama3.2:3b",
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_predict": 100, "temperature": 0.7}
        }

        print("Sending complex prompt...")
        async with session.post("http://localhost:11434/api/generate", json=payload) as response:
            if response.status == 200:
                result = await response.json()
                complex_response = result.get("response", "")
                print(f"Complex response: {complex_response[:200]}...")
            else:
                print(f"❌ Complex prompt failed: {response.status}")

        # Test simple prompt
        payload["prompt"] = simple_prompt
        print("\nSending simple prompt...")
        async with session.post("http://localhost:11434/api/generate", json=payload) as response:
            if response.status == 200:
                result = await response.json()
                simple_response = result.get("response", "")
                print(f"Simple response: {simple_response[:200]}...")
            else:
                print(f"❌ Simple prompt failed: {response.status}")

if __name__ == "__main__":
    asyncio.run(debug_prompt_construction())