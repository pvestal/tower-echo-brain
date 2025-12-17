#!/usr/bin/env python3
"""
Test the intelligence router directly to isolate corruption
"""

import asyncio
from src.core.intelligence import intelligence_router

async def test_direct_intelligence():
    """Test calling intelligence router directly"""

    # Test the same query that's failing
    query = "List all running services"

    # Build context like the API does
    context = {
        "conversation_history": [],
        "user_id": "test_user",
        "intent": "system_query",
        "intent_params": {}
    }

    print("=== TESTING INTELLIGENCE ROUTER DIRECTLY ===")

    # Test query_model directly
    result = await intelligence_router.query_model("llama3.2:3b", query, context)

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Response: {result['response'][:300]}...")
    else:
        print(f"Error: {result.get('error')}")

    # Test progressive_escalation
    print("\n=== TESTING PROGRESSIVE ESCALATION ===")

    result2 = await intelligence_router.progressive_escalation(query, context)

    print(f"Success: {result2['success']}")
    if result2['success']:
        print(f"Response: {result2['response'][:300]}...")
    else:
        print(f"Error: {result2.get('error')}")

if __name__ == "__main__":
    asyncio.run(test_direct_intelligence())