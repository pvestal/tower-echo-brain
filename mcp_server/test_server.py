#!/usr/bin/env python3
"""
Test script for Echo Brain MCP Server functionality.
Tests core tools without the MCP protocol overhead.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the server class directly
from mcp_server.main import EchoBrainMCPServer


async def test_server_functionality():
    """Test the core functionality of the MCP server."""
    print("🧪 Testing Echo Brain MCP Server functionality...\n")

    # Initialize server
    server = EchoBrainMCPServer()

    try:
        await server.initialize()
        print("✅ Server initialization successful\n")

        # Test 1: Search Memory
        print("🔍 Testing search_memory...")
        try:
            results = await server._search_vector_memory("tower services", 3)
            print(f"✅ Found {len(results)} memory results")
            if results:
                print(f"   Best match score: {results[0]['score']:.3f}")
        except Exception as e:
            print(f"❌ Search memory failed: {e}")

        # Test 2: Get Facts
        print("\n📚 Testing get_facts...")
        try:
            facts = await server._get_facts_from_db("tower")
            print(f"✅ Found {len(facts)} facts about 'tower'")
        except Exception as e:
            print(f"❌ Get facts failed: {e}")

        # Test 3: Store Fact
        print("\n💾 Testing store_fact...")
        try:
            success = await server._store_fact_in_db(
                "MCP Server",
                "runs on port",
                "stdio"
            )
            if success:
                print("✅ Successfully stored test fact")
            else:
                print("❌ Failed to store fact")
        except Exception as e:
            print(f"❌ Store fact failed: {e}")

        # Test 4: Recent Context
        print("\n⏰ Testing get_recent_context...")
        try:
            context = await server._get_recent_context_from_db(24)
            print(f"✅ Found {len(context)} recent context items")
        except Exception as e:
            print(f"❌ Get recent context failed: {e}")

        # Test 5: Embedding Generation
        print("\n🧠 Testing embedding generation...")
        try:
            embedding = await server._generate_embedding("test query")
            if embedding and len(embedding) > 0:
                print(f"✅ Generated embedding with {len(embedding)} dimensions")
            else:
                print("❌ Failed to generate embedding")
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")

        print("\n🎉 All tests completed!")

    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        return False

    finally:
        await server.cleanup()

    return True


if __name__ == "__main__":
    # Set environment variables if not set
    if not os.environ.get("ECHO_BRAIN_DB_PASSWORD"):
        os.environ["ECHO_BRAIN_DB_PASSWORD"] = os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")

    asyncio.run(test_server_functionality())