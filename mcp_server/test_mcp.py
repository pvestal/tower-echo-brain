#!/usr/bin/env python3
"""Test MCP server functionality without stdio complexity."""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the server
from mcp_server.main import EchoBrainMCPServer


async def test_server():
    """Test the MCP server functions."""
    print("Initializing Echo Brain MCP Server...")

    server = EchoBrainMCPServer()
    await server.initialize()

    print("✅ Server initialized successfully")

    # Test search_memory
    print("\nTesting search_memory...")
    result = await server.server.call_tool()({"query": "Patrick", "limit": 2})
    print(f"Search results: {len(result)} items")

    # Test get_facts
    print("\nTesting get_facts...")
    result = await server.server.call_tool()({"topic": "Tower"})
    print(f"Facts found: {result[0].text[:100] if result else 'None'}...")

    # Test get_recent_context
    print("\nTesting get_recent_context...")
    result = await server.server.call_tool()({"hours": 24})
    print(f"Context items: {result[0].text[:100] if result else 'None'}...")

    await server.cleanup()
    print("\n✅ All tests completed")


if __name__ == "__main__":
    asyncio.run(test_server())