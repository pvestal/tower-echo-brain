#!/usr/bin/env python3
"""Test script for DeepSeek Coding Agent"""

import asyncio
import sys
sys.path.insert(0, '/opt/tower-echo-brain')

from src.agents.deepseek_coding_agent import DeepSeekCodingAgent

async def test():
    print("Testing DeepSeek Coding Agent...")
    agent = DeepSeekCodingAgent()
    print(f"✅ Agent initialized: {agent.workspace}")
    
    # Test code generation
    result = await agent.generate_code("Create a hello world function")
    print(f"✅ Code generated: {len(result.get('code', ''))} chars")

asyncio.run(test())
