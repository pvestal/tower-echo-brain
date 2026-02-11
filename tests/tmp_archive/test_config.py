#!/usr/bin/env python3
"""Test if config manager loads properly"""
import sys
sys.path.insert(0, '/opt/tower-echo-brain')

try:
    from src.config.config_manager import config
    print("✅ Config manager imported successfully")

    # Test routing
    queries = [
        "Write a Python function",
        "Tell me about anime production",
        "What is 2+2?"
    ]

    for query in queries:
        agent = config.route_query(query)
        print(f"Query: '{query[:30]}...' → Agent: {agent}")

    # Test config retrieval
    print(f"\nOllama config: {config.get_ollama_config()}")
    print(f"Reasoning agent model: {config.get_agent_config('reasoning')['model']}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()