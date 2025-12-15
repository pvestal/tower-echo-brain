#!/usr/bin/env python3
"""
Echo Brain Multi-LLM Collaboration Demo
Demonstrates real-time collaboration between qwen-coder and deepseek-coder models
"""

import asyncio
import json
import sys
import time
from datetime import datetime

# Add src to path
sys.path.append('/opt/tower-echo-brain')

from src.collaboration_framework import collaborate_on_query

async def demo_collaboration():
    """Demonstrate the multi-LLM collaboration framework"""

    print("🧠 Echo Brain Multi-LLM Collaboration Framework Demo")
    print("=" * 60)
    print()

    # Test queries that benefit from collaboration
    test_queries = [
        "Create a Python function to implement binary search with error handling",
        "Design a simple web scraper for extracting product prices",
        "Write a script that analyzes CSV data and generates summary statistics"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"🎯 Test {i}: {query}")
        print("-" * 50)

        start_time = time.time()

        try:
            # Execute collaboration
            result = await collaborate_on_query(query)

            # Display results
            print(f"✅ Collaboration completed in {result.collaboration_time:.2f}s")
            print(f"🎭 Models involved: {', '.join(set(r.model for r in result.responses))}")
            print(f"📊 Confidence score: {result.confidence_score:.1f}%")
            print(f"🔍 Phases completed: {len(result.phases_completed)}")
            print(f"⚠️  Fabrication detected: {'Yes' if result.fabrication_detected else 'No'}")

            if result.inquisitive_validation:
                print(f"🔍 Inquisitive validation: {result.inquisitive_validation[:100]}...")

            print()
            print("📝 Consensus Summary:")
            print(result.consensus[:500] + "..." if len(result.consensus) > 500 else result.consensus)

            print()
            print("🤝 Individual Model Responses:")
            for response in result.responses:
                print(f"  • {response.model} ({response.phase.value}): {response.confidence}% confidence")
                print(f"    {response.response[:150]}...")
                print()

        except Exception as e:
            print(f"❌ Collaboration failed: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*60 + "\n")

        if i < len(test_queries):
            print("⏳ Waiting 5 seconds before next test...")
            await asyncio.sleep(5)

    print("🎉 Demo completed!")

    # Show backup file if database failed
    try:
        with open("/tmp/collaboration_backup.jsonl", "r") as f:
            lines = f.readlines()
            if lines:
                print(f"\n💾 {len(lines)} collaboration results saved to backup file")
                print("   (Database connection issue - results preserved locally)")
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    print("Starting Echo Brain Collaboration Demo...")
    asyncio.run(demo_collaboration())