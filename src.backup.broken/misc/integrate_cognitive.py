#!/usr/bin/env python3
"""
Integration script to connect cognitive model selection to Echo Brain's main service
"""

import os
import sys
import asyncio
import json
from typing import Dict, Optional, Tuple

# Add the fixed model selector to path
sys.path.insert(0, '/opt/tower-echo-brain')
from archive.fixed-naming-cleanup-20251030.fixed_model_selector import ModelSelector

class CognitiveIntegration:
    """Integrates cognitive model selection into Echo's main intelligence router"""

    def __init__(self):
        self.model_selector = ModelSelector()
        self.ollama_url = "http://localhost:11434/api/generate"

    async def process_with_cognitive_selection(self, query: str, context: Optional[Dict] = None) -> Dict:
        """Process query with intelligent model selection"""

        # Select the optimal model
        model, level, reasoning = self.model_selector.select_model(query)

        print(f"🧠 Cognitive Selection: {model} ({level}) - {reasoning}")

        # Call Ollama with selected model
        import aiohttp

        payload = {
            "model": model,
            "prompt": query,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

        start_time = asyncio.get_event_loop().time()

        async with aiohttp.ClientSession() as session:
            async with session.post(self.ollama_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    processing_time = asyncio.get_event_loop().time() - start_time

                    return {
                        "response": result.get("response", ""),
                        "model_used": model,
                        "intelligence_level": level,
                        "selection_reason": reasoning,
                        "processing_time": processing_time,
                        "success": True
                    }
                else:
                    return {
                        "response": f"Error: Ollama returned {response.status}",
                        "model_used": model,
                        "intelligence_level": level,
                        "selection_reason": reasoning,
                        "processing_time": 0,
                        "success": False
                    }

# Monkey-patch the existing intelligence router
def patch_intelligence_router():
    """Patches the existing intelligence router to use cognitive selection"""

    try:
        # Import the existing intelligence module
        from src.core.intelligence import EchoIntelligenceRouter

        # Save original method
        original_process = EchoIntelligenceRouter.process_query

        # Create cognitive integration
        cognitive = CognitiveIntegration()

        # Create new process method
        async def cognitive_process_query(self, query: str, context: Dict = None,
                                         intelligence_level: str = "auto",
                                         user_id: str = None) -> Dict:
            """Enhanced process_query with cognitive model selection"""

            # If auto mode, use cognitive selection
            if intelligence_level == "auto":
                result = await cognitive.process_with_cognitive_selection(query, context)

                # Format response to match expected structure
                return {
                    "response": result["response"],
                    "model_used": result["model_used"],
                    "intelligence_level": result["intelligence_level"],
                    "processing_time": result["processing_time"],
                    "escalation_path": [result["model_used"]],
                    "requires_clarification": False,
                    "clarifying_questions": [],
                    "conversation_id": context.get("conversation_id") if context else None,
                    "intent": None,
                    "confidence": 0.95
                }
            else:
                # Use original method for manual selection
                return await original_process(self, query, context, intelligence_level, user_id)

        # Replace method
        EchoIntelligenceRouter.process_query = cognitive_process_query

        print("✅ Cognitive integration patched successfully!")
        return True

    except Exception as e:
        print(f"❌ Failed to patch: {e}")
        return False

if __name__ == "__main__":
    print("🧠 COGNITIVE INTEGRATION FOR ECHO BRAIN")
    print("=" * 50)

    # Test the cognitive selection
    async def test():
        cognitive = CognitiveIntegration()

        test_queries = [
            "Hi there!",
            "Think harder about the nature of reality",
            "Write a Python function to calculate fibonacci numbers"
        ]

        for query in test_queries:
            print(f"\nTesting: {query}")
            result = await cognitive.process_with_cognitive_selection(query)
            print(f"  Model: {result['model_used']}")
            print(f"  Level: {result['intelligence_level']}")
            print(f"  Time: {result['processing_time']:.2f}s")
            print(f"  Response: {result['response'][:100]}...")

    # Run test
    asyncio.run(test())

    # Apply patch
    print("\n" + "=" * 50)
    print("Applying cognitive patch to Echo Brain...")
    if patch_intelligence_router():
        print("✅ Integration complete!")
    else:
        print("❌ Integration failed!")