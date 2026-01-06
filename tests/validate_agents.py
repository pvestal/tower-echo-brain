#!/usr/bin/env python3
"""
Direct validation of agents - see what they actually produce
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://localhost:8309"

def test_coding_agent():
    """Test CodingAgent with Tower-specific task"""
    print("\n" + "="*60)
    print("TESTING CODING AGENT")
    print("="*60)

    task = {
        "task": """Write a Python function to monitor Echo Brain agents. The function should:
        1. Check /api/echo/agents/status endpoint
        2. Parse response to get agent names and task counts
        3. Alert if any agent has 0 tasks processed (might be broken)
        4. Return a health score 0-100 based on agent activity""",
        "language": "python",
        "validate": True
    }

    start = time.time()
    try:
        response = requests.post(f"{BASE_URL}/api/echo/agents/coding", json=task, timeout=30)
        result = response.json()
        elapsed = time.time() - start

        print(f"Response Time: {elapsed:.2f}s")
        print(f"Code Valid: {result.get('validation', {}).get('valid', False)}")
        print(f"Code Length: {len(result.get('code', ''))} chars")
        print(f"Context Used: {result.get('context_used', {})}")

        if result.get('code'):
            print("\n--- Generated Code Preview (first 500 chars) ---")
            print(result['code'][:500])

            # Try to execute it
            try:
                exec(result['code'])
                print("\n✅ Code executes without errors")
            except Exception as e:
                print(f"\n❌ Execution error: {e}")

        return result

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def test_reasoning_agent():
    """Test ReasoningAgent with Tower-specific analysis"""
    print("\n" + "="*60)
    print("TESTING REASONING AGENT")
    print("="*60)

    task = {
        "task": "Analyze Echo Brain's architecture. Should we keep agents as singletons or use instance-per-request? Consider memory usage, concurrent requests, and state management.",
        "background": "Currently using singleton pattern with history tracking",
        "constraints": "Running on Tower server with 128GB RAM, handling ~100 req/s peak"
    }

    start = time.time()
    try:
        response = requests.post(f"{BASE_URL}/api/echo/agents/reasoning", json=task, timeout=90)
        result = response.json()
        elapsed = time.time() - start

        print(f"Response Time: {elapsed:.2f}s")
        print(f"Model: {result.get('model', 'unknown')}")
        print(f"Has Analysis: {'analysis' in result}")
        print(f"Has Reasoning: {'reasoning' in result}")
        print(f"Has Conclusion: {'conclusion' in result}")

        if result.get('conclusion'):
            print("\n--- Conclusion ---")
            print(result['conclusion'][:500])

            # Check if it actually addresses the question
            conclusion_lower = result['conclusion'].lower()
            if 'singleton' in conclusion_lower or 'instance' in conclusion_lower:
                print("\n✅ Conclusion addresses the specific question")
            else:
                print("\n⚠️  Conclusion might be generic")

        return result

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def test_narration_agent():
    """Test NarrationAgent with anime scene"""
    print("\n" + "="*60)
    print("TESTING NARRATION AGENT")
    print("="*60)

    task = {
        "scene": "A young programmer discovers their code has become sentient and is trying to improve itself",
        "genre": "sci-fi",
        "mood": "mysterious"
    }

    start = time.time()
    try:
        response = requests.post(f"{BASE_URL}/api/echo/agents/narration", json=task, timeout=30)
        result = response.json()
        elapsed = time.time() - start

        print(f"Response Time: {elapsed:.2f}s")
        print(f"Model: {result.get('model', 'unknown')}")
        print(f"Has Narration: {len(result.get('narration', '')) > 0}")
        print(f"Has Mood: {result.get('mood') is not None}")
        print(f"Has Visual Notes: {result.get('visual_notes') is not None}")
        print(f"Has ComfyUI Prompt: {result.get('comfyui_prompt') is not None}")

        if result.get('narration'):
            print("\n--- Narration Preview (first 300 chars) ---")
            print(result['narration'][:300])

        if result.get('comfyui_prompt'):
            print("\n--- ComfyUI Prompt Preview ---")
            print(result['comfyui_prompt'][:200])
            print("\n✅ ComfyUI prompt generated")
        else:
            print("\n⚠️  No ComfyUI prompt generated")

        return result

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def test_agent_history():
    """Check if agents are tracking their history"""
    print("\n" + "="*60)
    print("TESTING AGENT HISTORY TRACKING")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/api/echo/agents/status", timeout=10)
        result = response.json()

        print("Agent Status:")
        for agent in result.get('agents', []):
            name = agent['name']
            model = agent['model']
            tasks = agent['tasks_processed']
            status = "✅" if tasks > 0 else "⚠️"
            print(f"{status} {name}: {tasks} tasks processed (model: {model})")

        total_tasks = sum(a['tasks_processed'] for a in result.get('agents', []))
        print(f"\nTotal tasks processed across all agents: {total_tasks}")

        return result

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def test_concurrent_execution():
    """Test if agents can handle concurrent requests"""
    print("\n" + "="*60)
    print("TESTING CONCURRENT EXECUTION")
    print("="*60)

    import concurrent.futures
    import threading

    def call_agent(agent_type, task):
        url = f"{BASE_URL}/api/echo/agents/{agent_type}"
        try:
            start = time.time()
            response = requests.post(url, json=task, timeout=60)
            elapsed = time.time() - start
            return (agent_type, response.status_code, elapsed)
        except Exception as e:
            return (agent_type, "error", str(e))

    tasks = [
        ("coding", {"task": "Write a hello world function", "language": "python"}),
        ("reasoning", {"task": "What is 2+2 and why?"}),
        ("narration", {"scene": "A quiet forest", "genre": "fantasy"})
    ]

    print("Launching 3 concurrent agent requests...")
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(call_agent, t[0], t[1]) for t in tasks]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    total_time = time.time() - start

    print(f"\nTotal time for 3 concurrent requests: {total_time:.2f}s")
    for agent, status, time_or_error in results:
        if status == 200:
            print(f"✅ {agent}: {time_or_error:.2f}s")
        else:
            print(f"❌ {agent}: {status} - {time_or_error}")

    if all(r[1] == 200 for r in results):
        print("\n✅ All agents handled concurrent requests successfully")
    else:
        print("\n❌ Some agents failed under concurrent load")


def test_database_context():
    """Test if CodingAgent uses database context"""
    print("\n" + "="*60)
    print("TESTING DATABASE CONTEXT USAGE")
    print("="*60)

    task = {
        "task": "Write a function to search past_solutions table for Python-related fixes",
        "language": "python",
        "validate": True,
        "requirements": "Use psycopg2, connect to tower_consolidated database"
    }

    try:
        response = requests.post(f"{BASE_URL}/api/echo/agents/coding", json=task, timeout=30)
        result = response.json()

        code = result.get('code', '')
        context = result.get('context_used', {})

        print(f"Past solutions used: {context.get('past_solutions', 0)}")
        print(f"Codebase refs used: {context.get('codebase_refs', 0)}")

        # Check if code actually references the database
        if 'past_solutions' in code:
            print("✅ Code references past_solutions table")
        else:
            print("⚠️  Code doesn't reference past_solutions table")

        if 'psycopg2' in code or 'tower_consolidated' in code:
            print("✅ Code uses correct database connection")
        else:
            print("⚠️  Code doesn't use expected database connection")

        return result

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def main():
    print("\n" + "="*60)
    print("ECHO BRAIN AGENT VALIDATION SUITE")
    print(f"Started: {datetime.now()}")
    print("="*60)

    # Check service is running
    try:
        response = requests.get(f"{BASE_URL}/api/echo/agents/status", timeout=5)
        if response.status_code != 200:
            print("❌ Echo Brain service not responding properly")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot reach Echo Brain service: {e}")
        sys.exit(1)

    print("✅ Echo Brain service is running")

    # Run all tests
    tests = [
        test_coding_agent,
        test_reasoning_agent,
        test_narration_agent,
        test_agent_history,
        test_concurrent_execution,
        test_database_context
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result is not None))
        except Exception as e:
            print(f"\n❌ Test {test.__name__} crashed: {e}")
            results.append((test.__name__, False))

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} tests passed ({100*passed//total}%)")

    if passed == total:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
    elif passed >= total * 0.8:
        print("\n⚠️  Most tests passed but some issues remain")
    else:
        print("\n❌ Multiple failures detected - agents need fixing")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)