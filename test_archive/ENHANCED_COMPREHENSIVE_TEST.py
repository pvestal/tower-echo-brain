#!/usr/bin/env python3
"""
Enhanced Comprehensive Echo Brain Functionality Test
Tests all endpoints including newly discovered ones
"""

import asyncio
import httpx
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple

BASE_URL = "http://localhost:8309"
RESULTS = []

def log_result(feature: str, endpoint: str, status: str, details: str = ""):
    """Log test result"""
    result = {
        "feature": feature,
        "endpoint": endpoint,
        "status": status,
        "details": details
    }
    RESULTS.append(result)

    # Print with color
    if status == "✅ PASS":
        print(f"\033[92m{status}\033[0m {feature}: {endpoint}")
    elif status == "❌ FAIL":
        print(f"\033[91m{status}\033[0m {feature}: {endpoint}")
        if details:
            print(f"     Details: {details}")
    else:
        print(f"\033[93m{status}\033[0m {feature}: {endpoint}")


async def test_endpoint(client: httpx.AsyncClient, method: str, endpoint: str,
                        feature: str, data: Dict = None) -> Tuple[bool, str]:
    """Test a single endpoint"""
    try:
        if method == "GET":
            response = await client.get(f"{BASE_URL}{endpoint}", timeout=10)
        elif method == "POST":
            response = await client.post(f"{BASE_URL}{endpoint}", json=data, timeout=10)
        else:
            return False, f"Unknown method: {method}"

        if response.status_code == 200:
            return True, "Success"
        elif response.status_code == 404:
            return False, f"Endpoint not found (404)"
        elif response.status_code == 500:
            return False, f"Internal Server Error (500)"
        else:
            return False, f"HTTP {response.status_code}"

    except Exception as e:
        return False, str(e)


async def run_enhanced_tests():
    """Run enhanced tests including newly discovered endpoints"""

    async with httpx.AsyncClient() as client:

        print("=" * 80)
        print("ECHO BRAIN ENHANCED COMPREHENSIVE FUNCTIONALITY TEST")
        print("=" * 80)

        # 1. CORE ENDPOINTS
        print("\n1. CORE ENDPOINTS")
        print("-" * 50)

        tests = [
            ("GET", "/", "Core", None),
            ("GET", "/health", "Core", None),
            ("GET", "/api/status", "Core", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 2. CHAT & CONTEXT (including new endpoints)
        print("\n2. CHAT & CONTEXT")
        print("-" * 50)

        tests = [
            # Original simple chat
            ("POST", "/api/chat/simple", "Chat", {"query": "Test", "conversation_id": "test"}),
            # Echo API chat (newly restored)
            ("POST", "/api/echo/chat", "Echo Chat", {"query": "Test", "conversation_id": "test"}),
            ("POST", "/api/echo/query", "Echo Query", {"query": "Test", "conversation_id": "test"}),
            # Context endpoints
            ("POST", "/api/context", "Context", {"query": "test", "limit": 5}),
            ("POST", "/api/context/unified", "Context", {"query": "test"}),
            ("GET", "/api/collections", "Context", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 3. AGENTS ROUTER (NEWLY ADDED)
        print("\n3. AGENTS SYSTEM (NEWLY MOUNTED)")
        print("-" * 50)

        tests = [
            ("GET", "/api/echo/agents/status", "Agents", None),
            ("GET", "/api/echo/agents/coding/history", "Agents", None),
            ("GET", "/api/echo/agents/reasoning/history", "Agents", None),
            ("GET", "/api/echo/agents/narration/history", "Agents", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 4. AUTONOMOUS SYSTEM (NEWLY ADDED)
        print("\n4. AUTONOMOUS OPERATIONS (NEWLY MOUNTED)")
        print("-" * 50)

        tests = [
            ("GET", "/api/autonomous/status", "Autonomous", None),
            ("GET", "/api/autonomous/goals", "Autonomous", None),
            ("GET", "/api/autonomous/tasks", "Autonomous", None),
            ("GET", "/api/autonomous/approvals", "Autonomous", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 5. CODEBASE INTERACTION (NEWLY ADDED)
        print("\n5. CODEBASE INTERACTION (NEWLY MOUNTED)")
        print("-" * 50)

        tests = [
            ("GET", "/api/echo/codebase/stats", "Codebase", None),
            ("GET", "/api/echo/codebase/search?query=test", "Codebase", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 6. ANIME MODULE
        print("\n6. ANIME MODULE")
        print("-" * 50)

        tests = [
            ("POST", "/api/echo/anime/scene/plan", "Anime",
             {"session_id": "test", "scene_description": "test scene"}),
            ("POST", "/api/echo/anime/prompt/refine", "Anime",
             {"session_id": "test", "raw_prompt": "test prompt"}),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 7. MODEL MANAGEMENT
        print("\n7. MODEL MANAGEMENT")
        print("-" * 50)

        tests = [
            ("GET", "/api/echo/models/list", "Models", None),
            ("GET", "/api/models/list", "Models", None),
            ("GET", "/api/models/manifests", "Models", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 8. DIAGNOSTICS
        print("\n8. DIAGNOSTICS")
        print("-" * 50)

        tests = [
            ("GET", "/api/diagnostics/health", "Diagnostics", None),
            ("GET", "/api/diagnostics/full", "Diagnostics", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # SUMMARY
        print("\n" + "=" * 80)
        print("ENHANCED TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in RESULTS if "PASS" in r["status"])
        failed = sum(1 for r in RESULTS if "FAIL" in r["status"])

        print(f"Total tests: {len(RESULTS)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success rate: {passed/len(RESULTS)*100:.1f}%")

        if failed > 0:
            print("\n❌ FAILED ENDPOINTS:")
            for r in RESULTS:
                if "FAIL" in r["status"]:
                    print(f"  - {r['feature']}: {r['endpoint']}")
                    if r['details']:
                        print(f"    → {r['details']}")

        # Save results
        with open("/opt/tower-echo-brain/ENHANCED_TEST_RESULTS.json", "w") as f:
            json.dump(RESULTS, f, indent=2)

        print(f"\nResults saved to ENHANCED_TEST_RESULTS.json")

        print("\n🔍 KEY DISCOVERIES:")
        print("- Mounted new routers: agents, autonomous, codebase")
        print("- Echo Brain now has significantly more capabilities")
        print("- Some new endpoints work, others need database setup")


if __name__ == "__main__":
    asyncio.run(run_enhanced_tests())