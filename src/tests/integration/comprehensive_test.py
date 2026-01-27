#!/usr/bin/env python3
"""
Comprehensive Echo Brain Functionality Test
Tests all endpoints and features to identify what's broken
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
        else:
            return False, f"HTTP {response.status_code}"

    except Exception as e:
        return False, str(e)


async def run_tests():
    """Run all tests"""

    async with httpx.AsyncClient() as client:

        print("=" * 60)
        print("ECHO BRAIN COMPREHENSIVE FUNCTIONALITY TEST")
        print("=" * 60)

        # 1. CORE ENDPOINTS
        print("\n1. CORE ENDPOINTS")
        print("-" * 40)

        tests = [
            ("GET", "/", "Core", None),
            ("GET", "/health", "Core", None),
            ("GET", "/api/status", "Core", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 2. CHAT & CONTEXT
        print("\n2. CHAT & CONTEXT")
        print("-" * 40)

        tests = [
            ("POST", "/api/echo/chat", "Chat", {"query": "Test", "conversation_id": "test"}),
            ("POST", "/api/context", "Context", {"query": "test", "limit": 5}),
            ("POST", "/api/context/unified", "Context", {"query": "test"}),
            ("GET", "/api/collections", "Context", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 3. ANIME MODULE
        print("\n3. ANIME MODULE")
        print("-" * 40)

        tests = [
            ("POST", "/api/echo/anime/scene/plan", "Anime",
             {"session_id": "test", "scene_description": "test scene"}),
            ("POST", "/api/echo/anime/prompt/refine", "Anime",
             {"session_id": "test", "raw_prompt": "test prompt"}),
            ("POST", "/api/echo/anime/feedback/learn", "Anime",
             {"session_id": "test", "prompt_used": "test", "quality_scores": {"test": 0.5}, "generation_id": "test"}),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 4. MODEL MANAGEMENT
        print("\n4. MODEL MANAGEMENT")
        print("-" * 40)

        tests = [
            ("GET", "/api/echo/models/list", "Models", None),
            ("GET", "/api/models/list", "Models", None),
            ("GET", "/api/models/manifests", "Models", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 5. DIAGNOSTICS
        print("\n5. DIAGNOSTICS")
        print("-" * 40)

        tests = [
            ("GET", "/api/diagnostics/health", "Diagnostics", None),
            ("GET", "/api/diagnostics/full", "Diagnostics", None),
            ("GET", "/api/diagnostics/component/database", "Diagnostics", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 6. AGENT SYSTEM
        print("\n6. AGENT SYSTEM")
        print("-" * 40)

        tests = [
            ("POST", "/api/agent", "Agent", {"task": "test task"}),
            ("GET", "/api/agents/status", "Agent", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 7. AUTONOMOUS
        print("\n7. AUTONOMOUS FEATURES")
        print("-" * 40)

        tests = [
            ("GET", "/api/autonomous/status", "Autonomous", None),
        ]

        for method, endpoint, feature, data in tests:
            success, details = await test_endpoint(client, method, endpoint, feature, data)
            log_result(feature, endpoint, "✅ PASS" if success else "❌ FAIL", details)

        # 8. Check file dependencies
        print("\n8. FILE DEPENDENCIES")
        print("-" * 40)

        import os

        paths_to_check = [
            ("/opt/tower-echo-brain/model-manifests", "Model manifests directory"),
            ("/opt/tower-echo-brain/config/qdrant-config.yaml", "Qdrant config"),
            ("/opt/tower-echo-brain/scripts/extract_facts_background.py", "Background script"),
            ("/opt/tower-echo-brain/mcp_server/main.py", "MCP server"),
            ("/opt/tower-echo-brain/src/improvement/continuous_learning.py", "Continuous learning"),
        ]

        for path, description in paths_to_check:
            exists = os.path.exists(path)
            log_result("Files", path, "✅ PASS" if exists else "❌ FAIL",
                      f"{description} - {'exists' if exists else 'MISSING'}")

        # 9. Check running processes
        print("\n9. RUNNING PROCESSES")
        print("-" * 40)

        import subprocess

        processes_to_check = [
            ("echo.*main:app", "Echo Brain main service"),
            ("mcp_server/main.py", "MCP server"),
            ("extract_facts_background.py", "Background fact extraction"),
            ("continuous_learning.py", "Continuous learning"),
            ("qdrant", "Qdrant vector database"),
        ]

        for pattern, description in processes_to_check:
            try:
                result = subprocess.run(
                    f"ps aux | grep -E '{pattern}' | grep -v grep",
                    shell=True, capture_output=True, text=True
                )
                running = bool(result.stdout.strip())
                log_result("Process", description, "✅ PASS" if running else "⚠️ WARN",
                          "Running" if running else "Not running")
            except Exception as e:
                log_result("Process", description, "❌ FAIL", str(e))

        # SUMMARY
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in RESULTS if "PASS" in r["status"])
        failed = sum(1 for r in RESULTS if "FAIL" in r["status"])
        warned = sum(1 for r in RESULTS if "WARN" in r["status"])

        print(f"Total tests: {len(RESULTS)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️ Warnings: {warned}")

        if failed > 0:
            print("\n❌ FAILED ENDPOINTS:")
            for r in RESULTS:
                if "FAIL" in r["status"]:
                    print(f"  - {r['feature']}: {r['endpoint']}")
                    if r['details']:
                        print(f"    → {r['details']}")

        # Save results
        with open("/opt/tower-echo-brain/TEST_RESULTS.json", "w") as f:
            json.dump(RESULTS, f, indent=2)

        print("\nResults saved to TEST_RESULTS.json")


if __name__ == "__main__":
    asyncio.run(run_tests())