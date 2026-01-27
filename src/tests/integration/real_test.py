#!/usr/bin/env python3
"""
REAL Echo Brain Test Suite - No bullshit, no mocks, actual HTTP calls
Runs real tests against real endpoints and reports actual failures
"""

import asyncio
import httpx
import json
from datetime import datetime
import subprocess
import psutil
import os

BASE_URL = "http://localhost:8309"

class RealTest:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    async def test_endpoint(self, method, path, data=None, params=None, expected_status=None):
        """Actually test an endpoint - no faking"""
        url = f"{BASE_URL}{path}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if method == "GET":
                    response = await client.get(url, params=params)
                elif method == "POST":
                    response = await client.post(url, json=data)
                elif method == "PUT":
                    response = await client.put(url, json=data)
                elif method == "DELETE":
                    response = await client.delete(url)
                else:
                    return None, f"Unknown method: {method}"

                # Check if we got expected status
                if expected_status and response.status_code != expected_status:
                    return False, f"Expected {expected_status}, got {response.status_code}"

                # Try to parse JSON response
                try:
                    body = response.json()
                except:
                    body = response.text

                # Determine success
                success = response.status_code < 400

                return success, {
                    "status_code": response.status_code,
                    "body": body if success else str(body)[:200]
                }

        except httpx.TimeoutException:
            return False, "TIMEOUT after 10 seconds"
        except httpx.ConnectError as e:
            return False, f"CONNECTION FAILED: {str(e)}"
        except Exception as e:
            return False, f"ERROR: {str(e)}"

    def record_result(self, category, endpoint, method, success, details):
        """Record actual test result"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "endpoint": endpoint,
            "method": method,
            "success": success,
            "details": details
        }
        self.results.append(result)

        if success:
            self.passed += 1
            print(f"  ✓ {method} {endpoint}")
        else:
            self.failed += 1
            print(f"  ✗ {method} {endpoint}")
            if isinstance(details, dict):
                print(f"    → Status: {details.get('status_code', 'N/A')}")
            else:
                print(f"    → {details}")

async def main():
    print("=" * 80)
    print("REAL ECHO BRAIN TEST SUITE - ACTUAL HTTP CALLS")
    print("=" * 80)
    print(f"Started: {datetime.now()}")
    print(f"Target: {BASE_URL}")
    print()

    tester = RealTest()

    # 1. BASIC HEALTH CHECKS
    print("1. HEALTH & STATUS")
    print("-" * 40)

    success, details = await tester.test_endpoint("GET", "/")
    tester.record_result("Health", "/", "GET", success, details)

    success, details = await tester.test_endpoint("GET", "/health")
    tester.record_result("Health", "/health", "GET", success, details)

    success, details = await tester.test_endpoint("GET", "/api/status")
    tester.record_result("Health", "/api/status", "GET", success, details)

    # 2. CHAT ENDPOINTS - Test all variants
    print("\n2. CHAT ENDPOINTS")
    print("-" * 40)

    test_query = {"query": "test message", "conversation_id": "test123"}

    success, details = await tester.test_endpoint("POST", "/api/chat/simple", test_query)
    tester.record_result("Chat", "/api/chat/simple", "POST", success, details)

    success, details = await tester.test_endpoint("POST", "/api/echo/chat", test_query)
    tester.record_result("Chat", "/api/echo/chat", "POST", success, details)

    success, details = await tester.test_endpoint("POST", "/api/echo/query", test_query)
    tester.record_result("Chat", "/api/echo/query", "POST", success, details)

    # Code generation endpoint
    code_request = {"query": "write hello world", "mode": "code_only"}
    success, details = await tester.test_endpoint("POST", "/api/echo/code", code_request)
    tester.record_result("Chat", "/api/echo/code", "POST", success, details)

    # 3. CONTEXT & MEMORY
    print("\n3. CONTEXT & MEMORY")
    print("-" * 40)

    context_query = {"query": "test", "limit": 5}

    success, details = await tester.test_endpoint("POST", "/api/context", context_query)
    tester.record_result("Context", "/api/context", "POST", success, details)

    success, details = await tester.test_endpoint("POST", "/api/context/unified", context_query)
    tester.record_result("Context", "/api/context/unified", "POST", success, details)

    success, details = await tester.test_endpoint("GET", "/api/collections")
    tester.record_result("Context", "/api/collections", "GET", success, details)

    # 4. AGENT SYSTEM
    print("\n4. AGENT SYSTEM")
    print("-" * 40)

    success, details = await tester.test_endpoint("GET", "/api/echo/agents/status")
    tester.record_result("Agents", "/api/echo/agents/status", "GET", success, details)

    for agent_type in ["coding", "reasoning", "narration"]:
        endpoint = f"/api/echo/agents/{agent_type}/history"
        success, details = await tester.test_endpoint("GET", endpoint)
        tester.record_result("Agents", endpoint, "GET", success, details)

    # Test agent execution
    agent_task = {"task": "test task", "context": "test context"}
    success, details = await tester.test_endpoint("POST", "/api/echo/agents/coding", agent_task)
    tester.record_result("Agents", "/api/echo/agents/coding", "POST", success, details)

    # 5. MODEL MANAGEMENT
    print("\n5. MODEL MANAGEMENT")
    print("-" * 40)

    success, details = await tester.test_endpoint("GET", "/api/models/list")
    tester.record_result("Models", "/api/models/list", "GET", success, details)

    success, details = await tester.test_endpoint("GET", "/api/models/manifests")
    tester.record_result("Models", "/api/models/manifests", "GET", success, details)

    success, details = await tester.test_endpoint("GET", "/api/echo/models/list")
    tester.record_result("Models", "/api/echo/models/list", "GET", success, details)

    # 6. AUTONOMOUS SYSTEM
    print("\n6. AUTONOMOUS SYSTEM")
    print("-" * 40)

    success, details = await tester.test_endpoint("GET", "/api/autonomous/status")
    tester.record_result("Autonomous", "/api/autonomous/status", "GET", success, details)

    success, details = await tester.test_endpoint("GET", "/api/autonomous/goals")
    tester.record_result("Autonomous", "/api/autonomous/goals", "GET", success, details)

    success, details = await tester.test_endpoint("GET", "/api/autonomous/tasks")
    tester.record_result("Autonomous", "/api/autonomous/tasks", "GET", success, details)

    # 7. CODEBASE OPERATIONS
    print("\n7. CODEBASE OPERATIONS")
    print("-" * 40)

    success, details = await tester.test_endpoint("GET", "/api/echo/codebase/stats")
    tester.record_result("Codebase", "/api/echo/codebase/stats", "GET", success, details)

    success, details = await tester.test_endpoint("GET", "/api/echo/codebase/search", params={"q": "test"})
    tester.record_result("Codebase", "/api/echo/codebase/search?q=test", "GET", success, details)

    success, details = await tester.test_endpoint("GET", "/api/echo/codebase/file", params={"path": "src/main.py"})
    tester.record_result("Codebase", "/api/echo/codebase/file?path=src/main.py", "GET", success, details)

    # 8. DIAGNOSTICS
    print("\n8. DIAGNOSTICS")
    print("-" * 40)

    success, details = await tester.test_endpoint("GET", "/api/diagnostics/health")
    tester.record_result("Diagnostics", "/api/diagnostics/health", "GET", success, details)

    success, details = await tester.test_endpoint("GET", "/api/diagnostics/full")
    tester.record_result("Diagnostics", "/api/diagnostics/full", "GET", success, details)

    success, details = await tester.test_endpoint("GET", "/api/diagnostics/component/database")
    tester.record_result("Diagnostics", "/api/diagnostics/component/database", "GET", success, details)

    # 9. ANIME MODULE
    print("\n9. ANIME MODULE")
    print("-" * 40)

    scene_data = {"session_id": "test", "scene_description": "test scene"}
    success, details = await tester.test_endpoint("POST", "/api/echo/anime/scene/plan", scene_data)
    tester.record_result("Anime", "/api/echo/anime/scene/plan", "POST", success, details)

    prompt_data = {"session_id": "test", "raw_prompt": "test prompt"}
    success, details = await tester.test_endpoint("POST", "/api/echo/anime/prompt/refine", prompt_data)
    tester.record_result("Anime", "/api/echo/anime/prompt/refine", "POST", success, details)

    # 10. MCP ENDPOINTS
    print("\n10. MCP ENDPOINTS")
    print("-" * 40)

    # Test MCP server health
    mcp_url = "http://localhost:8312/health"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(mcp_url)
            success = response.status_code == 200
            details = {"status_code": response.status_code}
    except:
        success = False
        details = "MCP server not responding"
    tester.record_result("MCP", "localhost:8312/health", "GET", success, details)

    # 11. PROCESS & SERVICE CHECKS
    print("\n11. PROCESS & SERVICE CHECKS")
    print("-" * 40)

    # Check if Echo Brain process is running
    echo_brain_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'tower-echo-brain' in ' '.join(cmdline):
                echo_brain_running = True
                break
        except:
            pass

    tester.record_result("Process", "Echo Brain Service", "CHECK", echo_brain_running,
                         "Process running" if echo_brain_running else "Process NOT running")

    # Check Qdrant
    qdrant_url = "http://localhost:6333"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(qdrant_url)
            qdrant_running = response.status_code < 500
    except:
        qdrant_running = False

    tester.record_result("Process", "Qdrant", "CHECK", qdrant_running,
                         "Service running" if qdrant_running else "Service NOT running")

    # Check PostgreSQL
    try:
        result = subprocess.run(
            ["psql", "-h", "localhost", "-U", "patrick", "-d", "echo_brain", "-c", "SELECT 1"],
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "PGPASSWORD": os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")}
        )
        postgres_running = result.returncode == 0
    except:
        postgres_running = False

    tester.record_result("Process", "PostgreSQL", "CHECK", postgres_running,
                         "Database accessible" if postgres_running else "Database NOT accessible")

    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {tester.passed + tester.failed}")
    print(f"Passed: {tester.passed}")
    print(f"Failed: {tester.failed}")
    print(f"Success Rate: {(tester.passed / (tester.passed + tester.failed) * 100):.1f}%")

    # Save actual results to JSON
    output = {
        "test_run": {
            "timestamp": datetime.now().isoformat(),
            "base_url": BASE_URL,
            "total_tests": tester.passed + tester.failed,
            "passed": tester.passed,
            "failed": tester.failed,
            "success_rate": f"{(tester.passed / (tester.passed + tester.failed) * 100):.1f}%"
        },
        "results": tester.results,
        "failed_endpoints": [r for r in tester.results if not r["success"]]
    }

    with open("/opt/tower-echo-brain/ACTUAL_TEST_RESULTS.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to ACTUAL_TEST_RESULTS.json")

    # List failed endpoints
    if tester.failed > 0:
        print("\nFAILED ENDPOINTS:")
        for result in tester.results:
            if not result["success"]:
                print(f"  - {result['method']} {result['endpoint']}: {result['details']}")

if __name__ == "__main__":
    asyncio.run(main())