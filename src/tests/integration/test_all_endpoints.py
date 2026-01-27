#!/usr/bin/env python3
"""
Complete Echo Brain Endpoint Test Suite
Tests EVERY SINGLE endpoint discovered in the codebase
"""

import asyncio
import httpx
import json
from datetime import datetime
import subprocess
import os

BASE_URL = "http://localhost:8309"
MCP_URL = "http://localhost:8312"

# All endpoints discovered from source code
ENDPOINTS = [
    # Core Health
    ("GET", "/", None, "Core"),
    ("GET", "/health", None, "Core"),
    ("GET", "/ready", None, "Core"),
    ("GET", "/alive", None, "Core"),
    ("GET", "/metrics", None, "Core"),

    # Main API Status
    ("GET", "/api/status", None, "API"),

    # Echo API Routes
    ("POST", "/api/echo/query", {"query": "test", "conversation_id": "test"}, "Echo"),
    ("POST", "/api/echo/chat", {"query": "test", "conversation_id": "test"}, "Echo"),
    ("GET", "/api/echo/brain", None, "Echo"),
    ("GET", "/api/echo/thoughts/recent", None, "Echo"),
    ("GET", "/api/echo/thoughts/test123", None, "Echo"),
    ("GET", "/api/echo/conversation/test123", None, "Echo"),
    ("GET", "/api/echo/conversations", None, "Echo"),
    ("GET", "/api/echo/oversight/dashboard", None, "Echo"),
    ("GET", "/api/echo/users/patrick", None, "Echo"),
    ("POST", "/api/echo/users/patrick/preferences", {"theme": "dark"}, "Echo"),
    ("GET", "/api/echo/status", None, "Echo"),
    ("GET", "/api/echo/health", None, "Echo"),
    ("GET", "/api/echo/stats", None, "Echo"),
    ("GET", "/api/echo/tower/status", None, "Echo"),
    ("GET", "/api/echo/tower/health", None, "Echo"),
    ("POST", "/api/echo/code", {"query": "hello world", "mode": "code_only"}, "Echo"),

    # Chat endpoints
    ("POST", "/api/chat/simple", {"query": "test", "conversation_id": "test"}, "Chat"),

    # Context & Collections
    ("POST", "/api/context", {"query": "test", "limit": 5}, "Context"),
    ("POST", "/api/context/unified", {"query": "test"}, "Context"),
    ("GET", "/api/collections", None, "Context"),

    # Agent System
    ("GET", "/api/echo/agents/status", None, "Agents"),
    ("POST", "/api/echo/agents/coding", {"task": "test", "context": "test"}, "Agents"),
    ("GET", "/api/echo/agents/coding/history", None, "Agents"),
    ("POST", "/api/echo/agents/reasoning", {"task": "test", "context": "test"}, "Agents"),
    ("GET", "/api/echo/agents/reasoning/history", None, "Agents"),
    ("POST", "/api/echo/agents/narration", {"task": "test"}, "Agents"),
    ("GET", "/api/echo/agents/narration/history", None, "Agents"),
    ("POST", "/api/echo/agents/execute", {"agent_type": "coding", "task": "test"}, "Agents"),
    ("POST", "/api/echo/agents/narration/anime", {"scene": "test"}, "Agents"),

    # Model Management
    ("GET", "/api/models/list", None, "Models"),
    ("GET", "/api/models/manifests", None, "Models"),
    ("GET", "/api/echo/models/list", None, "Models"),
    ("POST", "/api/echo/models/pull/test-model", None, "Models"),
    ("DELETE", "/api/echo/models/test-model", None, "Models"),
    ("GET", "/api/echo/models/status/test-request", None, "Models"),

    # Autonomous System
    ("GET", "/api/autonomous/status", None, "Autonomous"),
    ("GET", "/api/autonomous/goals", None, "Autonomous"),
    ("POST", "/api/autonomous/goals", {"name": "test", "description": "test"}, "Autonomous"),
    ("PATCH", "/api/autonomous/goals/test-id", {"status": "active"}, "Autonomous"),
    ("DELETE", "/api/autonomous/goals/test-id", None, "Autonomous"),
    ("GET", "/api/autonomous/tasks", None, "Autonomous"),
    ("GET", "/api/autonomous/approvals", None, "Autonomous"),
    ("POST", "/api/autonomous/approvals/test-id/approve", None, "Autonomous"),
    ("POST", "/api/autonomous/approvals/test-id/reject", None, "Autonomous"),
    ("POST", "/api/autonomous/start", None, "Autonomous"),
    ("POST", "/api/autonomous/stop", None, "Autonomous"),
    ("POST", "/api/autonomous/pause", None, "Autonomous"),
    ("POST", "/api/autonomous/resume", None, "Autonomous"),
    ("POST", "/api/autonomous/kill", None, "Autonomous"),
    ("GET", "/api/autonomous/audit", None, "Autonomous"),
    ("GET", "/api/autonomous/notifications", None, "Autonomous"),
    ("GET", "/api/autonomous/notifications/count", None, "Autonomous"),
    ("POST", "/api/autonomous/notifications/test-id/read", None, "Autonomous"),
    ("POST", "/api/autonomous/notifications/mark-all-read", None, "Autonomous"),
    ("GET", "/api/autonomous/diagnosis", None, "Autonomous"),
    ("POST", "/api/autonomous/repair", None, "Autonomous"),
    ("POST", "/api/autonomous/cycle", None, "Autonomous"),

    # Codebase Operations
    ("GET", "/api/echo/codebase/search?q=test", None, "Codebase"),
    ("GET", "/api/echo/codebase/stats", None, "Codebase"),
    ("POST", "/api/echo/codebase/reindex", None, "Codebase"),
    ("GET", "/api/echo/codebase/file/src/main.py", None, "Codebase"),

    # Diagnostics
    ("GET", "/api/diagnostics/health", None, "Diagnostics"),
    ("GET", "/api/diagnostics/full", None, "Diagnostics"),
    ("GET", "/api/diagnostics/telegram", None, "Diagnostics"),
    ("GET", "/api/diagnostics/component/database", None, "Diagnostics"),

    # Anime Module
    ("POST", "/api/echo/anime/scene/plan", {"session_id": "test", "scene_description": "test"}, "Anime"),
    ("POST", "/api/echo/anime/prompt/refine", {"session_id": "test", "raw_prompt": "test"}, "Anime"),
    ("POST", "/api/echo/anime/feedback/learn", {"session_id": "test", "prompt_used": "test", "quality_scores": {"test": 0.5}, "generation_id": "test"}, "Anime"),
    ("GET", "/api/echo/anime/health", None, "Anime"),

    # Solutions
    ("GET", "/api/echo/solutions/search?query=test", None, "Solutions"),
    ("GET", "/api/echo/solutions/stats", None, "Solutions"),

    # Training Status
    ("GET", "/api/echo/training/status", None, "Training"),

    # Metrics
    ("GET", "/api/echo/metrics/system", None, "Metrics"),
    ("GET", "/api/echo/metrics/db", None, "Metrics"),

    # Takeout (stub)
    ("GET", "/api/echo/takeout/progress", None, "Takeout"),
    ("GET", "/api/echo/takeout/status", None, "Takeout"),
    ("GET", "/api/takeout/progress", None, "Takeout"),
    ("GET", "/api/takeout/status", None, "Takeout"),

    # Settings
    ("GET", "/settings", None, "Settings"),
    ("POST", "/settings", {"theme": "dark"}, "Settings"),

    # Integrations
    ("GET", "/integrations", None, "Integrations"),
    ("GET", "/integrations/github", None, "Integrations"),
    ("POST", "/integrations/github/connect", {"token": "test"}, "Integrations"),
    ("DELETE", "/integrations/github", None, "Integrations"),
    ("POST", "/integrations/github/sync", None, "Integrations"),

    # DeepSeek Coding Agent
    ("POST", "/analyze", {"code": "test code"}, "DeepSeek"),
    ("POST", "/generate", {"prompt": "test"}, "DeepSeek"),
    ("POST", "/generate-tests", {"code": "test"}, "DeepSeek"),
    ("POST", "/improve", {"code": "test"}, "DeepSeek"),
    ("GET", "/status", None, "DeepSeek"),
    ("POST", "/quick-fix", {"error": "test"}, "DeepSeek"),

    # Board API
    ("POST", "/deliberate", {"question": "test", "context": "test"}, "Board"),
    ("GET", "/history", None, "Board"),
    ("POST", "/override/test-id", None, "Board"),

    # Claude Bridge
    ("GET", "/session-context", None, "Claude"),
    ("POST", "/store-claude-context", {"context": "test"}, "Claude"),

    # Photo Comparison
    ("POST", "/api/photos/compare", {"photo1": "test", "photo2": "test"}, "Photos"),
    ("POST", "/api/photos/sync", None, "Photos"),

    # Home Assistant
    ("GET", "/home-assistant/status", None, "HomeAssistant"),
    ("GET", "/home-assistant/entities", None, "HomeAssistant"),
    ("GET", "/home-assistant/entities/light.living_room", None, "HomeAssistant"),
    ("POST", "/home-assistant/command", {"entity": "light.living_room", "action": "turn_on"}, "HomeAssistant"),
    ("POST", "/home-assistant/query", {"query": "temperature"}, "HomeAssistant"),

    # Secured Routes
    ("POST", "/auth/login", {"username": "test", "password": "test"}, "Auth"),
    ("POST", "/auth/refresh", {"token": "test"}, "Auth"),
    ("POST", "/auth/logout", None, "Auth"),
    ("POST", "/query", {"query": "test"}, "Secured"),
    ("POST", "/chat", {"message": "test"}, "Secured"),
    ("POST", "/execute", {"command": "test"}, "Secured"),
    ("POST", "/models/manage", {"action": "list"}, "Secured"),
    ("POST", "/files/analyze", {"path": "test"}, "Secured"),
    ("GET", "/system/metrics", None, "Secured"),
    ("GET", "/git/status", None, "Secured"),
    ("POST", "/git/commit", {"message": "test"}, "Secured"),
    ("GET", "/user/info", None, "Secured"),
    ("GET", "/user/rate-limit", None, "Secured"),
    ("GET", "/security", None, "Secured"),

    # Delegation Routes
    ("POST", "/api/delegation/to-tower", {"task": "test"}, "Delegation"),
    ("GET", "/api/delegation/capabilities", None, "Delegation"),
    ("GET", "/api/delegation/history", None, "Delegation"),
    ("POST", "/api/delegation/test", {"test": "test"}, "Delegation"),

    # Notification API
    ("GET", "/api/notifications/status", None, "Notifications"),
    ("POST", "/api/notifications/send", {"message": "test"}, "Notifications"),
    ("POST", "/api/notifications/send/bulk", {"messages": ["test1", "test2"]}, "Notifications"),
    ("POST", "/api/notifications/send/info", {"message": "test"}, "Notifications"),
    ("POST", "/api/notifications/send/warning", {"message": "test"}, "Notifications"),
    ("POST", "/api/notifications/send/error", {"message": "test"}, "Notifications"),
    ("POST", "/api/notifications/send/success", {"message": "test"}, "Notifications"),
    ("POST", "/api/notifications/test", None, "Notifications"),
    ("GET", "/api/notifications/channels", None, "Notifications"),

    # DB Metrics
    ("GET", "/api/db/health", None, "DBMetrics"),
    ("GET", "/api/db/pool/status", None, "DBMetrics"),
    ("GET", "/api/db/queries/slow", None, "DBMetrics"),
    ("GET", "/api/db/queries/stats", None, "DBMetrics"),
    ("GET", "/api/db/optimization/recommendations", None, "DBMetrics"),
    ("GET", "/api/db/analytics/interactions", None, "DBMetrics"),
    ("POST", "/api/db/maintenance/optimize", None, "DBMetrics"),
    ("POST", "/api/db/maintenance/cleanup", None, "DBMetrics"),
    ("POST", "/api/db/cache/invalidate", None, "DBMetrics"),
    ("GET", "/api/db/dashboard", None, "DBMetrics"),

    # Git Operations
    ("GET", "/api/git/status", None, "Git"),
    ("POST", "/api/git/commit", {"message": "test"}, "Git"),
    ("POST", "/api/git/branch", {"name": "test"}, "Git"),
    ("POST", "/api/git/pr", {"title": "test", "body": "test"}, "Git"),
    ("GET", "/api/git/tower/status", None, "Git"),
    ("POST", "/api/git/tower/sync", None, "Git"),
    ("GET", "/api/git/github/status", None, "Git"),
    ("POST", "/api/git/automation/enable", None, "Git"),
    ("POST", "/api/git/automation/disable", None, "Git"),
    ("GET", "/api/git/health", None, "Git"),
    ("POST", "/api/git/autonomous/quality-pr", None, "Git"),
    ("GET", "/api/git/logs", None, "Git"),

    # Vault
    ("GET", "/api/vault/keys", None, "Vault"),
    ("POST", "/api/vault/keys", {"name": "test", "value": "secret"}, "Vault"),
    ("DELETE", "/api/vault/keys/test", None, "Vault"),
    ("POST", "/api/vault/keys/test/test", None, "Vault"),
    ("GET", "/api/vault/health", None, "Vault"),

    # Model Manager
    ("GET", "/api/models/logs", None, "ModelManager"),
    ("GET", "/api/models/downloads/status", None, "ModelManager"),
    ("GET", "/api/models", None, "ModelManager"),
    ("GET", "/api/models/test/path", None, "ModelManager"),
    ("GET", "/api/models/test", None, "ModelManager"),
    ("POST", "/api/models/verify", None, "ModelManager"),
    ("POST", "/api/models/download", {"name": "test"}, "ModelManager"),
    ("POST", "/api/models/sync-manifests", None, "ModelManager"),
    ("POST", "/api/models/sync-ollama", None, "ModelManager"),

    # Knowledge
    ("GET", "/api/knowledge/facts", None, "Knowledge"),
    ("GET", "/api/knowledge/facts/test-id", None, "Knowledge"),
    ("POST", "/api/knowledge/facts", {"subject": "test", "fact": "test"}, "Knowledge"),
    ("PUT", "/api/knowledge/facts/test-id", {"fact": "updated"}, "Knowledge"),
    ("DELETE", "/api/knowledge/facts/test-id", None, "Knowledge"),
    ("GET", "/api/knowledge/subjects", None, "Knowledge"),
    ("GET", "/api/knowledge/about/patrick", None, "Knowledge"),

    # Tasks
    ("POST", "/api/tasks/execute", {"task": "test"}, "Tasks"),
    ("POST", "/api/tasks/queue", {"task": "test"}, "Tasks"),
    ("GET", "/api/tasks/status/test-id", None, "Tasks"),
    ("GET", "/api/tasks/results", None, "Tasks"),
    ("POST", "/api/tasks/anime/generate", {"prompt": "test"}, "Tasks"),
    ("POST", "/api/tasks/anime/scene", {"scene": "test"}, "Tasks"),

    # Resilience Status
    ("GET", "/api/resilience/health", None, "Resilience"),
    ("GET", "/api/resilience/services", None, "Resilience"),
    ("GET", "/api/resilience/services/echo-brain", None, "Resilience"),
    ("GET", "/api/resilience/circuit-breakers", None, "Resilience"),
    ("POST", "/api/resilience/circuit-breakers/echo-brain/reset", None, "Resilience"),
    ("POST", "/api/resilience/circuit-breakers/echo-brain/force-open", None, "Resilience"),
    ("POST", "/api/resilience/circuit-breakers/reset-all", None, "Resilience"),
    ("POST", "/api/resilience/test/echo-brain", None, "Resilience"),
    ("GET", "/api/resilience/fallback-cache", None, "Resilience"),
    ("POST", "/api/resilience/fallback-cache/clear", None, "Resilience"),
    ("GET", "/api/resilience/events", None, "Resilience"),
    ("GET", "/api/resilience/metrics/prometheus", None, "Resilience"),
    ("POST", "/api/resilience/monitoring/start", None, "Resilience"),
    ("POST", "/api/resilience/monitoring/stop", None, "Resilience"),
    ("GET", "/api/resilience/monitoring/status", None, "Resilience"),

    # Google Calendar
    ("GET", "/api/calendar/status", None, "Calendar"),
    ("GET", "/api/calendar/calendars", None, "Calendar"),
    ("GET", "/api/calendar/events/upcoming", None, "Calendar"),
    ("GET", "/api/calendar/events/today", None, "Calendar"),
    ("GET", "/api/calendar/events/date/2026-01-25", None, "Calendar"),
    ("POST", "/api/calendar/events", {"title": "test", "date": "2026-01-25"}, "Calendar"),
    ("POST", "/api/calendar/query", {"query": "meetings"}, "Calendar"),

    # Preferences
    ("GET", "/api/preferences", None, "Preferences"),
    ("GET", "/api/preferences/ui", None, "Preferences"),
    ("POST", "/api/preferences", {"category": "ui", "value": "dark"}, "Preferences"),
    ("PUT", "/api/preferences/test-id", {"value": "updated"}, "Preferences"),
    ("DELETE", "/api/preferences/test-id", None, "Preferences"),

    # Repair API
    ("GET", "/api/repair/status", None, "Repair"),
    ("POST", "/api/repair/trigger", None, "Repair"),
    ("GET", "/api/repair/history", None, "Repair"),
    ("POST", "/api/repair/diagnose", None, "Repair"),

    # Coordination & Theater
    ("GET", "/api/coordination/services", None, "Coordination"),
    ("GET", "/api/theater/agents", None, "Theater"),

    # Agent endpoints
    ("GET", "/api/agent", None, "Agent"),
    ("GET", "/api/agents/status", None, "Agent"),

    # Testing endpoints
    ("POST", "/test/echo-brain", None, "Testing"),
]

async def test_endpoint(client, method, path, data=None):
    """Test a single endpoint"""
    url = f"{BASE_URL}{path}"

    try:
        if method == "GET":
            response = await client.get(url)
        elif method == "POST":
            response = await client.post(url, json=data)
        elif method == "PUT":
            response = await client.put(url, json=data)
        elif method == "DELETE":
            response = await client.delete(url)
        elif method == "PATCH":
            response = await client.patch(url, json=data)
        else:
            return 999, "Unknown method"

        return response.status_code, None
    except httpx.TimeoutException:
        return 0, "TIMEOUT"
    except httpx.ConnectError:
        return 0, "CONNECT_ERROR"
    except Exception as e:
        return 0, str(e)

async def test_mcp_endpoints():
    """Test MCP server endpoints"""
    results = []

    async with httpx.AsyncClient(timeout=5.0) as client:
        # MCP health
        try:
            response = await client.get(f"{MCP_URL}/health")
            results.append(("MCP", "/health", response.status_code))
        except:
            results.append(("MCP", "/health", 0))

        # MCP search
        try:
            response = await client.post(f"{MCP_URL}/mcp", json={
                "method": "tools/call",
                "params": {
                    "name": "search_memory",
                    "arguments": {"query": "test", "limit": 5}
                }
            })
            results.append(("MCP", "/mcp search_memory", response.status_code))
        except:
            results.append(("MCP", "/mcp search_memory", 0))

        # MCP get facts
        try:
            response = await client.post(f"{MCP_URL}/mcp", json={
                "method": "tools/call",
                "params": {
                    "name": "get_facts",
                    "arguments": {"topic": "patrick"}
                }
            })
            results.append(("MCP", "/mcp get_facts", response.status_code))
        except:
            results.append(("MCP", "/mcp get_facts", 0))

        # MCP store fact
        try:
            response = await client.post(f"{MCP_URL}/mcp", json={
                "method": "tools/call",
                "params": {
                    "name": "store_fact",
                    "arguments": {"fact": "test fact", "category": "test"}
                }
            })
            results.append(("MCP", "/mcp store_fact", response.status_code))
        except:
            results.append(("MCP", "/mcp store_fact", 0))

    return results

async def main():
    print("=" * 80)
    print("COMPLETE ECHO BRAIN ENDPOINT TEST")
    print(f"Testing {len(ENDPOINTS)} endpoints")
    print("=" * 80)

    results = []
    status_counts = {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        for method, path, data, category in ENDPOINTS:
            status, error = await test_endpoint(client, method, path, data)

            # Track status codes
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1

            # Record result
            results.append({
                "category": category,
                "method": method,
                "endpoint": path,
                "status": status,
                "error": error
            })

            # Print result
            if status == 200:
                symbol = "✓"
            elif status == 404:
                symbol = "?"
            elif status == 500:
                symbol = "✗"
            elif status == 0:
                symbol = "⚠"
            else:
                symbol = "•"

            print(f"{symbol} [{status:3d}] {method:6s} {path[:60]:<60} ({category})")

    # Test MCP endpoints
    print("\nTesting MCP Server endpoints...")
    mcp_results = await test_mcp_endpoints()
    for category, endpoint, status in mcp_results:
        status_counts[status] = status_counts.get(status, 0) + 1
        results.append({
            "category": category,
            "method": "POST",
            "endpoint": endpoint,
            "status": status,
            "error": None if status > 0 else "Connection failed"
        })
        symbol = "✓" if status == 200 else "✗"
        print(f"{symbol} [{status:3d}] {category}: {endpoint}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    working = sum(1 for r in results if r["status"] in [200, 201, 204])
    not_found = sum(1 for r in results if r["status"] == 404)
    errors = sum(1 for r in results if r["status"] >= 500)
    failed = sum(1 for r in results if r["status"] == 0)

    print(f"Total endpoints tested: {total}")
    print(f"Working (2xx): {working} ({working/total*100:.1f}%)")
    print(f"Not found (404): {not_found} ({not_found/total*100:.1f}%)")
    print(f"Server errors (5xx): {errors} ({errors/total*100:.1f}%)")
    print(f"Connection failed: {failed} ({failed/total*100:.1f}%)")

    print("\nStatus code distribution:")
    for status in sorted(status_counts.keys()):
        count = status_counts[status]
        print(f"  {status}: {count} endpoints")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "working": working,
            "not_found": not_found,
            "errors": errors,
            "failed": failed
        },
        "status_distribution": status_counts,
        "results": results
    }

    with open("/opt/tower-echo-brain/COMPLETE_ENDPOINT_TEST_RESULTS.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to COMPLETE_ENDPOINT_TEST_RESULTS.json")

    # List all failed endpoints
    print("\nFAILED ENDPOINTS (status >= 500 or connection failed):")
    for r in results:
        if r["status"] >= 500 or r["status"] == 0:
            print(f"  [{r['status']}] {r['method']} {r['endpoint']}")
            if r["error"]:
                print(f"       Error: {r['error']}")

if __name__ == "__main__":
    asyncio.run(main())