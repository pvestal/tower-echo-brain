#!/usr/bin/env python3
"""Test all Echo Brain endpoints and generate status report"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8309"

# Define all endpoints from OpenAPI
endpoints = [
    ("GET", "/health", None, "Basic health check"),
    ("GET", "/api", None, "API info"),
    ("GET", "/api/echo/health/detailed", None, "Detailed health with Qdrant stats"),
    ("POST", "/api/echo/ask", {"question": "test"}, "Ask Echo Brain"),
    ("GET", "/api/echo/system/logs", None, "System logs"),
    ("GET", "/api/echo/memory/status", None, "Memory status"),
    ("GET", "/api/echo/memory/health", None, "Memory health"),
    ("POST", "/api/echo/memory/search", {"query": "test", "limit": 1}, "Memory search"),
    ("GET", "/api/echo/intelligence/status", None, "Intelligence status"),
    ("GET", "/api/echo/ingestion/status", None, "Ingestion status"),
    ("GET", "/api/echo/knowledge/facts", None, "Knowledge facts"),
    ("GET", "/api/echo/knowledge/stats", None, "Knowledge stats"),
    ("GET", "/api/workers/status", None, "Workers status"),
    ("POST", "/mcp", {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1}, "MCP tools list"),
    ("GET", "/mcp/health", None, "MCP health"),
    ("GET", "/api/autonomous/status", None, "Autonomous status"),
    ("GET", "/api/pipeline/health", None, "Pipeline health"),
]

print(f"\n{'='*80}")
print(f"ECHO BRAIN ENDPOINT STATUS REPORT - {datetime.now()}")
print(f"{'='*80}\n")

working = []
broken = []

for method, path, data, description in endpoints:
    try:
        if method == "GET":
            r = requests.get(f"{BASE_URL}{path}", timeout=2)
        else:
            r = requests.post(f"{BASE_URL}{path}", json=data, timeout=2)

        status = "✅" if r.status_code == 200 else f"⚠️ {r.status_code}"

        # Check for error in response
        try:
            resp_data = r.json()
            if resp_data.get("error") or resp_data.get("status") == "error":
                status = f"❌ {resp_data.get('error', 'error status')[:50]}"
                broken.append((path, description, resp_data.get('error', 'error')))
            else:
                working.append((path, description))
        except:
            if r.status_code == 200:
                working.append((path, description))

        print(f"{status:4} {method:4} {path:40} {description}")

    except Exception as e:
        print(f"❌   {method:4} {path:40} {description} - {str(e)[:30]}")
        broken.append((path, description, str(e)))

print(f"\n{'='*80}")
print(f"SUMMARY: {len(working)} working, {len(broken)} broken")
print(f"{'='*80}")

print("\n✅ WORKING ENDPOINTS:")
for path, desc in working:
    print(f"  - {path:35} {desc}")

print("\n❌ BROKEN ENDPOINTS:")
for path, desc, error in broken:
    print(f"  - {path:35} {desc}")
    print(f"    Error: {error[:60]}")

if broken:
    print("\n📊 ISSUES TO FIX:")
    for i, (path, desc, error) in enumerate(broken, 1):
        print(f"{i}. {path}: {error[:80]}")
else:
    print("\n✨ ALL SYSTEMS OPERATIONAL - No issues detected!")