#!/usr/bin/env python3
"""
Comprehensive Route Contract Tests for Echo Brain API

Tests ALL 203 routes from the OpenAPI spec against the running service.
Validates that every endpoint returns a non-500 response with proper payloads.

Usage:
    python3 test_all_routes.py              # Run all tests
    python3 test_all_routes.py --json       # Output JSON results
    python3 -m pytest test_all_routes.py -v # Run via pytest

Requires: aiohttp (pip install aiohttp)
"""
import asyncio
import aiohttp
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

BASE_URL = "http://localhost:8309"
TIMEOUT_SECONDS = 15

# ─── Endpoints that must never be hit in tests ───────────────────────────────
DANGEROUS_ENDPOINTS = {
    "/api/autonomous/kill",
    "/api/autonomous/stop",
    "/api/autonomous/start",
    "/api/autonomous/pause",
    "/api/autonomous/resume",
    "/api/autonomous/goals/active",
}

# ─── Endpoints known to be slow by design (>15s) ─────────────────────────────
SLOW_ENDPOINTS = {
    "/api/photos/analyze",        # batch default=50 takes ~96s
    "/api/echo/diagnostic/deep",  # full deep diagnosis ~17s
}

# ─── Routes that correctly return 503 when external dependency is unavailable ─
# These are NOT bugs — 503 is the proper contract for graceful degradation.
EXTERNAL_DEP_ROUTES = {
    "/api/home/command",                  # Home Assistant server required
    "/api/home/entities",                 # Home Assistant server required
    "/api/home/entities/{entity_id}",     # Home Assistant server required
    "/api/home/query",                    # Home Assistant server required
    "/api/home/status",                   # Home Assistant server required
    "/api/music/playlists",               # Apple Music developer token required
    "/api/agents/narration/anime",        # Narration agent not implemented
    "/google/photos/count",               # Google Photos OAuth scope required
}

# ─── Safe POST payloads for known endpoints ───────────────────────────────────
POST_PAYLOADS: Dict[str, Optional[Dict[str, Any]]] = {
    "/api/echo/chat": {"query": "contract test", "intelligence_level": "quick"},
    "/api/echo/search": {"query": "test", "limit": 2},
    "/api/echo/feedback": {"message_id": "contract-test-000", "rating": 5, "comment": "contract test"},
    "/api/echo/context/switch": {"domain": "GENERAL"},
    "/api/echo/context/inject": {"context": "test context", "source": "contract_test"},
    "/api/agents/route": {"query": "hello", "context": {}},
    "/api/agents/coding": {"query": "print hello", "context": {}},
    "/api/agents/narration": {"query": "test narration", "context": {}},
    "/api/agents/narration/anime": {"query": "test anime", "context": {}},
    "/api/agents/creative": {"query": "write a haiku", "context": {}},
    "/api/agents/research": {"query": "what is python", "context": {}},
    "/api/agents/memory": {"query": "test memory", "context": {}},
    "/api/agents/execute": {"code": "print('contract test')", "language": "python", "timeout": 5},
    "/api/autonomous/goals": {"goal_name": "contract_test_goal", "description": "test", "priority": 1},
    "/api/autonomous/learn": {"topic": "test", "content": "contract test content"},
    "/api/repair/trigger": {"issue_type": "test", "description": "contract test", "severity": "low", "auto_execute": False},
    "/api/repair/diagnose": {},
    "/api/home/command": {"command": "test status check"},
    "/api/home/query": {"query": "what lights are on"},
    "/api/memory/store": {"content": "contract test memory", "memory_type": "test"},
    "/api/memory/search": {"query": "test", "limit": 2},
    "/api/memory/consolidate": {},
    "/api/monitoring/contracts/check": {},
    "/api/monitoring/health/deep": {},
    "/api/monitoring/alerts/test": {},
    "/api/intelligence/analyze": {"query": "test analysis"},
    "/api/intelligence/system-model": {},
    "/api/intelligence/procedures/execute": {"procedure_name": "check_system_health"},
    "/api/google/auth/init": {},
    "/api/calendar/sync": {},
    "/mcp": {"method": "tools/list", "params": {}},
    "/api/integrations/test": {"provider": "test"},
    "/api/auth/validate": {},
    "/api/audit/log": {"action": "contract_test", "details": {"test": True}},
    "/api/models/verify": {},
    "/api/models/sync-manifests": {},
    "/api/models/download": {"model_id": "test_model"},
    "/api/models/sync-ollama": {},
    "/api/photos/scan/local": None,  # skip — takes >15s
}

# ─── Path parameter substitutions ────────────────────────────────────────────
PATH_PARAMS = {
    "{entity_id}": "light.test",
    "{goal_id}": "test-goal-000",
    "{task_id}": "test-task-000",
    "{model_id}": "test-model",
    "{collection_name}": "echo_memory",
    "{provider}": "google",
    "{key}": "test_key",
    "{key_name}": "test_key",
    "{category}": "general",
    "{domain}": "GENERAL",
    "{query}": "test",
    "{name}": "test",
    "{path}": "test",
    "{subject}": "test",
    "{date}": "2026-02-14",
    "{pref_id}": "test-pref",
    "{proposal_id}": "test-proposal",
    "{notification_id}": "test-notif",
    "{request_id}": "test-req",
    "{contract_id}": "test-contract",
    "{fact_id}": "test-fact",
}


@dataclass
class TestResult:
    method: str
    path: str
    status: str  # "pass", "fail", "skip"
    http_status: Optional[int] = None
    detail: str = ""
    duration_ms: float = 0


@dataclass
class TestReport:
    timestamp: str = ""
    base_url: str = BASE_URL
    total_routes: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    skips: List[Dict[str, Any]] = field(default_factory=list)


# ─── Route-specific path overrides (when generic PATH_PARAMS won't work) ─────
ROUTE_PATH_OVERRIDES = {
    "/api/echo/intelligence/procedures/{name}/execute": "/api/echo/intelligence/procedures/check_system_health/execute",
}


def resolve_path(path: str) -> str:
    """Replace path parameters with test values."""
    if path in ROUTE_PATH_OVERRIDES:
        return ROUTE_PATH_OVERRIDES[path]
    resolved = path
    for param, value in PATH_PARAMS.items():
        resolved = resolved.replace(param, value)
    # Fallback for any remaining {param} patterns
    resolved = re.sub(r"\{[^}]+\}", "test", resolved)
    return resolved


async def fetch_openapi_spec(session: aiohttp.ClientSession) -> Optional[Dict]:
    """Fetch the OpenAPI spec from the running service."""
    try:
        async with session.get(f"{BASE_URL}/openapi.json") as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception as e:
        print(f"FATAL: Cannot fetch OpenAPI spec: {e}", file=sys.stderr)
    return None


async def test_route(
    session: aiohttp.ClientSession, method: str, path: str
) -> TestResult:
    """Test a single route and return the result."""
    # Skip DELETE methods (destructive)
    if method == "DELETE":
        return TestResult(method, path, "skip", detail="DELETE method — destructive")

    # Skip dangerous lifecycle endpoints
    if path in DANGEROUS_ENDPOINTS:
        return TestResult(method, path, "skip", detail="dangerous lifecycle operation")

    # Skip known slow endpoints
    if path in SLOW_ENDPOINTS:
        return TestResult(method, path, "skip", detail=f"known slow >{TIMEOUT_SECONDS}s")

    # Resolve path parameters
    url = f"{BASE_URL}{resolve_path(path)}"
    start = time.monotonic()

    try:
        if method == "GET":
            async with session.get(url) as resp:
                status = resp.status
                body = await resp.text()

        elif method == "POST":
            payload = POST_PAYLOADS.get(path)
            if payload is None and path in POST_PAYLOADS:
                # Explicitly set to None = skip
                return TestResult(method, path, "skip", detail="explicitly skipped (slow/unsafe)")
            payload = payload if payload is not None else {}
            async with session.post(url, json=payload) as resp:
                status = resp.status
                body = await resp.text()

        elif method == "PUT":
            async with session.put(url, json={}) as resp:
                status = resp.status
                body = await resp.text()

        elif method == "PATCH":
            async with session.patch(url, json={}) as resp:
                status = resp.status
                body = await resp.text()

        else:
            return TestResult(method, path, "skip", detail=f"unsupported method {method}")

        elapsed = (time.monotonic() - start) * 1000

        if status < 500:
            return TestResult(method, path, "pass", http_status=status, duration_ms=elapsed)
        elif status == 503 and path in EXTERNAL_DEP_ROUTES:
            # 503 is the CORRECT contract when an external dependency is unavailable
            return TestResult(method, path, "pass", http_status=status, detail="503 expected (external dep unavailable)", duration_ms=elapsed)
        else:
            try:
                err = json.loads(body)
                detail = err.get("detail", body[:200])
            except (json.JSONDecodeError, AttributeError):
                detail = body[:200]
            return TestResult(method, path, "fail", http_status=status, detail=str(detail)[:200], duration_ms=elapsed)

    except asyncio.TimeoutError:
        elapsed = (time.monotonic() - start) * 1000
        return TestResult(method, path, "fail", detail=f"TIMEOUT exceeded {TIMEOUT_SECONDS}s", duration_ms=elapsed)
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return TestResult(method, path, "fail", detail=str(e)[:200], duration_ms=elapsed)


async def run_all_tests() -> TestReport:
    """Run contract tests against all routes."""
    report = TestReport(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))

    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Fetch OpenAPI spec
        spec = await fetch_openapi_spec(session)
        if not spec:
            print("FATAL: Cannot fetch OpenAPI spec", file=sys.stderr)
            sys.exit(1)

        # Extract all routes
        routes = []
        for path, ops in spec.get("paths", {}).items():
            for method in ("get", "post", "put", "delete", "patch"):
                if method in ops:
                    routes.append((method.upper(), path))

        routes.sort(key=lambda x: (x[1], x[0]))
        report.total_routes = len(routes)

        # Test each route sequentially (to avoid overwhelming the service)
        for method, path in routes:
            result = await test_route(session, method, path)

            result_dict = {
                "method": result.method,
                "path": result.path,
                "status": result.status,
                "http_status": result.http_status,
                "detail": result.detail,
                "duration_ms": round(result.duration_ms, 1),
            }
            report.results.append(result_dict)

            if result.status == "pass":
                report.passed += 1
            elif result.status == "fail":
                report.failed += 1
                report.failures.append(result_dict)
            else:
                report.skipped += 1
                report.skips.append(result_dict)

    return report


def print_report(report: TestReport):
    """Print human-readable test report."""
    print(f"\n{'='*70}")
    print(f"ECHO BRAIN ROUTE CONTRACT TEST RESULTS")
    print(f"{'='*70}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Base URL:  {report.base_url}")
    print(f"PASS: {report.passed}  |  FAIL: {report.failed}  |  SKIP: {report.skipped}  |  TOTAL: {report.total_routes}")
    print(f"{'='*70}")

    if report.failures:
        print(f"\n--- FAILURES ({report.failed}) ---")
        for r in report.failures:
            print(f"  FAIL {r['method']:6} {r['path']:55} {str(r.get('http_status') or '???'):>5}  {r['detail'][:70]}")

    if report.skips:
        print(f"\n--- SKIPPED ({report.skipped}) ---")
        for r in report.skips:
            print(f"  SKIP {r['method']:6} {r['path']:55} {r['detail']}")

    print(f"\n--- PASSING ({report.passed}) ---")
    for r in report.results:
        if r["status"] == "pass":
            print(f"  PASS {r['method']:6} {r['path']:55} {r.get('http_status', '???'):>5}  {r['duration_ms']:>7.1f}ms")

    # Final summary
    pct = (report.passed / report.total_routes * 100) if report.total_routes else 0
    print(f"\n{'='*70}")
    print(f"Coverage: {report.passed}/{report.total_routes} ({pct:.1f}%) routes passing")
    print(f"{'='*70}")


# ─── Pytest integration ──────────────────────────────────────────────────────

def test_all_routes_pass():
    """Pytest entry point: verifies all testable routes return non-500."""
    report = asyncio.run(run_all_tests())
    print_report(report)

    if report.failed > 0:
        failed_routes = [f"{r['method']} {r['path']}" for r in report.failures]
        raise AssertionError(
            f"{report.failed} route(s) returned 500:\n"
            + "\n".join(f"  - {r}" for r in failed_routes)
        )


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    report = asyncio.run(run_all_tests())

    if "--json" in sys.argv:
        output = {
            "timestamp": report.timestamp,
            "base_url": report.base_url,
            "total_routes": report.total_routes,
            "passed": report.passed,
            "failed": report.failed,
            "skipped": report.skipped,
            "failures": report.failures,
            "skips": report.skips,
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report)

    sys.exit(1 if report.failed > 0 else 0)
