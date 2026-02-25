#!/usr/bin/env python3
"""
Echo Brain — Contract Monitor Service
======================================
Permanent self-review system that validates the API contract between
frontend and backend. Runs as a scheduled worker, stores results in
PostgreSQL, and exposes diagnostic endpoints.

Installation:
  1. Copy to /opt/tower-echo-brain/src/monitoring/contract_monitor.py
  2. Run the schema migration (see setup_schema())
  3. Register the worker and API routes in main.py (see bottom of file)

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Worker Scheduler (every 5 min)                     │
  │    → ContractMonitor.run_all()                      │
  │      → Tests each contract against live endpoints   │
  │      → Stores results in contract_monitor_results   │
  │      → Stores issues in contract_monitor_issues     │
  │      → Updates contract_monitor_snapshots (latest)  │
  └─────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────┐
  │  API Endpoints                                      │
  │    GET  /api/echo/diagnostics/contracts             │
  │         → Latest snapshot + all findings            │
  │    GET  /api/echo/diagnostics/contracts/history     │
  │         → Trend data for dashboard graphs           │
  │    POST /api/echo/diagnostics/contracts/run         │
  │         → Trigger immediate test run                │
  │    GET  /api/echo/diagnostics/contracts/issues      │
  │         → Open issues only (WARN/FAIL/ERROR)        │
  └─────────────────────────────────────────────────────┘
"""

import asyncio
import json
import time
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

import httpx
import asyncpg
from fastapi import APIRouter

logger = logging.getLogger(__name__)

# Global monitor instance for route handlers
_monitor_instance = None

# Create a router for contract monitoring endpoints
contract_router = APIRouter(prefix="/api/echo/diagnostics")


# ═══════════════════════════════════════════════════════════════════════════════
# Severity Levels
# ═══════════════════════════════════════════════════════════════════════════════

class Severity(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    ERROR = "ERROR"

    @property
    def rank(self) -> int:
        return {"PASS": 0, "WARN": 1, "FAIL": 2, "ERROR": 3}[self.value]

    @property
    def icon(self) -> str:
        return {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "ERROR": "💀"}[self.value]


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Finding:
    severity: str
    message: str

@dataclass
class ContractResult:
    contract_id: str
    description: str
    vue_component: str
    method: str
    path: str
    severity: str = Severity.PASS.value
    findings: List[Dict[str, str]] = field(default_factory=list)
    response_time_ms: Optional[int] = None
    http_status: Optional[int] = None
    response_size: Optional[int] = None
    response_keys: List[str] = field(default_factory=list)
    raw_response: Optional[dict] = None
    tested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_finding(self, severity: Severity, message: str):
        self.findings.append({"severity": severity.value, "message": message})
        if severity.rank > Severity(self.severity).rank:
            self.severity = severity.value

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't store full raw responses in DB — too large
        d.pop("raw_response", None)
        return d


@dataclass
class MonitorSnapshot:
    """Summary of a full test run"""
    run_id: str
    timestamp: str
    total_contracts: int
    passed: int
    warned: int
    failed: int
    errored: int
    total_response_time_ms: int
    verdict: str  # "healthy", "degraded", "broken"
    results: List[dict] = field(default_factory=list)
    issues: List[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Contract Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Internal base URL — monitor runs inside the service, no nginx proxy needed
# But we ALSO test through nginx to catch proxy issues
INTERNAL_BASE = "http://localhost:8309"
EXTERNAL_BASE = os.getenv("ECHO_BRAIN_EXTERNAL_URL", "https://tower.local")
REQUEST_TIMEOUT = 15.0

CONTRACTS = [
    {
        "id": "health_detailed",
        "method": "GET",
        "path": "/api/echo/health/detailed",
        "test_external": True,  # Also test through nginx
        "body": None,
        "description": "Detailed health with Qdrant stats",
        "vue_component": "DashboardView.vue → healthStore",
        "expected_keys": {
            "status": str,
            "timestamp": str,
            "workers": dict,
            "knowledge": dict,
            "quality": dict,
            "issues": dict,
            "self_awareness": dict,
        },
        "health_keys": ["status"],
        "healthy_values": ["healthy", "ok", "operational"],
        "degraded_values": ["degraded", "partial", "warning"],
        "count_keys": [],
        "list_keys": [],
        "nested_checks": {
            "self_awareness.vector_count": {"type": int, "min": 1},
            "self_awareness.facts_count": {"type": int, "min": 1},
            "knowledge.total_facts": {"type": int, "min": 1},
        },
        "content_checks": {},
    },
    {
        "id": "ask",
        "method": "POST",
        "path": "/api/echo/ask",
        "test_external": True,
        "body": {"question": "Echo self-test: confirm operational status"},
        "description": "Ask Echo Brain a question",
        "vue_component": "AskView.vue → askApi.ask()",
        "expected_keys": {
            "answer": str,
            "question": str,
            "confidence": None,
            "memories_used": int,
            "sources": list,
            "model_used": str,
        },
        "health_keys": [],
        "count_keys": ["memories_used"],
        "list_keys": [],
        "content_checks": {
            "answer": {"min_length": 10, "label": "Answer too short — likely error"},
        },
    },
    {
        "id": "memory_search",
        "method": "POST",
        "path": "/api/echo/memory/search",
        "test_external": False,
        "body": {"query": "Echo Brain system", "limit": 3},
        "description": "Memory vector search",
        "vue_component": "MemoryView.vue → memoryApi.search()",
        "expected_keys": {
            "results": list,
        },
        "health_keys": [],
        "count_keys": [],
        "list_keys": ["results"],
        "content_checks": {
            "results": {"min_length": 1, "label": "No search results returned"},
        },
    },
    {
        "id": "system_logs",
        "method": "GET",
        "path": "/api/echo/system/logs?limit=5",
        "test_external": False,
        "body": None,
        "description": "System logs from journalctl",
        "vue_component": "LogsView.vue → systemApi.logs()",
        "expected_keys": {
            "logs": list,
            "total": int,
            "filtered": bool,
        },
        "health_keys": [],
        "count_keys": ["total"],
        "list_keys": ["logs"],
    },
    {
        "id": "knowledge_facts",
        "method": "GET",
        "path": "/api/echo/knowledge/facts?limit=2",
        "test_external": False,
        "body": None,
        "description": "Knowledge facts from PostgreSQL",
        "vue_component": "KnowledgeView.vue → knowledgeApi",
        "expected_keys": {
            "total": int,
            "facts": list,
        },
        "health_keys": [],
        "count_keys": ["total"],
        "list_keys": ["facts"],
    },
    {
        "id": "knowledge_stats",
        "method": "GET",
        "path": "/api/echo/knowledge/stats",
        "test_external": False,
        "body": None,
        "description": "Knowledge statistics",
        "vue_component": "KnowledgeView.vue → knowledgeApi",
        "expected_keys": {},
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "id": "memory_status",
        "method": "GET",
        "path": "/api/echo/memory/status",
        "test_external": False,
        "body": None,
        "description": "Memory/ingestion status",
        "vue_component": "MemoryView.vue → memoryApi.status()",
        "expected_keys": {
            "conversations_processed": int,
            "embeddings_created": int,
            "is_running": bool,
            "config": dict,
        },
        "health_keys": [],
        "count_keys": ["embeddings_created"],
        "list_keys": [],
    },
    {
        "id": "intelligence_status",
        "method": "GET",
        "path": "/api/echo/intelligence/status",
        "test_external": False,
        "body": None,
        "description": "Intelligence engine status",
        "vue_component": "DashboardView.vue → intelligenceApi",
        "expected_keys": {
            "components": dict,
            "database_connectivity": None,
            "timestamp": str,
        },
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "id": "ingestion_status",
        "method": "GET",
        "path": "/api/echo/ingestion/status",
        "test_external": False,
        "body": None,
        "description": "Ingestion pipeline status",
        "vue_component": "DashboardView.vue → ingestionApi",
        "expected_keys": {},
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "id": "workers_status",
        "method": "GET",
        "path": "/api/workers/status",
        "test_external": True,
        "body": None,
        "description": "Background workers status",
        "vue_component": "DashboardView.vue → workersApi",
        "expected_keys": {},
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "id": "basic_health",
        "method": "GET",
        "path": "/health",
        "test_external": False,
        "body": None,
        "description": "Basic health check",
        "vue_component": "N/A — infrastructure",
        "expected_keys": {
            "status": str,
            "service": str,
            "timestamp": str,
        },
        "health_keys": ["status"],
        "healthy_values": ["healthy", "ok"],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "id": "mcp_tools",
        "method": "POST",
        "path": "/mcp",
        "test_external": False,
        "body": {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1},
        "description": "MCP tools list",
        "vue_component": "N/A — MCP interface",
        "expected_keys": {
            "tools": list,
        },
        "health_keys": [],
        "count_keys": [],
        "list_keys": ["tools"],
    },
    {
        "id": "mcp_health",
        "method": "GET",
        "path": "/mcp/health",
        "test_external": False,
        "body": None,
        "description": "MCP health endpoint",
        "vue_component": "N/A — MCP interface",
        "expected_keys": {},
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "id": "autonomous_status",
        "method": "GET",
        "path": "/api/autonomous/status",
        "test_external": True,
        "body": None,
        "description": "Autonomous agent status",
        "vue_component": "DashboardView.vue → autonomousApi",
        "expected_keys": {},
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "id": "pipeline_health",
        "method": "GET",
        "path": "/api/pipeline/health",
        "test_external": True,
        "body": None,
        "description": "Full pipeline health",
        "vue_component": "DashboardView.vue → pipelineApi",
        "expected_keys": {},
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Database Schema
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- Stores every individual contract test result
CREATE TABLE IF NOT EXISTS contract_monitor_results (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,
    contract_id     TEXT NOT NULL,
    description     TEXT,
    vue_component   TEXT,
    method          TEXT,
    path            TEXT,
    severity        TEXT NOT NULL,  -- PASS, WARN, FAIL, ERROR
    findings        JSONB DEFAULT '[]',
    response_time_ms INTEGER,
    http_status     INTEGER,
    response_size   INTEGER,
    response_keys   JSONB DEFAULT '[]',
    tested_at       TIMESTAMPTZ DEFAULT NOW(),
    test_target     TEXT DEFAULT 'internal'  -- 'internal' or 'external'
);

-- Stores run-level snapshots for trending
CREATE TABLE IF NOT EXISTS contract_monitor_snapshots (
    id                  SERIAL PRIMARY KEY,
    run_id              TEXT UNIQUE NOT NULL,
    timestamp           TIMESTAMPTZ DEFAULT NOW(),
    total_contracts     INTEGER,
    passed              INTEGER,
    warned              INTEGER,
    failed              INTEGER,
    errored             INTEGER,
    total_response_time_ms INTEGER,
    verdict             TEXT,  -- 'healthy', 'degraded', 'broken'
    issues_json         JSONB DEFAULT '[]'
);

-- Tracks open issues for the dashboard (persists until resolved)
CREATE TABLE IF NOT EXISTS contract_monitor_issues (
    id              SERIAL PRIMARY KEY,
    contract_id     TEXT NOT NULL,
    path            TEXT NOT NULL,
    severity        TEXT NOT NULL,
    message         TEXT NOT NULL,
    first_seen      TIMESTAMPTZ DEFAULT NOW(),
    last_seen       TIMESTAMPTZ DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ,
    occurrences     INTEGER DEFAULT 1,
    UNIQUE(contract_id, message)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_cmr_run_id ON contract_monitor_results(run_id);
CREATE INDEX IF NOT EXISTS idx_cmr_contract_id ON contract_monitor_results(contract_id);
CREATE INDEX IF NOT EXISTS idx_cmr_severity ON contract_monitor_results(severity);
CREATE INDEX IF NOT EXISTS idx_cms_timestamp ON contract_monitor_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_cmi_open ON contract_monitor_issues(resolved_at) WHERE resolved_at IS NULL;
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Nested dict access via dot notation
# ═══════════════════════════════════════════════════════════════════════════════

def get_nested(data: dict, dotted_key: str) -> Any:
    keys = dotted_key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current


# ═══════════════════════════════════════════════════════════════════════════════
# Contract Monitor Core
# ═══════════════════════════════════════════════════════════════════════════════

class ContractMonitor:
    """
    Runs frontend contract validation tests and persists results.

    Usage:
        monitor = ContractMonitor(db_pool)
        await monitor.setup_schema()
        snapshot = await monitor.run_all()
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self._http_client: Optional[httpx.AsyncClient] = None

    async def setup_schema(self):
        """Create monitoring tables if they don't exist"""
        async with self.db_pool.acquire() as conn:
            # AGE puts ag_catalog first in search_path; pin DDL to public
            await conn.execute("SET search_path TO public")
            await conn.execute(SCHEMA_SQL)
        logger.info("[ContractMonitor] Schema ready")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                verify=False,  # Self-signed cert for external
            )
        return self._http_client

    async def close(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    # ─── Run a single contract ───────────────────────────────────────────

    async def _test_contract(
        self, contract: dict, base_url: str, target_label: str
    ) -> ContractResult:
        """Execute one contract test against a base URL"""

        method = contract["method"]
        path = contract["path"]
        body = contract.get("body")
        cid = contract["id"]
        desc = contract["description"]
        expected = contract.get("expected_keys", {})
        health_keys = contract.get("health_keys", [])
        healthy_values = contract.get("healthy_values", ["healthy", "ok", "operational"])
        degraded_values = contract.get("degraded_values", ["degraded", "partial", "warning"])
        count_keys = contract.get("count_keys", [])
        list_keys = contract.get("list_keys", [])
        nested_checks = contract.get("nested_checks", {})
        content_checks = contract.get("content_checks", {})

        result = ContractResult(
            contract_id=f"{cid}_{target_label}",
            description=f"{desc} ({target_label})",
            vue_component=contract.get("vue_component", ""),
            method=method,
            path=path,
        )

        url = f"{base_url}{path}"
        client = await self._get_client()

        try:
            t0 = time.time()
            if method == "GET":
                resp = await client.get(url)
            else:
                resp = await client.post(url, json=body)
            elapsed_ms = round((time.time() - t0) * 1000)

            result.response_time_ms = elapsed_ms
            result.http_status = resp.status_code
            result.response_size = len(resp.content)

            # ── Timing thresholds ──
            if elapsed_ms > 10000:
                result.add_finding(Severity.FAIL,
                    f"Response took {elapsed_ms}ms (>10s) — frontend axios will timeout")
            elif elapsed_ms > 5000:
                result.add_finding(Severity.WARN,
                    f"Response took {elapsed_ms}ms (>5s) — slow for UI")
            elif elapsed_ms > 2000:
                result.add_finding(Severity.WARN,
                    f"Response took {elapsed_ms}ms (>2s) — noticeable delay")

            # ── HTTP status ──
            if resp.status_code != 200:
                result.add_finding(Severity.FAIL,
                    f"HTTP {resp.status_code} — frontend expects 200")
                return result

            # ── Parse JSON ──
            try:
                data = resp.json()
                result.raw_response = data
            except Exception:
                result.add_finding(Severity.ERROR, "Response is not valid JSON")
                return result

            if not isinstance(data, dict):
                result.add_finding(Severity.WARN,
                    f"Response is {type(data).__name__}, not dict")
                return result

            result.response_keys = list(data.keys())

            # ── Error fields in response ──
            if data.get("error"):
                result.add_finding(Severity.FAIL,
                    f"Response contains error: {str(data['error'])[:120]}")

            if data.get("detail") and "not found" in str(data.get("detail", "")).lower():
                result.add_finding(Severity.FAIL,
                    f"Endpoint returned not found: {data['detail']}")

            # ── Key presence + type checking ──
            for key, expected_type in expected.items():
                if key not in data:
                    result.add_finding(Severity.FAIL,
                        f"Missing key '{key}' — frontend will break")
                elif expected_type is not None and not isinstance(data[key], expected_type):
                    actual = type(data[key]).__name__
                    result.add_finding(Severity.WARN,
                        f"Key '{key}' type mismatch: expected {expected_type.__name__}, "
                        f"got {actual} = {str(data[key])[:60]}")

            # ── Extra keys (informational) ──
            if expected:
                extra = set(data.keys()) - set(expected.keys())
                if extra:
                    result.add_finding(Severity.PASS,
                        f"Extra keys: {', '.join(sorted(extra))}")

            # ── Health value assertions ──
            for hk in health_keys:
                val = str(data.get(hk, "")).lower().strip()
                if val in [v.lower() for v in healthy_values]:
                    result.add_finding(Severity.PASS, f"'{hk}' = '{val}' ✓")
                elif val in [v.lower() for v in degraded_values]:
                    result.add_finding(Severity.WARN,
                        f"'{hk}' = '{val}' — DEGRADED")
                elif val in ["error", "critical", "down", "offline", "unavailable"]:
                    result.add_finding(Severity.FAIL,
                        f"'{hk}' = '{val}' — ERROR STATE")
                else:
                    result.add_finding(Severity.WARN,
                        f"'{hk}' = '{val}' — unknown health value")

            # ── Count assertions ──
            for ck in count_keys:
                val = data.get(ck)
                if val is not None:
                    try:
                        if int(val) <= 0:
                            result.add_finding(Severity.WARN,
                                f"'{ck}' = {val} — expected > 0")
                        else:
                            result.add_finding(Severity.PASS, f"'{ck}' = {val}")
                    except (ValueError, TypeError):
                        result.add_finding(Severity.WARN,
                            f"'{ck}' = {val} — not a number")

            # ── List assertions ──
            for lk in list_keys:
                val = data.get(lk)
                if isinstance(val, list):
                    if len(val) == 0:
                        result.add_finding(Severity.WARN,
                            f"'{lk}' is empty — frontend may show blank")
                    else:
                        result.add_finding(Severity.PASS,
                            f"'{lk}' has {len(val)} item(s)")

            # ── Nested checks ──
            for dotted_key, check in nested_checks.items():
                val = get_nested(data, dotted_key)
                if val is None:
                    result.add_finding(Severity.FAIL,
                        f"Nested key '{dotted_key}' not found or null")
                else:
                    exp_type = check.get("type")
                    min_val = check.get("min")
                    if exp_type and not isinstance(val, exp_type):
                        result.add_finding(Severity.WARN,
                            f"'{dotted_key}' type: expected {exp_type.__name__}, "
                            f"got {type(val).__name__}")
                    elif min_val is not None:
                        try:
                            if int(val) < min_val:
                                result.add_finding(Severity.FAIL,
                                    f"'{dotted_key}' = {val} — below min {min_val}")
                            else:
                                result.add_finding(Severity.PASS,
                                    f"'{dotted_key}' = {val:,}")
                        except (ValueError, TypeError):
                            result.add_finding(Severity.WARN,
                                f"'{dotted_key}' = {val} — cannot compare")

            # ── Content checks ──
            for ck_key, ck_rule in content_checks.items():
                val = data.get(ck_key)
                if val is not None:
                    min_len = ck_rule.get("min_length", 0)
                    label = ck_rule.get("label", f"'{ck_key}' content check failed")
                    if hasattr(val, '__len__') and len(val) < min_len:
                        result.add_finding(Severity.WARN,
                            f"{label} (size={len(val)}, min={min_len})")
                    else:
                        size = len(val) if hasattr(val, '__len__') else val
                        result.add_finding(Severity.PASS,
                            f"'{ck_key}' content OK (size={size})")

        except httpx.ConnectError:
            result.add_finding(Severity.ERROR,
                f"Connection refused at {base_url} — service down?")
        except httpx.TimeoutException:
            result.add_finding(Severity.ERROR,
                f"Timed out after {REQUEST_TIMEOUT}s")
        except Exception as e:
            result.add_finding(Severity.ERROR,
                f"{type(e).__name__}: {str(e)[:100]}")

        return result

    # ─── Run all contracts ───────────────────────────────────────────────

    async def run_all(self, include_external: bool = True) -> MonitorSnapshot:
        """
        Run every contract test and persist results.
        Returns a MonitorSnapshot with the full picture.
        """
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        all_results: List[ContractResult] = []
        total_time = 0

        logger.info(f"[ContractMonitor] Starting run {run_id} "
                     f"({len(CONTRACTS)} contracts)")

        for contract in CONTRACTS:
            # ── Internal test (always) ──
            result = await self._test_contract(
                contract, INTERNAL_BASE, "internal"
            )
            all_results.append(result)
            total_time += result.response_time_ms or 0

            # ── External/nginx test (if enabled and contract requests it) ──
            if include_external and contract.get("test_external"):
                ext_result = await self._test_contract(
                    contract, EXTERNAL_BASE, "nginx"
                )
                all_results.append(ext_result)
                total_time += ext_result.response_time_ms or 0

        # ── Aggregate counts ──
        counts = {s.value: 0 for s in Severity}
        for r in all_results:
            counts[r.severity] += 1

        # ── Determine verdict ──
        if counts["ERROR"] > 0 or counts["FAIL"] > 0:
            verdict = "broken"
        elif counts["WARN"] > 0:
            verdict = "degraded"
        else:
            verdict = "healthy"

        # ── Collect all non-PASS issues ──
        issues = []
        for r in all_results:
            for f in r.findings:
                if f["severity"] != Severity.PASS.value:
                    issues.append({
                        "contract_id": r.contract_id,
                        "path": r.path,
                        "description": r.description,
                        "severity": f["severity"],
                        "message": f["message"],
                    })

        snapshot = MonitorSnapshot(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_contracts=len(all_results),
            passed=counts["PASS"],
            warned=counts["WARN"],
            failed=counts["FAIL"],
            errored=counts["ERROR"],
            total_response_time_ms=total_time,
            verdict=verdict,
            results=[r.to_dict() for r in all_results],
            issues=issues,
        )

        # ── Persist to DB ──
        await self._persist_results(snapshot, all_results)

        logger.info(
            f"[ContractMonitor] Run {run_id} complete: "
            f"{counts['PASS']}✅ {counts['WARN']}⚠️ "
            f"{counts['FAIL']}❌ {counts['ERROR']}💀 "
            f"→ {verdict}"
        )

        return snapshot

    # ─── Persistence ─────────────────────────────────────────────────────

    async def _persist_results(
        self, snapshot: MonitorSnapshot, results: List[ContractResult]
    ):
        """Store results, snapshot, and update issue tracker"""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # ── Store individual results ──
                for r in results:
                    await conn.execute("""
                        INSERT INTO contract_monitor_results
                            (run_id, contract_id, description, vue_component,
                             method, path, severity, findings,
                             response_time_ms, http_status, response_size,
                             response_keys, test_target)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                    """,
                        snapshot.run_id, r.contract_id, r.description,
                        r.vue_component, r.method, r.path, r.severity,
                        json.dumps(r.findings), r.response_time_ms,
                        r.http_status, r.response_size,
                        json.dumps(r.response_keys),
                        "nginx" if "_nginx" in r.contract_id else "internal",
                    )

                # ── Store snapshot ──
                await conn.execute("""
                    INSERT INTO contract_monitor_snapshots
                        (run_id, total_contracts, passed, warned, failed,
                         errored, total_response_time_ms, verdict, issues_json)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                """,
                    snapshot.run_id, snapshot.total_contracts,
                    snapshot.passed, snapshot.warned, snapshot.failed,
                    snapshot.errored, snapshot.total_response_time_ms,
                    snapshot.verdict, json.dumps(snapshot.issues),
                )

                # ── Update issue tracker ──
                # Mark all existing issues as potentially resolved
                current_issue_keys = set()
                for issue in snapshot.issues:
                    key = (issue["contract_id"], issue["message"])
                    current_issue_keys.add(key)

                    await conn.execute("""
                        INSERT INTO contract_monitor_issues
                            (contract_id, path, severity, message, last_seen)
                        VALUES ($1, $2, $3, $4, NOW())
                        ON CONFLICT (contract_id, message) DO UPDATE SET
                            severity = EXCLUDED.severity,
                            last_seen = NOW(),
                            occurrences = contract_monitor_issues.occurrences + 1,
                            resolved_at = NULL
                    """, issue["contract_id"], issue["path"],
                         issue["severity"], issue["message"])

                # Resolve issues that didn't appear this run
                await conn.execute("""
                    UPDATE contract_monitor_issues
                    SET resolved_at = NOW()
                    WHERE resolved_at IS NULL
                      AND last_seen < NOW() - INTERVAL '1 minute'
                """)

    # ─── Query Methods (for API endpoints) ───────────────────────────────

    async def get_latest_snapshot(self) -> Optional[dict]:
        """Get the most recent test run snapshot with full results"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM contract_monitor_snapshots
                ORDER BY timestamp DESC LIMIT 1
            """)
            if not row:
                return None

            # Get the individual results for this run
            results = await conn.fetch("""
                SELECT * FROM contract_monitor_results
                WHERE run_id = $1
                ORDER BY contract_id
            """, row["run_id"])

            return {
                "run_id": row["run_id"],
                "timestamp": row["timestamp"].isoformat(),
                "total_contracts": row["total_contracts"],
                "passed": row["passed"],
                "warned": row["warned"],
                "failed": row["failed"],
                "errored": row["errored"],
                "total_response_time_ms": row["total_response_time_ms"],
                "verdict": row["verdict"],
                "issues": json.loads(row["issues_json"]) if row["issues_json"] else [],
                "results": [
                    {
                        "contract_id": r["contract_id"],
                        "description": r["description"],
                        "vue_component": r["vue_component"],
                        "method": r["method"],
                        "path": r["path"],
                        "severity": r["severity"],
                        "findings": json.loads(r["findings"]) if r["findings"] else [],
                        "response_time_ms": r["response_time_ms"],
                        "http_status": r["http_status"],
                        "response_size": r["response_size"],
                        "response_keys": json.loads(r["response_keys"]) if r["response_keys"] else [],
                        "test_target": r["test_target"],
                    }
                    for r in results
                ],
            }

    async def get_history(self, hours: int = 24) -> List[dict]:
        """Get snapshot history for trending charts"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT run_id, timestamp, total_contracts,
                       passed, warned, failed, errored,
                       total_response_time_ms, verdict
                FROM contract_monitor_snapshots
                WHERE timestamp > NOW() - ($1 || ' hours')::INTERVAL
                ORDER BY timestamp DESC
            """, str(hours))

            return [
                {
                    "run_id": r["run_id"],
                    "timestamp": r["timestamp"].isoformat(),
                    "total_contracts": r["total_contracts"],
                    "passed": r["passed"],
                    "warned": r["warned"],
                    "failed": r["failed"],
                    "errored": r["errored"],
                    "total_response_time_ms": r["total_response_time_ms"],
                    "verdict": r["verdict"],
                }
                for r in rows
            ]

    async def get_open_issues(self) -> List[dict]:
        """Get all currently open (unresolved) issues"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT contract_id, path, severity, message,
                       first_seen, last_seen, occurrences
                FROM contract_monitor_issues
                WHERE resolved_at IS NULL
                ORDER BY
                    CASE severity
                        WHEN 'ERROR' THEN 0
                        WHEN 'FAIL' THEN 1
                        WHEN 'WARN' THEN 2
                        ELSE 3
                    END,
                    occurrences DESC
            """)

            return [
                {
                    "contract_id": r["contract_id"],
                    "path": r["path"],
                    "severity": r["severity"],
                    "message": r["message"],
                    "first_seen": r["first_seen"].isoformat(),
                    "last_seen": r["last_seen"].isoformat(),
                    "occurrences": r["occurrences"],
                    "age_hours": round(
                        (datetime.now(timezone.utc) - r["first_seen"])
                        .total_seconds() / 3600, 1
                    ),
                }
                for r in rows
            ]

    async def get_contract_history(
        self, contract_id: str, hours: int = 24
    ) -> List[dict]:
        """Get history for a single contract (for per-endpoint trending)"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT severity, response_time_ms, http_status,
                       findings, tested_at
                FROM contract_monitor_results
                WHERE contract_id = $1
                  AND tested_at > NOW() - ($2 || ' hours')::INTERVAL
                ORDER BY tested_at DESC
            """, contract_id, str(hours))

            return [
                {
                    "severity": r["severity"],
                    "response_time_ms": r["response_time_ms"],
                    "http_status": r["http_status"],
                    "findings": json.loads(r["findings"]) if r["findings"] else [],
                    "tested_at": r["tested_at"].isoformat(),
                }
                for r in rows
            ]


# ═══════════════════════════════════════════════════════════════════════════════
# Route Handlers (using router)
# ═══════════════════════════════════════════════════════════════════════════════

@contract_router.get("/contracts")
async def get_contract_status():
    """
    Latest contract test snapshot with all results and findings.
    This is what the frontend dashboard should poll for self-review.
    """
    if not _monitor_instance:
        return {"status": "error", "message": "Contract monitor not initialized"}

    snapshot = await _monitor_instance.get_latest_snapshot()
    if not snapshot:
        return {
            "status": "no_data",
            "message": "No contract tests have run yet. POST /api/echo/diagnostics/contracts/run to trigger.",
        }

    open_issues = await _monitor_instance.get_open_issues()

    return {
        "status": snapshot["verdict"],
        "snapshot": snapshot,
        "open_issues": open_issues,
        "open_issue_count": len(open_issues),
        "summary": {
            "total": snapshot["total_contracts"],
            "passed": snapshot["passed"],
            "warned": snapshot["warned"],
            "failed": snapshot["failed"],
            "errored": snapshot["errored"],
            "response_time_ms": snapshot["total_response_time_ms"],
        },
    }

@contract_router.get("/contracts/history")
async def get_contract_history(hours: int = 24):
    """
    Trend data for the dashboard. Returns snapshots over time
    so you can graph pass/fail/warn counts and response times.
    """
    if not _monitor_instance:
        return {"status": "error", "message": "Contract monitor not initialized"}

    history = await _monitor_instance.get_history(hours=hours)
    return {
        "hours": hours,
        "snapshots": history,
        "total_runs": len(history),
    }

@contract_router.post("/contracts/run")
async def trigger_contract_run(include_external: bool = True):
    """
    Trigger an immediate contract test run.
    Returns the full snapshot when complete.
    """
    if not _monitor_instance:
        return {"status": "error", "message": "Contract monitor not initialized"}

    snapshot = await _monitor_instance.run_all(include_external=include_external)
    return {
        "status": snapshot.verdict,
        "run_id": snapshot.run_id,
        "total_contracts": snapshot.total_contracts,
        "passed": snapshot.passed,
        "warned": snapshot.warned,
        "failed": snapshot.failed,
        "errored": snapshot.errored,
        "total_response_time_ms": snapshot.total_response_time_ms,
        "issues": snapshot.issues,
        "results": snapshot.results,
    }

@contract_router.get("/contracts/issues")
async def get_open_issues():
    """
    All currently open (unresolved) issues, sorted by severity.
    Issues auto-resolve when they stop appearing in test runs.
    """
    if not _monitor_instance:
        return {"status": "error", "message": "Contract monitor not initialized"}

    issues = await _monitor_instance.get_open_issues()
    return {
        "total_open": len(issues),
        "issues": issues,
        "by_severity": {
            "error": len([i for i in issues if i["severity"] == "ERROR"]),
            "fail": len([i for i in issues if i["severity"] == "FAIL"]),
            "warn": len([i for i in issues if i["severity"] == "WARN"]),
        },
    }

@contract_router.get("/contracts/{contract_id}")
async def get_single_contract_history(contract_id: str, hours: int = 24):
    """
    History for a single contract endpoint.
    Useful for debugging intermittent failures.
    """
    if not _monitor_instance:
        return {"status": "error", "message": "Contract monitor not initialized"}

    history = await _monitor_instance.get_contract_history(contract_id, hours)
    return {
        "contract_id": contract_id,
        "hours": hours,
        "total_checks": len(history),
        "history": history,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI Route Registration
# ═══════════════════════════════════════════════════════════════════════════════
#
# Add this to main.py after initializing the DB pool:
#
#   from src.monitoring.contract_monitor import ContractMonitor, register_contract_routes
#
#   # During startup:
#   contract_monitor = ContractMonitor(db_pool)
#   await contract_monitor.setup_schema()
#   register_contract_routes(app, contract_monitor)
#
#   # Register with worker scheduler:
#   worker_scheduler.register_task(
#       name="contract_monitor",
#       interval_seconds=300,  # Every 5 minutes
#       callback=lambda: asyncio.create_task(contract_monitor.run_all())
#   )

def initialize_contract_monitor(monitor: ContractMonitor):
    """Initialize the global contract monitor instance for route handlers"""
    global _monitor_instance
    _monitor_instance = monitor
    logger.info("[ContractMonitor] Monitor instance initialized for route handlers")


# ═══════════════════════════════════════════════════════════════════════════════
# Integration with /health/detailed
# ═══════════════════════════════════════════════════════════════════════════════
#
# To include contract status in the main health endpoint, add this to
# the health/detailed handler in main.py:
#
#   # Inside get_detailed_health():
#   contract_snapshot = await contract_monitor.get_latest_snapshot()
#   contract_issues = await contract_monitor.get_open_issues()
#
#   # Add to the response dict:
#   "contract_health": {
#       "verdict": contract_snapshot["verdict"] if contract_snapshot else "unknown",
#       "last_run": contract_snapshot["timestamp"] if contract_snapshot else None,
#       "passed": contract_snapshot["passed"] if contract_snapshot else 0,
#       "total": contract_snapshot["total_contracts"] if contract_snapshot else 0,
#       "open_issues": len(contract_issues),
#       "critical_issues": len([i for i in contract_issues if i["severity"] in ("FAIL", "ERROR")]),
#   }
#
# This way, when the dashboard calls /health/detailed, it sees the
# contract monitor status as part of the overall health picture.
# If contract_health.verdict != "healthy", the dashboard knows
# something is wrong even if the basic health check passes.


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone CLI Mode (for manual runs / cron)
# ═══════════════════════════════════════════════════════════════════════════════

async def cli_main():
    """Run contract tests from command line with formatted output"""
    import sys

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    json_out = "--json" in sys.argv
    no_db = "--no-db" in sys.argv

    # ── Connect to DB (unless --no-db) ──
    db_pool = None
    monitor = None

    if not no_db:
        try:
            # Get password from environment (should be set by Vault)
            db_password = os.getenv("PGPASSWORD", os.getenv("DB_PASSWORD", ""))
            if not db_password:
                print("⚠️  Database password not set in environment. Use PGPASSWORD or DB_PASSWORD env var.")
                no_db = True
                raise Exception("No database password in environment")

            db_url = os.getenv(
                "DATABASE_URL",
                f"postgresql://patrick:{db_password}@localhost/echo_brain"
            )
            db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
            monitor = ContractMonitor(db_pool)
            await monitor.setup_schema()
        except Exception as e:
            print(f"⚠️  DB connection failed ({e}) — running without persistence")
            no_db = True

    if no_db or monitor is None:
        # Lightweight run without DB
        monitor = ContractMonitor.__new__(ContractMonitor)
        monitor.db_pool = None
        monitor._http_client = None

        # Override persistence to no-op
        async def _noop(*a, **kw):
            pass
        monitor._persist_results = _noop

    # ── Run tests ──
    snapshot = await monitor.run_all(include_external=True)
    await monitor.close()

    if json_out:
        print(json.dumps(asdict(snapshot), indent=2, default=str))
        return 1 if snapshot.verdict == "broken" else 0

    # ── Formatted output ──
    W = 90
    print(f"\n{'═' * W}")
    print("  ECHO BRAIN — CONTRACT MONITOR REPORT")
    print(f"{'═' * W}")
    print(f"  Run ID:     {snapshot.run_id}")
    print(f"  Timestamp:  {snapshot.timestamp}")
    print(f"  Contracts:  {snapshot.total_contracts}")
    print(f"  Verdict:    {snapshot.verdict.upper()}")
    print(f"  Mode:       {'VERBOSE' if verbose else 'NORMAL'}")

    # ── Summary table ──
    print(f"\n{'═' * W}")
    print(f"  {'#':>3}  {'Sev':5}  {'Time':>7}  {'Size':>7}  {'HTTP':>4}  "
          f"{'Issues':>6}  {'Target':8}  Description")
    print(f"  {'─'*3}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*4}  "
          f"{'─'*6}  {'─'*8}  {'─'*35}")

    for i, r in enumerate(snapshot.results, 1):
        sev = r["severity"]
        icon = Severity(sev).icon
        timing = f"{r['response_time_ms']}ms" if r.get("response_time_ms") else "N/A"
        size_b = r.get("response_size", 0) or 0
        if size_b < 1024:
            size = f"{size_b}B"
        else:
            size = f"{size_b/1024:.1f}KB"
        http = str(r.get("http_status", "N/A"))
        n_issues = len([f for f in r.get("findings", []) if f["severity"] != "PASS"])
        target = r["contract_id"].split("_")[-1]
        desc = r["description"][:35]
        print(f"  {i:>3}  {icon:3}  {timing:>7}  {size:>7}  {http:>4}  "
              f"{n_issues:>6}  {target:8}  {desc}")

    # ── Detailed findings ──
    if verbose or snapshot.verdict != "healthy":
        print(f"\n{'═' * W}")
        print("  DETAILED FINDINGS")
        print(f"{'═' * W}")

        for r in snapshot.results:
            non_pass = [f for f in r.get("findings", [])
                        if f["severity"] != "PASS"]
            if not non_pass and not verbose:
                continue

            sev_icon = Severity(r["severity"]).icon
            print(f"\n  {sev_icon} {r['description']}")
            print(f"    Route: {r['method']} {r['path']}")
            print(f"    Vue:   {r.get('vue_component', 'N/A')}")

            for f in r.get("findings", []):
                if f["severity"] == "PASS" and not verbose:
                    continue
                ficon = Severity(f["severity"]).icon
                print(f"    {ficon} {f['message']}")

    # ── Issues rollup ──
    if snapshot.issues:
        print(f"\n{'═' * W}")
        print(f"  ALL ISSUES ({len(snapshot.issues)})")
        print(f"{'═' * W}")
        for i, issue in enumerate(snapshot.issues, 1):
            icon = Severity(issue["severity"]).icon
            print(f"  {i:>2}. {icon} [{issue['severity']}] "
                  f"{issue['description']}")
            print(f"      {issue['path']}")
            print(f"      → {issue['message']}")

    # ── Verdict ──
    print(f"\n{'═' * W}")
    print(f"  ✅ PASS:  {snapshot.passed}    ⚠️  WARN:  {snapshot.warned}    "
          f"❌ FAIL:  {snapshot.failed}    💀 ERROR: {snapshot.errored}")
    print(f"  Total time: {snapshot.total_response_time_ms}ms")

    if snapshot.verdict == "healthy":
        print("\n  🟢 ALL CONTRACTS VALID")
    elif snapshot.verdict == "degraded":
        print("\n  🟡 DEGRADED — review warnings above")
    else:
        print("\n  🔴 BROKEN — fix FAIL/ERROR items above")

    print(f"{'═' * W}\n")

    if db_pool:
        await db_pool.close()

    return 1 if snapshot.verdict == "broken" else 0


if __name__ == "__main__":
    exit(asyncio.run(cli_main()))