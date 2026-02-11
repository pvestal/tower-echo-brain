#!/usr/bin/env python3
"""
Echo Brain — Frontend Contract Validator (Verbose Diagnostic)
=============================================================
Tests every API call the Vue frontend makes, through the actual nginx proxy.

Validates:
  - HTTP status code and response time
  - Response JSON shape (expected keys present)
  - Value type correctness (str, int, list, dict, bool)
  - Health assertions (status fields must be "healthy"/"ok", not "degraded"/"error")
  - Content assertions (lists must not be empty, counts must be > 0)
  - Extra/unexpected keys flagged as warnings
  - Full response body dump in verbose mode

Usage:
  python3 test_echo_frontend_contracts.py              # Normal run
  python3 test_echo_frontend_contracts.py --verbose     # Full response dumps
  python3 test_echo_frontend_contracts.py --json        # Machine-readable JSON output
"""
import requests
import json
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Configuration ───────────────────────────────────────────────────────────
FRONTEND_BASE = "https://tower.local"
DIRECT_BASE = "http://localhost:8309"
REQUEST_TIMEOUT = 15  # seconds — /ask can be slow

# ─── Severity Levels ────────────────────────────────────────────────────────
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
ERROR = "ERROR"

SEVERITY_ICON = {PASS: "✅", WARN: "⚠️ ", FAIL: "❌", ERROR: "💀"}
SEVERITY_RANK = {PASS: 0, WARN: 1, FAIL: 2, ERROR: 3}


# ─── Contract Definitions ───────────────────────────────────────────────────
# Each contract defines:
#   method, path, request_body, description, vue_component,
#   expected_keys: {key: expected_type} — use None for any type
#   health_keys: list of keys whose value must indicate healthy state
#   count_keys: list of keys whose numeric value must be > 0
#   list_keys: list of keys whose array value must not be empty

CONTRACTS = [
    {
        "method": "GET",
        "path": "/api/echo/health/detailed",
        "body": None,
        "description": "Detailed health with Qdrant stats",
        "vue_component": "DashboardView.vue → healthStore",
        "expected_keys": {
            "status": str,
            "timestamp": str,
            "workers": dict,
            "knowledge": dict,
            "quality": dict,
            "issues": list,
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
    },
    {
        "method": "POST",
        "path": "/api/echo/ask",
        "body": {"question": "What is Echo Brain?"},
        "description": "Ask Echo Brain a question",
        "vue_component": "AskView.vue → askApi.ask()",
        "expected_keys": {
            "answer": str,
            "question": str,
            "confidence": None,  # could be float or str
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
        "method": "POST",
        "path": "/api/echo/memory/search",
        "body": {"query": "Mei Kobayashi character", "limit": 3},
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
        "method": "GET",
        "path": "/api/echo/system/logs?limit=10",
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
        "method": "GET",
        "path": "/api/echo/knowledge/facts?limit=5",
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
        "method": "GET",
        "path": "/api/echo/knowledge/stats",
        "body": None,
        "description": "Knowledge statistics",
        "vue_component": "KnowledgeView.vue → knowledgeApi",
        "expected_keys": {},  # shape unknown — just validate HTTP 200 + valid JSON
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "method": "GET",
        "path": "/api/echo/memory/status",
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
        "method": "GET",
        "path": "/api/echo/intelligence/status",
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
        "method": "GET",
        "path": "/api/echo/ingestion/status",
        "body": None,
        "description": "Ingestion pipeline status",
        "vue_component": "DashboardView.vue → ingestionApi",
        "expected_keys": {},
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "method": "GET",
        "path": "/api/workers/status",
        "body": None,
        "description": "Background workers status",
        "vue_component": "DashboardView.vue → workersApi",
        "expected_keys": {},
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "method": "GET",
        "path": f"{DIRECT_BASE}/health",
        "body": None,
        "description": "Basic health (direct, bypass nginx)",
        "vue_component": "N/A — infrastructure check",
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
        "method": "POST",
        "path": "/mcp",
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
        "method": "GET",
        "path": "/mcp/health",
        "body": None,
        "description": "MCP health endpoint",
        "vue_component": "N/A — MCP interface",
        "expected_keys": {},
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "method": "GET",
        "path": "/api/autonomous/status",
        "body": None,
        "description": "Autonomous agent status",
        "vue_component": "DashboardView.vue → autonomousApi",
        "expected_keys": {},
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
    {
        "method": "GET",
        "path": "/api/pipeline/health",
        "body": None,
        "description": "Full pipeline health",
        "vue_component": "DashboardView.vue → pipelineApi",
        "expected_keys": {},
        "health_keys": [],
        "count_keys": [],
        "list_keys": [],
    },
]


# ─── Deep Value Access ───────────────────────────────────────────────────────
def get_nested(data: dict, dotted_key: str) -> Any:
    """Access nested dict values with dot notation: 'self_awareness.vector_count'"""
    keys = dotted_key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current


# ─── Single Contract Test ────────────────────────────────────────────────────
def run_contract(contract: dict, verbose: bool = False) -> dict:
    """
    Execute one contract test. Returns a result dict with:
      severity, findings (list of issues), timing, raw response, etc.
    """
    method = contract["method"]
    path = contract["path"]
    body = contract.get("body")
    desc = contract["description"]
    expected = contract.get("expected_keys", {})
    health_keys = contract.get("health_keys", [])
    healthy_values = contract.get("healthy_values", ["healthy", "ok", "operational"])
    degraded_values = contract.get("degraded_values", ["degraded", "partial", "warning"])
    count_keys = contract.get("count_keys", [])
    list_keys = contract.get("list_keys", [])
    nested_checks = contract.get("nested_checks", {})
    content_checks = contract.get("content_checks", {})

    result = {
        "description": desc,
        "vue_component": contract.get("vue_component", ""),
        "method": method,
        "path": path,
        "severity": PASS,
        "findings": [],
        "response_time_ms": None,
        "http_status": None,
        "response_size": None,
        "response_keys": [],
        "raw_response": None,
    }

    def add_finding(severity: str, message: str):
        result["findings"].append({"severity": severity, "message": message})
        if SEVERITY_RANK[severity] > SEVERITY_RANK[result["severity"]]:
            result["severity"] = severity

    # ── Build URL ──
    if path.startswith("http://") or path.startswith("https://"):
        url = path
    else:
        url = f"{FRONTEND_BASE}{path}"

    # ── Make Request ──
    try:
        t0 = time.time()
        if method == "GET":
            resp = requests.get(url, verify=False, timeout=REQUEST_TIMEOUT)
        else:
            resp = requests.post(url, json=body, verify=False, timeout=REQUEST_TIMEOUT)
        elapsed_ms = round((time.time() - t0) * 1000)

        result["response_time_ms"] = elapsed_ms
        result["http_status"] = resp.status_code
        result["response_size"] = len(resp.content)

        # ── Timing thresholds ──
        if elapsed_ms > 10000:
            add_finding(FAIL, f"Response took {elapsed_ms}ms (>10s) — will cause frontend timeout")
        elif elapsed_ms > 5000:
            add_finding(WARN, f"Response took {elapsed_ms}ms (>5s) — slow for UI")
        elif elapsed_ms > 2000:
            add_finding(WARN, f"Response took {elapsed_ms}ms (>2s) — noticeable delay")

        # ── HTTP Status ──
        if resp.status_code != 200:
            add_finding(FAIL, f"HTTP {resp.status_code} — frontend expects 200")
            result["raw_response"] = resp.text[:500]
            return result

        # ── Parse JSON ──
        try:
            data = resp.json()
            result["raw_response"] = data
        except json.JSONDecodeError:
            add_finding(ERROR, "Response is not valid JSON")
            result["raw_response"] = resp.text[:500]
            return result

        # ── Check for error fields in response ──
        if isinstance(data, dict):
            result["response_keys"] = list(data.keys())

            if data.get("error"):
                add_finding(FAIL, f"Response contains error: {str(data['error'])[:120]}")

            if data.get("detail") and "not found" in str(data.get("detail", "")).lower():
                add_finding(FAIL, f"Endpoint returned 'not found': {data['detail']}")

            # ── Key Presence ──
            for key, expected_type in expected.items():
                if key not in data:
                    add_finding(FAIL, f"Missing key '{key}' — frontend will break")
                elif expected_type is not None and not isinstance(data[key], expected_type):
                    actual_type = type(data[key]).__name__
                    add_finding(WARN,
                        f"Key '{key}' type mismatch: expected {expected_type.__name__}, got {actual_type} "
                        f"(value: {str(data[key])[:60]})"
                    )

            # ── Extra keys (informational) ──
            if expected:
                extra = set(data.keys()) - set(expected.keys())
                if extra:
                    add_finding(PASS, f"Extra keys not in contract: {', '.join(sorted(extra))}")

            # ── Health value assertions ──
            for hk in health_keys:
                val = data.get(hk, "")
                val_lower = str(val).lower().strip()
                if val_lower in [v.lower() for v in healthy_values]:
                    add_finding(PASS, f"'{hk}' = '{val}' ✓")
                elif val_lower in [v.lower() for v in degraded_values]:
                    add_finding(WARN, f"'{hk}' = '{val}' — DEGRADED (not healthy)")
                elif val_lower in ["error", "critical", "down", "offline", "unavailable"]:
                    add_finding(FAIL, f"'{hk}' = '{val}' — ERROR STATE")
                else:
                    add_finding(WARN, f"'{hk}' = '{val}' — unknown health value")

            # ── Count assertions (must be > 0) ──
            for ck in count_keys:
                val = data.get(ck)
                if val is not None:
                    try:
                        if int(val) <= 0:
                            add_finding(WARN, f"'{ck}' = {val} — expected > 0")
                        else:
                            add_finding(PASS, f"'{ck}' = {val}")
                    except (ValueError, TypeError):
                        add_finding(WARN, f"'{ck}' = {val} — not a valid number")

            # ── List assertions (must not be empty) ──
            for lk in list_keys:
                val = data.get(lk)
                if isinstance(val, list):
                    if len(val) == 0:
                        add_finding(WARN, f"'{lk}' is empty list — frontend may show blank")
                    else:
                        add_finding(PASS, f"'{lk}' has {len(val)} item(s)")
                elif val is None:
                    pass  # already caught by key presence check
                else:
                    add_finding(WARN, f"'{lk}' expected list, got {type(val).__name__}")

            # ── Nested checks (dot-notation deep inspection) ──
            for dotted_key, check in nested_checks.items():
                val = get_nested(data, dotted_key)
                if val is None:
                    add_finding(FAIL, f"Nested key '{dotted_key}' not found or null")
                else:
                    expected_type = check.get("type")
                    min_val = check.get("min")
                    if expected_type and not isinstance(val, expected_type):
                        add_finding(WARN,
                            f"'{dotted_key}' type mismatch: expected {expected_type.__name__}, "
                            f"got {type(val).__name__} = {val}"
                        )
                    elif min_val is not None:
                        try:
                            if int(val) < min_val:
                                add_finding(FAIL, f"'{dotted_key}' = {val} — below minimum {min_val}")
                            else:
                                add_finding(PASS, f"'{dotted_key}' = {val:,}")
                        except (ValueError, TypeError):
                            add_finding(WARN, f"'{dotted_key}' = {val} — cannot compare to min")
                    else:
                        add_finding(PASS, f"'{dotted_key}' = {val}")

            # ── Content checks (string length, etc.) ──
            for ck_key, ck_rule in content_checks.items():
                val = data.get(ck_key)
                if val is not None:
                    min_len = ck_rule.get("min_length", 0)
                    label = ck_rule.get("label", f"'{ck_key}' content check failed")
                    if isinstance(val, str) and len(val) < min_len:
                        add_finding(WARN, f"{label} (length={len(val)}, min={min_len})")
                    elif isinstance(val, list) and len(val) < min_len:
                        add_finding(WARN, f"{label} (count={len(val)}, min={min_len})")
                    else:
                        size = len(val) if hasattr(val, '__len__') else val
                        add_finding(PASS, f"'{ck_key}' content OK (size={size})")

    except requests.exceptions.ConnectionError as e:
        add_finding(ERROR, f"Connection refused — is the service running? ({str(e)[:80]})")
    except requests.exceptions.Timeout:
        add_finding(ERROR, f"Request timed out after {REQUEST_TIMEOUT}s")
    except Exception as e:
        add_finding(ERROR, f"Unexpected error: {type(e).__name__}: {str(e)[:100]}")

    return result


# ─── Output Formatting ──────────────────────────────────────────────────────
def print_divider(char="─", width=90):
    print(char * width)

def print_header(title: str, width=90):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")

def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024*1024):.1f}MB"

def print_result_verbose(r: dict, index: int, verbose: bool = False):
    """Print a single test result with full detail"""
    icon = SEVERITY_ICON[r["severity"]]
    timing = f"{r['response_time_ms']}ms" if r["response_time_ms"] is not None else "N/A"
    size = format_bytes(r["response_size"]) if r["response_size"] is not None else "N/A"
    http = r["http_status"] if r["http_status"] is not None else "N/A"

    print(f"\n┌{'─'*88}┐")
    print(f"│ {icon} [{index:02d}] {r['description']:<72} {r['severity']:>5} │")
    print(f"├{'─'*88}┤")
    print(f"│  Route:     {r['method']} {r['path']:<70} │")
    print(f"│  Component: {r['vue_component']:<72} │")
    print(f"│  HTTP:      {str(http):<10}  Time: {timing:<10}  Size: {size:<10}              │")

    if r["response_keys"]:
        keys_str = ", ".join(r["response_keys"])
        # Wrap long key lists
        while len(keys_str) > 72:
            cut = keys_str[:72].rfind(",")
            if cut == -1:
                cut = 72
            print(f"│  Keys:      {keys_str[:cut+1]:<72} │")
            keys_str = keys_str[cut+1:].strip()
        print(f"│  Keys:      {keys_str:<72} │")

    if r["findings"]:
        print(f"├{'─'*88}┤")
        print(f"│  {'Findings:':<84} │")
        for f in r["findings"]:
            ficon = SEVERITY_ICON[f["severity"]]
            msg = f["message"]
            # Wrap long messages
            first_line = True
            while len(msg) > 74:
                cut = msg[:74].rfind(" ")
                if cut == -1:
                    cut = 74
                if first_line:
                    print(f"│    {ficon} {msg[:cut]:<78} │")
                    first_line = False
                else:
                    print(f"│      {'':2}{msg[:cut]:<78} │")
                msg = msg[cut:].strip()
            if first_line:
                print(f"│    {ficon} {msg:<78} │")
            else:
                print(f"│      {'':2}{msg:<78} │")

    # ── Verbose: dump raw response ──
    if verbose and r["raw_response"] is not None:
        print(f"├{'─'*88}┤")
        print(f"│  {'Raw Response:':<84} │")
        if isinstance(r["raw_response"], dict):
            dumped = json.dumps(r["raw_response"], indent=2, default=str)
        else:
            dumped = str(r["raw_response"])
        for line in dumped.split("\n")[:40]:  # cap at 40 lines
            truncated = line[:82]
            print(f"│    {truncated:<82} │")
        if dumped.count("\n") > 40:
            print(f"│    {'... (truncated)':<82} │")

    print(f"└{'─'*88}┘")


def print_summary_table(results: list):
    """Print a compact summary table of all results"""
    print_header("SUMMARY TABLE")
    print(f"  {'#':>3}  {'Status':6}  {'Time':>7}  {'Size':>7}  {'HTTP':>4}  {'Findings':>8}  Description")
    print(f"  {'─'*3}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*4}  {'─'*8}  {'─'*40}")
    for i, r in enumerate(results, 1):
        icon = SEVERITY_ICON[r["severity"]]
        timing = f"{r['response_time_ms']}ms" if r["response_time_ms"] else "N/A"
        size = format_bytes(r["response_size"]) if r["response_size"] else "N/A"
        http = str(r["http_status"]) if r["http_status"] else "N/A"
        n_findings = len([f for f in r["findings"] if f["severity"] != PASS])
        desc = r["description"][:42]
        print(f"  {i:>3}  {icon:4}    {timing:>7}  {size:>7}  {http:>4}  {n_findings:>8}  {desc}")


def print_issues_only(results: list):
    """Print only WARN/FAIL/ERROR findings across all tests"""
    issues = []
    for r in results:
        for f in r["findings"]:
            if f["severity"] != PASS:
                issues.append({
                    "severity": f["severity"],
                    "endpoint": r["path"],
                    "description": r["description"],
                    "message": f["message"],
                })

    if not issues:
        print("\n  ✨ No issues detected across all contracts.\n")
        return

    print_header("ALL ISSUES (non-PASS findings)")
    # Sort by severity (ERROR > FAIL > WARN)
    issues.sort(key=lambda x: -SEVERITY_RANK[x["severity"]])

    for i, issue in enumerate(issues, 1):
        icon = SEVERITY_ICON[issue["severity"]]
        print(f"  {i:>2}. {icon} [{issue['severity']}] {issue['description']}")
        print(f"      {issue['endpoint']}")
        print(f"      → {issue['message']}")
        print()


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    json_output = "--json" in sys.argv

    print_header(f"ECHO BRAIN — FRONTEND CONTRACT VALIDATOR")
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Proxy:      {FRONTEND_BASE}")
    print(f"  Direct:     {DIRECT_BASE}")
    print(f"  Contracts:  {len(CONTRACTS)}")
    print(f"  Mode:       {'VERBOSE' if verbose else 'NORMAL'}")

    results = []
    total_time = 0

    for i, contract in enumerate(CONTRACTS, 1):
        desc = contract["description"]
        sys.stdout.write(f"  [{i:02d}/{len(CONTRACTS):02d}] {desc:40} ")
        sys.stdout.flush()

        r = run_contract(contract, verbose)
        results.append(r)

        icon = SEVERITY_ICON[r["severity"]]
        timing = f"{r['response_time_ms']}ms" if r["response_time_ms"] else "ERR"
        total_time += r["response_time_ms"] or 0
        print(f"{icon} {timing}")

    # ── Summary Table ──
    print_summary_table(results)

    # ── Detailed Results ──
    print_header("DETAILED RESULTS")
    for i, r in enumerate(results, 1):
        print_result_verbose(r, i, verbose)

    # ── Issues Rollup ──
    print_issues_only(results)

    # ── Final Scoreboard ──
    counts = {PASS: 0, WARN: 0, FAIL: 0, ERROR: 0}
    for r in results:
        counts[r["severity"]] += 1

    print_header("FINAL VERDICT")
    print(f"  Total Contracts: {len(results)}")
    print(f"  ✅ PASS:  {counts[PASS]}")
    print(f"  ⚠️  WARN:  {counts[WARN]}")
    print(f"  ❌ FAIL:  {counts[FAIL]}")
    print(f"  💀 ERROR: {counts[ERROR]}")
    print(f"  Total Response Time: {total_time}ms")
    print()

    if counts[ERROR] > 0 or counts[FAIL] > 0:
        print("  🔴 FRONTEND WILL HAVE ISSUES — fix FAIL/ERROR items above")
        exit_code = 1
    elif counts[WARN] > 0:
        print("  🟡 FRONTEND WORKS BUT WITH DEGRADED DATA — review WARN items")
        exit_code = 0
    else:
        print("  🟢 ALL CONTRACTS VALID — frontend should work correctly")
        exit_code = 0

    print(f"{'═' * 90}\n")

    # ── Optional JSON dump ──
    if json_output:
        json_path = "/tmp/echo_brain_contract_results.json"
        serializable = []
        for r in results:
            entry = dict(r)
            # raw_response may not be serializable
            if isinstance(entry.get("raw_response"), dict):
                entry["raw_response"] = json.loads(json.dumps(entry["raw_response"], default=str))
            else:
                entry["raw_response"] = str(entry.get("raw_response", ""))[:500]
            serializable.append(entry)
        with open(json_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "proxy": FRONTEND_BASE,
                "total_contracts": len(results),
                "counts": counts,
                "total_time_ms": total_time,
                "results": serializable,
            }, f, indent=2, default=str)
        print(f"  JSON results written to {json_path}")

    return exit_code


if __name__ == "__main__":
    exit(main())