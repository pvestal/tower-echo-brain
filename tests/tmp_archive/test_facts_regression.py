#!/usr/bin/env python3
"""
Echo Brain Facts Regression Test
=================================
Run this BEFORE and AFTER any Echo Brain changes to ensure fact retrieval
hasn't regressed. Designed to catch the recurring problem where sessions
accidentally delete curated facts or break retrieval paths.

Exit codes:
  0 = all tests passed
  1 = one or more tests failed

Usage:
  cd /opt/tower-echo-brain && ./venv/bin/python scripts/test_facts_regression.py

  # Quick mode (just pass/fail, no detail):
  cd /opt/tower-echo-brain && ./venv/bin/python scripts/test_facts_regression.py --quick

  # As a systemd pre-flight or cron:
  /opt/tower-echo-brain/venv/bin/python /opt/tower-echo-brain/scripts/test_facts_regression.py
"""

import sys
import json
import time
import argparse
from datetime import datetime

import httpx

# ── Config ──────────────────────────────────────────────────────────
API_BASE = "http://localhost:8309"
ASK_ENDPOINT = f"{API_BASE}/api/echo/ask"
HEALTH_ENDPOINT = f"{API_BASE}/health"
TIMEOUT = 30

# ── Test Cases ──────────────────────────────────────────────────────
# (query, [required_substrings], description)
# At least ONE required_substring must appear in the answer (case-insensitive).

FACT_TESTS = [
    (
        "What vehicles does Patrick own?",
        ["tundra", "1794"],
        "Patrick's 2022 Toyota Tundra 1794 Edition",
    ),
    (
        "What GPUs are in Tower?",
        ["3060", "9070"],
        "RTX 3060 12GB and RX 9070 XT 16GB",
    ),
    (
        "What embedding model does Echo Brain use?",
        ["nomic-embed-text", "768"],
        "nomic-embed-text with 768 dimensions",
    ),
    (
        "What anime projects is Patrick working on?",
        ["tokyo debt desire", "cyberpunk goblin slayer"],
        "Tokyo Debt Desire and/or Cyberpunk Goblin Slayer",
    ),
    (
        "What RV does Patrick have?",
        ["sundowner", "trailblazer"],
        "2021 Sundowner Trailblazer 2286TB",
    ),
    (
        "What processor does Tower have?",
        ["ryzen 9", "24-core", "24 core"],
        "AMD Ryzen 9 24-core",
    ),
    (
        "How much RAM does Tower have?",
        ["96", "ddr6"],
        "96GB DDR6",
    ),
]


def check_health() -> bool:
    """Verify Echo Brain API is reachable."""
    try:
        resp = httpx.get(HEALTH_ENDPOINT, timeout=5)
        return resp.status_code == 200 and resp.json().get("status") == "healthy"
    except Exception:
        return False


def query_echo(question: str) -> dict:
    """Send a question to /api/echo/ask and return the full response."""
    try:
        resp = httpx.post(
            ASK_ENDPOINT,
            json={"question": question},
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}", "answer": ""}
    except Exception as e:
        return {"error": str(e), "answer": ""}


def test_contains_any(answer: str, required: list[str]) -> tuple[bool, str]:
    """Check if the answer contains at least one of the required substrings."""
    answer_lower = answer.lower()
    for term in required:
        if term.lower() in answer_lower:
            return True, term
    return False, ""


def run_tests(quick: bool = False) -> int:
    """Run all fact regression tests. Returns exit code."""
    timestamp = datetime.now().isoformat()

    if not quick:
        print("=" * 70)
        print("ECHO BRAIN FACTS REGRESSION TEST")
        print(f"Started: {timestamp}")
        print("=" * 70)

    # Pre-flight: health check
    if not check_health():
        print("❌ ABORT: Echo Brain API is not healthy")
        print(f"   Check: curl {HEALTH_ENDPOINT}")
        return 1

    if not quick:
        print(f"✓ API healthy\n")

    passed = 0
    failed = 0
    results = []

    for question, required, description in FACT_TESTS:
        if not quick:
            print(f"Testing: {description}")
            print(f"  Query: {question}")

        response = query_echo(question)
        answer = response.get("answer", "")

        if "error" in response and response["error"]:
            success = False
            match = ""
            if not quick:
                print(f"  ❌ ERROR: {response['error']}")
        else:
            success, match = test_contains_any(answer, required)
            if not quick:
                # Truncate answer for display
                display = answer[:150].replace("\n", " ")
                if len(answer) > 150:
                    display += "..."
                print(f"  Answer: {display}")

                if success:
                    print(f"  ✅ PASS (matched: '{match}')")
                else:
                    print(f"  ❌ FAIL (needed one of: {required})")
                print()

        if success:
            passed += 1
        else:
            failed += 1

        results.append({
            "question": question,
            "description": description,
            "passed": success,
            "matched_term": match,
            "answer_length": len(answer),
        })

    # ── Summary ────────────────────────────────────────────────────
    total = passed + failed
    rate = (passed / total * 100) if total else 0

    if quick:
        status = "PASS" if failed == 0 else "FAIL"
        print(f"{status}: {passed}/{total} ({rate:.0f}%)")
        if failed > 0:
            for r in results:
                if not r["passed"]:
                    print(f"  ❌ {r['description']}")
    else:
        print("=" * 70)
        print(f"RESULTS: {passed}/{total} passed ({rate:.0f}%)")
        print("=" * 70)
        if failed > 0:
            print("\nFailed tests:")
            for r in results:
                if not r["passed"]:
                    print(f"  ❌ {r['description']}")
                    print(f"     Query: {r['question']}")
        print()

    # ── Write JSON report ──────────────────────────────────────────
    report = {
        "timestamp": timestamp,
        "total": total,
        "passed": passed,
        "failed": failed,
        "success_rate": rate,
        "results": results,
    }
    report_path = "/tmp/echo_brain_regression_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    if not quick:
        print(f"Report saved: {report_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo Brain facts regression test")
    parser.add_argument("--quick", action="store_true", help="Minimal output, just pass/fail")
    args = parser.parse_args()

    exit_code = run_tests(quick=args.quick)
    sys.exit(exit_code)