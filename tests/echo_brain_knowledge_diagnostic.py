#!/usr/bin/env python3
"""
Echo Brain Knowledge Quality Diagnostic
=========================================
Goes beyond smoke tests (is it alive?) to answer:
  "Does Echo Brain actually KNOW things correctly?"

Three test tiers:
  1. FACTUAL   — Questions with known correct answers (scored automatically)
  2. NAVIGATION — "Where is X in the codebase?" (scored by keyword presence)
  3. REASONING  — Architecture, history, recommendations (human-reviewed)

Two test domains:
  A. Echo Brain self-knowledge
  B. Tower Anime Production knowledge

Usage:
    pip install requests rich --break-system-packages

    # Full diagnostic
    python3 echo_brain_knowledge_diagnostic.py

    # Specific domain
    python3 echo_brain_knowledge_diagnostic.py --domain echo
    python3 echo_brain_knowledge_diagnostic.py --domain anime

    # Specific tier
    python3 echo_brain_knowledge_diagnostic.py --tier factual
    python3 echo_brain_knowledge_diagnostic.py --tier navigation
    python3 echo_brain_knowledge_diagnostic.py --tier reasoning

    # Save JSON report
    python3 echo_brain_knowledge_diagnostic.py --output /tmp/knowledge_report.json
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

import requests

# ─── Configuration ───────────────────────────────────────────────────────────

ECHO_BRAIN_URL = "http://localhost:8309"
QUERY_ENDPOINT = "/api/echo/query"
QUERY_TIMEOUT = 45  # LLM responses can be slow


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class TestCase:
    """One knowledge question to ask Echo Brain."""
    domain: str          # "echo" or "anime"
    tier: str            # "factual", "navigation", "reasoning"
    question: str
    expected_keywords: list[str] = field(default_factory=list)
    anti_keywords: list[str] = field(default_factory=list)  # should NOT appear
    expected_answer: str = ""  # human-readable expected answer
    weight: float = 1.0       # importance multiplier

@dataclass
class TestResult:
    """Result of running one test case."""
    test: TestCase
    response: str = ""
    response_time_ms: float = 0.0
    keyword_hits: list[str] = field(default_factory=list)
    keyword_misses: list[str] = field(default_factory=list)
    anti_keyword_hits: list[str] = field(default_factory=list)  # bad - these appeared
    score: float = 0.0        # 0.0 to 1.0
    status: str = "PENDING"   # PASS / PARTIAL / FAIL / ERROR / SKIP
    notes: str = ""
    sources_found: int = 0


# ─── Test Definitions ───────────────────────────────────────────────────────
# Each test has:
#   - question: what to ask Echo Brain
#   - expected_keywords: words/phrases that SHOULD appear in a correct answer
#   - anti_keywords: words that should NOT appear (contamination check)
#   - expected_answer: human-readable correct answer for the report

ECHO_BRAIN_TESTS = [
    # ── FACTUAL: Known answers, auto-scored ──────────────────────────────
    TestCase(
        domain="echo",
        tier="factual",
        question="What port does Echo Brain run on?",
        expected_keywords=["8309"],
        expected_answer="8309",
        weight=1.0,
    ),
    TestCase(
        domain="echo",
        tier="factual",
        question="What embedding model does Echo Brain use and what are its dimensions?",
        expected_keywords=["nomic-embed-text", "768"],
        anti_keywords=["mxbai-embed-large", "1024"],
        expected_answer="nomic-embed-text with 768 dimensions",
        weight=2.0,  # Critical — this was a known confusion point
    ),
    TestCase(
        domain="echo",
        tier="factual",
        question="What are the three agent types in Echo Brain and what models do they use?",
        expected_keywords=["coding", "deepseek-coder", "reasoning", "deepseek-r1", "narration", "gemma"],
        expected_answer="CodingAgent (deepseek-coder-v2:16b), ReasoningAgent (deepseek-r1:8b), NarrationAgent (gemma2:9b)",
        weight=1.5,
    ),
    TestCase(
        domain="echo",
        tier="factual",
        question="What databases does Echo Brain use?",
        expected_keywords=["postgres", "qdrant"],
        expected_answer="PostgreSQL (echo_brain database) and Qdrant (vector store on port 6333)",
        weight=1.0,
    ),
    TestCase(
        domain="echo",
        tier="factual",
        question="How many modules and directories does Echo Brain have?",
        expected_keywords=["108", "29"],
        expected_answer="108 modules across 29 directories",
        weight=0.5,
    ),
    TestCase(
        domain="echo",
        tier="factual",
        question="What frontend stack does Echo Brain use?",
        expected_keywords=["vue", "typescript", "tailwind"],
        expected_answer="Vue 3 + TypeScript + Tailwind CSS",
        weight=0.5,
    ),

    # ── NAVIGATION: "Where is X?" — tests codebase retrieval ────────────
    TestCase(
        domain="echo",
        tier="navigation",
        question="Where is the agent routing logic defined? Show me the file path.",
        expected_keywords=["route", "router", "agent", ".py"],
        expected_answer="Should reference actual file path containing routing/classification logic",
        weight=1.5,
    ),
    TestCase(
        domain="echo",
        tier="navigation",
        question="Where is the MCP server implementation?",
        expected_keywords=["mcp", ".py"],
        expected_answer="Should reference file path to MCP server code",
        weight=1.0,
    ),
    TestCase(
        domain="echo",
        tier="navigation",
        question="What file handles the embedding/ingestion pipeline?",
        expected_keywords=["ingest", "embed", ".py"],
        expected_answer="Should reference the ingestion pipeline source file",
        weight=1.0,
    ),
    TestCase(
        domain="echo",
        tier="navigation",
        question="List all API endpoints that Echo Brain exposes.",
        expected_keywords=["health", "echo", "query"],
        expected_answer="Should list /health, /api/echo/query, /api/echo/chat, and others",
        weight=1.5,
    ),

    # ── REASONING: Architecture, history, recommendations ────────────────
    TestCase(
        domain="echo",
        tier="reasoning",
        question="What was the context contamination bug between Echo Brain and the anime system?",
        expected_keywords=["anime", "contamination", "separate", "shared"],
        anti_keywords=[],
        expected_answer="Anime content was bleeding into technical queries due to shared database/embeddings",
        weight=2.0,
    ),
    TestCase(
        domain="echo",
        tier="reasoning",
        question="Why did we switch from mxbai-embed-large to nomic-embed-text?",
        expected_keywords=["768", "token", "dimension"],
        anti_keywords=[],
        expected_answer="mxbai only supports 512 tokens context; nomic supports 8192 tokens with 768D vectors",
        weight=2.0,
    ),
    TestCase(
        domain="echo",
        tier="reasoning",
        question="What happens if Qdrant crashes? What is the failure mode?",
        expected_keywords=["vector", "search", "retrieval", "fail"],
        expected_answer="Should describe impact on retrieval/memory — queries degrade to no context",
        weight=1.0,
    ),
    TestCase(
        domain="echo",
        tier="reasoning",
        question="What are the top 3 architectural weaknesses in Echo Brain right now?",
        expected_keywords=[],  # Open-ended — human review
        expected_answer="Should mention: embedding confusion, no temporal ordering, code-heavy/context-light retrieval, deduplication",
        weight=1.0,
    ),
    TestCase(
        domain="echo",
        tier="reasoning",
        question="What environment variables are required to run Echo Brain?",
        expected_keywords=["env", "port", "database"],
        expected_answer="Should list actual env vars needed for deployment",
        weight=1.0,
    ),
]

ANIME_TESTS = [
    # ── FACTUAL ──────────────────────────────────────────────────────────
    TestCase(
        domain="anime",
        tier="factual",
        question="What are the two anime projects in the Tower production system?",
        expected_keywords=["tokyo debt desire", "cyberpunk goblin slayer"],
        anti_keywords=[],
        expected_answer="Tokyo Debt Desire (photorealistic) and Cyberpunk Goblin Slayer (arcane style)",
        weight=1.5,
    ),
    TestCase(
        domain="anime",
        tier="factual",
        question="What port does ComfyUI run on?",
        expected_keywords=["8188"],
        expected_answer="8188",
        weight=1.0,
    ),
    TestCase(
        domain="anime",
        tier="factual",
        question="What video generation tool is used for anime clips?",
        expected_keywords=["framepack"],
        expected_answer="FramePack for 60-second video clips",
        weight=1.0,
    ),
    TestCase(
        domain="anime",
        tier="factual",
        question="What visual style does each anime project use?",
        expected_keywords=["photorealistic", "arcane"],
        expected_answer="Tokyo Debt Desire = photorealistic, Cyberpunk Goblin Slayer = arcane style",
        weight=1.0,
    ),

    # ── NAVIGATION ───────────────────────────────────────────────────────
    TestCase(
        domain="anime",
        tier="navigation",
        question="Where is the production orchestrator code?",
        expected_keywords=["orchestrat", ".py"],
        expected_answer="Should reference the anime production orchestrator file path",
        weight=1.0,
    ),
    TestCase(
        domain="anime",
        tier="navigation",
        question="What ComfyUI workflows exist and where are they stored?",
        expected_keywords=["workflow", "comfyui", "json"],
        expected_answer="Should list workflow files and their storage location",
        weight=1.0,
    ),
    TestCase(
        domain="anime",
        tier="navigation",
        question="Where are LoRA models stored and what characters do they represent?",
        expected_keywords=["lora", "model", "character"],
        expected_answer="Should reference model storage paths and character-to-LoRA mapping",
        weight=1.5,
    ),

    # ── REASONING ────────────────────────────────────────────────────────
    TestCase(
        domain="anime",
        tier="reasoning",
        question="How was the cross-contamination between Echo Brain and the anime system resolved?",
        expected_keywords=["separate", "database", "split"],
        expected_answer="Systems were separated — distinct databases/collections to prevent anime bleeding into technical queries",
        weight=2.0,
    ),
    TestCase(
        domain="anime",
        tier="reasoning",
        question="What is the biggest bottleneck in anime production throughput?",
        expected_keywords=["gpu", "render", "vram", "time", "video"],
        expected_answer="Should identify GPU/VRAM constraints, render time, or video generation quality",
        weight=1.0,
    ),
    TestCase(
        domain="anime",
        tier="reasoning",
        question="How is GPU memory split between the RTX 3060 and RX 9070 XT for production?",
        expected_keywords=["3060", "9070", "gpu", "vram"],
        expected_answer="Should describe which GPU handles what workload",
        weight=1.0,
    ),
    TestCase(
        domain="anime",
        tier="reasoning",
        question="What video generation quality problems have been encountered?",
        expected_keywords=["quality", "video", "framepack"],
        expected_answer="Should describe specific quality issues with generated video",
        weight=1.0,
    ),
]


# ─── Query Engine ────────────────────────────────────────────────────────────

def query_echo_brain(question: str) -> tuple[str, float, dict]:
    """
    Send a question to Echo Brain and return (response_text, time_ms, raw_json).
    """
    url = f"{ECHO_BRAIN_URL}{QUERY_ENDPOINT}"
    payload = {"query": question}

    start = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=QUERY_TIMEOUT)
        elapsed_ms = (time.monotonic() - start) * 1000

        if resp.status_code != 200:
            return f"HTTP {resp.status_code}: {resp.text[:200]}", elapsed_ms, {}

        data = resp.json()
        # Echo Brain returns response in various fields — try common ones
        response_text = (
            data.get("response", "")
            or data.get("answer", "")
            or data.get("content", "")
            or data.get("message", "")
            or json.dumps(data)
        )
        return response_text, elapsed_ms, data

    except requests.exceptions.Timeout:
        elapsed_ms = (time.monotonic() - start) * 1000
        return "TIMEOUT", elapsed_ms, {}
    except requests.exceptions.ConnectionError:
        return "CONNECTION_REFUSED", 0.0, {}
    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        return f"ERROR: {e}", elapsed_ms, {}


# ─── Scoring Engine ──────────────────────────────────────────────────────────

def score_response(test: TestCase, response: str) -> TestResult:
    """
    Score a response against expected keywords and anti-keywords.

    Scoring:
      - Each expected keyword found = points toward 1.0
      - Each anti-keyword found = penalty
      - TIMEOUT/ERROR = 0.0
    """
    result = TestResult(test=test, response=response)

    if not response or response in ("TIMEOUT", "CONNECTION_REFUSED") or response.startswith("ERROR:"):
        result.status = "ERROR"
        result.score = 0.0
        result.notes = f"No valid response: {response[:100]}"
        return result

    resp_lower = response.lower()

    # Check expected keywords
    if test.expected_keywords:
        for kw in test.expected_keywords:
            if kw.lower() in resp_lower:
                result.keyword_hits.append(kw)
            else:
                result.keyword_misses.append(kw)

        keyword_score = len(result.keyword_hits) / len(test.expected_keywords)
    else:
        # No keywords to check (reasoning questions) — score as PARTIAL for human review
        keyword_score = 0.5  # neutral — needs human review

    # Check anti-keywords (contamination)
    for akw in test.anti_keywords:
        if akw.lower() in resp_lower:
            result.anti_keyword_hits.append(akw)

    # Anti-keyword penalty: each hit reduces score by 0.2
    penalty = len(result.anti_keyword_hits) * 0.2
    result.score = max(0.0, min(1.0, keyword_score - penalty))

    # Status thresholds
    if result.score >= 0.8:
        result.status = "PASS"
    elif result.score >= 0.4:
        result.status = "PARTIAL"
    else:
        result.status = "FAIL"

    # Flag contamination explicitly
    if result.anti_keyword_hits:
        result.notes = f"⚠️ CONTAMINATION: found [{', '.join(result.anti_keyword_hits)}]"

    # Flag no-keyword reasoning questions for human review
    if not test.expected_keywords:
        result.status = "REVIEW"
        result.notes = "No auto-score keywords — needs human review"

    return result


# ─── Report Generator ───────────────────────────────────────────────────────

STATUS_ICONS = {
    "PASS": "✅",
    "PARTIAL": "🟡",
    "FAIL": "❌",
    "ERROR": "💥",
    "SKIP": "⏭️",
    "REVIEW": "👁️",
}

def print_header(text: str):
    print(f"\n{'═' * 70}")
    print(f"  {text}")
    print(f"{'═' * 70}")

def print_subheader(text: str):
    print(f"\n{'─' * 50}")
    print(f"  {text}")
    print(f"{'─' * 50}")

def truncate(text: str, max_len: int = 200) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text

def print_result(i: int, result: TestResult):
    """Print one test result with context."""
    icon = STATUS_ICONS.get(result.status, "?")
    t = result.test

    print(f"\n  {icon} [{result.status}] Q{i}: {t.question}")
    print(f"     Expected: {t.expected_answer}")
    print(f"     Got:      {truncate(result.response)}")
    print(f"     Score:    {result.score:.0%}  |  Time: {result.response_time_ms:.0f}ms")

    if result.keyword_hits:
        print(f"     ✓ Found:  {', '.join(result.keyword_hits)}")
    if result.keyword_misses:
        print(f"     ✗ Missing: {', '.join(result.keyword_misses)}")
    if result.anti_keyword_hits:
        print(f"     ⚠ Contamination: {', '.join(result.anti_keyword_hits)}")
    if result.notes:
        print(f"     Note: {result.notes}")

def print_summary(results: list[TestResult], domain: str):
    """Print aggregate scores by tier."""
    print_header(f"SUMMARY — {domain.upper()}")

    tiers = {}
    for r in results:
        if r.test.domain != domain:
            continue
        tier = r.test.tier
        if tier not in tiers:
            tiers[tier] = {"pass": 0, "partial": 0, "fail": 0, "error": 0, "review": 0, "total": 0, "weighted_score": 0.0, "total_weight": 0.0}
        t = tiers[tier]
        t["total"] += 1
        t["weighted_score"] += r.score * r.test.weight
        t["total_weight"] += r.test.weight
        status_key = r.status.lower()
        if status_key in t:
            t[status_key] += 1

    overall_weighted = 0.0
    overall_weight = 0.0

    for tier_name in ["factual", "navigation", "reasoning"]:
        if tier_name not in tiers:
            continue
        t = tiers[tier_name]
        pct = (t["weighted_score"] / t["total_weight"] * 100) if t["total_weight"] > 0 else 0
        overall_weighted += t["weighted_score"]
        overall_weight += t["total_weight"]

        bar_filled = int(pct / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)

        print(f"\n  {tier_name.upper():12s}  {bar}  {pct:5.1f}%")
        print(f"    ✅ {t['pass']} pass  🟡 {t['partial']} partial  ❌ {t['fail']} fail  💥 {t['error']} error  👁️ {t['review']} review")

    if overall_weight > 0:
        overall_pct = overall_weighted / overall_weight * 100
        print(f"\n  {'OVERALL':12s}  {'━' * 20}  {overall_pct:5.1f}%  (weighted)")

        # Diagnosis
        print(f"\n  DIAGNOSIS:")
        if overall_pct >= 80:
            print("    Echo Brain has strong knowledge retrieval. Focus on edge cases.")
        elif overall_pct >= 50:
            print("    Mixed results. Code retrieval works but conceptual/historical knowledge is weak.")
            print("    → Ingest ADRs, README docs, and architectural summaries.")
        else:
            print("    Retrieval quality is poor. Likely causes:")
            print("    → Embedding dimension mismatch (check Qdrant vs model)")
            print("    → Missing ingestion of key documents")
            print("    → Deduplication issues flooding results with identical chunks")


def generate_json_report(results: list[TestResult]) -> dict:
    """Generate a machine-readable JSON report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "echo_brain_url": ECHO_BRAIN_URL,
        "total_tests": len(results),
        "results": [],
        "summary": {},
    }

    for r in results:
        report["results"].append({
            "domain": r.test.domain,
            "tier": r.test.tier,
            "question": r.test.question,
            "expected_answer": r.test.expected_answer,
            "actual_response": r.response[:500],
            "score": r.score,
            "status": r.status,
            "keyword_hits": r.keyword_hits,
            "keyword_misses": r.keyword_misses,
            "anti_keyword_hits": r.anti_keyword_hits,
            "response_time_ms": r.response_time_ms,
            "notes": r.notes,
        })

    # Aggregate by domain+tier
    for domain in ["echo", "anime"]:
        domain_results = [r for r in results if r.test.domain == domain]
        if not domain_results:
            continue
        report["summary"][domain] = {}
        for tier in ["factual", "navigation", "reasoning"]:
            tier_results = [r for r in domain_results if r.test.tier == tier]
            if not tier_results:
                continue
            weighted = sum(r.score * r.test.weight for r in tier_results)
            total_w = sum(r.test.weight for r in tier_results)
            report["summary"][domain][tier] = {
                "weighted_score_pct": round(weighted / total_w * 100, 1) if total_w else 0,
                "tests": len(tier_results),
                "pass": sum(1 for r in tier_results if r.status == "PASS"),
                "fail": sum(1 for r in tier_results if r.status in ("FAIL", "ERROR")),
            }

    return report


# ─── Main Runner ─────────────────────────────────────────────────────────────

def preflight_check() -> bool:
    """Verify Echo Brain is reachable before running tests."""
    print("\n🔍 Preflight Check")
    print(f"   Target: {ECHO_BRAIN_URL}")

    try:
        resp = requests.get(f"{ECHO_BRAIN_URL}/health", timeout=5)
        if resp.status_code == 200:
            health = resp.json()
            print(f"   Status: ✅ UP")
            print(f"   Health: {json.dumps(health, indent=2)[:200]}")
            return True
        else:
            print(f"   Status: ⚠️ HTTP {resp.status_code}")
            return True  # Still reachable, just maybe unhealthy
    except requests.exceptions.ConnectionError:
        print(f"   Status: ❌ UNREACHABLE")
        print(f"   → Is Echo Brain running? Try: sudo systemctl start tower-echo-brain")
        return False
    except Exception as e:
        print(f"   Status: ❌ ERROR: {e}")
        return False


def run_tests(tests: list[TestCase], domain_filter: str = None, tier_filter: str = None) -> list[TestResult]:
    """Run all matching tests and return results."""
    filtered = tests
    if domain_filter:
        filtered = [t for t in filtered if t.domain == domain_filter]
    if tier_filter:
        filtered = [t for t in filtered if t.tier == tier_filter]

    if not filtered:
        print("  No tests match the given filters.")
        return []

    results = []
    total = len(filtered)

    for i, test in enumerate(filtered, 1):
        label = f"[{test.domain}/{test.tier}]"
        print(f"\n  ({i}/{total}) {label} {test.question[:60]}...", end="", flush=True)

        response, time_ms, raw = query_echo_brain(test.question)
        result = score_response(test, response)
        result.response_time_ms = time_ms

        icon = STATUS_ICONS.get(result.status, "?")
        print(f" {icon} {result.score:.0%} ({time_ms:.0f}ms)")

        results.append(result)

        # Small delay to avoid hammering the LLM
        time.sleep(0.5)

    return results


def main():
    parser = argparse.ArgumentParser(description="Echo Brain Knowledge Quality Diagnostic")
    parser.add_argument("--domain", choices=["echo", "anime"], help="Test only one domain")
    parser.add_argument("--tier", choices=["factual", "navigation", "reasoning"], help="Test only one tier")
    parser.add_argument("--output", "-o", help="Save JSON report to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full responses")
    args = parser.parse_args()

    print_header("ECHO BRAIN KNOWLEDGE QUALITY DIAGNOSTIC")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target:    {ECHO_BRAIN_URL}")

    if not preflight_check():
        sys.exit(1)

    all_tests = ECHO_BRAIN_TESTS + ANIME_TESTS
    all_results = []

    # ── Run Echo Brain tests ─────────────────────────────────────────────
    if args.domain in (None, "echo"):
        print_header("DOMAIN: ECHO BRAIN SELF-KNOWLEDGE")

        for tier in ["factual", "navigation", "reasoning"]:
            if args.tier and args.tier != tier:
                continue
            print_subheader(f"Tier: {tier.upper()}")
            results = run_tests(ECHO_BRAIN_TESTS, domain_filter="echo", tier_filter=tier)
            all_results.extend(results)

            # Print detailed results for this tier
            for i, r in enumerate(results, 1):
                if args.verbose:
                    print_result(i, r)
                elif r.status in ("FAIL", "ERROR"):
                    print_result(i, r)  # Always show failures

    # ── Run Anime Production tests ───────────────────────────────────────
    if args.domain in (None, "anime"):
        print_header("DOMAIN: TOWER ANIME PRODUCTION")

        for tier in ["factual", "navigation", "reasoning"]:
            if args.tier and args.tier != tier:
                continue
            print_subheader(f"Tier: {tier.upper()}")
            results = run_tests(ANIME_TESTS, domain_filter="anime", tier_filter=tier)
            all_results.extend(results)

            for i, r in enumerate(results, 1):
                if args.verbose:
                    print_result(i, r)
                elif r.status in ("FAIL", "ERROR"):
                    print_result(i, r)

    # ── Summaries ────────────────────────────────────────────────────────
    if args.domain in (None, "echo"):
        print_summary(all_results, "echo")
    if args.domain in (None, "anime"):
        print_summary(all_results, "anime")

    # ── Cross-domain contamination check ─────────────────────────────────
    print_header("CROSS-DOMAIN CONTAMINATION CHECK")
    echo_results = [r for r in all_results if r.test.domain == "echo"]
    anime_leaks = [r for r in echo_results if any(
        kw in r.response.lower() for kw in ["lora", "comfyui", "anime", "waifu", "checkpoint"]
        if r.test.tier == "factual"  # Only check factual echo questions for anime bleed
    )]
    if anime_leaks:
        print(f"  ⚠️  {len(anime_leaks)} Echo Brain factual responses contain anime terminology!")
        for r in anime_leaks:
            print(f"     → Q: {r.test.question[:50]}...")
    else:
        print("  ✅ No anime content detected in Echo Brain factual responses.")

    # ── Response time analysis ───────────────────────────────────────────
    print_header("RESPONSE TIME ANALYSIS")
    valid_times = [r.response_time_ms for r in all_results if r.response_time_ms > 0]
    if valid_times:
        avg_ms = sum(valid_times) / len(valid_times)
        max_ms = max(valid_times)
        min_ms = min(valid_times)
        slowest = max(all_results, key=lambda r: r.response_time_ms)
        print(f"  Avg: {avg_ms:.0f}ms  |  Min: {min_ms:.0f}ms  |  Max: {max_ms:.0f}ms")
        print(f"  Slowest: {slowest.test.question[:50]}... ({slowest.response_time_ms:.0f}ms)")
        if avg_ms > 10000:
            print("  ⚠️ Average response time > 10s — check Ollama model loading")

    # ── JSON report ──────────────────────────────────────────────────────
    if args.output:
        report = generate_json_report(all_results)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  📄 JSON report saved to: {args.output}")

    # ── Final verdict ────────────────────────────────────────────────────
    print_header("NEXT STEPS BASED ON RESULTS")

    failed = [r for r in all_results if r.status in ("FAIL", "ERROR")]
    contaminated = [r for r in all_results if r.anti_keyword_hits]

    if failed:
        print(f"\n  {len(failed)} tests FAILED — questions Echo Brain can't answer:")
        for r in failed:
            print(f"    → {r.test.question[:60]}")
        print(f"\n  ACTION: These represent ingestion gaps. Create documents")
        print(f"  covering these topics and re-ingest into echo_memory collection.")

    if contaminated:
        print(f"\n  {len(contaminated)} responses had CONTAMINATION:")
        for r in contaminated:
            print(f"    → {r.test.question[:50]}... (found: {', '.join(r.anti_keyword_hits)})")
        print(f"\n  ACTION: Check vector collection separation. Anime vectors")
        print(f"  may still be in echo_memory collection.")

    review_items = [r for r in all_results if r.status == "REVIEW"]
    if review_items:
        print(f"\n  {len(review_items)} reasoning questions need HUMAN REVIEW.")
        print(f"  Run with --verbose to see full responses, then judge quality.")

    print(f"\n{'═' * 70}")
    print(f"  Done. {len(all_results)} tests completed.")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()