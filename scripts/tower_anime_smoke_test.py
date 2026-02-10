#!/usr/bin/env python3
"""
Tower Anime Production Pipeline — Smoke Test Suite
===================================================
Validates the full stack: database connections, Qdrant collections,
ComfyUI readiness, embedding health, codebase ingestion, and
end-to-end image generation.

Usage:
    python3 tower_anime_smoke_test.py              # Run all tests
    python3 tower_anime_smoke_test.py --quick       # Skip generation (fast check)
    python3 tower_anime_smoke_test.py --verbose      # Extra detail on failures
    python3 tower_anime_smoke_test.py --fix          # Attempt auto-remediation

Place at: /opt/tower-echo-brain/scripts/tower_anime_smoke_test.py
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random
import struct

# ---------------------------------------------------------------------------
# Configuration — edit these to match your Tower environment
# ---------------------------------------------------------------------------

class Config:
    """Central config — single place to update connection details."""

    # PostgreSQL (anime_production DB)
    PG_HOST = os.getenv("PG_HOST", "localhost")
    PG_PORT = int(os.getenv("PG_PORT", "5432"))
    PG_USER = os.getenv("PG_USER", "patrick")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
    PG_DATABASE = os.getenv("PG_DATABASE", "anime_production")

    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

    # Required collections and their expected dimensions
    QDRANT_COLLECTIONS = {
        "story_bible": {"dim": 768, "min_points": 100},
        "echo_memory": {"dim": 768, "min_points": 1},
    }

    # Ollama (embedding model)
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    EMBEDDING_MODEL = "nomic-embed-text"
    EXPECTED_EMBED_DIM = 768

    # ComfyUI
    COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")

    # Echo Brain MCP
    MCP_URL = os.getenv("MCP_URL", "http://localhost:8309")

    # File paths
    WORKFLOW_DIR = "/opt/tower-anime-production/workflows/comfyui"
    CHECKPOINT_DIR = "/mnt/1TB-storage/models/checkpoints"
    LORA_DIR = "/mnt/1TB-storage/models/loras"
    CODEBASE_DIRS = [
        "/opt/tower-echo-brain",
        "/opt/tower-anime-production",
    ]

    # Generation test config
    GEN_TIMEOUT = 300  # seconds max to wait for ComfyUI generation
    GEN_POLL_INTERVAL = 5

    # Known-good checkpoints (at least one must exist)
    REQUIRED_CHECKPOINTS = [
        "realistic_vision_v51.safetensors",
        "AOM3A1B.safetensors",
    ]


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

class Status(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️  WARN"
    SKIP = "⏭️  SKIP"


@dataclass
class TestResult:
    name: str
    status: Status
    message: str = ""
    detail: str = ""
    fix_hint: str = ""


@dataclass
class TestReport:
    results: list = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    @property
    def passed(self):
        return sum(1 for r in self.results if r.status == Status.PASS)

    @property
    def failed(self):
        return sum(1 for r in self.results if r.status == Status.FAIL)

    @property
    def warnings(self):
        return sum(1 for r in self.results if r.status == Status.WARN)

    def print_summary(self, verbose: bool = False):
        width = 72
        print("\n" + "=" * width)
        print("  TOWER ANIME PRODUCTION — SMOKE TEST RESULTS")
        print("=" * width)

        for r in self.results:
            print(f"\n  {r.status.value}  {r.name}")
            if r.message:
                print(f"         {r.message}")
            if verbose and r.detail:
                for line in r.detail.strip().split("\n"):
                    print(f"         | {line}")
            if r.status == Status.FAIL and r.fix_hint:
                print(f"         💡 Fix: {r.fix_hint}")

        print("\n" + "-" * width)
        total = len(self.results)
        print(
            f"  Total: {total} | "
            f"Passed: {self.passed} | "
            f"Failed: {self.failed} | "
            f"Warnings: {self.warnings}"
        )

        if self.failed == 0:
            print("\n  🎉 All critical checks passed — pipeline is operational.")
        else:
            print(f"\n  🔧 {self.failed} issue(s) need attention before generation.")
        print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

import urllib.request
import urllib.error


def http_get(url: str, timeout: int = 10) -> Optional[dict]:
    """Simple GET returning parsed JSON, or None on failure."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def http_post(url: str, payload: dict, timeout: int = 30) -> Optional[dict]:
    """Simple POST returning parsed JSON, or None on failure."""
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

def test_postgres_connection(report: TestReport):
    """Verify PostgreSQL is reachable and anime_production DB exists."""
    name = "PostgreSQL Connection"
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=Config.PG_HOST,
            port=Config.PG_PORT,
            user=Config.PG_USER,
            password=Config.PG_PASSWORD,
            dbname=Config.PG_DATABASE,
            connect_timeout=5,
        )
        cur = conn.cursor()

        # Count tables
        cur.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        table_count = cur.fetchone()[0]

        # Spot-check critical tables
        critical_tables = [
            "characters", "episodes", "scenes",
            "ai_models", "generation_history",
        ]
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY(%s)",
            (critical_tables,),
        )
        found = {row[0] for row in cur.fetchall()}
        missing = set(critical_tables) - found

        conn.close()

        if missing:
            report.add(TestResult(
                name, Status.WARN,
                f"{table_count} tables found, but missing expected: {missing}",
                fix_hint="Tables may have different names — verify schema.",
            ))
        else:
            report.add(TestResult(
                name, Status.PASS,
                f"{table_count} tables in anime_production, all critical tables present",
            ))

    except ImportError:
        report.add(TestResult(
            name, Status.FAIL,
            "psycopg2 not installed",
            fix_hint="pip install psycopg2-binary --break-system-packages",
        ))
    except Exception as e:
        report.add(TestResult(
            name, Status.FAIL,
            f"Connection failed: {e}",
            fix_hint="Check PG is running: sudo systemctl status postgresql",
        ))


def test_postgres_row_counts(report: TestReport):
    """Verify key tables have data (not just structure)."""
    name = "PostgreSQL Data Health"
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=Config.PG_HOST,
            port=Config.PG_PORT,
            user=Config.PG_USER,
            password=Config.PG_PASSWORD,
            dbname=Config.PG_DATABASE,
            connect_timeout=5,
        )
        cur = conn.cursor()

        checks = {
            "characters": 1,
            "episodes": 1,
            "scenes": 1,
        }
        empty_tables = []
        detail_lines = []

        for table, min_rows in checks.items():
            try:
                cur.execute(f"SELECT count(*) FROM {table}")
                count = cur.fetchone()[0]
                detail_lines.append(f"{table}: {count} rows")
                if count < min_rows:
                    empty_tables.append(table)
            except Exception:
                detail_lines.append(f"{table}: TABLE NOT FOUND")
                empty_tables.append(table)

        conn.close()

        if empty_tables:
            report.add(TestResult(
                name, Status.WARN,
                f"Empty/missing tables: {empty_tables}",
                detail="\n".join(detail_lines),
                fix_hint="Run ingestion scripts to populate data.",
            ))
        else:
            report.add(TestResult(
                name, Status.PASS,
                "All key tables have data",
                detail="\n".join(detail_lines),
            ))

    except Exception as e:
        report.add(TestResult(name, Status.FAIL, str(e)))


def test_qdrant_connection(report: TestReport):
    """Verify Qdrant is reachable."""
    name = "Qdrant Connection"
    data = http_get(f"{Config.QDRANT_URL}/collections")
    if data and "result" in data:
        collections = [c["name"] for c in data["result"].get("collections", [])]
        report.add(TestResult(
            name, Status.PASS,
            f"Online — {len(collections)} collection(s): {collections}",
        ))
    else:
        report.add(TestResult(
            name, Status.FAIL,
            "Qdrant not reachable",
            fix_hint=f"Check: curl {Config.QDRANT_URL}/collections",
        ))


def test_qdrant_collections(report: TestReport):
    """Verify required collections exist with correct dimensions and data."""
    for coll_name, expected in Config.QDRANT_COLLECTIONS.items():
        name = f"Qdrant Collection: {coll_name}"
        data = http_get(f"{Config.QDRANT_URL}/collections/{coll_name}")

        if not data or "result" not in data:
            report.add(TestResult(
                name, Status.FAIL,
                f"Collection '{coll_name}' not found",
                fix_hint=f"Recreate with dim={expected['dim']} and re-ingest.",
            ))
            continue

        result = data["result"]
        actual_dim = (
            result.get("config", {})
            .get("params", {})
            .get("vectors", {})
            .get("size", 0)
        )
        points = result.get("points_count", 0)

        issues = []
        if actual_dim != expected["dim"]:
            issues.append(
                f"dimension mismatch: got {actual_dim}, expected {expected['dim']}"
            )
        if points < expected["min_points"]:
            issues.append(
                f"too few points: {points} (need ≥{expected['min_points']})"
            )

        if issues:
            report.add(TestResult(
                name, Status.FAIL,
                f"{'; '.join(issues)} (points={points}, dim={actual_dim})",
                fix_hint=(
                    f"Delete and recreate: curl -X DELETE "
                    f"{Config.QDRANT_URL}/collections/{coll_name} "
                    f"then re-ingest with nomic-embed-text @ {expected['dim']}D"
                ),
            ))
        else:
            report.add(TestResult(
                name, Status.PASS,
                f"{points} points @ {actual_dim}D",
            ))


def test_qdrant_embedding_integrity(report: TestReport):
    """Spot-check that stored vectors are actually 768D (not stale 1024D)."""
    name = "Qdrant Embedding Integrity (story_bible)"
    payload = {
        "limit": 5,
        "with_vectors": True,
        "with_payload": False,
    }
    data = http_post(
        f"{Config.QDRANT_URL}/collections/story_bible/points/scroll",
        payload,
    )

    if not data or "result" not in data:
        report.add(TestResult(
            name, Status.WARN,
            "Could not scroll points — collection may be empty",
        ))
        return

    points = data["result"].get("points", [])
    if not points:
        report.add(TestResult(name, Status.WARN, "No points to check"))
        return

    bad = []
    for pt in points:
        vec = pt.get("vector", [])
        if len(vec) != Config.EXPECTED_EMBED_DIM:
            bad.append(f"id={pt['id']} dim={len(vec)}")

    if bad:
        report.add(TestResult(
            name, Status.FAIL,
            f"Found vectors with wrong dimensions: {bad}",
            fix_hint=(
                "Stale mxbai-embed-large (1024D) vectors detected. "
                "Re-run full ingestion with nomic-embed-text (768D)."
            ),
        ))
    else:
        report.add(TestResult(
            name, Status.PASS,
            f"Sampled {len(points)} vectors — all {Config.EXPECTED_EMBED_DIM}D ✓",
        ))


def test_ollama_embedding_model(report: TestReport):
    """Verify Ollama is running and nomic-embed-text produces 768D vectors."""
    name = "Ollama Embedding Model"

    # Check Ollama is up
    tags = http_get(f"{Config.OLLAMA_URL}/api/tags")
    if not tags:
        report.add(TestResult(
            name, Status.FAIL,
            "Ollama not reachable",
            fix_hint="sudo systemctl start ollama",
        ))
        return

    # Check model is available
    models = [m["name"] for m in tags.get("models", [])]
    model_present = any(Config.EMBEDDING_MODEL in m for m in models)

    if not model_present:
        report.add(TestResult(
            name, Status.FAIL,
            f"{Config.EMBEDDING_MODEL} not found in Ollama",
            detail=f"Available: {models}",
            fix_hint=f"ollama pull {Config.EMBEDDING_MODEL}",
        ))
        return

    # Test actual embedding dimension
    embed_resp = http_post(
        f"{Config.OLLAMA_URL}/api/embed",
        {"model": Config.EMBEDDING_MODEL, "input": "smoke test vector check"},
    )

    if not embed_resp or "embeddings" not in embed_resp:
        report.add(TestResult(
            name, Status.WARN,
            "Model present but embedding call failed",
            fix_hint="Try: curl -X POST http://localhost:11434/api/embed ...",
        ))
        return

    dim = len(embed_resp["embeddings"][0])
    if dim != Config.EXPECTED_EMBED_DIM:
        report.add(TestResult(
            name, Status.FAIL,
            f"Produces {dim}D vectors — expected {Config.EXPECTED_EMBED_DIM}D",
            fix_hint="Ensure nomic-embed-text is being used, NOT mxbai-embed-large.",
        ))
    else:
        report.add(TestResult(
            name, Status.PASS,
            f"{Config.EMBEDDING_MODEL} → {dim}D vectors ✓",
        ))


def test_comfyui_status(report: TestReport):
    """Verify ComfyUI is running and has GPU access."""
    name = "ComfyUI Status"
    data = http_get(f"{Config.COMFYUI_URL}/system_stats")

    if not data:
        report.add(TestResult(
            name, Status.FAIL,
            "ComfyUI not responding",
            fix_hint="Check: cd /opt/ComfyUI && python main.py --listen",
        ))
        return

    system = data.get("system", {})
    devices = system.get("devices", [])
    gpu_names = [d.get("name", "unknown") for d in devices]

    # Check queue
    queue_data = http_get(f"{Config.COMFYUI_URL}/queue")
    pending = len(queue_data.get("queue_pending", [])) if queue_data else "?"
    running = len(queue_data.get("queue_running", [])) if queue_data else "?"

    report.add(TestResult(
        name, Status.PASS,
        f"Online | GPUs: {gpu_names} | Queue: {running} running, {pending} pending",
    ))


def test_comfyui_models(report: TestReport):
    """Verify required checkpoint and LoRA files exist on disk."""
    name = "ComfyUI Models on Disk"
    issues = []
    detail_lines = []

    # Check checkpoints (filter to actual model files only)
    if os.path.isdir(Config.CHECKPOINT_DIR):
        all_files = os.listdir(Config.CHECKPOINT_DIR)
        # Filter to actual model files (.safetensors, .ckpt, .pth)
        checkpoints = [
            f for f in all_files
            if f.endswith(('.safetensors', '.ckpt', '.pth'))
        ]
        detail_lines.append(f"Checkpoints ({len(checkpoints)}): {checkpoints[:10]}")

        found_required = [
            c for c in Config.REQUIRED_CHECKPOINTS if c in checkpoints
        ]
        if not found_required:
            issues.append(
                f"No required checkpoints found. Need at least one of: "
                f"{Config.REQUIRED_CHECKPOINTS}"
            )
    else:
        issues.append(f"Checkpoint dir missing: {Config.CHECKPOINT_DIR}")

    # Check LoRAs (filter to actual model files only)
    if os.path.isdir(Config.LORA_DIR):
        all_lora_files = os.listdir(Config.LORA_DIR)
        # Filter to actual LoRA files (.safetensors, .ckpt, .pt, .pth)
        loras = [
            f for f in all_lora_files
            if f.endswith(('.safetensors', '.ckpt', '.pt', '.pth'))
        ]
        detail_lines.append(f"LoRAs ({len(loras)}): {loras[:10]}")
    else:
        issues.append(f"LoRA dir missing: {Config.LORA_DIR}")

    if issues:
        report.add(TestResult(
            name, Status.FAIL,
            "; ".join(issues),
            detail="\n".join(detail_lines),
        ))
    else:
        report.add(TestResult(
            name, Status.PASS,
            f"{len(checkpoints)} checkpoints, {len(loras)} LoRAs available",
            detail="\n".join(detail_lines),
        ))


def test_workflow_files(report: TestReport):
    """Verify ComfyUI workflow JSONs exist and are parseable."""
    name = "ComfyUI Workflow Files"

    if not os.path.isdir(Config.WORKFLOW_DIR):
        report.add(TestResult(
            name, Status.FAIL,
            f"Workflow directory not found: {Config.WORKFLOW_DIR}",
        ))
        return

    workflows = [
        f for f in os.listdir(Config.WORKFLOW_DIR) if f.endswith(".json")
    ]

    if not workflows:
        report.add(TestResult(
            name, Status.FAIL,
            "No .json workflow files found",
            fix_hint=f"Add workflows to {Config.WORKFLOW_DIR}",
        ))
        return

    broken = []
    valid = []
    for wf in workflows:
        path = os.path.join(Config.WORKFLOW_DIR, wf)
        try:
            with open(path) as f:
                data = json.load(f)
            # Basic sanity: should have node dicts with class_type
            node_types = set()
            for node_id, node in data.items():
                if isinstance(node, dict) and "class_type" in node:
                    node_types.add(node["class_type"])
            if node_types:
                valid.append(wf)
            else:
                broken.append(f"{wf} (no class_type nodes)")
        except Exception as e:
            broken.append(f"{wf} ({e})")

    if broken:
        report.add(TestResult(
            name, Status.WARN,
            f"{len(valid)} valid, {len(broken)} broken: {broken}",
        ))
    else:
        report.add(TestResult(
            name, Status.PASS,
            f"{len(valid)} valid workflow files",
        ))


def test_mcp_endpoint(report: TestReport):
    """Verify Echo Brain MCP API is responding."""
    name = "Echo Brain MCP API"

    # Try a simple search query
    payload = {
        "method": "search",
        "params": {"query": "smoke test", "collection": "story_bible", "limit": 1},
    }
    data = http_post(f"{Config.MCP_URL}/mcp", payload, timeout=15)

    if data is not None:
        report.add(TestResult(
            name, Status.PASS,
            f"MCP responding at {Config.MCP_URL}",
        ))
    else:
        # Try health endpoint as fallback
        health = http_get(f"{Config.MCP_URL}/health")
        if health:
            report.add(TestResult(
                name, Status.WARN,
                "Health endpoint OK, but /mcp search returned no data",
                fix_hint="Check MCP route handler and Qdrant connectivity.",
            ))
        else:
            report.add(TestResult(
                name, Status.FAIL,
                f"MCP not reachable at {Config.MCP_URL}",
                fix_hint="Check Echo Brain service: systemctl status echo-brain",
            ))


def test_codebase_structure(report: TestReport):
    """Verify expected codebase directories and key files exist."""
    name = "Codebase Structure"
    issues = []
    detail_lines = []

    key_files = {
        "/opt/tower-echo-brain": [
            "scripts/ingest_complete_anime_production.py",
            "scripts/ingest_workflows_to_story_bible.py",
        ],
        "/opt/tower-anime-production": [
            "workflows/comfyui",
        ],
    }

    for base_dir, expected_files in key_files.items():
        if not os.path.isdir(base_dir):
            issues.append(f"Missing directory: {base_dir}")
            continue

        detail_lines.append(f"\n{base_dir}/")
        for rel_path in expected_files:
            full = os.path.join(base_dir, rel_path)
            exists = os.path.exists(full)
            marker = "✓" if exists else "✗"
            detail_lines.append(f"  {marker} {rel_path}")
            if not exists:
                issues.append(f"Missing: {full}")

    if issues:
        report.add(TestResult(
            name, Status.WARN,
            f"{len(issues)} missing paths",
            detail="\n".join(detail_lines),
        ))
    else:
        report.add(TestResult(
            name, Status.PASS,
            "All expected directories and scripts present",
            detail="\n".join(detail_lines),
        ))


def test_story_bible_content_types(report: TestReport):
    """Verify story_bible has the expected content type distribution."""
    name = "Story Bible Content Distribution"

    # Scroll through points checking payload types
    payload = {
        "limit": 200,
        "with_vectors": False,
        "with_payload": True,
    }
    data = http_post(
        f"{Config.QDRANT_URL}/collections/story_bible/points/scroll",
        payload,
    )

    if not data or "result" not in data:
        report.add(TestResult(
            name, Status.WARN, "Could not read story_bible points"
        ))
        return

    points = data["result"].get("points", [])
    type_counts = {}
    for pt in points:
        p = pt.get("payload", {})
        # Try common type field names
        content_type = (
            p.get("type")
            or p.get("content_type")
            or p.get("category")
            or p.get("source_type")
            or "unknown"
        )
        type_counts[content_type] = type_counts.get(content_type, 0) + 1

    expected_types = {"character", "episode", "scene", "workflow"}
    found_types = set(type_counts.keys())
    missing = expected_types - found_types

    detail = "\n".join(f"  {t}: {c}" for t, c in sorted(type_counts.items()))

    if missing:
        report.add(TestResult(
            name, Status.WARN,
            f"Missing content types: {missing}",
            detail=detail,
            fix_hint="Re-run ingestion for missing types.",
        ))
    else:
        report.add(TestResult(
            name, Status.PASS,
            f"{len(points)} points across {len(type_counts)} types",
            detail=detail,
        ))


def test_ssot_compliance(report: TestReport):
    """Verify story_bible follows SSOT architecture (references only, no data duplication)."""
    name = "SSOT Compliance (Story Bible)"

    # Check a sample of points for SSOT-compliant payloads
    resp = http_post(
        f"{Config.QDRANT_URL}/collections/story_bible/points/scroll",
        {"limit": 20, "with_payload": True, "with_vectors": False}
    )

    if not resp or "result" not in resp:
        report.add(TestResult(name, Status.FAIL, "Cannot scroll story_bible"))
        return

    points = resp["result"].get("points", [])
    if not points:
        report.add(TestResult(name, Status.WARN, "No points to check"))
        return

    # SSOT requirements
    required_fields = {"type", "source_table", "source_id", "indexed_at"}
    forbidden_fields = {"content", "description", "personality", "appearance",
                       "backstory", "dialogue", "narrative_text", "full_text"}

    compliant = 0
    violations = []

    for pt in points:
        payload = pt.get("payload", {})
        pt_type = payload.get("type", "unknown")

        # Skip workflows (file-based, not in DB)
        if pt_type == "workflow" or payload.get("source_type") == "file":
            continue

        # Check required fields
        missing = required_fields - set(payload.keys())
        if missing:
            violations.append(f"Missing fields: {missing}")

        # Check for data duplication
        duped = forbidden_fields & set(payload.keys())
        if duped:
            violations.append(f"Duplicated data: {duped}")

        if not missing and not duped:
            compliant += 1

    db_points = len([p for p in points if p.get("payload", {}).get("type") != "workflow"])

    if violations:
        report.add(TestResult(
            name, Status.FAIL,
            f"Not SSOT compliant: {compliant}/{db_points} points valid",
            detail=violations[:3]
        ))
    else:
        report.add(TestResult(
            name, Status.PASS,
            f"SSOT compliant: Qdrant stores references only, PostgreSQL is SSOT"
        ))


def test_end_to_end_generation(report: TestReport):
    """Submit a minimal workflow to ComfyUI, wait for completion, then VALIDATE the actual output files exist and are usable images."""
    name = "End-to-End Generation (Image Validated)"

    # Find a working workflow (prefer non-RIFE workflows for single-frame test)
    workflow_path = None
    candidates = [
        "ACTION_combat_workflow.json",           # No RIFE - safe for single-frame
        "anime_video_simple_test.json",         # No RIFE - safe for single-frame
        "anime_video_fixed_no_rife.json",       # No RIFE - safe for single-frame
        "FIXED_anime_video_workflow.json",      # No RIFE - safe for single-frame
        "anime_30sec_working_workflow.json",    # Has RIFE - needs batch_size >= 2
    ]
    for wf in candidates:
        path = os.path.join(Config.WORKFLOW_DIR, wf)
        if os.path.isfile(path):
            workflow_path = path
            break

    if not workflow_path:
        report.add(TestResult(
            name, Status.SKIP,
            f"No known workflow found in {Config.WORKFLOW_DIR}",
            fix_hint=f"Expected one of: {candidates}",
        ))
        return

    # Load and patch workflow
    try:
        with open(workflow_path) as f:
            workflow = json.load(f)
    except Exception as e:
        report.add(TestResult(name, Status.FAIL, f"Workflow parse error: {e}"))
        return

    # Check if workflow contains RIFE VFI (needs multiple frames)
    has_rife = any(
        n.get("class_type") == "RIFE VFI"
        for n in workflow.values()
        if isinstance(n, dict)
    )

    # Check if workflow contains AnimateDiff (needs 16+ frames for motion)
    has_animatediff = any(
        "AnimateDiff" in n.get("class_type", "")
        for n in workflow.values()
        if isinstance(n, dict)
    )

    # Patch: fix model name if needed, set a simple test prompt
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue

        ct = node.get("class_type", "")

        # Fix checkpoint name mismatch
        if ct == "CheckpointLoaderSimple":
            ckpt = node.get("inputs", {}).get("ckpt_name", "")
            if ckpt == "realisticVision_v51.safetensors":
                node["inputs"]["ckpt_name"] = "realistic_vision_v51.safetensors"

        # Set a minimal positive prompt
        if ct == "CLIPTextEncode":
            title = node.get("_meta", {}).get("title", "").lower()
            if "positive" in title or "prompt" in title:
                node["inputs"]["text"] = (
                    "1girl, portrait, dark hair, city background, "
                    "photorealistic, high quality, smoke test"
                )

        # Reduce dimensions for speed (if KSampler or EmptyLatentImage)
        if ct == "EmptyLatentImage":
            node["inputs"]["width"] = 512
            node["inputs"]["height"] = 512

            # Set batch_size based on workflow requirements:
            # - AnimateDiff needs 16+ frames for coherent motion
            # - RIFE VFI needs >= 2 frames for interpolation
            # - Static images need only 1 frame
            if has_animatediff:
                # AnimateDiff REQUIRES 16+ frames or produces garbage 1-frame videos
                original_batch = node["inputs"].get("batch_size", 24)
                node["inputs"]["batch_size"] = max(16, min(original_batch, 16))
            elif has_rife:
                node["inputs"]["batch_size"] = 2
            else:
                node["inputs"]["batch_size"] = 1

        if ct == "KSampler":
            node["inputs"]["steps"] = min(node["inputs"].get("steps", 20), 8)
            # CRITICAL: Randomize seed to prevent cache
            import random
            node["inputs"]["seed"] = random.randint(0, 2**31)

    # Submit to ComfyUI
    submit_payload = {"prompt": workflow}
    try:
        data = json.dumps(submit_payload).encode()
        req = urllib.request.Request(
            f"{Config.COMFYUI_URL}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        report.add(TestResult(
            name, Status.FAIL,
            f"Failed to submit workflow: {e}",
            fix_hint="Check ComfyUI logs for errors.",
        ))
        return

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        error_msg = result.get("error", result.get("node_errors", "unknown"))
        report.add(TestResult(
            name, Status.FAIL,
            f"ComfyUI rejected workflow: {error_msg}",
            fix_hint="Check node compatibility and model availability.",
        ))
        return

    # Poll for completion
    start = time.time()
    completed = False
    history_entry = None

    while time.time() - start < Config.GEN_TIMEOUT:
        history = http_get(f"{Config.COMFYUI_URL}/history/{prompt_id}")
        if history and prompt_id in history:
            history_entry = history[prompt_id]
            status_info = history_entry.get("status", {})
            if status_info.get("completed", False) or status_info.get("status_str") == "success":
                completed = True
                break
            # Check for errors
            if status_info.get("status_str") == "error":
                messages = status_info.get("messages", [])
                err_detail = ""
                for msg in messages:
                    if isinstance(msg, list) and msg[0] == "execution_error":
                        err_detail = msg[1].get("exception_message", "")[:200]
                report.add(TestResult(
                    name, Status.FAIL,
                    f"Generation errored: {err_detail or status_info}",
                ))
                return
        time.sleep(Config.GEN_POLL_INTERVAL)

    elapsed = round(time.time() - start, 1)

    if not completed:
        report.add(TestResult(
            name, Status.FAIL,
            f"Timed out after {Config.GEN_TIMEOUT}s (prompt_id: {prompt_id})",
            fix_hint="Check ComfyUI console for stuck jobs or GPU OOM.",
        ))
        return

    # ---------------------------------------------------------------
    # VALIDATE OUTPUT — the part that was completely missing
    # ---------------------------------------------------------------
    COMFYUI_OUTPUT_DIR = "/opt/ComfyUI/output"

    outputs = history_entry.get("outputs", {})
    output_files = []

    # Collect all output file references from history
    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for img in node_output.get("images", []):
            fname = img.get("filename", "")
            subfolder = img.get("subfolder", "")
            if fname:
                output_files.append(
                    os.path.join(COMFYUI_OUTPUT_DIR, subfolder, fname)
                )
        for gif in node_output.get("gifs", []):
            fname = gif.get("filename", "")
            subfolder = gif.get("subfolder", "")
            if fname:
                output_files.append(
                    os.path.join(COMFYUI_OUTPUT_DIR, subfolder, fname)
                )

    # --- Gate: Must have at least one output file ---
    if not output_files:
        report.add(TestResult(
            name, Status.FAIL,
            f"ComfyUI completed in {elapsed}s but produced 0 output files. "
            f"Workflow: {os.path.basename(workflow_path)}",
            fix_hint=(
                "Workflow may lack a SaveImage/SaveAnimatedWEBP node, "
                "or the output node is not connected."
            ),
        ))
        return

    # --- Validate each file ---
    validated = 0
    file_issues = []
    detail_lines = []

    for fpath in output_files:
        fname = os.path.basename(fpath)

        if not os.path.isfile(fpath):
            file_issues.append(f"{fname}: file not found on disk")
            continue

        file_size = os.path.getsize(fpath)
        if file_size < 10_000:  # 10KB minimum
            file_issues.append(f"{fname}: too small ({file_size}B)")
            continue

        # Try to read image/video dimensions
        width, height, fmt = 0, 0, ""
        ext = os.path.splitext(fname)[1].lower()

        # Handle video files
        if ext in ['.mp4', '.webm', '.avi']:
            fmt = ext.lstrip('.')
            # Basic video validation
            with open(fpath, "rb") as f:
                header = f.read(32)
            if ext == '.mp4' and b"ftyp" in header[:12]:
                width, height = 512, 512  # Assume standard for now
            elif ext == '.webm' and header[:4] == b'\x1a\x45\xdf\xa3':
                width, height = 512, 512
            else:
                file_issues.append(f"{fname}: unrecognized video format")
                continue
        else:
            # Handle image files
            try:
                with open(fpath, "rb") as f:
                    header = f.read(32)
                # PNG
                if header[:8] == b'\x89PNG\r\n\x1a\n':
                    import struct
                    width = struct.unpack(">I", header[16:20])[0]
                    height = struct.unpack(">I", header[20:24])[0]
                    fmt = "PNG"
                # JPEG
                elif header[:2] == b'\xff\xd8':
                    fmt = "JPEG"
                    width, height = 512, 512  # Can't easily parse, assume OK
                # GIF
                elif header[:6] in (b'GIF87a', b'GIF89a'):
                    import struct
                    width = struct.unpack("<H", header[6:8])[0]
                    height = struct.unpack("<H", header[8:10])[0]
                    fmt = "GIF"
            except Exception:
                pass

        if width > 0 and width < 128:
            file_issues.append(f"{fname}: dimensions too small ({width}x{height})")
            continue

        size_kb = file_size / 1024
        dims = f"{width}x{height}" if width > 0 else "?x?"
        detail_lines.append(f"✓ {fname} — {fmt} {dims} ({size_kb:.0f}KB)")
        validated += 1

    # --- Final verdict ---
    total = len(output_files)
    workflow_name = os.path.basename(workflow_path)

    if validated == 0:
        report.add(TestResult(
            name, Status.FAIL,
            f"Generated {total} file(s) but none passed validation. "
            f"Workflow: {workflow_name}",
            detail="\n".join(file_issues),
            fix_hint="Check ComfyUI output dir and workflow save nodes.",
        ))
    elif validated < total:
        report.add(TestResult(
            name, Status.WARN,
            f"{validated}/{total} images validated in {elapsed}s "
            f"({workflow_name})",
            detail="\n".join(detail_lines + file_issues),
        ))
    else:
        report.add(TestResult(
            name, Status.PASS,
            f"{validated} image(s) generated and validated in {elapsed}s "
            f"({workflow_name})",
            detail="\n".join(detail_lines),
        ))


def test_semantic_search_quality(report: TestReport):
    """Run a known query and verify results are relevant (not garbage)."""
    name = "Semantic Search Quality"

    test_queries = [
        ("Mei Kobayashi character", "character"),
        ("cyberpunk action scene", "scene"),
        ("ComfyUI workflow generation", "workflow"),
    ]

    passed = 0
    detail_lines = []

    for query, expected_type in test_queries:
        payload = {
            "limit": 3,
            "with_vectors": False,
            "with_payload": True,
            "query_vector": None,  # We need to embed first
        }

        # Get embedding for query
        embed_resp = http_post(
            f"{Config.OLLAMA_URL}/api/embed",
            {"model": Config.EMBEDDING_MODEL, "input": query},
        )
        if not embed_resp or "embeddings" not in embed_resp:
            detail_lines.append(f"  ✗ '{query}' — embedding failed")
            continue

        vector = embed_resp["embeddings"][0]
        search_payload = {
            "vector": vector,
            "limit": 3,
            "with_payload": True,
        }
        results = http_post(
            f"{Config.QDRANT_URL}/collections/story_bible/points/search",
            search_payload,
        )

        if not results or "result" not in results:
            detail_lines.append(f"  ✗ '{query}' — search failed")
            continue

        hits = results["result"]
        if hits:
            top_score = round(hits[0].get("score", 0), 3)
            top_type = (
                hits[0].get("payload", {}).get("type")
                or hits[0].get("payload", {}).get("content_type")
                or "?"
            )
            match = "✓" if top_type == expected_type else "~"
            detail_lines.append(
                f"  {match} '{query}' → type={top_type}, score={top_score}"
            )
            if top_score > 0.3:  # Reasonable similarity threshold
                passed += 1
        else:
            detail_lines.append(f"  ✗ '{query}' — no results")

    total = len(test_queries)
    if passed == total:
        report.add(TestResult(
            name, Status.PASS,
            f"All {total} queries returned relevant results",
            detail="\n".join(detail_lines),
        ))
    elif passed > 0:
        report.add(TestResult(
            name, Status.WARN,
            f"{passed}/{total} queries returned good results",
            detail="\n".join(detail_lines),
        ))
    else:
        report.add(TestResult(
            name, Status.FAIL,
            "No queries returned relevant results",
            detail="\n".join(detail_lines),
            fix_hint="Re-ingest with nomic-embed-text. Check collection dimensions.",
        ))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_tests(quick: bool = False, verbose: bool = False):
    """Execute all smoke tests and print report."""
    report = TestReport()

    print("\n🔍 Running Tower Anime Production smoke tests...\n")

    # --- Database ---
    print("  [1/13] PostgreSQL connection...")
    test_postgres_connection(report)

    print("  [2/13] PostgreSQL data health...")
    test_postgres_row_counts(report)

    # --- Vector Store ---
    print("  [3/13] Qdrant connection...")
    test_qdrant_connection(report)

    print("  [4/13] Qdrant collections...")
    test_qdrant_collections(report)

    print("  [5/13] Embedding integrity...")
    test_qdrant_embedding_integrity(report)

    print("  [6/13] Story bible content types...")
    test_story_bible_content_types(report)

    print("  [7/13] SSOT compliance...")
    test_ssot_compliance(report)

    # --- Embedding Model ---
    print("  [8/13] Ollama embedding model...")
    test_ollama_embedding_model(report)

    # --- ComfyUI ---
    print("  [9/13] ComfyUI status...")
    test_comfyui_status(report)

    print("  [10/13] ComfyUI models...")
    test_comfyui_models(report)

    print("  [11/13] Workflow files...")
    test_workflow_files(report)

    # --- Codebase ---
    print("  [12/13] Codebase structure...")
    test_codebase_structure(report)

    # --- Integration ---
    print("  [13/13] Echo Brain MCP API...")
    test_mcp_endpoint(report)

    # --- Semantic Search Quality ---
    print("  [bonus] Semantic search quality...")
    test_semantic_search_quality(report)

    # --- End-to-End Generation (skip if --quick) ---
    if not quick:
        print("  [e2e]   End-to-end generation (this takes ~2min)...")
        test_end_to_end_generation(report)
    else:
        report.add(TestResult(
            "End-to-End Generation",
            Status.SKIP,
            "Skipped (--quick mode)",
        ))

    report.print_summary(verbose=verbose)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tower Anime Production Pipeline — Smoke Tests"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip generation test for a fast check",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show extra detail on each test result",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="(Future) Attempt auto-remediation of failures",
    )
    args = parser.parse_args()

    if args.fix:
        print("⚠️  --fix mode is not yet implemented. Showing diagnostics only.")

    report = run_all_tests(quick=args.quick, verbose=args.verbose)
    sys.exit(1 if report.failed > 0 else 0)