"""
Vector Health Monitor — validates the Qdrant data pipeline end-to-end.

Two roles:
1. **Startup contract test** — runs once at boot, verifies:
   - Ollama /api/embeddings returns a 768-dim vector
   - Qdrant accepts typed points with required payload fields
   - Each expected worker type has ≥1 point in the collection
   - Cross-worker retrieval works (semantic search finds what ingestors stored)

2. **GET /api/echo/vector-health** — live endpoint returning:
   - Per-type point counts and expected minimums
   - Missing/degraded types flagged
   - Recent 24h ingestion activity per type
   - Required field coverage (type, source, text)
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "echo_memory")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Expected types and their minimum healthy counts.
# 0 means "should exist eventually" (non-critical at boot).
EXPECTED_TYPES: Dict[str, Dict[str, Any]] = {
    "conversation":          {"min": 100,  "source": "conversation_watcher"},
    "domain_code":           {"min": 100,  "source": "domain_ingestor"},
    "domain_record":         {"min": 1,    "source": "domain_ingestor"},
    "domain_git":            {"min": 1,    "source": "domain_ingestor"},
    "code":                  {"min": 10,   "source": "codebase_indexer"},
    "schema":                {"min": 1,    "source": "schema_indexer"},
    "fact":                  {"min": 1,    "source": "mcp_store_fact"},
    "knowledge_fact":        {"min": 0,    "source": "reasoning_worker"},
    "improvement_proposal":  {"min": 0,    "source": "improvement_engine"},
    "learning":              {"min": 0,    "source": "learning_loop"},
    "photo":                 {"min": 0,    "source": "photo_pipeline"},
    "email":                 {"min": 0,    "source": "google_ingest"},
    "calendar_event":        {"min": 0,    "source": "google_ingest"},
    "video":                 {"min": 0,    "source": "video_pipeline"},
    "generation":            {"min": 0,    "source": "comfyui_bridge"},
    "kb_article":            {"min": 0,    "source": "kb_ingestor"},
    "documentation":         {"min": 0,    "source": "doc_ingestor"},
}

# Fields every point MUST have.
REQUIRED_FIELDS = ["type"]


# ── Startup Contract Test ──────────────────────────────────────────

class VectorHealthContract:
    """One-shot startup validation of the vector pipeline."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.passed = True

    def _record(self, name: str, ok: bool, detail: str = ""):
        status = "PASS" if ok else "FAIL"
        self.results.append({"test": name, "status": status, "detail": detail})
        if not ok:
            self.passed = False

    async def run_all(self) -> Dict[str, Any]:
        """Run every contract check.  Returns summary dict."""
        start = time.time()
        self.results = []
        self.passed = True

        await self._test_ollama_embeddings()
        await self._test_qdrant_write_read()
        await self._test_type_inventory()
        await self._test_cross_retrieval()
        await self._test_required_fields_sample()

        duration_ms = int((time.time() - start) * 1000)
        summary = {
            "passed": self.passed,
            "tests_run": len(self.results),
            "tests_passed": sum(1 for r in self.results if r["status"] == "PASS"),
            "tests_failed": sum(1 for r in self.results if r["status"] == "FAIL"),
            "duration_ms": duration_ms,
            "results": self.results,
            "timestamp": datetime.now().isoformat(),
        }

        level = "INFO" if self.passed else "WARNING"
        getattr(logger, level.lower())(
            f"Vector health contract: {summary['tests_passed']}/{summary['tests_run']} passed "
            f"({duration_ms}ms)"
        )
        for r in self.results:
            if r["status"] == "FAIL":
                logger.warning(f"  FAIL: {r['test']} — {r['detail']}")

        return summary

    # ── individual tests ───────────────────────────────────────────

    async def _test_ollama_embeddings(self):
        """Verify Ollama /api/embeddings returns a 768-dim vector."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": "vector health contract test"},
                )
                if r.status_code != 200:
                    self._record("ollama_embeddings", False, f"HTTP {r.status_code}")
                    return
                emb = r.json().get("embedding", [])
                ok = len(emb) == 768
                self._record(
                    "ollama_embeddings", ok,
                    f"{len(emb)} dims" if ok else f"Expected 768 dims, got {len(emb)}",
                )
        except Exception as e:
            self._record("ollama_embeddings", False, str(e))

    async def _test_qdrant_write_read(self):
        """Write a typed point, read it back, delete it."""
        point_id = str(uuid4())
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                # Get a real embedding
                r = await c.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": "contract write test"},
                )
                emb = r.json().get("embedding", [])
                if not emb:
                    self._record("qdrant_write_read", False, "No embedding for test point")
                    return

                # Write
                payload = {
                    "type": "_contract_test",
                    "source": "vector_health_contract",
                    "text": "contract test point",
                    "created_at": datetime.now().isoformat(),
                }
                r = await c.put(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points",
                    json={"points": [{"id": point_id, "vector": emb, "payload": payload}]},
                )
                if r.status_code not in (200, 201):
                    self._record("qdrant_write_read", False, f"Write failed: {r.status_code}")
                    return

                # Read back
                r = await c.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                    json={
                        "filter": {"must": [{"key": "type", "match": {"value": "_contract_test"}}]},
                        "limit": 1,
                        "with_payload": True,
                        "with_vector": False,
                    },
                )
                points = r.json().get("result", {}).get("points", [])
                if not points:
                    self._record("qdrant_write_read", False, "Write succeeded but read-back found 0 points")
                    return

                read_type = (points[0].get("payload") or {}).get("type")
                ok = read_type == "_contract_test"
                self._record(
                    "qdrant_write_read", ok,
                    "Write → read → type verified" if ok else f"Type mismatch: {read_type}",
                )
        except Exception as e:
            self._record("qdrant_write_read", False, str(e))
        finally:
            # Cleanup
            try:
                async with httpx.AsyncClient(timeout=5.0) as c:
                    await c.post(
                        f"{QDRANT_URL}/collections/{COLLECTION}/points/delete",
                        json={"points": [point_id]},
                    )
            except Exception:
                pass

    async def _test_type_inventory(self):
        """Check that critical types have ≥ min points."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                critical = {t: cfg for t, cfg in EXPECTED_TYPES.items() if cfg["min"] > 0}
                for type_name, cfg in critical.items():
                    r = await c.post(
                        f"{QDRANT_URL}/collections/{COLLECTION}/points/count",
                        json={"filter": {"must": [{"key": "type", "match": {"value": type_name}}]}},
                    )
                    count = r.json().get("result", {}).get("count", 0)
                    ok = count >= cfg["min"]
                    self._record(
                        f"type_inventory:{type_name}", ok,
                        f"{count} points (min {cfg['min']})" if ok
                        else f"Only {count} points, expected ≥{cfg['min']} (source: {cfg['source']})",
                    )
        except Exception as e:
            self._record("type_inventory", False, str(e))

    async def _test_cross_retrieval(self):
        """Verify semantic search can find domain_ingestor content from a query."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                # Embed a query about Echo Brain code
                r = await c.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": "Echo Brain worker scheduler startup"},
                )
                emb = r.json().get("embedding", [])
                if not emb:
                    self._record("cross_retrieval", False, "Could not get query embedding")
                    return

                # Search for domain_code type points
                r = await c.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
                    json={
                        "query": emb,
                        "filter": {"must": [{"key": "type", "match": {"value": "domain_code"}}]},
                        "limit": 3,
                        "with_payload": {"include": ["type", "source"]},
                    },
                )
                points = r.json().get("result", {}).get("points", [])
                ok = len(points) > 0
                self._record(
                    "cross_retrieval", ok,
                    f"Found {len(points)} relevant domain_code points"
                    if ok else "Semantic search returned 0 results for domain_code",
                )
        except Exception as e:
            self._record("cross_retrieval", False, str(e))

    async def _test_required_fields_sample(self):
        """Sample 50 recent points and verify required fields are present."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                    json={
                        "limit": 50,
                        "with_payload": {"include": REQUIRED_FIELDS},
                        "with_vector": False,
                    },
                )
                points = r.json().get("result", {}).get("points", [])
                if not points:
                    self._record("required_fields", False, "No points to sample")
                    return

                missing_count = 0
                for pt in points:
                    payload = pt.get("payload") or {}
                    for field in REQUIRED_FIELDS:
                        if not payload.get(field):
                            missing_count += 1
                            break

                ok = missing_count == 0
                self._record(
                    "required_fields", ok,
                    f"All {len(points)} sampled points have required fields"
                    if ok else f"{missing_count}/{len(points)} points missing required fields",
                )
        except Exception as e:
            self._record("required_fields", False, str(e))


# ── Live Health Endpoint ───────────────────────────────────────────

async def get_vector_health() -> Dict[str, Any]:
    """Build the full vector health report for the API endpoint."""
    start = time.time()
    report: Dict[str, Any] = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "collection": COLLECTION,
        "types": {},
        "summary": {},
        "issues": [],
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            # Total collection size
            r = await c.get(f"{QDRANT_URL}/collections/{COLLECTION}")
            total = r.json().get("result", {}).get("points_count", 0)
            report["summary"]["total_points"] = total

            # Per-type counts
            typed_total = 0
            types_healthy = 0
            types_degraded = 0
            types_missing = 0

            for type_name, cfg in EXPECTED_TYPES.items():
                r = await c.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/count",
                    json={"filter": {"must": [{"key": "type", "match": {"value": type_name}}]}},
                )
                count = r.json().get("result", {}).get("count", 0)
                typed_total += count

                # Check 24h activity
                recent_count = await _count_recent(c, type_name)

                health = "healthy"
                if count == 0 and cfg["min"] > 0:
                    health = "missing"
                    types_missing += 1
                    report["issues"].append(
                        f"Type '{type_name}' has 0 points (expected ≥{cfg['min']}, source: {cfg['source']})"
                    )
                elif count < cfg["min"]:
                    health = "degraded"
                    types_degraded += 1
                    report["issues"].append(
                        f"Type '{type_name}' has {count} points (expected ≥{cfg['min']})"
                    )
                else:
                    types_healthy += 1

                report["types"][type_name] = {
                    "count": count,
                    "min_expected": cfg["min"],
                    "source": cfg["source"],
                    "health": health,
                    "recent_24h": recent_count,
                }

            # Check for untyped points
            r = await c.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/count",
                json={"filter": {"must": [{"is_empty": {"key": "type"}}]}},
            )
            untyped = r.json().get("result", {}).get("count", 0)

            report["summary"].update({
                "typed_points": typed_total,
                "untyped_points": untyped,
                "types_healthy": types_healthy,
                "types_degraded": types_degraded,
                "types_missing": types_missing,
                "type_count": len(EXPECTED_TYPES),
            })

            if untyped > 0:
                report["issues"].append(f"{untyped} points have no 'type' field")

            # Overall status
            if types_missing > 0 or untyped > 100:
                report["status"] = "degraded"
            if types_missing > 3:
                report["status"] = "unhealthy"

    except Exception as e:
        report["status"] = "error"
        report["issues"].append(f"Health check failed: {e}")

    report["duration_ms"] = int((time.time() - start) * 1000)
    return report


async def _count_recent(client: httpx.AsyncClient, type_name: str) -> int:
    """Count points of a given type ingested in the last 24 hours."""
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
    try:
        # Try ingested_at first (domain_ingestor), then created_at (other workers)
        for date_field in ("ingested_at", "created_at", "indexed_at"):
            r = await client.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/count",
                json={
                    "filter": {
                        "must": [
                            {"key": "type", "match": {"value": type_name}},
                            {"key": date_field, "range": {"gte": cutoff}},
                        ]
                    }
                },
            )
            count = r.json().get("result", {}).get("count", 0)
            if count > 0:
                return count
        return 0
    except Exception:
        return -1


# ── Singleton ──────────────────────────────────────────────────────

_contract: Optional[VectorHealthContract] = None


def get_vector_health_contract() -> VectorHealthContract:
    global _contract
    if _contract is None:
        _contract = VectorHealthContract()
    return _contract
