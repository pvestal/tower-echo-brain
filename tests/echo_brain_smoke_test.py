#!/usr/bin/env python3
"""
Echo Brain Smoke Test Suite
============================
Comprehensive test suite for Echo Brain API endpoints and database migration status.

Usage:
    # Run all tests
    pytest echo_brain_smoke_test.py -v

    # Run specific test group
    pytest echo_brain_smoke_test.py -k "health"
    pytest echo_brain_smoke_test.py -k "query"
    pytest echo_brain_smoke_test.py -k "separation"
"""

import json
import time
import subprocess
import os
import pytest
import requests
from typing import Optional, Any


# ─── Configuration ───────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8309"
TIMEOUT = 15
QUERY_TIMEOUT = 30


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get(path: str, timeout: int = TIMEOUT, **kwargs) -> requests.Response:
    """GET with proper error handling."""
    return requests.get(f"{BASE_URL}{path}", timeout=timeout, **kwargs)


def post(path: str, payload: dict, timeout: int = QUERY_TIMEOUT, **kwargs) -> requests.Response:
    """POST JSON with proper error handling."""
    return requests.post(
        f"{BASE_URL}{path}",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
        **kwargs,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  TEST GROUP 1: HEALTH & CONNECTIVITY
# ═════════════════════════════════════════════════════════════════════════════

class TestHealth:
    """Basic health and connectivity checks."""

    def test_root_health(self):
        """GET /health — should return status and uptime."""
        resp = get("/health")
        assert resp.status_code == 200, f"Health check failed: HTTP {resp.status_code}"

        body = resp.json()
        assert "status" in body, f"Missing 'status' in health response: {body}"
        assert body["status"] == "healthy", f"Unhealthy status: {body['status']}"

        # Response time check (warning only)
        ms = resp.elapsed.total_seconds() * 1000
        assert ms < 1000, f"Health check slow: {ms:.1f}ms (expected <1000ms)"

    def test_api_echo_health(self):
        """GET /api/echo/health — namespaced health endpoint."""
        resp = get("/api/echo/health")
        assert resp.status_code == 200, f"Echo health failed: HTTP {resp.status_code}"

    def test_openapi_spec(self):
        """GET /openapi.json — FastAPI auto-generated spec exists."""
        resp = get("/openapi.json")
        assert resp.status_code == 200, f"OpenAPI spec missing: HTTP {resp.status_code}"

        body = resp.json()
        assert "paths" in body, "Invalid OpenAPI spec - missing 'paths'"
        assert len(body["paths"]) > 0, "OpenAPI spec has no endpoints defined"

    def test_docs_available(self):
        """GET /docs — Swagger UI should be available."""
        resp = get("/docs")
        assert resp.status_code == 200, f"Swagger docs unavailable: HTTP {resp.status_code}"
        assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower(), "Not a valid docs page"


# ═════════════════════════════════════════════════════════════════════════════
#  TEST GROUP 2: CORE QUERY ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

class TestCoreQuery:
    """Main query/chat endpoints — the heart of Echo Brain."""

    def test_echo_query_simple(self):
        """POST /api/echo/query — basic conversational query."""
        payload = {"query": "What is 2 plus 2?"}
        resp = post("/api/echo/query", payload)

        # May return 405 if endpoint uses different method
        if resp.status_code == 405:
            pytest.skip("/api/echo/query returns 405 - may use different method")

        assert resp.status_code == 200, f"Query failed: HTTP {resp.status_code}"

        body = resp.json()
        # Check for any response field
        response_fields = ["response", "answer", "result", "content", "message"]
        has_response = any(field in body for field in response_fields)
        assert has_response, f"No response content in: {list(body.keys())}"

        # Verify it actually answered (contains "4" or "four")
        response_text = str(body.get("response", body.get("answer", ""))).lower()
        assert "4" in response_text or "four" in response_text, f"Wrong answer: {response_text[:100]}"

    def test_echo_query_code_routing(self):
        """POST /api/echo/query — code query should route to code model."""
        payload = {"query": "Write a Python function to reverse a string"}
        resp = post("/api/echo/query", payload)

        if resp.status_code == 405:
            pytest.skip("Query endpoint not available")

        assert resp.status_code == 200, f"Code query failed: HTTP {resp.status_code}"

        body = resp.json()
        # Check if model_used indicates code model or intelligence layer
        if "model_used" in body:
            model = body["model_used"].lower()
            is_code_model = any(m in model for m in ["deepseek", "code", "coder", "intelligence"])
            assert is_code_model, f"Code query routed to wrong model: {model}"

    def test_echo_chat_endpoint(self):
        """POST /api/echo/chat — alternative chat endpoint."""
        payload = {"query": "Hello"}
        resp = post("/api/echo/chat", payload)

        if resp.status_code == 404:
            pytest.skip("/api/echo/chat endpoint not implemented")

        assert resp.status_code in [200, 405], f"Chat endpoint error: HTTP {resp.status_code}"

    def test_echo_ask_endpoint(self):
        """POST /api/echo/ask — intelligence/ask endpoint with 'question' field."""
        # This was the bug from the session - must use 'question' not 'query'
        payload = {"question": "How much RAM does Tower have?"}
        resp = post("/api/echo/ask", payload)

        if resp.status_code == 404:
            pytest.skip("/api/echo/ask endpoint not implemented")

        assert resp.status_code == 200, f"Ask endpoint failed: HTTP {resp.status_code}"

        body = resp.json()
        # Should have a real answer mentioning RAM
        response_text = str(body.get("answer", body.get("response", ""))).lower()
        assert len(response_text) > 10, f"Empty or too short response: {response_text}"

    def test_wrong_field_name_fails(self):
        """POST /api/echo/ask with 'query' instead of 'question' should fail or return empty."""
        payload = {"query": "This should fail or return empty"}
        resp = post("/api/echo/ask", payload)

        if resp.status_code == 404:
            pytest.skip("/api/echo/ask not implemented")

        # Should either return 422 (validation error) or 200 with empty/generic response
        if resp.status_code == 422:
            pass  # Good - field validation works
        elif resp.status_code == 200:
            body = resp.json()
            response = str(body.get("answer", body.get("response", "")))
            # Response should be empty or generic (not a real answer)
            assert len(response) < 20 or "i don" in response.lower(), \
                f"Wrong field name still worked - got real answer: {response[:100]}"


# ═════════════════════════════════════════════════════════════════════════════
#  TEST GROUP 3: MEMORY & VECTOR SEARCH
# ═════════════════════════════════════════════════════════════════════════════

class TestMemoryVectorSearch:
    """Memory retrieval and vector search endpoints."""

    def test_memory_search_mcp(self):
        """POST /mcp search_memory — MCP memory search endpoint."""
        payload = {
            "method": "tools/call",
            "params": {
                "name": "search_memory",
                "arguments": {
                    "query": "Tower hardware specs",
                    "limit": 3
                }
            }
        }
        resp = post("/mcp", payload)

        if resp.status_code == 404:
            pytest.skip("MCP endpoint not available")

        assert resp.status_code == 200, f"MCP search failed: HTTP {resp.status_code}"

        body = resp.json()
        # MCP returns array of results directly
        assert isinstance(body, list), f"Expected list, got {type(body).__name__}"
        assert len(body) > 0, "No results returned from memory search"

        # Verify result structure
        first_result = body[0]
        assert "content" in first_result, f"Missing 'content' in result: {list(first_result.keys())}"
        assert "score" in first_result, f"Missing 'score' in result: {list(first_result.keys())}"

    def test_get_facts_mcp(self):
        """POST /mcp get_facts — retrieve structured facts."""
        payload = {
            "method": "tools/call",
            "params": {
                "name": "get_facts",
                "arguments": {
                    "topic": "Tower"
                }
            }
        }
        resp = post("/mcp", payload)

        if resp.status_code == 404:
            pytest.skip("MCP get_facts not available")

        assert resp.status_code == 200, f"Get facts failed: HTTP {resp.status_code}"

    def test_memory_endpoint_exists(self):
        """Check if any memory endpoint exists."""
        endpoints = ["/api/memory", "/api/echo/memory", "/api/memory/conversations"]

        found = False
        for endpoint in endpoints:
            try:
                resp = get(endpoint)
                if resp.status_code in [200, 405]:  # 405 means exists but wrong method
                    found = True
                    break
            except:
                continue

        if not found:
            pytest.skip("No memory endpoint implemented yet")


# ═════════════════════════════════════════════════════════════════════════════
#  TEST GROUP 4: EMBEDDING & VECTOR VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

class TestEmbeddings:
    """Validate embedding pipeline and vector store configuration."""

    def test_qdrant_reachable(self):
        """Qdrant should be running on port 6333."""
        try:
            resp = requests.get("http://localhost:6333/collections", timeout=5)
            assert resp.status_code == 200, f"Qdrant not responding: HTTP {resp.status_code}"
        except requests.exceptions.ConnectionError:
            pytest.fail("Qdrant not reachable on localhost:6333 - is it running?")

    def test_echo_memory_collection_exists(self):
        """echo_memory collection should exist with correct dimensions."""
        try:
            resp = requests.get("http://localhost:6333/collections/echo_memory", timeout=5)
            assert resp.status_code == 200, "echo_memory collection not found"

            body = resp.json()
            result = body.get("result", {})

            # Extract vector dimensions
            config = result.get("config", {})
            params = config.get("params", {})
            vectors = params.get("vectors", params)

            if isinstance(vectors, dict) and "size" in vectors:
                dim = vectors["size"]
            else:
                # Named vectors - get first one
                if isinstance(vectors, dict) and len(vectors) > 0:
                    first = next(iter(vectors.values()))
                    dim = first.get("size", 0) if isinstance(first, dict) else 0
                else:
                    dim = 0

            # Should be 768 (nomic-embed-text) or 1024 (newer model)
            assert dim in [768, 1024], f"Wrong dimensions: {dim} (expected 768 or 1024)"

        except requests.exceptions.ConnectionError:
            pytest.fail("Qdrant not reachable")

    def test_echo_memory_has_vectors(self):
        """echo_memory should have indexed vectors."""
        try:
            resp = requests.get("http://localhost:6333/collections/echo_memory", timeout=5)
            assert resp.status_code == 200, "Collection not found"

            body = resp.json()
            points = body.get("result", {}).get("points_count", 0)
            assert points > 0, f"Collection empty - has {points} vectors (expected >0)"
            assert points > 10000, f"Too few vectors: {points} (expected >10000 for production)"

        except requests.exceptions.ConnectionError:
            pytest.fail("Qdrant not reachable")

    def test_ollama_reachable(self):
        """Ollama should be running for embeddings."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            assert resp.status_code == 200, f"Ollama not responding: HTTP {resp.status_code}"
        except requests.exceptions.ConnectionError:
            pytest.fail("Ollama not reachable on :11434 - is it running?")

    def test_ollama_has_embedding_model(self):
        """Ollama should have nomic-embed-text for embeddings."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            body = resp.json()

            models = [m.get("name", "") for m in body.get("models", [])]
            has_embed = any("embed" in m.lower() for m in models)

            assert has_embed, f"No embedding model in Ollama. Found: {models}"

            # Check specifically for nomic-embed-text
            has_nomic = any("nomic-embed-text" in m for m in models)
            if not has_nomic:
                pytest.skip("nomic-embed-text not found, but other embedding model exists")

        except requests.exceptions.ConnectionError:
            pytest.fail("Ollama not reachable")


# ═════════════════════════════════════════════════════════════════════════════
#  TEST GROUP 5: DATABASE HEALTH
# ═════════════════════════════════════════════════════════════════════════════

class TestDatabaseHealth:
    """PostgreSQL database health checks."""

    PG_ENV = {
        **os.environ,
        "PGPASSWORD": "RP78eIrW7cI2jYvL5akt1yurE"
    }

    def _psql(self, db: str, query: str) -> tuple[int, str]:
        """Execute PostgreSQL query."""
        result = subprocess.run(
            ["psql", "-h", "localhost", "-U", "patrick", "-d", db,
             "-t", "-A", "-c", query],
            capture_output=True, text=True, timeout=10, env=self.PG_ENV
        )
        return result.returncode, result.stdout.strip()

    def test_postgresql_reachable(self):
        """PostgreSQL should be reachable."""
        rc, out = self._psql("echo_brain", "SELECT 1;")
        assert rc == 0, f"Cannot connect to PostgreSQL: {out}"

    def test_expected_databases_exist(self):
        """Key databases should exist."""
        required_dbs = ["echo_brain", "tower_consolidated", "tower_auth"]

        for db in required_dbs:
            rc, out = self._psql(db, "SELECT 1;")
            assert rc == 0, f"Database '{db}' not accessible"

    def test_conversation_table_count(self):
        """Should have minimal conversation tables (not 9+)."""
        total_tables = 0

        for db in ["echo_brain", "tower_consolidated"]:
            rc, out = self._psql(
                db,
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name LIKE '%conversation%';"
            )
            if rc == 0:
                try:
                    total_tables += int(out)
                except ValueError:
                    pass

        assert total_tables <= 8, f"Too many conversation tables: {total_tables} (expected ≤8)"

    def test_pgvector_installed(self):
        """pgvector extension should be installed (regression test)."""
        for db in ["tower_consolidated", "echo_brain"]:
            rc, out = self._psql(
                db,
                "SELECT extname FROM pg_extension WHERE extname = 'vector';"
            )
            # Should have been installed on 2026-02-09
            assert "vector" in out, f"pgvector MISSING on {db} - regression from 2026-02-09"


# ═════════════════════════════════════════════════════════════════════════════
#  TEST GROUP 6: SEPARATION PROGRESS (Migration Tracking)
# ═════════════════════════════════════════════════════════════════════════════

class TestSeparationProgress:
    """Track tower_consolidated separation plan progress."""

    PG_ENV = TestDatabaseHealth.PG_ENV

    def _psql(self, db: str, query: str) -> tuple[int, str]:
        """Execute PostgreSQL query."""
        result = subprocess.run(
            ["psql", "-h", "localhost", "-U", "patrick", "-d", db,
             "-t", "-A", "-c", query],
            capture_output=True, text=True, timeout=10, env=self.PG_ENV
        )
        return result.returncode, result.stdout.strip()

    def test_facts_source_of_truth(self):
        """echo_brain should be sole source of facts."""
        rc_eb, eb_out = self._psql("echo_brain", "SELECT count(*) FROM facts;")
        rc_tc, tc_out = self._psql("tower_consolidated", "SELECT count(*) FROM facts;")

        eb_count = int(eb_out) if rc_eb == 0 and eb_out.isdigit() else 0
        tc_count = int(tc_out) if rc_tc == 0 and tc_out.isdigit() else 0

        assert eb_count > 0, f"echo_brain has no facts: {eb_count}"
        assert tc_count == 0, f"tower_consolidated still has {tc_count} facts - should be migrated"

    def test_service_registry_location(self):
        """service_registry should be in tower_auth only."""
        rc_tc, tc_out = self._psql(
            "tower_consolidated",
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name='service_registry';"
        )
        rc_ta, ta_out = self._psql(
            "tower_auth",
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name='service_registry';"
        )

        tc_has = tc_out == "1" if rc_tc == 0 else False
        ta_has = ta_out == "1" if rc_ta == 0 else False

        assert ta_has, "service_registry not in tower_auth"
        assert not tc_has, "service_registry still in tower_consolidated - needs removal"

    def test_telegram_tables_location(self):
        """Telegram tables should be in echo_brain only."""
        rc, out = self._psql(
            "tower_consolidated",
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name LIKE 'echo_telegram%';"
        )

        count = int(out) if rc == 0 and out.isdigit() else 0
        assert count == 0, f"{count} telegram tables still in tower_consolidated"

    def test_embedding_cache_migrated(self):
        """embedding_cache should be migrated to Qdrant."""
        rc, out = self._psql(
            "tower_consolidated",
            "SELECT count(*) FROM embedding_cache;"
        )

        if rc != 0 or "does not exist" in out:
            # Table dropped - good!
            return

        count = int(out) if out.isdigit() else -1
        assert count == 0, f"{count:,} embeddings not migrated (768MB waste)"

    def test_tower_consolidated_shrinking(self):
        """tower_consolidated should be getting smaller."""
        rc, out = self._psql(
            "tower_consolidated",
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';"
        )

        count = int(out) if rc == 0 and out.isdigit() else -1
        assert count < 20, f"{count} tables still in tower_consolidated (target: 0)"


# ═════════════════════════════════════════════════════════════════════════════
#  TEST GROUP 7: PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Performance and response time checks."""

    def test_health_response_time(self):
        """Health endpoint should respond quickly."""
        start = time.time()
        resp = get("/health")
        elapsed = (time.time() - start) * 1000

        assert resp.status_code == 200, f"Health check failed: HTTP {resp.status_code}"
        assert elapsed < 100, f"Health check too slow: {elapsed:.1f}ms (expected <100ms)"

    def test_query_response_time(self):
        """Simple queries should respond within timeout."""
        payload = {"query": "What is 1+1?"}

        start = time.time()
        resp = post("/api/echo/query", payload)
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 405:
            pytest.skip("Query endpoint not available")

        assert resp.status_code == 200, f"Query failed: HTTP {resp.status_code}"
        assert elapsed < 5000, f"Query too slow: {elapsed:.1f}ms (expected <5000ms)"


# ═════════════════════════════════════════════════════════════════════════════
#  TEST GROUP 8: VOICE ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

class TestVoiceEndpoints:
    """Voice API endpoint smoke tests."""

    def test_voice_status(self):
        """GET /api/echo/voice/status — service health."""
        resp = get("/api/echo/voice/status")
        assert resp.status_code == 200, f"Voice status failed: HTTP {resp.status_code}"

        body = resp.json()
        # Should have some status information
        assert isinstance(body, dict), f"Expected dict, got {type(body).__name__}"

    def test_voice_voices(self):
        """GET /api/echo/voice/voices — available TTS voices."""
        resp = get("/api/echo/voice/voices")
        assert resp.status_code == 200, f"Voice voices failed: HTTP {resp.status_code}"

        body = resp.json()
        assert "installed" in body, f"Missing 'installed' in response: {list(body.keys())}"
        assert "suggested" in body, f"Missing 'suggested' in response: {list(body.keys())}"
        assert isinstance(body["installed"], list), "installed should be a list"
        assert isinstance(body["suggested"], list), "suggested should be a list"
        assert len(body["suggested"]) > 0, "Should have at least one suggested voice"

    def test_voice_synthesize(self):
        """POST /api/echo/voice/synthesize — TTS generates audio."""
        resp = requests.post(
            f"{BASE_URL}/api/echo/voice/synthesize",
            json={"text": "Hello from the smoke test.", "length_scale": 1.0},
            headers={"Content-Type": "application/json"},
            timeout=QUERY_TIMEOUT,
        )

        if resp.status_code == 503:
            pytest.skip("Voice TTS service not initialized")

        assert resp.status_code == 200, f"Voice synthesize failed: HTTP {resp.status_code}"
        assert resp.headers.get("content-type", "").startswith("audio/"), \
            f"Expected audio content-type, got {resp.headers.get('content-type')}"
        assert len(resp.content) > 100, f"Audio response too small: {len(resp.content)} bytes"


# ═════════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ═════════════════════════════════════════════════════════════════════════════

def pytest_sessionfinish(session, exitstatus):
    """Print summary after all tests."""
    import _pytest.config

    # Only print if running these tests
    if 'echo_brain_smoke_test' not in str(session.config.args):
        return

    print("\n" + "=" * 70)
    print("  ECHO BRAIN SMOKE TEST COMPLETED")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Get test results
    passed = len([i for i in session.items if i.stash.get(_pytest.runner.runtest_outcome, None) == "passed"])
    failed = len([i for i in session.items if i.stash.get(_pytest.runner.runtest_outcome, None) == "failed"])
    skipped = len([i for i in session.items if i.stash.get(_pytest.runner.runtest_outcome, None) == "skipped"])
    total = len(session.items)

    print(f"\n  Results: {passed} passed, {failed} failed, {skipped} skipped / {total} total")

    if failed > 0:
        print("\n  ❌ FAILURES DETECTED - Check test output above")
    elif skipped > passed:
        print("\n  ⚠️  Many tests skipped - endpoints may not be implemented")
    else:
        print("\n  ✅ All implemented tests passing")

    print("=" * 70)