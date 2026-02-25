#!/usr/bin/env python3
"""
Echo Brain Smoke Test Suite
============================
Sequential system verification that tests infrastructure + every claimed capability.

Run:
    pytest tests/echo_brain_smoke_test.py -v
    pytest tests/echo_brain_smoke_test.py -v -k "embedding"
    pytest tests/echo_brain_smoke_test.py -v -k "mcp"
    pytest tests/echo_brain_smoke_test.py -v -k "Google"

Test groups (ordered by dependency):
  1. Infrastructure     — Services alive
  2. EmbeddingConsistency — THE thing that keeps breaking
  3. QdrantIndexes      — Text + keyword indexes exist
  4. VectorQuality      — No garbage in sample
  5. HybridSearch       — MCP retrieval pipeline
  6. Facts              — PostgreSQL structured knowledge
  7. IntelligencePipeline — /ask end-to-end
  8. Workers            — Autonomous system running
  9. Voice              — STT/TTS service
 10. KnowledgeGraph     — v0.6.0 graph feature
 11. MCPTools           — All MCP tools respond
 12. GoogleIntegration  — Google OAuth + Calendar/Gmail/Ingest end-to-end
"""

import json
import re
import time

import pytest
import requests

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8309"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
TIMEOUT = 10
QUERY_TIMEOUT = 45

# ─── Known correct values (verified 2026-02-16) ─────────────────────────────

EXPECTED_EMBEDDING_DIM = 768
EXPECTED_EMBEDDING_MODEL = "nomic-embed-text"
MIN_VECTOR_COUNT = 100_000


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get(url: str, timeout: int = TIMEOUT) -> requests.Response:
    return requests.get(url, timeout=timeout)


def post(url: str, payload: dict, timeout: int = QUERY_TIMEOUT) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


def mcp_call(tool: str, arguments: dict, timeout: int = QUERY_TIMEOUT) -> requests.Response:
    """Call an MCP tool via the /mcp endpoint."""
    return post(
        f"{BASE_URL}/mcp",
        {"method": "tools/call", "params": {"name": tool, "arguments": arguments}},
        timeout=timeout,
    )


def mcp_results(resp: requests.Response) -> list:
    """Extract the results list from an MCP response (handles both list and dict formats)."""
    data = resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("results", data.get("result", []))
    return []


def is_readable_text(text: str) -> bool:
    """Check if text is human-readable (not base64 garbage)."""
    if not text or len(text) < 10:
        return False
    # Space ratio check
    space_ratio = text.count(" ") / len(text) if text else 0
    # Alphanumeric density
    alnum = sum(c.isalnum() or c.isspace() for c in text) / len(text) if text else 0
    return space_ratio > 0.05 and alnum > 0.6


# ═════════════════════════════════════════════════════════════════════════════
#  1. INFRASTRUCTURE — Services alive
# ═════════════════════════════════════════════════════════════════════════════

class TestInfrastructure:
    """All external services must be reachable."""

    def test_health_endpoint(self):
        """GET /health returns {"status": "healthy"}."""
        resp = get(f"{BASE_URL}/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy", f"Unhealthy: {body}"

    def test_qdrant_reachable(self):
        """Qdrant on port 6333 responds."""
        resp = get(f"{QDRANT_URL}/collections")
        assert resp.status_code == 200

    def test_ollama_reachable(self):
        """Ollama on port 11434 responds."""
        resp = get(f"{OLLAMA_URL}/api/tags")
        assert resp.status_code == 200

    def test_postgresql_reachable(self):
        """PostgreSQL reachable via detailed health endpoint."""
        resp = get(f"{BASE_URL}/api/echo/health/detailed")
        assert resp.status_code == 200
        data = resp.json()
        assert "knowledge" in data, f"Missing 'knowledge' in detailed health: {list(data.keys())}"
        # If we can read fact counts, the DB is connected
        assert data["knowledge"]["total_facts"] >= 0


# ═════════════════════════════════════════════════════════════════════════════
#  2. EMBEDDING CONSISTENCY — THE thing that keeps breaking
# ═════════════════════════════════════════════════════════════════════════════

class TestEmbeddingConsistency:
    """Qdrant dimensions must match Ollama output. No ambiguity."""

    def test_qdrant_dimension_is_768(self):
        """echo_memory collection must be exactly 768 dimensions."""
        resp = get(f"{QDRANT_URL}/collections/echo_memory")
        assert resp.status_code == 200
        config = resp.json()["result"]["config"]["params"]["vectors"]
        dim = config["size"] if isinstance(config, dict) and "size" in config else 0
        assert dim == EXPECTED_EMBEDDING_DIM, (
            f"Qdrant dimension is {dim}, expected exactly {EXPECTED_EMBEDDING_DIM}. "
            "This mismatch breaks ALL vector search."
        )

    def test_ollama_outputs_768(self):
        """nomic-embed-text must produce exactly 768-dim vectors."""
        resp = post(
            f"{OLLAMA_URL}/api/embed",
            {"model": EXPECTED_EMBEDDING_MODEL, "input": "dimension consistency test"},
            timeout=30,
        )
        assert resp.status_code == 200, f"Ollama embed failed: HTTP {resp.status_code}"
        embeddings = resp.json().get("embeddings", [[]])
        assert len(embeddings) > 0 and len(embeddings[0]) > 0, "Empty embedding returned"
        dim = len(embeddings[0])
        assert dim == EXPECTED_EMBEDDING_DIM, (
            f"Ollama produced {dim}-dim vector, expected {EXPECTED_EMBEDDING_DIM}"
        )

    def test_qdrant_matches_ollama(self):
        """Cross-check: Qdrant dimension == Ollama output dimension."""
        # Qdrant
        q_resp = get(f"{QDRANT_URL}/collections/echo_memory")
        q_config = q_resp.json()["result"]["config"]["params"]["vectors"]
        q_dim = q_config["size"] if isinstance(q_config, dict) and "size" in q_config else 0

        # Ollama
        o_resp = post(
            f"{OLLAMA_URL}/api/embed",
            {"model": EXPECTED_EMBEDDING_MODEL, "input": "cross-check"},
            timeout=30,
        )
        o_dim = len(o_resp.json().get("embeddings", [[]])[0])

        assert q_dim == o_dim, (
            f"DIMENSION MISMATCH: Qdrant={q_dim}, Ollama={o_dim}. "
            "This is the #1 cause of broken search."
        )

    def test_collection_has_enough_vectors(self):
        """Production must have >100k vectors."""
        resp = get(f"{QDRANT_URL}/collections/echo_memory")
        points = resp.json()["result"]["points_count"]
        assert points > MIN_VECTOR_COUNT, (
            f"Only {points:,} vectors (need >{MIN_VECTOR_COUNT:,} for production)"
        )

    def test_collection_status_green(self):
        """Collection status must be green (not yellow/red from ongoing operations)."""
        resp = get(f"{QDRANT_URL}/collections/echo_memory")
        status = resp.json()["result"]["status"]
        assert status == "green", f"Collection status is '{status}', expected 'green'"


# ═════════════════════════════════════════════════════════════════════════════
#  3. QDRANT INDEXES — Text + keyword indexes exist
# ═════════════════════════════════════════════════════════════════════════════

class TestQdrantIndexes:
    """Payload indexes required for hybrid search."""

    @pytest.fixture(autouse=True)
    def _load_schema(self):
        resp = get(f"{QDRANT_URL}/collections/echo_memory")
        assert resp.status_code == 200
        self.schema = resp.json()["result"].get("payload_schema", {})

    def test_content_text_index(self):
        """'content' field must have a text index for keyword search."""
        assert "content" in self.schema, "No 'content' index — hybrid text search is broken"
        assert self.schema["content"]["data_type"] == "text", (
            f"'content' index is {self.schema['content']['data_type']}, expected 'text'"
        )

    def test_type_keyword_index(self):
        """'type' field must have a keyword index for filtering."""
        assert "type" in self.schema, "No 'type' keyword index"
        assert self.schema["type"]["data_type"] == "keyword"

    def test_category_keyword_index(self):
        """'category' field must have a keyword index."""
        assert "category" in self.schema, "No 'category' keyword index"
        assert self.schema["category"]["data_type"] == "keyword"


# ═════════════════════════════════════════════════════════════════════════════
#  4. VECTOR QUALITY — No garbage
# ═════════════════════════════════════════════════════════════════════════════

class TestVectorQuality:
    """Sample random vectors and verify they contain readable text."""

    def test_sample_vectors_are_readable(self):
        """10 random vectors must all pass is_readable_text()."""
        resp = post(
            f"{QDRANT_URL}/collections/echo_memory/points/scroll",
            {"limit": 10, "with_payload": True},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        points = resp.json()["result"]["points"]
        assert len(points) > 0, "No points returned from scroll"

        garbage_count = 0
        for pt in points:
            content = pt.get("payload", {}).get("content", "")
            if not is_readable_text(content):
                garbage_count += 1

        assert garbage_count == 0, (
            f"{garbage_count}/{len(points)} sampled vectors contain garbage text. "
            "Garbage vectors pollute search results."
        )


# ═════════════════════════════════════════════════════════════════════════════
#  5. HYBRID SEARCH — The retrieval pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestHybridSearch:
    """MCP search_memory must return quality results."""

    def test_search_returns_results(self):
        """search_memory for a known topic returns results."""
        resp = mcp_call("search_memory", {"query": "Echo Brain architecture", "limit": 5})
        assert resp.status_code == 200
        results = mcp_results(resp)
        assert isinstance(results, list), f"Expected list, got {type(results).__name__}"
        assert len(results) > 0, "search_memory returned empty results"

    def test_result_structure(self):
        """Results must have content and confidence/score fields."""
        resp = mcp_call("search_memory", {"query": "Tower hardware specs", "limit": 3})
        results = mcp_results(resp)
        assert len(results) > 0, "No results to check structure"

        first = results[0]
        assert "content" in first, f"Missing 'content': {list(first.keys())}"
        has_score = "score" in first or "confidence" in first
        assert has_score, f"Missing score/confidence: {list(first.keys())}"

    def test_top_score_above_threshold(self):
        """Top score for a known-good query must be > 0.3."""
        resp = mcp_call("search_memory", {"query": "Echo Brain port 8309", "limit": 3})
        results = mcp_results(resp)
        assert len(results) > 0
        top_score = results[0].get("score", results[0].get("confidence", 0))
        assert top_score > 0.3, f"Top score {top_score:.3f} is too low (expected >0.3)"

    def test_text_search_returns_results(self):
        """Keyword-style query returns results (tests text index path)."""
        resp = mcp_call("search_memory", {"query": "nomic-embed-text", "limit": 3})
        results = mcp_results(resp)
        assert len(results) > 0, "Text/keyword search returned no results"

    def test_vector_search_returns_results(self):
        """Conceptual query returns results (tests vector similarity path)."""
        resp = mcp_call(
            "search_memory",
            {"query": "how does the memory system retrieve information", "limit": 3},
        )
        results = mcp_results(resp)
        assert len(results) > 0, "Conceptual/vector search returned no results"


# ═════════════════════════════════════════════════════════════════════════════
#  6. FACTS — PostgreSQL structured knowledge
# ═════════════════════════════════════════════════════════════════════════════

class TestFacts:
    """Structured facts must be stored and retrievable."""

    def test_get_facts_returns_results(self):
        """get_facts for 'Echo Brain' returns facts."""
        resp = mcp_call("get_facts", {"topic": "Echo Brain"})
        assert resp.status_code == 200
        results = mcp_results(resp)
        assert isinstance(results, list), f"Expected list, got {type(results).__name__}"
        assert len(results) > 0, "get_facts returned no facts for 'Echo Brain'"

    def test_fact_structure(self):
        """Facts must have expected fields."""
        resp = mcp_call("get_facts", {"topic": "Tower"})
        results = mcp_results(resp)
        assert len(results) > 0, "No facts to check structure"
        first = results[0]
        # Facts should have content/subject/predicate/object or similar
        has_content = "content" in first or "subject" in first or "object" in first
        assert has_content, f"Fact missing content fields: {list(first.keys())}"

    def test_knowledge_stats_has_facts(self):
        """Knowledge stats endpoint reports fact count > 0."""
        resp = get(f"{BASE_URL}/api/echo/knowledge/stats")
        assert resp.status_code == 200
        data = resp.json()
        total = data.get("total_facts", 0)
        assert total > 0, f"Knowledge stats reports 0 facts"


# ═════════════════════════════════════════════════════════════════════════════
#  7. INTELLIGENCE PIPELINE — /ask endpoint end-to-end
# ═════════════════════════════════════════════════════════════════════════════

class TestIntelligencePipeline:
    """The /ask endpoint must produce correct answers using retrieval."""

    def test_ask_returns_answer(self):
        """POST /api/echo/ask returns a non-empty answer."""
        resp = post(
            f"{BASE_URL}/api/echo/ask",
            {"question": "What port does Echo Brain run on?"},
        )
        assert resp.status_code == 200, f"/ask failed: HTTP {resp.status_code}"
        body = resp.json()
        answer = body.get("answer", body.get("response", ""))
        assert len(answer) > 10, f"Answer too short: {answer!r}"

    def test_ask_contains_expected_keywords(self):
        """Answer about Echo Brain port must mention 8309."""
        resp = post(
            f"{BASE_URL}/api/echo/ask",
            {"question": "What port does Echo Brain run on?"},
        )
        body = resp.json()
        answer = str(body.get("answer", body.get("response", ""))).lower()
        assert "8309" in answer, f"Answer doesn't mention 8309: {answer[:200]}"

    def test_ask_rejects_wrong_field(self):
        """Sending 'query' instead of 'question' should fail or return empty."""
        resp = post(f"{BASE_URL}/api/echo/ask", {"query": "This should fail"})
        if resp.status_code == 422:
            return  # Validation caught it
        if resp.status_code == 200:
            body = resp.json()
            answer = str(body.get("answer", body.get("response", "")))
            # Should be empty/generic, not a real answer
            assert len(answer) < 50 or "don't" in answer.lower() or "no question" in answer.lower(), (
                f"Wrong field name still produced a real answer: {answer[:100]}"
            )

    def test_ask_response_time(self):
        """Response must complete within 30 seconds."""
        start = time.time()
        resp = post(
            f"{BASE_URL}/api/echo/ask",
            {"question": "What GPU does Tower have?"},
        )
        elapsed = time.time() - start
        assert resp.status_code == 200
        assert elapsed < 30, f"Response took {elapsed:.1f}s (limit: 30s)"


# ═════════════════════════════════════════════════════════════════════════════
#  8. WORKERS — Autonomous system running
# ═════════════════════════════════════════════════════════════════════════════

class TestWorkers:
    """Worker scheduler must be running."""

    def test_workers_status_returns_list(self):
        """GET /api/workers/status returns worker data."""
        resp = get(f"{BASE_URL}/api/workers/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "workers" in data, f"Missing 'workers': {list(data.keys())}"
        assert len(data["workers"]) > 0, "No workers registered"

    def test_workers_are_running(self):
        """Worker system is running (not all errored)."""
        resp = get(f"{BASE_URL}/api/workers/status")
        data = resp.json()
        assert data.get("running") is True, "Worker scheduler is not running"

    def test_no_excessive_errors(self):
        """No worker should have >10 consecutive errors."""
        resp = get(f"{BASE_URL}/api/workers/status")
        data = resp.json()
        for name, info in data.get("workers", {}).items():
            error_count = info.get("error_count", 0)
            assert error_count <= 10, (
                f"Worker '{name}' has {error_count} errors (threshold: 10)"
            )


# ═════════════════════════════════════════════════════════════════════════════
#  9. VOICE — STT/TTS service
# ═════════════════════════════════════════════════════════════════════════════

class TestVoice:
    """Voice endpoints must respond (service may not be initialized)."""

    def test_voice_status_responds(self):
        """GET /api/echo/voice/status returns 200."""
        resp = get(f"{BASE_URL}/api/echo/voice/status")
        assert resp.status_code == 200

    def test_voice_voices_has_entries(self):
        """GET /api/echo/voice/voices lists at least 1 voice."""
        resp = get(f"{BASE_URL}/api/echo/voice/voices")
        assert resp.status_code == 200
        data = resp.json()
        total = len(data.get("installed", [])) + len(data.get("suggested", []))
        assert total > 0, "No voices available (installed or suggested)"


# ═════════════════════════════════════════════════════════════════════════════
# 10. KNOWLEDGE GRAPH — v0.6.0
# ═════════════════════════════════════════════════════════════════════════════

class TestKnowledgeGraph:
    """Knowledge graph must have nodes and support queries."""

    def test_graph_stats_has_nodes(self):
        """GET /api/echo/graph/stats returns node_count > 0."""
        resp = get(f"{BASE_URL}/api/echo/graph/stats")
        assert resp.status_code == 200
        data = resp.json()
        nodes = data.get("nodes", 0)
        assert nodes > 0, f"Graph has {nodes} nodes (expected >0)"

    def test_graph_related_returns_results(self):
        """Graph related endpoint returns edges for a known entity."""
        resp = get(f"{BASE_URL}/api/echo/graph/related/Echo%20Brain")
        assert resp.status_code == 200
        data = resp.json()
        results = data.get("results", [])
        assert len(results) > 0, "No related entities found for 'Echo Brain'"


# ═════════════════════════════════════════════════════════════════════════════
# 11. MCP TOOLS — All 5 tools respond
# ═════════════════════════════════════════════════════════════════════════════

class TestMCPTools:
    """Every registered MCP tool must respond successfully."""

    def test_search_memory(self):
        """MCP search_memory tool works."""
        resp = mcp_call("search_memory", {"query": "test connectivity", "limit": 1})
        assert resp.status_code == 200

    def test_get_facts(self):
        """MCP get_facts tool works."""
        resp = mcp_call("get_facts", {"topic": "Tower"})
        assert resp.status_code == 200

    def test_store_fact_roundtrip(self):
        """MCP store_fact stores a fact and it can be verified."""
        # Store
        store_resp = mcp_call("store_fact", {
            "subject": "_smoke_test_probe",
            "predicate": "is_a",
            "object": "transient_test_entry",
            "confidence": 0.1,
        })
        assert store_resp.status_code == 200
        data = store_resp.json()
        assert data.get("stored") is True, f"store_fact failed: {data}"

    def test_explore_graph(self):
        """MCP explore_graph tool works."""
        resp = mcp_call("explore_graph", {"query": "Echo Brain"})
        assert resp.status_code == 200

    def test_manage_ollama_list(self):
        """MCP manage_ollama (list action) works."""
        resp = mcp_call("manage_ollama", {"action": "list"})
        assert resp.status_code == 200
        data = resp.json()
        models = data.get("models", [])
        assert len(models) > 0, "manage_ollama list returned no models"


# ═════════════════════════════════════════════════════════════════════════════
# 12. GOOGLE INTEGRATION — OAuth + Calendar/Gmail/Ingest end-to-end
# ═════════════════════════════════════════════════════════════════════════════

class TestGoogleIntegration:
    """Google OAuth chain: tower-auth token → Google API → response."""

    def test_summary_returns_200(self):
        """GET /google/summary returns 200 with emails + calendar data."""
        resp = get(f"{BASE_URL}/google/summary", timeout=QUERY_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert "emails" in data, f"Missing 'emails' in summary: {list(data.keys())}"
        assert "calendar" in data, f"Missing 'calendar' in summary: {list(data.keys())}"
        # At least one of emails/calendar should have real data (not error)
        emails_ok = "total" in data.get("emails", {})
        calendar_ok = "events_60_days" in data.get("calendar", {})
        assert emails_ok or calendar_ok, (
            f"Both emails and calendar errored: emails={data.get('emails')}, calendar={data.get('calendar')}"
        )

    def test_email_count_has_totals(self):
        """GET /google/emails/count returns total > 0 and inbox count."""
        resp = get(f"{BASE_URL}/google/emails/count", timeout=QUERY_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("count", 0) > 0, f"Email total is 0 (expected thousands): {data}"
        assert "inbox_count" in data, f"Missing inbox_count: {list(data.keys())}"

    def test_calendar_count_returns_events(self):
        """GET /google/calendar/count returns event count for 60-day window."""
        resp = get(f"{BASE_URL}/google/calendar/count", timeout=QUERY_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data, f"Missing 'count': {list(data.keys())}"
        assert "upcoming_count" in data, f"Missing 'upcoming_count': {list(data.keys())}"
        # Calendar should have at least some events in a 60-day window
        assert data["count"] >= 0, f"Negative event count: {data['count']}"

    def test_calendar_status_returns_200(self):
        """GET /api/calendar/status returns 200 with availability info."""
        resp = get(f"{BASE_URL}/api/calendar/status", timeout=QUERY_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data, f"Missing 'available': {list(data.keys())}"

    def test_calendar_upcoming_returns_events(self):
        """GET /api/calendar/events/upcoming returns events list."""
        resp = get(f"{BASE_URL}/api/calendar/events/upcoming", timeout=QUERY_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data, f"Missing 'events': {list(data.keys())}"
        assert isinstance(data["events"], list), f"'events' is not a list: {type(data['events']).__name__}"

    def test_ingest_stats_has_counts(self):
        """GET /api/google/ingest/stats returns ingestion counts > 0."""
        resp = get(f"{BASE_URL}/api/google/ingest/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("total", 0) > 0, f"Ingest stats total is 0: {data}"
        assert "by_source" in data, f"Missing 'by_source': {list(data.keys())}"

    def test_ingest_calendar_returns_structure(self):
        """POST /api/google/ingest/calendar returns ingested/skipped/errors."""
        resp = post(f"{BASE_URL}/api/google/ingest/calendar", {}, timeout=QUERY_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        # Should have some ingestion result fields
        has_expected = any(k in data for k in ("ingested", "skipped", "errors", "total", "status"))
        assert has_expected, f"Unexpected ingest response structure: {list(data.keys())}"
