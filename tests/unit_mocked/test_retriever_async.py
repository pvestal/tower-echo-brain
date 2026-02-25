"""Mocked async tests for ParallelRetriever.

Covers: _get_embedding, _qdrant_vector_search, _qdrant_text_search, _search_qdrant
Uses mocked httpx to avoid live service dependencies.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from src.context_assembly.retriever import ParallelRetriever
from src.context_assembly.classifier import Domain

pytestmark = pytest.mark.mocked


def _make_httpx_response(json_data, status_code=200):
    """Create a mock httpx Response with sync .json() method."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = json_data
    return resp


@pytest.fixture
def retriever_with_client():
    """ParallelRetriever with a mocked http_client."""
    r = ParallelRetriever()
    r.http_client = AsyncMock()
    return r


# ── _get_embedding ────────────────────────────────────────────────────

class TestGetEmbedding:
    async def test_success_returns_vector(self, retriever_with_client):
        r = retriever_with_client
        r.http_client.post.return_value = _make_httpx_response(
            {"embeddings": [[0.1, 0.2, 0.3]]}
        )
        result = await r._get_embedding("test query")
        assert result == [0.1, 0.2, 0.3]

    async def test_failure_returns_none(self, retriever_with_client):
        r = retriever_with_client
        r.http_client.post.side_effect = Exception("connection refused")
        result = await r._get_embedding("test query")
        assert result is None

    async def test_empty_embeddings_returns_empty_list(self, retriever_with_client):
        r = retriever_with_client
        r.http_client.post.return_value = _make_httpx_response({"embeddings": []})
        result = await r._get_embedding("test query")
        assert result == []


# ── _qdrant_vector_search ─────────────────────────────────────────────

class TestQdrantVectorSearch:
    async def test_garbage_filtered_out(self, retriever_with_client):
        r = retriever_with_client
        r.http_client.post.return_value = _make_httpx_response({
            "result": [
                {"id": 1, "score": 0.9, "payload": {"content": "Good readable text, with spaces — and punctuation! (yes)"}},
                {"id": 2, "score": 0.8, "payload": {"content": "a" * 200}},  # Garbage
            ]
        })
        results = await r._qdrant_vector_search([0.1, 0.2], "echo_memory", 0.3)
        assert len(results) == 1
        assert results[0]["point_id"] == 1

    async def test_max_30_cap(self, retriever_with_client):
        r = retriever_with_client
        points = [
            {"id": i, "score": 0.9, "payload": {"content": f"Good readable text number {i} here"}}
            for i in range(50)
        ]
        r.http_client.post.return_value = _make_httpx_response({"result": points})
        results = await r._qdrant_vector_search([0.1], "echo_memory", 0.3)
        assert len(results) <= 30

    async def test_readable_results_returned(self, retriever_with_client):
        r = retriever_with_client
        r.http_client.post.return_value = _make_httpx_response({
            "result": [
                {"id": "abc", "score": 0.75, "payload": {"text": "This is a readable text payload"}}
            ]
        })
        results = await r._qdrant_vector_search([0.1], "echo_memory", 0.3)
        assert len(results) == 1
        assert results[0]["content"] == "This is a readable text payload"
        assert results[0]["point_id"] == "abc"


# ── _qdrant_text_search ──────────────────────────────────────────────

class TestQdrantTextSearch:
    async def test_term_overlap_scoring(self, retriever_with_client):
        r = retriever_with_client
        r.http_client.post.return_value = _make_httpx_response({
            "result": {
                "points": [
                    {"id": 1, "payload": {"content": "tower: server service, port = 8309 — configuration details."}},
                    {"id": 2, "payload": {"content": "tower is a great system, with many features (here)."}},
                ]
            }
        })
        results = await r._qdrant_text_search("tower server port", "echo_memory")
        assert len(results) == 2
        assert results[0]["point_id"] == 1  # Higher score (3/3 terms)
        assert results[0]["score"] > results[1]["score"]

    async def test_empty_terms_returns_empty(self, retriever_with_client):
        r = retriever_with_client
        results = await r._qdrant_text_search("what is the", "echo_memory")
        assert results == []

    async def test_readable_filter_applied(self, retriever_with_client):
        r = retriever_with_client
        r.http_client.post.return_value = _make_httpx_response({
            "result": {
                "points": [
                    {"id": 1, "payload": {"content": "a" * 200}},  # Garbage
                ]
            }
        })
        results = await r._qdrant_text_search("tower server", "echo_memory")
        assert results == []


# ── _search_qdrant ────────────────────────────────────────────────────

class TestSearchQdrant:
    async def test_authoritative_boost(self, retriever_with_client):
        r = retriever_with_client

        async def mock_vector_search(embedding, collection, min_score):
            return [
                {"point_id": 1, "content": "Auth content here", "score": 0.5,
                 "payload": {"authoritative": True}},
            ]

        async def mock_text_search(query, collection):
            return []

        r._qdrant_vector_search = mock_vector_search
        r._qdrant_text_search = mock_text_search

        results = await r._search_qdrant("test", [0.1], "echo_memory", Domain.GENERAL)
        # Base fused score = 0.7 * 0.5 = 0.35, then * 2.5 = 0.875
        assert len(results) == 1
        assert abs(results[0]["score"] - 0.875) < 0.01

    async def test_architecture_doc_boost(self, retriever_with_client):
        r = retriever_with_client

        async def mock_vector_search(embedding, collection, min_score):
            return [
                {"point_id": 1, "content": "Arch doc content here", "score": 0.5,
                 "payload": {"source": "architecture_doc"}},
            ]

        async def mock_text_search(query, collection):
            return []

        r._qdrant_vector_search = mock_vector_search
        r._qdrant_text_search = mock_text_search

        results = await r._search_qdrant("test", [0.1], "echo_memory", Domain.GENERAL)
        # Base = 0.7 * 0.5 = 0.35, then * 2.0 = 0.70
        assert abs(results[0]["score"] - 0.70) < 0.01

    async def test_documentation_boost(self, retriever_with_client):
        r = retriever_with_client

        async def mock_vector_search(embedding, collection, min_score):
            return [
                {"point_id": 1, "content": "Doc content with details", "score": 0.5,
                 "payload": {"content_type": "documentation"}},
            ]

        async def mock_text_search(query, collection):
            return []

        r._qdrant_vector_search = mock_vector_search
        r._qdrant_text_search = mock_text_search

        results = await r._search_qdrant("test", [0.1], "echo_memory", Domain.GENERAL)
        # Base = 0.7 * 0.5 = 0.35, then * 1.5 = 0.525
        assert abs(results[0]["score"] - 0.525) < 0.01

    async def test_no_vector_fallback_text_full_weight(self, retriever_with_client):
        r = retriever_with_client

        async def mock_vector_search(embedding, collection, min_score):
            return []

        async def mock_text_search(query, collection):
            return [
                {"point_id": 1, "content": "Text only result with details", "score": 0.8,
                 "payload": {}},
            ]

        r._qdrant_vector_search = mock_vector_search
        r._qdrant_text_search = mock_text_search

        results = await r._search_qdrant("test", [0.1], "echo_memory", Domain.GENERAL)
        # No vector results → text gets full weight (1.0)
        assert len(results) == 1
        assert abs(results[0]["score"] - 0.8) < 0.01
