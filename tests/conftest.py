"""Shared fixtures for Echo Brain unit tests."""
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Timestamp helpers ─────────────────────────────────────────────────

@pytest.fixture
def recent_timestamp() -> str:
    """ISO timestamp from ~1 day ago."""
    return (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


@pytest.fixture
def old_timestamp() -> str:
    """ISO timestamp from ~90 days ago."""
    return (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()


# ── Instance fixtures (no live connections) ───────────────────────────

@pytest.fixture
def retriever():
    """ParallelRetriever instance without calling initialize()."""
    from src.context_assembly.retriever import ParallelRetriever
    return ParallelRetriever()


@pytest.fixture
def compiler():
    """ContextCompiler instance."""
    from src.context_assembly.compiler import ContextCompiler
    return ContextCompiler()


@pytest.fixture
def domain_classifier():
    """DomainClassifier instance."""
    from src.context_assembly.classifier import DomainClassifier
    return DomainClassifier()


@pytest.fixture
def intelligence():
    """IntelligenceEngine instance (no live connections)."""
    from src.core.intelligence_engine import IntelligenceEngine
    return IntelligenceEngine()


# ── Factory fixtures ──────────────────────────────────────────────────

@pytest.fixture
def make_source():
    """Factory that builds a source dict with sensible defaults."""
    def _make(
        content: str = "Some relevant content",
        score: float = 0.75,
        source_type: str = "hybrid",
        source_name: str = "qdrant/echo_memory",
        metadata: Dict[str, Any] | None = None,
        point_id: Any = None,
    ) -> Dict[str, Any]:
        return {
            "type": source_type,
            "source": source_name,
            "content": content,
            "score": score,
            "metadata": metadata or {},
            **({"point_id": point_id} if point_id is not None else {}),
        }
    return _make


@pytest.fixture
def make_memory_context():
    """Factory that builds a MemoryContext dataclass."""
    from src.core.intelligence_engine import MemoryContext, KnowledgeDomain

    def _make(
        content: str = "Some memory content",
        score: float = 0.8,
        source: str = "echo_memory",
        domain: KnowledgeDomain = KnowledgeDomain.UNKNOWN,
        metadata: Dict[str, Any] | None = None,
    ) -> MemoryContext:
        return MemoryContext(
            content=content,
            score=score,
            source=source,
            domain=domain,
            metadata=metadata or {},
        )
    return _make


# ── Search result fixtures ────────────────────────────────────────────

@pytest.fixture
def vector_results() -> List[Dict]:
    """Sample vector search results for fusion tests."""
    return [
        {"point_id": 1, "content": "Tower API docs", "score": 0.9, "payload": {"source": "docs"}},
        {"point_id": 2, "content": "Echo Brain arch", "score": 0.7, "payload": {"source": "code"}},
        {"point_id": 3, "content": "Only in vector", "score": 0.5, "payload": {"source": "misc"}},
    ]


@pytest.fixture
def text_results() -> List[Dict]:
    """Sample text search results for fusion tests."""
    return [
        {"point_id": 1, "content": "Tower API docs", "score": 0.8, "payload": {"source": "docs"}},
        {"point_id": 4, "content": "Only in text", "score": 0.6, "payload": {"source": "logs"}},
    ]
