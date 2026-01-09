"""
Comprehensive Test Suite - Shared Fixtures
Provides fixtures for all Echo Brain tests including mocks, database, API client, etc.
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import httpx


# ============================================================
# Environment Configuration
# ============================================================

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration matching production settings"""
    return {
        "echo_brain_url": os.getenv("ECHO_BRAIN_URL", "http://localhost:8309"),
        "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
        "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
        "db_host": os.getenv("DB_HOST", "localhost"),
        "db_name": os.getenv("DB_NAME", "echo_brain"),
        "db_user": os.getenv("DB_USER", "echo_user"),
        "anime_db_name": "anime_production",
        "tower_ip": "192.168.50.135",
    }


# ============================================================
# Event Loop
# ============================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================
# Mock Responses
# ============================================================

@dataclass
class MockOllamaResponse:
    """Mock Ollama API response"""
    model: str
    response: str
    done: bool = True
    context: List[int] = None

    def json(self):
        return {
            "model": self.model,
            "response": self.response,
            "done": self.done,
            "context": self.context or []
        }


@dataclass
class MockEmbeddingResponse:
    """Mock embedding response"""
    embedding: List[float]

    def json(self):
        return {"embedding": self.embedding}


@pytest.fixture
def mock_ollama_response():
    """Factory for mock Ollama responses"""
    def _create(model: str = "qwen2.5-coder:7b", response: str = "Test response"):
        return MockOllamaResponse(model=model, response=response)
    return _create


@pytest.fixture
def mock_embedding():
    """Mock 768-dimensional embedding vector"""
    import random
    return [random.uniform(-1, 1) for _ in range(768)]


# ============================================================
# HTTP Client Fixtures
# ============================================================

@pytest.fixture
async def async_client(test_config):
    """Async HTTP client for API tests"""
    async with httpx.AsyncClient(
        base_url=test_config["echo_brain_url"],
        timeout=30.0
    ) as client:
        yield client


@pytest.fixture
def sync_client(test_config):
    """Sync HTTP client for API tests"""
    with httpx.Client(
        base_url=test_config["echo_brain_url"],
        timeout=30.0
    ) as client:
        yield client


# ============================================================
# Database Fixtures
# ============================================================

@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock())
    conn.cursor.return_value.__exit__ = MagicMock(return_value=None)
    return conn


@pytest.fixture
def mock_async_db():
    """Mock async database pool"""
    pool = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


# ============================================================
# Qdrant Fixtures
# ============================================================

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client"""
    client = MagicMock()
    client.get_collections.return_value.collections = []
    client.create_collection.return_value = True
    client.upsert.return_value = MagicMock(status="completed")
    client.search.return_value = []
    return client


# ============================================================
# Model Router Fixtures
# ============================================================

@pytest.fixture
def sample_queries():
    """Sample queries for testing model routing"""
    return {
        "simple": "What time is it?",
        "medium": "Explain how Python decorators work with examples",
        "complex": """
            I need to design a microservices architecture for a real-time
            video processing pipeline that handles 4K streams, applies ML
            inference for object detection, and stores results in a distributed
            database. Consider scalability, fault tolerance, and cost optimization.
            What are the trade-offs between different approaches?
        """,
        "reasoning": "Think through the pros and cons of using PostgreSQL vs MongoDB for this use case",
        "code": "Write a Python function to merge two sorted arrays",
        "anime": "Generate an anime character with blue hair and red eyes",
    }


@pytest.fixture
def complexity_expectations():
    """Expected complexity scores for sample queries"""
    return {
        "simple": {"min": 0, "max": 15, "model": "qwen2.5-coder:7b"},
        "medium": {"min": 10, "max": 40, "model": "qwen2.5-coder:7b"},
        "complex": {"min": 40, "max": 100, "model": "qwen2.5-coder:32b"},
        "reasoning": {"min": 30, "max": 80, "model": "deepseek-r1"},
    }


# ============================================================
# API Test Data
# ============================================================

@pytest.fixture
def chat_request():
    """Sample chat request payload"""
    return {
        "query": "Hello, how are you?",
        "user_id": "test_user",
        "conversation_id": "test_conv_001",
        "intelligence_level": "auto"
    }


@pytest.fixture
def delegation_request():
    """Sample delegation request payload"""
    return {
        "task": "List all Python files in the current directory",
        "context": {"base_path": "/tmp"},
        "model": "qwen2.5-coder:7b",
        "priority": "normal"
    }


# ============================================================
# Anime Integration Fixtures
# ============================================================

@pytest.fixture
def anime_character_data():
    """Sample anime character data"""
    return {
        "name": "Kai Nakamura",
        "description": "A young warrior with spiky black hair and determined eyes",
        "style_elements": {
            "hair_color": "black",
            "eye_color": "brown",
            "outfit": "traditional warrior garb"
        },
        "consistency_score": 0.85,
        "generation_count": 15
    }


@pytest.fixture
def anime_generation_request():
    """Sample anime generation request"""
    return {
        "prompt": "Generate Kai in a battle stance",
        "character_id": 1,
        "style": "action",
        "duration": 2
    }


# ============================================================
# Auth Fixtures
# ============================================================

@pytest.fixture
def test_user():
    """Test user data"""
    return {
        "username": "test_patrick",
        "user_id": "test_user_001",
        "permissions": ["read", "write", "execute"],
        "access_level": "admin"
    }


@pytest.fixture
def auth_headers(test_user):
    """Authentication headers"""
    return {
        "X-Username": test_user["username"],
        "X-User-Id": test_user["user_id"],
        "Authorization": "Bearer test_token_12345"
    }


# ============================================================
# Worker Fixtures
# ============================================================

@pytest.fixture
def worker_task():
    """Sample worker task"""
    return {
        "task_id": "task_001",
        "task_type": "code_refactor",
        "payload": {
            "file_path": "/tmp/test.py",
            "operation": "optimize"
        },
        "priority": 1,
        "status": "pending"
    }


# ============================================================
# Celery Fixtures (if using Celery)
# ============================================================

@pytest.fixture
def mock_celery_app():
    """Mock Celery application"""
    app = MagicMock()
    app.send_task.return_value = MagicMock(id="task_123")
    return app


# ============================================================
# Utility Functions
# ============================================================

@pytest.fixture
def assert_response_ok():
    """Helper to assert successful API response"""
    def _assert(response, expected_keys: List[str] = None):
        assert response.status_code in [200, 201], f"Expected 2xx, got {response.status_code}: {response.text}"
        if expected_keys:
            data = response.json()
            for key in expected_keys:
                assert key in data, f"Missing key: {key}"
    return _assert


@pytest.fixture
def assert_error_response():
    """Helper to assert error response"""
    def _assert(response, expected_status: int, error_contains: str = None):
        assert response.status_code == expected_status
        if error_contains:
            assert error_contains.lower() in response.text.lower()
    return _assert
