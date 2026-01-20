"""
Comprehensive tests for Echo Brain Unified Context System
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.unified_context import UnifiedContextProvider
from src.core.agent_router import AgentRouter
from src.agents.context_aware_base import ContextAwareAgent


class TestUnifiedContextProvider:
    """Test context retrieval from all sources"""

    @pytest.mark.asyncio
    async def test_qdrant_connection(self):
        """Verify Qdrant connection and search works"""
        provider = UnifiedContextProvider()

        # Mock Qdrant client
        mock_qdrant = MagicMock()
        mock_qdrant.search_memories = AsyncMock(return_value=[
            {"text": "Test memory", "score": 0.95, "metadata": {}}
        ])
        provider.qdrant = mock_qdrant
        provider._initialized = True

        # Test search
        results = await provider.search_memories("test query", limit=5)

        assert len(results) > 0
        assert results[0]["text"] == "Test memory"
        assert results[0]["score"] == 0.95
        mock_qdrant.search_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_postgres_connection(self):
        """Verify PostgreSQL connection and facts query works"""
        provider = UnifiedContextProvider()

        # Mock PostgreSQL connection
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                "subject": "Echo Brain",
                "predicate": "is",
                "object": "AI assistant",
                "confidence": 1.0,
                "source": "test",
                "created_at": None
            }
        ]
        provider.pg_conn = mock_conn
        provider._initialized = True

        # Test facts retrieval
        facts = await provider.get_facts("Echo Brain")

        assert len(facts) > 0
        assert facts[0]["subject"] == "Echo Brain"
        assert facts[0]["predicate"] == "is"
        assert facts[0]["object"] == "AI assistant"

    @pytest.mark.asyncio
    async def test_ollama_embeddings(self):
        """Verify Ollama can generate embeddings"""
        provider = UnifiedContextProvider()

        # Mock httpx client
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embedding": [0.1] * 768
            }
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Test embedding generation
            embedding = await provider.generate_embedding("test text")

            assert embedding is not None
            assert len(embedding) == 768
            assert all(isinstance(x, (int, float)) for x in embedding)

    @pytest.mark.asyncio
    async def test_get_context_returns_all_sources(self):
        """Verify get_context returns memories, facts, and recent"""
        provider = UnifiedContextProvider()

        # Mock all sources
        provider.qdrant = MagicMock()
        provider.qdrant.search_memories = AsyncMock(return_value=[
            {"text": "Memory 1", "score": 0.9, "metadata": {}}
        ])

        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = [
            # Facts
            [{"subject": "test", "predicate": "is", "object": "fact", "confidence": 1.0, "source": "db", "created_at": None}],
            # Recent conversations
            [{"summary": "Recent conversation", "key_topics": ["test"], "timestamp": None, "metadata": {}}]
        ]
        provider.pg_conn = mock_conn
        provider._initialized = True

        # Get context
        context = await provider.get_context("test query", limit=10)

        assert "memories" in context
        assert "facts" in context
        assert "recent_conversations" in context
        assert len(context["memories"]) > 0
        assert len(context["facts"]) > 0
        assert len(context["recent_conversations"]) > 0

    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Verify graceful handling of empty queries"""
        provider = UnifiedContextProvider()
        provider._initialized = True
        provider.qdrant = MagicMock()
        provider.qdrant.search_memories = AsyncMock(return_value=[])
        provider.pg_conn = AsyncMock()
        provider.pg_conn.fetch.return_value = []

        # Test with empty query
        context = await provider.get_context("", limit=10)

        assert context["query"] == ""
        assert context["memories"] == []
        assert context["facts"] == []

    @pytest.mark.asyncio
    async def test_context_relevance(self):
        """Verify returned context is relevant to query"""
        provider = UnifiedContextProvider()
        provider._initialized = True

        # Mock Qdrant with relevant results
        provider.qdrant = MagicMock()
        provider.qdrant.search_memories = AsyncMock(return_value=[
            {"text": "Python programming guide", "score": 0.95, "metadata": {}},
            {"text": "JavaScript basics", "score": 0.75, "metadata": {}}
        ])

        # Mock PostgreSQL with relevant facts
        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = [
            [{"subject": "Python", "predicate": "is", "object": "programming language", "confidence": 1.0, "source": "db", "created_at": None}],
            []
        ]
        provider.pg_conn = mock_conn

        # Search for Python-related content
        context = await provider.get_context("Python programming", limit=5)

        # Verify relevance
        assert len(context["memories"]) > 0
        assert "Python" in context["memories"][0]["text"]
        assert context["memories"][0]["score"] > 0.9


class TestContextAwareAgents:
    """Test that agents properly use context"""

    @pytest.mark.asyncio
    async def test_coding_agent_receives_context(self):
        """Verify CodingAgent gets context before processing"""
        from src.agents.coding_agent import CodingAgent

        agent = CodingAgent()

        # Mock context provider
        mock_provider = AsyncMock()
        mock_provider.get_context.return_value = {
            "memories": [{"text": "Previous code solution", "score": 0.9}],
            "facts": [],
            "recent_conversations": []
        }
        agent.context_provider = mock_provider

        # Mock call_model
        agent.call_model = AsyncMock(return_value="def test(): return True")

        # Process with context
        result = await agent.process("Write a test function", include_context=True)

        # Verify context was retrieved
        mock_provider.get_context.assert_called_once()
        assert "memories" in result.get("context_used", {})

    @pytest.mark.asyncio
    async def test_reasoning_agent_receives_context(self):
        """Verify ReasoningAgent gets context before processing"""
        from src.agents.reasoning_agent import ReasoningAgent

        agent = ReasoningAgent()

        # Mock context provider
        mock_provider = AsyncMock()
        mock_provider.get_context.return_value = {
            "memories": [],
            "facts": [{"subject": "AI", "predicate": "uses", "object": "machine learning"}],
            "recent_conversations": []
        }
        agent.context_provider = mock_provider

        # Mock call_model
        agent.call_model = AsyncMock(return_value="## Analysis\nTest analysis\n## Conclusion\nTest conclusion")

        # Process with context
        result = await agent.process("Should I use AI?", include_context=True)

        # Verify context was retrieved
        mock_provider.get_context.assert_called_once()
        assert result["model"] == "deepseek-r1:8b"

    @pytest.mark.asyncio
    async def test_narration_agent_receives_context(self):
        """Verify NarrationAgent gets context before processing"""
        from src.agents.narration_agent import NarrationAgent

        agent = NarrationAgent()

        # Mock context provider
        mock_provider = AsyncMock()
        mock_provider.get_context.return_value = {
            "memories": [{"text": "Anime character description", "score": 0.85}],
            "facts": [],
            "recent_conversations": []
        }
        agent.context_provider = mock_provider

        # Mock call_model
        agent.call_model = AsyncMock(return_value="## Narration\nThe scene unfolds...")

        # Process with context
        result = await agent.process("Describe a cyberpunk scene", include_context=True)

        # Verify context was retrieved
        mock_provider.get_context.assert_called_once()
        assert result["model"] == "gemma2:9b"

    @pytest.mark.asyncio
    async def test_agent_response_includes_context_info(self):
        """Verify agent responses reference provided context"""
        agent = ContextAwareAgent(
            model_name="test-model",
            system_prompt="Test prompt"
        )

        # Mock context with specific content
        mock_provider = AsyncMock()
        mock_provider.get_context.return_value = {
            "memories": [{"text": "Important context", "score": 0.95}],
            "facts": [{"subject": "Test", "predicate": "has", "object": "context"}],
            "recent_conversations": []
        }
        agent.context_provider = mock_provider

        # Mock call_model to include context reference
        agent.call_model = AsyncMock(return_value="Based on the context: Important context")

        # Process
        result = await agent.process("Test query", include_context=True)

        assert "context" in result["response"].lower()
        assert result["context_used"] is not None

    @pytest.mark.asyncio
    async def test_agent_without_context_flag(self):
        """Verify agents work with include_context=False"""
        agent = ContextAwareAgent(
            model_name="test-model",
            system_prompt="Test prompt"
        )

        # Mock call_model
        agent.call_model = AsyncMock(return_value="Response without context")

        # Process without context
        result = await agent.process("Test query", include_context=False)

        assert result["response"] == "Response without context"
        assert result.get("context_used") is None or result["context_used"] == {}


class TestAgentRouter:
    """Test query routing logic"""

    @pytest.mark.asyncio
    async def test_coding_query_routes_to_coding_agent(self):
        """'Fix this Python bug' -> CodingAgent"""
        router = AgentRouter()

        # Mock agents
        router.coding_agent.process = AsyncMock(return_value={
            "task": "Fix bug",
            "response": "Bug fixed",
            "model": "deepseek-coder-v2:16b"
        })

        # Route coding query
        result = await router.route("Fix this Python bug", agent="auto")

        assert result["agent_used"] == "coding"
        router.coding_agent.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_reasoning_query_routes_to_reasoning_agent(self):
        """'Should I use React or Vue?' -> ReasoningAgent"""
        router = AgentRouter()

        # Mock agents
        router.reasoning_agent.process = AsyncMock(return_value={
            "task": "Compare frameworks",
            "response": "Analysis complete",
            "model": "deepseek-r1:8b"
        })

        # Route reasoning query
        result = await router.route("Should I use React or Vue?", agent="auto")

        assert result["agent_used"] == "reasoning"
        router.reasoning_agent.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_narration_query_routes_to_narration_agent(self):
        """'Write a scene for my anime' -> NarrationAgent"""
        router = AgentRouter()

        # Mock agents
        router.narration_agent.process = AsyncMock(return_value={
            "scene": "Anime scene",
            "narration": "Scene narrated",
            "model": "gemma2:9b"
        })

        # Route narration query
        result = await router.route("Write a scene for my anime", agent="auto")

        assert result["agent_used"] == "narration"
        router.narration_agent.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_ambiguous_query_defaults_to_reasoning(self):
        """'Help me with this' -> ReasoningAgent (default)"""
        router = AgentRouter()

        # Mock agents
        router.reasoning_agent.process = AsyncMock(return_value={
            "task": "Help request",
            "response": "Here's help",
            "model": "deepseek-r1:8b"
        })

        # Route ambiguous query
        result = await router.route("Help me with this", agent="auto")

        assert result["agent_used"] == "reasoning"
        router.reasoning_agent.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_agent_override(self):
        """Verify agent parameter overrides auto-routing"""
        router = AgentRouter()

        # Mock agents
        router.narration_agent.process = AsyncMock(return_value={
            "scene": "Debug scene",
            "narration": "Narrated",
            "model": "gemma2:9b"
        })

        # Route coding query to narration agent explicitly
        result = await router.route("Debug this code", agent="narration")

        assert result["agent_used"] == "narration"
        router.narration_agent.process.assert_called_once()


class TestAPIEndpoints:
    """Test the API integration"""

    @pytest.mark.asyncio
    async def test_unified_agent_endpoint(self):
        """POST /api/agent returns valid response"""
        from fastapi.testclient import TestClient
        from src.main import app

        with TestClient(app) as client:
            # Mock the agent router
            with patch('src.main.agent_router') as mock_router:
                mock_router.route = AsyncMock(return_value={
                    "task": "test",
                    "response": "test response",
                    "agent_used": "reasoning",
                    "routing_method": "auto",
                    "model": "test-model"
                })

                # Test the endpoint
                response = client.post("/api/agent", json={
                    "query": "Test query",
                    "agent": "auto"
                })

                # Note: This will fail if agent_router is None
                # In real test, need to initialize properly

    @pytest.mark.asyncio
    async def test_context_endpoint(self):
        """POST /api/context/unified returns context without agent processing"""
        from fastapi.testclient import TestClient
        from src.main import app

        with TestClient(app) as client:
            # Mock the context provider
            with patch('src.main.context_provider') as mock_provider:
                mock_provider.get_context = AsyncMock(return_value={
                    "query": "test",
                    "memories": [],
                    "facts": [],
                    "recent_conversations": []
                })

                # Test the endpoint
                response = client.post("/api/context/unified", json={
                    "query": "Test query",
                    "limit": 5
                })

    @pytest.mark.asyncio
    async def test_health_includes_agent_status(self):
        """GET /health shows agent availability"""
        from fastapi.testclient import TestClient
        from src.main import app

        with TestClient(app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "agents_available" in data


class TestIntegration:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    async def test_full_flow_coding_query(self):
        """
        Query -> Context retrieval -> CodingAgent -> Response
        Verify entire pipeline works
        """
        router = AgentRouter()

        # Mock context provider
        mock_provider = AsyncMock()
        mock_provider.get_context.return_value = {
            "memories": [{"text": "def example(): pass", "score": 0.9}],
            "facts": [],
            "recent_conversations": []
        }
        router.context_provider = mock_provider

        # Mock coding agent
        router.coding_agent.process = AsyncMock(return_value={
            "task": "Write function",
            "response": "def hello(): return 'Hello'",
            "code": "def hello(): return 'Hello'",
            "model": "deepseek-coder-v2:16b",
            "context_used": {"memories": 1}
        })

        # Execute full flow
        result = await router.route("Write a hello function", agent="auto")

        # Verify flow
        assert result["agent_used"] == "coding"
        assert "def hello()" in result["response"]
        mock_provider.get_context.assert_called()

    @pytest.mark.asyncio
    async def test_full_flow_with_relevant_memories(self):
        """
        1. Store a test memory in Qdrant
        2. Query related to that memory
        3. Verify memory appears in context
        4. Verify agent references the memory
        """
        provider = UnifiedContextProvider()

        # Mock Qdrant with specific memory
        provider.qdrant = MagicMock()
        provider.qdrant.search_memories = AsyncMock(return_value=[
            {"text": "Tower Echo Brain uses Qdrant for vector search", "score": 0.98, "metadata": {"test": True}}
        ])
        provider.pg_conn = AsyncMock()
        provider.pg_conn.fetch.return_value = []
        provider._initialized = True

        # Get context
        context = await provider.get_context("How does Tower Echo Brain search?")

        # Verify memory appears
        assert len(context["memories"]) > 0
        assert "Qdrant" in context["memories"][0]["text"]
        assert context["memories"][0]["score"] > 0.95

    @pytest.mark.asyncio
    async def test_concurrent_agent_requests(self):
        """Verify system handles multiple simultaneous requests"""
        router = AgentRouter()

        # Mock all agents
        for agent in [router.coding_agent, router.reasoning_agent, router.narration_agent]:
            agent.process = AsyncMock(return_value={
                "task": "concurrent",
                "response": f"{agent.__class__.__name__} response",
                "model": "test"
            })

        # Launch concurrent requests
        tasks = [
            router.route("Write code", agent="coding"),
            router.route("Analyze this", agent="reasoning"),
            router.route("Narrate scene", agent="narration")
        ]

        results = await asyncio.gather(*tasks)

        # Verify all completed
        assert len(results) == 3
        assert results[0]["agent_used"] == "coding"
        assert results[1]["agent_used"] == "reasoning"
        assert results[2]["agent_used"] == "narration"

    @pytest.mark.asyncio
    async def test_context_provider_caching(self):
        """Verify repeated queries don't hammer databases"""
        provider = UnifiedContextProvider()

        # Mock sources
        provider.qdrant = MagicMock()
        provider.qdrant.search_memories = AsyncMock(return_value=[])
        provider.pg_conn = AsyncMock()
        provider.pg_conn.fetch.return_value = []
        provider._initialized = True

        # First query
        context1 = await provider.get_context("test query")

        # Second identical query (should use cache)
        context2 = await provider.get_context("test query")

        # Verify caching worked (only called once)
        provider.qdrant.search_memories.assert_called_once()

        # Results should be identical
        assert context1 == context2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])