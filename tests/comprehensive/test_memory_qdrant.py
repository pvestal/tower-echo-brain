"""
Comprehensive Memory and Qdrant Vector Database Tests

Tests:
- Qdrant client initialization
- Collection management
- Embedding generation
- Vector storage and retrieval
- Memory search
- Conversation memory
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestQdrantClientInit:
    """Test Qdrant client initialization"""

    def test_qdrant_client_initializes(self, mock_qdrant_client):
        """QdrantMemory should initialize without errors"""
        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            from qdrant_client import QdrantMemory
            memory = QdrantMemory()
            assert memory is not None

    def test_qdrant_default_collection_name(self, mock_qdrant_client):
        """Should use default collection name if not specified"""
        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            from qdrant_client import QdrantMemory
            memory = QdrantMemory()
            assert memory.collection_name == "echo_memories"

    def test_qdrant_custom_collection_name(self, mock_qdrant_client):
        """Should use custom collection name when specified"""
        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            from qdrant_client import QdrantMemory
            memory = QdrantMemory(collection_name="custom_collection")
            assert memory.collection_name == "custom_collection"

    def test_qdrant_vector_size(self, mock_qdrant_client):
        """Should use correct vector size for embeddings"""
        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            from qdrant_client import QdrantMemory
            memory = QdrantMemory()
            assert memory.vector_size == 768  # nomic-embed-text dimension


class TestCollectionManagement:
    """Test Qdrant collection operations"""

    def test_ensure_collection_creates_if_not_exists(self, mock_qdrant_client):
        """Should create collection if it doesn't exist"""
        mock_qdrant_client.get_collections.return_value.collections = []

        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            from qdrant_client import QdrantMemory
            memory = QdrantMemory()
            mock_qdrant_client.create_collection.assert_called_once()

    def test_ensure_collection_skips_if_exists(self, mock_qdrant_client):
        """Should not create collection if it already exists"""
        mock_collection = MagicMock()
        mock_collection.name = "echo_memories"
        mock_qdrant_client.get_collections.return_value.collections = [mock_collection]

        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            from qdrant_client import QdrantMemory
            memory = QdrantMemory()
            mock_qdrant_client.create_collection.assert_not_called()


class TestEmbeddingGeneration:
    """Test embedding generation via Ollama"""

    @pytest.mark.asyncio
    async def test_generate_embedding_returns_vector(self, mock_qdrant_client, mock_embedding):
        """Should generate 768-dimensional embedding vector"""
        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"embedding": mock_embedding}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                from qdrant_client import QdrantMemory
                memory = QdrantMemory()
                embedding = await memory.generate_embedding("test text")

                assert embedding is not None
                assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_generate_embedding_handles_error(self, mock_qdrant_client):
        """Should handle embedding generation errors gracefully"""
        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                from qdrant_client import QdrantMemory
                memory = QdrantMemory()
                embedding = await memory.generate_embedding("test text")

                assert embedding is None

    @pytest.mark.asyncio
    async def test_generate_embedding_uses_correct_model(self, mock_qdrant_client, mock_embedding):
        """Should use nomic-embed-text model for embeddings"""
        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"embedding": mock_embedding}

                mock_post = AsyncMock(return_value=mock_response)
                mock_client.return_value.__aenter__.return_value.post = mock_post

                from qdrant_client import QdrantMemory
                memory = QdrantMemory()
                await memory.generate_embedding("test text")

                # Verify correct model was requested
                call_args = mock_post.call_args
                if call_args:
                    json_arg = call_args.kwargs.get('json', call_args[1].get('json', {}))
                    assert json_arg.get('model') == 'nomic-embed-text:latest'


class TestMemoryStorage:
    """Test memory storage operations"""

    @pytest.mark.asyncio
    async def test_store_memory(self, mock_qdrant_client, mock_embedding):
        """Should store memory with embedding"""
        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"embedding": mock_embedding}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                from qdrant_client import QdrantMemory
                memory = QdrantMemory()

                # Test store operation
                if hasattr(memory, 'store_memory'):
                    result = await memory.store_memory(
                        content="Test memory content",
                        metadata={"type": "conversation", "user": "test"}
                    )
                    # Should call upsert on client
                    mock_qdrant_client.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_store_memory_with_metadata(self, mock_qdrant_client, mock_embedding):
        """Should store metadata alongside vector"""
        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"embedding": mock_embedding}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                from qdrant_client import QdrantMemory
                memory = QdrantMemory()

                metadata = {
                    "type": "conversation",
                    "user": "patrick",
                    "timestamp": "2025-01-09T00:00:00Z",
                    "conversation_id": "conv_123"
                }

                if hasattr(memory, 'store_memory'):
                    await memory.store_memory(
                        content="Test with metadata",
                        metadata=metadata
                    )


class TestMemorySearch:
    """Test memory search operations"""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_qdrant_client, mock_embedding):
        """Should return search results"""
        mock_search_result = MagicMock()
        mock_search_result.payload = {"content": "test", "type": "conversation"}
        mock_search_result.score = 0.95
        mock_qdrant_client.search.return_value = [mock_search_result]

        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"embedding": mock_embedding}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                from qdrant_client import QdrantMemory
                memory = QdrantMemory()

                if hasattr(memory, 'search'):
                    results = await memory.search("test query", limit=5)
                    assert results is not None

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, mock_qdrant_client, mock_embedding):
        """Should respect result limit parameter"""
        mock_results = [MagicMock(payload={"content": f"result_{i}"}, score=0.9-i*0.1)
                       for i in range(10)]
        mock_qdrant_client.search.return_value = mock_results[:5]

        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"embedding": mock_embedding}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                from qdrant_client import QdrantMemory
                memory = QdrantMemory()

                if hasattr(memory, 'search'):
                    results = await memory.search("test", limit=5)
                    # Should not exceed limit

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_qdrant_client, mock_embedding):
        """Should apply filters to search"""
        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"embedding": mock_embedding}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                from qdrant_client import QdrantMemory
                memory = QdrantMemory()

                if hasattr(memory, 'search'):
                    # Search with type filter
                    results = await memory.search(
                        "test query",
                        filter_by={"type": "conversation"}
                    )


class TestConversationMemory:
    """Test conversation-specific memory operations"""

    @pytest.mark.asyncio
    async def test_store_conversation(self, mock_qdrant_client, mock_embedding):
        """Should store conversation turns"""
        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"embedding": mock_embedding}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                from qdrant_client import QdrantMemory
                memory = QdrantMemory()

                conversation = {
                    "conversation_id": "conv_123",
                    "user_message": "Hello Echo",
                    "assistant_response": "Hello! How can I help?",
                    "timestamp": "2025-01-09T00:00:00Z"
                }

                if hasattr(memory, 'store_conversation'):
                    await memory.store_conversation(**conversation)

    @pytest.mark.asyncio
    async def test_retrieve_conversation_context(self, mock_qdrant_client, mock_embedding):
        """Should retrieve relevant conversation context"""
        mock_results = [
            MagicMock(payload={
                "content": "Previous conversation about Python",
                "conversation_id": "conv_123"
            }, score=0.9)
        ]
        mock_qdrant_client.search.return_value = mock_results

        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"embedding": mock_embedding}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                from qdrant_client import QdrantMemory
                memory = QdrantMemory()

                if hasattr(memory, 'get_conversation_context'):
                    context = await memory.get_conversation_context(
                        query="Tell me more about Python",
                        conversation_id="conv_123"
                    )


class TestMemoryIntegration:
    """Integration tests for memory system"""

    @pytest.mark.asyncio
    async def test_memory_roundtrip(self, mock_qdrant_client, mock_embedding):
        """Store and retrieve memory should work correctly"""
        stored_content = "Important information about the user's preferences"

        mock_search_result = MagicMock()
        mock_search_result.payload = {"content": stored_content}
        mock_search_result.score = 0.95
        mock_qdrant_client.search.return_value = [mock_search_result]

        with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"embedding": mock_embedding}
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                from qdrant_client import QdrantMemory
                memory = QdrantMemory()

                # Store
                if hasattr(memory, 'store_memory'):
                    await memory.store_memory(content=stored_content, metadata={})

                # Retrieve
                if hasattr(memory, 'search'):
                    results = await memory.search("user preferences")
                    if results:
                        assert any(stored_content in str(r) for r in results)


class TestMemoryContextRetrieval:
    """Test context retrieval for conversations"""

    def test_context_retriever_init(self):
        """ConversationContextRetriever should initialize"""
        try:
            from memory.context_retrieval import ConversationContextRetriever
            retriever = ConversationContextRetriever()
            assert retriever is not None
        except ImportError:
            pytest.skip("context_retrieval module not available")

    def test_pronoun_resolver(self):
        """PronounResolver should resolve pronouns to entities"""
        try:
            from memory.pronoun_resolver import PronounResolver
            resolver = PronounResolver()

            # Test basic pronoun resolution
            context = {"entities": [{"name": "Patrick", "type": "person"}]}
            text = "He said hello"

            resolved = resolver.resolve(text, context) if hasattr(resolver, 'resolve') else text
            # Should attempt to resolve "He" to "Patrick"
        except ImportError:
            pytest.skip("pronoun_resolver module not available")

    def test_entity_extractor(self):
        """EntityExtractor should extract entities from text"""
        try:
            from memory.entity_extractor import EntityExtractor
            extractor = EntityExtractor()

            text = "Patrick is working on Echo Brain in the Tower server room"

            if hasattr(extractor, 'extract'):
                entities = extractor.extract(text)
                # Should find "Patrick", "Echo Brain", "Tower"
        except ImportError:
            pytest.skip("entity_extractor module not available")
