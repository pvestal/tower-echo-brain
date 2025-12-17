"""
Test suite for VectorConnector - comprehensive failing tests for Qdrant integration.
These tests will fail until VectorConnector is implemented.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import numpy as np
from typing import List, Dict, Any

from src.connectors.vector_connector import VectorConnector
from src.models.learning_item import LearningItem, LearningItemType
from src.config.settings import VectorDatabaseConfig


@pytest.fixture
def qdrant_config():
    """Qdrant configuration for localhost."""
    return VectorDatabaseConfig(
        host="localhost",
        port=6333,
        collection_name="claude_conversations",
        embedding_dimension=384,
        batch_size=100,
        timeout=60
    )


@pytest.mark.asyncio
class TestVectorConnector:
    """Test suite for Qdrant vector database connector."""

    async def test_vector_connector_initialization(self, qdrant_config):
        """Test connector initializes with proper configuration."""
        connector = VectorConnector(qdrant_config)

        assert connector.config.host == "localhost"
        assert connector.config.port == 6333
        assert connector.config.collection_name == "claude_conversations"
        assert connector.config.embedding_dimension == 384
        assert connector.client is None  # Not connected yet

    async def test_connect_to_qdrant(self, qdrant_config):
        """Test connector connects to Qdrant server."""
        connector = VectorConnector(qdrant_config)

        # Should connect without errors
        await connector.connect()

        assert connector.client is not None
        assert connector.is_connected is True

        # Clean up
        await connector.disconnect()

    async def test_collection_creation(self, qdrant_config):
        """Test automatic collection creation if it doesn't exist."""
        connector = VectorConnector(qdrant_config)
        await connector.connect()

        # Should ensure collection exists
        collection_exists = await connector.ensure_collection_exists()
        assert collection_exists is True

        # Should be able to get collection info
        collection_info = await connector.get_collection_info()
        assert collection_info is not None
        assert collection_info['dimension'] == 384
        assert collection_info['name'] == "claude_conversations"

        await connector.disconnect()

    async def test_add_learning_items_with_embeddings(self, qdrant_config):
        """Test adding learning items with automatic embedding generation."""
        connector = VectorConnector(qdrant_config)
        await connector.connect()

        # Create test learning items
        learning_items = [
            LearningItem(
                content="Database connection patterns for PostgreSQL localhost setup",
                item_type=LearningItemType.INSIGHT,
                title="Database Connection Best Practice",
                source_file="test_conversation.md",
                confidence_score=0.9
            ),
            LearningItem(
                content="async def connect_db(): return await asyncpg.connect(host='localhost')",
                item_type=LearningItemType.CODE_EXAMPLE,
                title="Async Database Connection",
                source_file="test_conversation.md",
                confidence_score=0.85
            ),
            LearningItem(
                content="Circuit breaker pattern prevents cascade failures in distributed systems",
                item_type=LearningItemType.SOLUTION,
                title="Circuit Breaker Pattern",
                source_file="test_conversation.md",
                confidence_score=0.95
            )
        ]

        # Should add items and return count
        added_count = await connector.add_learning_items(learning_items)
        assert added_count == 3

        # Should be able to search for similar content
        search_results = await connector.search_similar(
            "database connection postgresql",
            limit=2
        )
        assert len(search_results) > 0
        assert search_results[0]['score'] > 0.5  # Should have reasonable similarity

        await connector.disconnect()

    async def test_search_functionality(self, qdrant_config):
        """Test vector search with various parameters."""
        connector = VectorConnector(qdrant_config)
        await connector.connect()

        # Add some test data first
        test_items = [
            LearningItem(
                content="Python asyncio patterns for concurrent processing",
                item_type=LearningItemType.INSIGHT,
                title="Asyncio Best Practices"
            ),
            LearningItem(
                content="SQL query optimization techniques for PostgreSQL",
                item_type=LearningItemType.INSIGHT,
                title="SQL Optimization"
            ),
            LearningItem(
                content="Machine learning model deployment strategies",
                item_type=LearningItemType.SOLUTION,
                title="ML Deployment"
            )
        ]
        await connector.add_learning_items(test_items)

        # Test basic search
        results = await connector.search_similar("python asyncio", limit=5)
        assert len(results) > 0
        assert results[0]['score'] > 0.3

        # Test search with filters
        insight_results = await connector.search_similar(
            "optimization",
            limit=5,
            filters={"item_type": "insight"}
        )
        assert len(insight_results) > 0

        # Test search by item type
        code_results = await connector.search_by_type(LearningItemType.CODE_EXAMPLE, limit=3)
        # May be empty if no code examples were added

        await connector.disconnect()

    async def test_batch_processing(self, qdrant_config):
        """Test batch processing of large numbers of learning items."""
        connector = VectorConnector(qdrant_config)
        await connector.connect()

        # Create large batch (more than batch_size)
        large_batch = []
        for i in range(150):  # More than batch_size of 100
            item = LearningItem(
                content=f"Test learning item {i} with unique content for batch processing",
                item_type=LearningItemType.INSIGHT,
                title=f"Batch Item {i}",
                source_file=f"batch_test_{i}.md"
            )
            large_batch.append(item)

        # Should process in batches
        added_count = await connector.add_learning_items(large_batch)
        assert added_count == 150

        # Should be able to search for items from the batch
        search_results = await connector.search_similar("batch processing", limit=10)
        assert len(search_results) > 0

        await connector.disconnect()

    async def test_update_existing_vectors(self, qdrant_config):
        """Test updating existing vectors when content is modified."""
        connector = VectorConnector(qdrant_config)
        await connector.connect()

        # Add initial item
        original_item = LearningItem(
            content="Original content about vector databases",
            item_type=LearningItemType.INSIGHT,
            title="Vector DB Concept",
            source_file="update_test.md",
            unique_id="update-test-123"
        )
        await connector.add_learning_items([original_item])

        # Update the item with new content
        updated_item = LearningItem(
            content="Updated content about advanced vector database operations",
            item_type=LearningItemType.INSIGHT,
            title="Advanced Vector DB Concept",
            source_file="update_test.md",
            unique_id="update-test-123"  # Same ID
        )

        # Should update existing vector instead of creating new one
        update_count = await connector.update_learning_item(updated_item)
        assert update_count == 1

        # Search should return updated content
        results = await connector.search_similar("advanced vector database", limit=1)
        assert len(results) > 0
        assert "advanced" in results[0]['content'].lower()

        await connector.disconnect()

    async def test_delete_by_source_file(self, qdrant_config):
        """Test deleting vectors by source file."""
        connector = VectorConnector(qdrant_config)
        await connector.connect()

        # Add items from different source files
        items = [
            LearningItem(
                content="Content from file A",
                item_type=LearningItemType.INSIGHT,
                source_file="file_a.md"
            ),
            LearningItem(
                content="Content from file B",
                item_type=LearningItemType.INSIGHT,
                source_file="file_b.md"
            ),
            LearningItem(
                content="More content from file A",
                item_type=LearningItemType.CODE_EXAMPLE,
                source_file="file_a.md"
            )
        ]
        await connector.add_learning_items(items)

        # Delete all vectors from file_a.md
        deleted_count = await connector.delete_by_source_file("file_a.md")
        assert deleted_count == 2

        # Search should only find content from file_b.md
        all_results = await connector.search_similar("content", limit=10)
        remaining_files = [r.get('source_file', '') for r in all_results]
        assert "file_a.md" not in remaining_files

        await connector.disconnect()

    async def test_get_collection_stats(self, qdrant_config):
        """Test getting collection statistics."""
        connector = VectorConnector(qdrant_config)
        await connector.connect()

        # Get initial stats
        initial_stats = await connector.get_collection_stats()
        initial_count = initial_stats.get('vectors_count', 0)

        # Add some items
        test_items = [
            LearningItem(content=f"Stats test item {i}", item_type=LearningItemType.INSIGHT)
            for i in range(5)
        ]
        await connector.add_learning_items(test_items)

        # Stats should reflect new items
        updated_stats = await connector.get_collection_stats()
        assert updated_stats['vectors_count'] >= initial_count + 5
        assert 'memory_usage' in updated_stats
        assert 'indexed_vectors_count' in updated_stats

        await connector.disconnect()

    async def test_health_check(self, qdrant_config):
        """Test Qdrant health check functionality."""
        connector = VectorConnector(qdrant_config)

        # Health check should fail when not connected
        health = await connector.health_check()
        assert health is False

        # Connect and health check should pass
        await connector.connect()
        health = await connector.health_check()
        assert health is True

        # Should be able to get detailed health info
        health_info = await connector.get_health_info()
        assert 'status' in health_info
        assert 'version' in health_info
        assert 'collections' in health_info

        await connector.disconnect()

    async def test_connection_error_handling(self, qdrant_config):
        """Test proper error handling for connection failures."""
        # Use wrong port
        bad_config = VectorDatabaseConfig(
            host="localhost",
            port=9999,  # Wrong port
            collection_name="test"
        )
        connector = VectorConnector(bad_config)

        # Should raise appropriate exception
        with pytest.raises(Exception):  # Connection error
            await connector.connect()

    async def test_embedding_generation_fallback(self, qdrant_config):
        """Test fallback embedding generation when semantic service is unavailable."""
        connector = VectorConnector(qdrant_config)
        await connector.connect()

        # Mock semantic service failure
        with patch.object(connector, '_generate_embeddings_via_service', side_effect=Exception("Service down")):
            # Should fall back to local embedding generation
            test_item = LearningItem(
                content="Test content for embedding generation fallback",
                item_type=LearningItemType.INSIGHT,
                title="Fallback Test"
            )

            # Should still work with fallback
            added_count = await connector.add_learning_items([test_item])
            assert added_count == 1

        await connector.disconnect()

    async def test_concurrent_operations(self, qdrant_config):
        """Test handling multiple concurrent vector operations."""
        connector = VectorConnector(qdrant_config)
        await connector.connect()

        # Perform multiple concurrent operations
        tasks = []
        for i in range(5):
            items = [
                LearningItem(
                    content=f"Concurrent test item {i}-{j}",
                    item_type=LearningItemType.INSIGHT,
                    title=f"Concurrent {i}-{j}"
                )
                for j in range(3)
            ]
            tasks.append(connector.add_learning_items(items))

        # Should handle concurrent operations
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (no exceptions)
        for result in results:
            assert not isinstance(result, Exception)
            assert result == 3  # 3 items per task

        await connector.disconnect()

    async def test_vector_dimension_validation(self, qdrant_config):
        """Test validation of embedding dimensions."""
        connector = VectorConnector(qdrant_config)
        await connector.connect()

        # Mock embedding service to return wrong dimensions
        async def wrong_dimension_embeddings(texts):
            return [np.random.random(256) for _ in texts]  # Wrong dimension (should be 384)

        with patch.object(connector, '_generate_embeddings', side_effect=wrong_dimension_embeddings):
            test_item = LearningItem(
                content="Test dimension validation",
                item_type=LearningItemType.INSIGHT
            )

            # Should raise validation error
            with pytest.raises(ValueError, match="dimension"):
                await connector.add_learning_items([test_item])

        await connector.disconnect()