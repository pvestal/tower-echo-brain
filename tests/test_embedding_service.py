#!/usr/bin/env python3
"""
Test suite for OpenAI Embedding Service
Tests embedding generation, caching, and batch operations
Created: January 6, 2026
"""

import pytest
import asyncio
import json
import os
import tempfile
from unittest.mock import patch, AsyncMock, MagicMock
from src.services.embedding_service import EmbeddingService, create_embedding_service


class TestEmbeddingService:
    """Test suite for EmbeddingService"""

    @pytest.fixture
    def mock_vault_file(self):
        """Create a temporary vault.json file for testing"""
        vault_data = {
            "openai": {
                "api_key": "sk-test-key-12345",
                "model": "text-embedding-3-small",
                "dimensions": 1536
            },
            "postgresql": {
                "host": "localhost",
                "database": "test_db",
                "user": "test_user",
                "password": "test_pass",
                "port": 5432
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(vault_data, f)
            vault_path = f.name

        yield vault_path
        os.unlink(vault_path)

    @pytest.fixture
    def embedding_service(self, mock_vault_file):
        """Create EmbeddingService instance for testing"""
        return EmbeddingService(vault_path=mock_vault_file)

    def test_credential_loading(self, embedding_service):
        """Test that credentials are loaded correctly from vault"""
        assert embedding_service.api_key == "sk-test-key-12345"
        assert embedding_service.model == "text-embedding-3-small"
        assert embedding_service.dimensions == 1536
        assert embedding_service.pg_config["host"] == "localhost"

    def test_invalid_vault_path(self):
        """Test handling of invalid vault path"""
        with pytest.raises(FileNotFoundError):
            EmbeddingService(vault_path="/nonexistent/path.json")

    def test_text_hashing(self, embedding_service):
        """Test text hash generation"""
        text1 = "Hello world"
        text2 = "Hello world"
        text3 = "Different text"

        hash1 = embedding_service._get_text_hash(text1)
        hash2 = embedding_service._get_text_hash(text2)
        hash3 = embedding_service._get_text_hash(text3)

        assert hash1 == hash2  # Same text should have same hash
        assert hash1 != hash3  # Different text should have different hash
        assert len(hash1) == 64  # SHA-256 hex digest

    @pytest.mark.asyncio
    async def test_database_pool_initialization(self, embedding_service):
        """Test database pool initialization"""
        # Mock asyncpg.create_pool and table creation
        with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_create_pool, \
             patch.object(embedding_service, '_create_cache_table', new_callable=AsyncMock) as mock_create_table:

            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool

            await embedding_service._init_db_pool()

            assert embedding_service.db_pool == mock_pool
            mock_create_pool.assert_called_once()
            mock_create_table.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_operations(self, embedding_service):
        """Test embedding cache storage and retrieval"""
        # Mock database operations
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_acquire = AsyncMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire
        embedding_service.db_pool = mock_pool

        text = "Test text for caching"
        embedding = [0.1, 0.2, 0.3] * 512  # 1536D mock embedding

        # Test cache miss
        mock_conn.fetchrow.return_value = None
        cached = await embedding_service._get_cached_embedding(text)
        assert cached is None

        # Test cache storage
        await embedding_service._cache_embedding(text, embedding)
        mock_conn.execute.assert_called()

        # Test cache hit
        mock_conn.fetchrow.return_value = {'embedding': embedding}
        cached = await embedding_service._get_cached_embedding(text)
        assert cached == embedding

    @pytest.mark.asyncio
    async def test_openai_api_call(self, embedding_service):
        """Test OpenAI API call with mocked response"""
        texts = ["Hello world", "Testing embeddings"]

        # Mock aiohttp response
        mock_response_data = {
            "data": [
                {"index": 0, "embedding": [0.1] * 1536},
                {"index": 1, "embedding": [0.2] * 1536}
            ]
        }

        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data

            # Mock the session context manager
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Mock the post context manager
            mock_post = AsyncMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.post.return_value = mock_post

            embeddings = await embedding_service._call_openai_api(texts)

            assert len(embeddings) == 2
            assert len(embeddings[0]) == 1536
            assert len(embeddings[1]) == 1536
            assert embeddings[0] == [0.1] * 1536
            assert embeddings[1] == [0.2] * 1536

    @pytest.mark.asyncio
    async def test_openai_api_rate_limit(self, embedding_service):
        """Test OpenAI API rate limit handling"""
        texts = ["Test text"]

        with patch('aiohttp.ClientSession') as mock_session, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

            # Mock rate limit response then success
            mock_response_429 = AsyncMock()
            mock_response_429.status = 429

            mock_response_200 = AsyncMock()
            mock_response_200.status = 200
            mock_response_200.json.return_value = {
                "data": [{"index": 0, "embedding": [0.1] * 1536}]
            }

            # Mock the session context manager
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Mock the post context manager with side effects
            mock_posts = []
            for resp in [mock_response_429, mock_response_200]:
                mock_post = AsyncMock()
                mock_post.__aenter__ = AsyncMock(return_value=resp)
                mock_post.__aexit__ = AsyncMock(return_value=None)
                mock_posts.append(mock_post)

            mock_session_instance.post.side_effect = mock_posts

            embeddings = await embedding_service._call_openai_api(texts)

            assert len(embeddings) == 1
            mock_sleep.assert_called_once()  # Should have waited for rate limit

    @pytest.mark.asyncio
    async def test_single_embedding_with_cache(self, embedding_service):
        """Test single text embedding with caching"""
        text = "Test embedding text"
        expected_embedding = [0.5] * 1536

        # Mock database and API
        with patch.object(embedding_service, '_get_cached_embedding', return_value=None) as mock_get_cache, \
             patch.object(embedding_service, '_call_openai_api', return_value=[expected_embedding]) as mock_api, \
             patch.object(embedding_service, '_cache_embedding') as mock_cache:

            result = await embedding_service.embed_single(text)

            assert result == expected_embedding
            mock_get_cache.assert_called_once_with(text)
            mock_api.assert_called_once_with([text])
            mock_cache.assert_called_once_with(text, expected_embedding)

    @pytest.mark.asyncio
    async def test_single_embedding_cache_hit(self, embedding_service):
        """Test single text embedding with cache hit"""
        text = "Cached text"
        cached_embedding = [0.7] * 1536

        with patch.object(embedding_service, '_get_cached_embedding', return_value=cached_embedding) as mock_get_cache, \
             patch.object(embedding_service, '_call_openai_api') as mock_api:

            result = await embedding_service.embed_single(text)

            assert result == cached_embedding
            mock_get_cache.assert_called_once_with(text)
            mock_api.assert_not_called()  # Should not call API when cached

    @pytest.mark.asyncio
    async def test_batch_embedding(self, embedding_service):
        """Test batch text embedding"""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

        # Mock cache misses for all texts
        with patch.object(embedding_service, '_get_cached_embedding', return_value=None), \
             patch.object(embedding_service, '_call_openai_api', return_value=embeddings) as mock_api, \
             patch.object(embedding_service, '_cache_embedding') as mock_cache:

            results = await embedding_service.embed_batch(texts)

            assert len(results) == 3
            assert results == embeddings
            mock_api.assert_called_once_with(texts)
            assert mock_cache.call_count == 3  # Each text should be cached

    @pytest.mark.asyncio
    async def test_batch_embedding_mixed_cache(self, embedding_service):
        """Test batch embedding with mixed cache hits/misses"""
        texts = ["Cached text", "New text", "Another cached"]
        cached_embeddings = {
            "Cached text": [0.1] * 1536,
            "Another cached": [0.3] * 1536
        }
        new_embedding = [0.2] * 1536

        async def mock_get_cache(text):
            return cached_embeddings.get(text)

        with patch.object(embedding_service, '_get_cached_embedding', side_effect=mock_get_cache), \
             patch.object(embedding_service, '_call_openai_api', return_value=[new_embedding]) as mock_api, \
             patch.object(embedding_service, '_cache_embedding') as mock_cache:

            results = await embedding_service.embed_batch(texts)

            assert len(results) == 3
            assert results[0] == cached_embeddings["Cached text"]
            assert results[1] == new_embedding
            assert results[2] == cached_embeddings["Another cached"]
            mock_api.assert_called_once_with(["New text"])  # Only uncached text
            mock_cache.assert_called_once()  # Only new embedding cached

    @pytest.mark.asyncio
    async def test_batch_size_limit(self, embedding_service):
        """Test batch size limit enforcement"""
        # Create a list larger than max batch size
        large_batch = ["text"] * 150  # Larger than default 100

        with pytest.raises(ValueError, match="Batch size 150 exceeds maximum 100"):
            await embedding_service.embed_batch(large_batch)

    @pytest.mark.asyncio
    async def test_empty_text_validation(self, embedding_service):
        """Test validation of empty or whitespace-only text"""
        # Mock database to prevent actual DB initialization
        mock_pool = AsyncMock()
        embedding_service.db_pool = mock_pool

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedding_service.embed_single("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedding_service.embed_single("   ")  # Whitespace only

        with pytest.raises(ValueError, match="Text at index 1 cannot be empty"):
            await embedding_service.embed_batch(["Valid text", "", "Another valid"])

    @pytest.mark.asyncio
    async def test_cache_stats(self, embedding_service):
        """Test cache statistics retrieval"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_acquire = AsyncMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire
        embedding_service.db_pool = mock_pool

        mock_stats = {
            'total_entries': 100,
            'unique_models': 2,
            'avg_dimensions': 1536.0,
            'oldest_entry': '2026-01-01T00:00:00',
            'newest_access': '2026-01-06T12:00:00',
            'accessed_today': 50
        }
        mock_conn.fetchrow.return_value = mock_stats

        stats = await embedding_service.get_cache_stats()

        assert stats['total_entries'] == 100
        assert stats['unique_models'] == 2
        assert stats['avg_dimensions'] == 1536.0
        assert stats['accessed_today'] == 50

    @pytest.mark.asyncio
    async def test_cache_cleanup(self, embedding_service):
        """Test cache cleanup functionality"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_acquire = AsyncMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire
        embedding_service.db_pool = mock_pool

        # Test cleanup with age limit
        await embedding_service.clear_cache(older_than_days=7)
        mock_conn.execute.assert_called()

        # Test complete cleanup
        await embedding_service.clear_cache()
        mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_service_lifecycle(self, embedding_service):
        """Test service initialization and cleanup"""
        # Mock pool
        mock_pool = AsyncMock()
        embedding_service.db_pool = mock_pool

        await embedding_service.close()

        mock_pool.close.assert_called_once()
        assert embedding_service.db_pool is None

    @pytest.mark.asyncio
    async def test_convenience_functions(self, mock_vault_file):
        """Test convenience functions for easy usage"""
        expected_embedding = [0.5] * 1536

        with patch('src.services.embedding_service.EmbeddingService') as mock_service_class:
            mock_service = AsyncMock()
            mock_service.embed_single.return_value = expected_embedding
            mock_service.embed_batch.return_value = [expected_embedding]
            mock_service_class.return_value = mock_service

            # Mock _init_db_pool
            mock_service._init_db_pool = AsyncMock()

            # Test single embedding function
            from src.services.embedding_service import embed_text
            result = await embed_text("Test text")
            assert result == expected_embedding
            mock_service.close.assert_called()

            # Test batch embedding function
            from src.services.embedding_service import embed_texts
            results = await embed_texts(["Test text"])
            assert results == [expected_embedding]

    def test_dimensions_validation(self, embedding_service):
        """Test that service is configured for correct dimensions"""
        assert embedding_service.dimensions == 1536
        assert embedding_service.model == "text-embedding-3-small"


# Integration test (requires actual database and API - run separately)
@pytest.mark.integration
class TestEmbeddingServiceIntegration:
    """Integration tests requiring actual database and API access"""

    @pytest.mark.asyncio
    async def test_real_embedding_generation(self):
        """Test with real OpenAI API (requires valid API key)"""
        # Only run if API key is available
        try:
            service = await create_embedding_service()

            text = "This is a test for real OpenAI embeddings"
            embedding = await service.embed_single(text)

            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)

            await service.close()

        except Exception as e:
            pytest.skip(f"Skipping integration test: {e}")

    @pytest.mark.asyncio
    async def test_real_cache_operations(self):
        """Test with real database connection"""
        try:
            service = await create_embedding_service()

            # Clear any existing cache
            await service.clear_cache()

            # Test cache miss and storage
            text = f"Integration test text {asyncio.get_event_loop().time()}"
            embedding = await service.embed_single(text)

            # Test cache hit
            cached_embedding = await service.embed_single(text)
            assert embedding == cached_embedding

            # Check cache stats
            stats = await service.get_cache_stats()
            assert stats['total_entries'] >= 1

            await service.close()

        except Exception as e:
            pytest.skip(f"Skipping integration test: {e}")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])