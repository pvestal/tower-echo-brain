#!/usr/bin/env python3
"""
OpenAI Embedding Service for Echo Brain
Uses OpenAI text-embedding-3-small (1536 dimensions) with caching and batch support
Created: January 6, 2026
"""

import json
import hashlib
import asyncio
import aiohttp
import asyncpg
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    OpenAI Embedding Service with PostgreSQL caching and batch support

    Features:
    - OpenAI text-embedding-3-small (1536 dimensions)
    - Async API calls with retry logic
    - PostgreSQL embedding cache
    - Batch support (max 100 texts per API call)
    - Automatic cache expiration (30 days)
    """

    def __init__(self, vault_path: str = "/home/patrick/.tower_credentials/vault.json"):
        self.vault_path = vault_path
        self.api_key = None
        self.model = "text-embedding-3-small"
        self.dimensions = 1536
        self.max_batch_size = 100
        self.cache_expiry_days = 30
        self.base_url = "https://api.openai.com/v1"
        self.db_pool = None
        self._load_credentials()

    def _load_credentials(self):
        """Load OpenAI API key and PostgreSQL credentials from vault"""
        try:
            with open(self.vault_path, 'r') as f:
                vault = json.load(f)

            # OpenAI credentials
            openai_config = vault.get('openai', {})
            self.api_key = openai_config.get('api_key')
            if not self.api_key:
                raise ValueError("OpenAI API key not found in vault")

            # PostgreSQL credentials
            self.pg_config = vault.get('postgresql', {})
            if not self.pg_config:
                raise ValueError("PostgreSQL configuration not found in vault")

            logger.info(f"Loaded credentials from {self.vault_path}")

        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            raise

    async def _init_db_pool(self):
        """Initialize PostgreSQL connection pool"""
        if self.db_pool is None:
            try:
                self.db_pool = await asyncpg.create_pool(
                    host=self.pg_config['host'],
                    database=self.pg_config['database'],
                    user=self.pg_config['user'],
                    password=self.pg_config['password'],
                    port=self.pg_config['port'],
                    min_size=2,
                    max_size=10
                )
                await self._create_cache_table()
                logger.info("PostgreSQL connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise

    async def _create_cache_table(self):
        """Create embedding cache table if it doesn't exist"""
        sql = """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            id SERIAL PRIMARY KEY,
            text_hash VARCHAR(64) UNIQUE NOT NULL,
            text TEXT NOT NULL,
            model VARCHAR(100) NOT NULL,
            embedding FLOAT8[] NOT NULL,
            dimensions INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_embedding_cache_hash ON embedding_cache(text_hash);
        CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache(model);
        CREATE INDEX IF NOT EXISTS idx_embedding_cache_accessed ON embedding_cache(accessed_at);
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(sql)

        logger.info("Embedding cache table created/verified")

    def _get_text_hash(self, text: str) -> str:
        """Generate SHA-256 hash for text"""
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    async def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Retrieve embedding from cache"""
        if not self.db_pool:
            await self._init_db_pool()

        text_hash = self._get_text_hash(text)

        # Clean old cache entries
        cutoff_date = datetime.now() - timedelta(days=self.cache_expiry_days)

        async with self.db_pool.acquire() as conn:
            # Update access time and get embedding
            result = await conn.fetchrow(
                """
                UPDATE embedding_cache
                SET accessed_at = CURRENT_TIMESTAMP
                WHERE text_hash = $1 AND model = $2 AND created_at > $3
                RETURNING embedding
                """,
                text_hash, self.model, cutoff_date
            )

            if result:
                logger.debug(f"Cache hit for text hash: {text_hash[:8]}...")
                return list(result['embedding'])

            return None

    async def _cache_embedding(self, text: str, embedding: List[float]):
        """Store embedding in cache"""
        if not self.db_pool:
            await self._init_db_pool()

        text_hash = self._get_text_hash(text)

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO embedding_cache (text_hash, text, model, embedding, dimensions)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (text_hash) DO UPDATE SET
                    accessed_at = CURRENT_TIMESTAMP,
                    embedding = EXCLUDED.embedding
                """,
                text_hash, text, self.model, embedding, self.dimensions
            )

        logger.debug(f"Cached embedding for text hash: {text_hash[:8]}...")

    async def _cleanup_old_cache(self):
        """Remove old cache entries"""
        if not self.db_pool:
            return

        cutoff_date = datetime.now() - timedelta(days=self.cache_expiry_days)

        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM embedding_cache WHERE accessed_at < $1",
                cutoff_date
            )

        logger.info(f"Cleaned up old cache entries: {result}")

    async def _call_openai_api(self, texts: List[str], retry_count: int = 3) -> List[List[float]]:
        """Call OpenAI embeddings API with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "input": texts,
            "model": self.model,
            "dimensions": self.dimensions
        }

        for attempt in range(retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/embeddings",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:

                        if response.status == 200:
                            result = await response.json()
                            embeddings = []

                            # Sort by index to maintain order
                            data_items = sorted(result['data'], key=lambda x: x['index'])
                            for item in data_items:
                                embeddings.append(item['embedding'])

                            logger.info(f"Generated {len(embeddings)} embeddings via OpenAI API")
                            return embeddings

                        elif response.status == 429:  # Rate limit
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                            await asyncio.sleep(wait_time)

                        else:
                            error_text = await response.text()
                            logger.error(f"OpenAI API error {response.status}: {error_text}")
                            if attempt == retry_count - 1:
                                raise Exception(f"OpenAI API error {response.status}: {error_text}")

            except Exception as e:
                logger.error(f"API call attempt {attempt + 1} failed: {e}")
                if attempt == retry_count - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise Exception("Failed to get embeddings after all retries")

    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            1536-dimensional embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Check cache first
        cached = await self._get_cached_embedding(text)
        if cached:
            return cached

        # Get from OpenAI API
        embeddings = await self._call_openai_api([text])
        embedding = embeddings[0]

        # Cache the result
        await self._cache_embedding(text, embedding)

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed (max 100)

        Returns:
            List of 1536-dimensional embedding vectors
        """
        if not texts:
            return []

        if len(texts) > self.max_batch_size:
            raise ValueError(f"Batch size {len(texts)} exceeds maximum {self.max_batch_size}")

        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")

            cached = await self._get_cached_embedding(text)
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Get uncached embeddings from API
        if uncached_texts:
            new_embeddings = await self._call_openai_api(uncached_texts)

            # Cache and insert new embeddings
            for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                await self._cache_embedding(text, embedding)
                embeddings[uncached_indices[i]] = embedding

        return embeddings

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics"""
        if not self.db_pool:
            await self._init_db_pool()

        async with self.db_pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT model) as unique_models,
                    AVG(array_length(embedding, 1)) as avg_dimensions,
                    MIN(created_at) as oldest_entry,
                    MAX(accessed_at) as newest_access,
                    COUNT(*) FILTER (WHERE accessed_at > CURRENT_TIMESTAMP - INTERVAL '24 hours') as accessed_today
                FROM embedding_cache
            """)

            return {
                "total_entries": stats['total_entries'],
                "unique_models": stats['unique_models'],
                "avg_dimensions": float(stats['avg_dimensions'] or 0),
                "oldest_entry": stats['oldest_entry'].isoformat() if stats['oldest_entry'] else None,
                "newest_access": stats['newest_access'].isoformat() if stats['newest_access'] else None,
                "accessed_today": stats['accessed_today'],
                "cache_expiry_days": self.cache_expiry_days
            }

    async def clear_cache(self, older_than_days: Optional[int] = None):
        """Clear embedding cache"""
        if not self.db_pool:
            await self._init_db_pool()

        async with self.db_pool.acquire() as conn:
            if older_than_days:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)
                result = await conn.execute(
                    "DELETE FROM embedding_cache WHERE created_at < $1",
                    cutoff_date
                )
            else:
                result = await conn.execute("DELETE FROM embedding_cache")

        logger.info(f"Cleared cache entries: {result}")

    async def close(self):
        """Close database connections"""
        if self.db_pool:
            await self.db_pool.close()
            self.db_pool = None
            logger.info("Database pool closed")

    def __del__(self):
        """Cleanup on destruction"""
        if self.db_pool:
            # Note: In a real async context, you should call close() explicitly
            logger.warning("EmbeddingService destroyed without calling close()")


# Convenience functions for backward compatibility and easier usage
async def create_embedding_service() -> EmbeddingService:
    """Create and initialize embedding service"""
    service = EmbeddingService()
    await service._init_db_pool()
    return service


async def embed_text(text: str) -> List[float]:
    """Quick function to embed a single text"""
    service = await create_embedding_service()
    try:
        return await service.embed_single(text)
    finally:
        await service.close()


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Quick function to embed multiple texts"""
    service = await create_embedding_service()
    try:
        return await service.embed_batch(texts)
    finally:
        await service.close()


if __name__ == "__main__":
    async def test_embedding_service():
        """Test the embedding service"""
        print("Testing OpenAI Embedding Service...")

        service = await create_embedding_service()

        try:
            # Test single embedding
            print("\n=== Testing Single Embedding ===")
            text = "Hello, this is a test for OpenAI embeddings"
            embedding = await service.embed_single(text)
            print(f"Text: {text}")
            print(f"Embedding dimensions: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")

            # Test batch embedding
            print("\n=== Testing Batch Embedding ===")
            texts = [
                "Artificial intelligence and machine learning",
                "Vector databases and semantic search",
                "OpenAI embeddings and natural language processing"
            ]
            embeddings = await service.embed_batch(texts)
            print(f"Batch size: {len(texts)}")
            print(f"Results: {len(embeddings)} embeddings")
            for i, (text, emb) in enumerate(zip(texts, embeddings)):
                print(f"  {i+1}. {text[:50]}... -> {len(emb)}D")

            # Test cache
            print("\n=== Testing Cache ===")
            cached_embedding = await service.embed_single(text)  # Should hit cache
            print(f"Cache hit: {embedding == cached_embedding}")

            # Get cache stats
            print("\n=== Cache Statistics ===")
            stats = await service.get_cache_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")

        finally:
            await service.close()

        print("\n=== Test Complete ===")

    # Run the test
    asyncio.run(test_embedding_service())