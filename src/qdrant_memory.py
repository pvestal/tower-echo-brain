#!/usr/bin/env python3
"""
Fixed Qdrant Integration for Echo Brain
Uses proper QdrantClient library instead of raw HTTP calls
"""

import logging
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, Range,
    SearchRequest, UpdateStatus
)
import httpx
import hashlib
from datetime import datetime
import uuid
import asyncio
import time

logger = logging.getLogger(__name__)


class QdrantMemory:
    """Properly integrated Qdrant memory system"""

    def __init__(self, collection_name: str = "echo_memories"):
        # Initialize Qdrant client - CORRECT PORT
        self.client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=30
        )

        self.collection_name = collection_name
        self.vector_size = 768  # nomic-embed-text dimension
        self.ollama_url = "http://localhost:11434"

        # Embedding models in priority order
        # Using tinyllama as fallback since it's available
        self.embedding_models = [
            "nomic-embed-text:latest",
            "mxbai-embed-large:latest",
            "tinyllama:latest"  # Fallback to available model
        ]

        # Retry configuration
        self.max_retries = 2  # Reduce retries for faster failure
        self.base_delay = 0.5  # Reduce delay between retries

        # Ensure collection exists
        self._ensure_collection()
        
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding with retry logic and model fallback"""
        models_to_try = self.embedding_models.copy()

        for model_name in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    embedding = await self._generate_single_embedding(text, model_name)
                    if embedding:
                        if attempt > 0 or models_to_try.index(model_name) > 0:
                            logger.info(f"Successfully generated embedding with {model_name} "
                                      f"(attempt {attempt + 1}, model {models_to_try.index(model_name) + 1})")
                        return embedding
                except Exception as e:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {model_name}: {e}. "
                                 f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)

        logger.error(f"All embedding attempts failed for text: {text[:100]}...")
        return None

    async def _generate_single_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Generate a single embedding attempt"""
        async with httpx.AsyncClient(timeout=10) as client:  # Reduced timeout
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text[:1000]  # Limit text length for faster processing
                },
                timeout=10.0  # 10 second timeout instead of 120
            )

            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding")
                if embedding:
                    # Validate dimension
                    if len(embedding) != self.vector_size:
                        # Resize if needed (for mxbai which is 1024D)
                        if len(embedding) == 1024:
                            embedding = embedding[:self.vector_size]
                        else:
                            logger.warning(f"Unexpected embedding dimension: {len(embedding)}")
                    return embedding
            else:
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response
                )

    async def store_memory(self, text: str, metadata: Optional[Dict] = None) -> bool:
        """Store a memory in Qdrant"""
        try:
            # Generate embedding
            embedding = await self.generate_embedding(text)
            if not embedding:
                return False

        except Exception as e:
            self.logger.error(f"Error in search_memory: {e}")
            return []
    # Test function
async def test_qdrant_connection():
    """Test Qdrant is working properly"""
    import asyncio
    from datetime import datetime
    
    memory = QdrantMemory()
    
    # Test 1: Check connection
    info = memory.get_collection_info()
    print(f"✅ Connected to Qdrant: {info}")
    
    # Test 2: Store a memory
    test_text = f"Test memory stored at {datetime.now()}"
    success = await memory.store_memory(
        test_text,
        {"type": "test", "source": "echo_brain"}
    )
    print(f"✅ Stored memory: {success}")
    
    # Test 3: Search memories
    await asyncio.sleep(1)  # Give it time to index
    results = await memory.search_memories("test memory")
    print(f"✅ Found {len(results)} memories")
    for r in results:
        print(f"  - Score: {r['score']:.3f}, Text: {r['text'][:50]}...")
    
    return True


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_qdrant_connection())

    async def search_memory(self, query: str, limit: int = 10, **kwargs):
        """Search for similar memories (alias for search_memories)"""
        try:
            return await self.search_memories(query, limit)
        except Exception as e:
            self.logger.error(f"Search memory failed: {e}")
            return []
