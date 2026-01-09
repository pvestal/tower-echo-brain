#!/usr/bin/env python3
"""
Resilient Qdrant Integration for Echo Brain
Enhanced with circuit breaker patterns for improved reliability
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

from .resilience.service_breakers import get_qdrant_breaker
from .resilience.fallback_handlers import get_fallback_handler

logger = logging.getLogger(__name__)


class ResilientQdrantMemory:
    """Circuit breaker protected Qdrant memory system"""

    def __init__(self, collection_name: str = "echo_memories"):
        self.collection_name = collection_name
        self.vector_size = 768  # nomic-embed-text dimension
        self.ollama_url = "http://localhost:11434"

        # Initialize Qdrant client - will be protected by circuit breaker
        self._client = None
        self._connection_params = {
            "host": "localhost",
            "port": 6333,
            "timeout": 30
        }

        # Get circuit breaker and fallback handler
        self.circuit_breaker = get_qdrant_breaker()
        self.fallback_handler = get_fallback_handler()

        logger.info(f"Initialized resilient Qdrant memory system with circuit breaker protection")

    async def _get_client(self) -> QdrantClient:
        """Get Qdrant client with lazy initialization"""
        if self._client is None:
            self._client = QdrantClient(**self._connection_params)
        return self._client

    async def _ensure_collection_internal(self):
        """Internal method to ensure collection exists (protected by circuit breaker)"""
        client = await self._get_client()

        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
        else:
            logger.debug(f"Qdrant collection already exists: {self.collection_name}")

    async def ensure_collection(self):
        """Ensure collection exists with circuit breaker protection"""
        try:
            # Define fallback for collection creation
            async def collection_fallback(*args, **kwargs):
                return await self.fallback_handler.execute_fallback(
                    "qdrant", "ensure_collection",
                    collection_name=self.collection_name
                )

            # Execute with circuit breaker protection
            await self.circuit_breaker.call(
                self._ensure_collection_internal,
                fallback=collection_fallback
            )

        except Exception as e:
            logger.error(f"Failed to ensure collection even with fallback: {e}")
            raise

    async def _embed_text_internal(self, text: str) -> List[float]:
        """Internal method to embed text using Ollama (protected by circuit breaker)"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text
                },
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("embedding", [])
            else:
                raise httpx.HTTPStatusError(
                    message=f"Ollama embedding failed with {response.status_code}",
                    request=response.request,
                    response=response
                )

    async def embed_text(self, text: str) -> List[float]:
        """Embed text with circuit breaker protection"""
        try:
            # Define fallback for text embedding
            async def embedding_fallback(*args, **kwargs):
                # Return empty embedding as fallback
                logger.warning("Ollama unavailable - returning empty embedding")
                return [0.0] * self.vector_size

            # Execute with circuit breaker protection
            embedding = await self.circuit_breaker.call(
                self._embed_text_internal,
                text,
                fallback=embedding_fallback
            )

            return embedding

        except Exception as e:
            logger.error(f"Failed to embed text even with fallback: {e}")
            # Return empty embedding as last resort
            return [0.0] * self.vector_size

    async def _upsert_memory_internal(
        self,
        content: str,
        metadata: Dict[str, Any],
        memory_id: Optional[str] = None
    ) -> str:
        """Internal method to upsert memory (protected by circuit breaker)"""
        client = await self._get_client()

        # Generate embedding
        embedding = await self.embed_text(content)

        # Generate memory ID if not provided
        if memory_id is None:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            memory_id = f"memory_{content_hash}_{int(datetime.utcnow().timestamp())}"

        # Create point
        point = PointStruct(
            id=memory_id,
            vector=embedding,
            payload={
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
                **metadata
            }
        )

        # Upsert to collection
        result = client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        if result.status == UpdateStatus.COMPLETED:
            logger.debug(f"Successfully upserted memory: {memory_id}")
            return memory_id
        else:
            raise Exception(f"Failed to upsert memory: {result}")

    async def upsert_memory(
        self,
        content: str,
        metadata: Dict[str, Any],
        memory_id: Optional[str] = None
    ) -> str:
        """Store or update memory with circuit breaker protection"""
        try:
            # Define fallback for memory upsert
            async def upsert_fallback(*args, **kwargs):
                fallback_result = await self.fallback_handler.execute_fallback(
                    "qdrant", "upsert",
                    content=content,
                    metadata=metadata,
                    memory_id=memory_id
                )
                # Return a fallback memory ID
                return f"fallback_{int(datetime.utcnow().timestamp())}"

            # Execute with circuit breaker protection
            memory_id = await self.circuit_breaker.call(
                self._upsert_memory_internal,
                content,
                metadata,
                memory_id,
                fallback=upsert_fallback
            )

            return memory_id

        except Exception as e:
            logger.error(f"Failed to upsert memory even with fallback: {e}")
            # Return fallback ID as last resort
            return f"error_{int(datetime.utcnow().timestamp())}"

    async def _search_memories_internal(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Internal method to search memories (protected by circuit breaker)"""
        client = await self._get_client()

        # Generate query embedding
        query_embedding = await self.embed_text(query)

        # Build filter if provided
        search_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match={"value": value})
                )
            search_filter = Filter(must=conditions)

        # Search
        results = client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=search_filter
        )

        # Convert results
        memories = []
        for result in results:
            memories.append({
                "id": result.id,
                "content": result.payload.get("content", ""),
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k != "content"},
                "timestamp": result.payload.get("timestamp")
            })

        return memories

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search memories with circuit breaker protection"""
        try:
            # Define fallback for memory search
            async def search_fallback(*args, **kwargs):
                fallback_result = await self.fallback_handler.execute_fallback(
                    "qdrant", "search",
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold,
                    filters=filters
                )
                return []  # Return empty results as fallback

            # Execute with circuit breaker protection
            memories = await self.circuit_breaker.call(
                self._search_memories_internal,
                query,
                limit,
                score_threshold,
                filters,
                fallback=search_fallback
            )

            return memories

        except Exception as e:
            logger.error(f"Failed to search memories even with fallback: {e}")
            # Return empty results as last resort
            return []

    async def _get_memory_by_id_internal(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Internal method to get memory by ID (protected by circuit breaker)"""
        client = await self._get_client()

        # Retrieve point by ID
        result = client.retrieve(
            collection_name=self.collection_name,
            ids=[memory_id],
            with_payload=True
        )

        if result:
            point = result[0]
            return {
                "id": point.id,
                "content": point.payload.get("content", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "content"},
                "timestamp": point.payload.get("timestamp")
            }

        return None

    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get memory by ID with circuit breaker protection"""
        try:
            # Define fallback for memory retrieval
            async def get_fallback(*args, **kwargs):
                await self.fallback_handler.execute_fallback(
                    "qdrant", "get_by_id",
                    memory_id=memory_id
                )
                return None  # Return None as fallback

            # Execute with circuit breaker protection
            memory = await self.circuit_breaker.call(
                self._get_memory_by_id_internal,
                memory_id,
                fallback=get_fallback
            )

            return memory

        except Exception as e:
            logger.error(f"Failed to get memory by ID even with fallback: {e}")
            return None

    async def _delete_memory_internal(self, memory_id: str) -> bool:
        """Internal method to delete memory (protected by circuit breaker)"""
        client = await self._get_client()

        result = client.delete(
            collection_name=self.collection_name,
            points_selector=[memory_id]
        )

        return result.status == UpdateStatus.COMPLETED

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory with circuit breaker protection"""
        try:
            # Define fallback for memory deletion
            async def delete_fallback(*args, **kwargs):
                await self.fallback_handler.execute_fallback(
                    "qdrant", "delete",
                    memory_id=memory_id
                )
                return False  # Return False as fallback

            # Execute with circuit breaker protection
            success = await self.circuit_breaker.call(
                self._delete_memory_internal,
                memory_id,
                fallback=delete_fallback
            )

            return success

        except Exception as e:
            logger.error(f"Failed to delete memory even with fallback: {e}")
            return False

    async def _get_collection_info_internal(self) -> Dict[str, Any]:
        """Internal method to get collection info (protected by circuit breaker)"""
        client = await self._get_client()

        collection = client.get_collection(self.collection_name)

        return {
            "name": collection.name,
            "vectors_count": collection.vectors_count or 0,
            "indexed_vectors_count": collection.indexed_vectors_count or 0,
            "points_count": collection.points_count or 0,
            "status": collection.status,
            "vector_size": self.vector_size
        }

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information with circuit breaker protection"""
        try:
            # Define fallback for collection info
            async def info_fallback(*args, **kwargs):
                fallback_result = await self.fallback_handler.execute_fallback(
                    "qdrant", "get_collection_info",
                    collection_name=self.collection_name
                )
                return {
                    "name": self.collection_name,
                    "vectors_count": 0,
                    "status": "unavailable",
                    "fallback": True
                }

            # Execute with circuit breaker protection
            info = await self.circuit_breaker.call(
                self._get_collection_info_internal,
                fallback=info_fallback
            )

            return info

        except Exception as e:
            logger.error(f"Failed to get collection info even with fallback: {e}")
            return {
                "name": self.collection_name,
                "vectors_count": 0,
                "status": "error",
                "error": str(e)
            }

    async def check_service_health(self) -> Dict[str, Any]:
        """Check Qdrant service health"""
        try:
            async def health_check():
                client = await self._get_client()
                # Try to get collections as health check
                collections = client.get_collections()
                return {
                    "success": True,
                    "status": "healthy",
                    "collections_count": len(collections.collections),
                    "connection_params": self._connection_params
                }

            # Use circuit breaker for health check
            result = await self.circuit_breaker.call(health_check)
            return result

        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return {
                "success": False,
                "status": "unreachable",
                "error": str(e),
                "connection_params": self._connection_params
            }

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return self.circuit_breaker.get_metrics()


async def test_resilient_qdrant_memory():
    """Test resilient Qdrant memory system"""

    print("=" * 60)
    print("RESILIENT QDRANT MEMORY TEST")
    print("=" * 60)

    memory = ResilientQdrantMemory()

    # Check circuit breaker status
    cb_status = memory.get_circuit_breaker_status()
    print(f"\nCircuit Breaker Status: {cb_status['state']}")
    print(f"Total Requests: {cb_status['metrics']['total_requests']}")
    print(f"Failure Rate: {cb_status['metrics']['failure_rate']}%")

    # Test service health
    print("\nTesting service health...")
    health = await memory.check_service_health()
    if health['success']:
        print(f"✅ Qdrant is healthy: {health['status']}")
        print(f"   Collections: {health['collections_count']}")
    else:
        print(f"❌ Qdrant health check failed: {health['error']}")

    # Ensure collection exists
    print("\nEnsuring collection exists...")
    try:
        await memory.ensure_collection()
        print("✅ Collection ensured")
    except Exception as e:
        print(f"❌ Failed to ensure collection: {e}")

    # Test memory operations
    print("\nTesting memory operations...")

    # Store a memory
    test_content = "This is a test memory for resilient Qdrant integration"
    test_metadata = {"type": "test", "importance": "high"}

    try:
        memory_id = await memory.upsert_memory(test_content, test_metadata)
        print(f"✅ Stored memory: {memory_id}")
    except Exception as e:
        print(f"❌ Failed to store memory: {e}")
        return False

    # Search memories
    try:
        results = await memory.search_memories("test memory resilient", limit=5)
        print(f"✅ Search returned {len(results)} results")
        for result in results:
            print(f"   - {result['id']}: {result['content'][:50]}... (score: {result['score']:.3f})")
    except Exception as e:
        print(f"❌ Failed to search memories: {e}")

    # Get collection info
    try:
        info = await memory.get_collection_info()
        print(f"✅ Collection info: {info['vectors_count']} vectors, status: {info['status']}")
    except Exception as e:
        print(f"❌ Failed to get collection info: {e}")

    # Final circuit breaker status
    final_cb_status = memory.get_circuit_breaker_status()
    print(f"\nFinal Circuit Breaker Status:")
    print(f"   State: {final_cb_status['state']}")
    print(f"   Total Requests: {final_cb_status['metrics']['total_requests']}")
    print(f"   Success Rate: {final_cb_status['metrics']['success_rate']}%")
    print(f"   Failure Rate: {final_cb_status['metrics']['failure_rate']}%")

    print("\n" + "=" * 60)
    return True


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_resilient_qdrant_memory())