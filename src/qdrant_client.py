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
        """Generate embedding using Ollama"""
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": "nomic-embed-text:latest",
                        "prompt": text
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("embedding")
                else:
                    logger.error(f"Embedding generation failed: {response.status_code}")
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
        return None

    async def store_memory(self, text: str, metadata: Optional[Dict] = None) -> bool:
        """Store a memory in Qdrant"""
        try:
            # Generate embedding
            embedding = await self.generate_embedding(text)
            if not embedding:
                return False
            
            # Create unique ID
            memory_id = str(uuid.uuid4())
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadata["text"] = text[:500]  # Store first 500 chars for reference
            metadata["timestamp"] = str(datetime.now())
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload=metadata
                    )
                ]
            )
            
            logger.info(f"Stored memory {memory_id} in Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    async def search_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant memories"""
        try:
            # Generate query embedding
            embedding = await self.generate_embedding(query)
            if not embedding:
                return []
            
            # Search in Qdrant
            results = self.client.search_points(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit
            )
            
            # Format results
            memories = []
            for result in results:
                memory = {
                    "id": str(result.id),
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload
                }
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    def get_collection_info(self) -> Dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


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
