"""
Unified Memory Service for Echo Brain
Consolidates all memory/vector operations into a single service.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for memory service"""
    vector_db_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = "echo_memory"
    embedding_model: str = "all-MiniLM-L6-v2"

class UnifiedMemoryService:
    """
    Single source of truth for all memory operations.
    Combines functionality from multiple memory implementations.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all memory components"""
        # Try to import Qdrant for vector storage
        try:
            from qdrant_client import QdrantClient
            self.vector_client = QdrantClient(url=self.config.vector_db_url)
            logger.info(f"Connected to Qdrant at {self.config.vector_db_url}")
        except ImportError:
            logger.warning("Qdrant not available, using in-memory storage")
            self.vector_client = None
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.vector_client = None

        # Initialize embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")
        except ImportError:
            logger.warning("SentenceTransformers not available, using simple embeddings")
            self.embedding_model = None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def store_memory(self, text: str, metadata: Dict[str, Any]) -> str:
        """Store a memory with text and metadata"""
        import uuid
        memory_id = str(uuid.uuid4())

        if self.embedding_model and self.vector_client:
            try:
                # Generate embedding
                embedding = self.embedding_model.encode(text).tolist()

                # Store in Qdrant
                from qdrant_client.models import PointStruct
                self.vector_client.upsert(
                    collection_name=self.config.collection_name,
                    points=[PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload={"text": text, **metadata}
                    )]
                )
                logger.info(f"Stored memory {memory_id}")
            except Exception as e:
                logger.error(f"Failed to store memory: {e}")

        return memory_id

    def search_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for memories similar to query"""
        results = []

        if self.embedding_model and self.vector_client:
            try:
                # Generate query embedding
                query_embedding = self.embedding_model.encode(query).tolist()

                # Search in Qdrant
                search_results = self.vector_client.search(
                    collection_name=self.config.collection_name,
                    query_vector=query_embedding,
                    limit=limit
                )

                for hit in search_results:
                    results.append({
                        "id": hit.id,
                        "score": hit.score,
                        "text": hit.payload.get("text", ""),
                        "metadata": hit.payload
                    })

                logger.info(f"Found {len(results)} memories for query")
            except Exception as e:
                logger.error(f"Memory search failed: {e}")

        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        stats = {
            "status": "operational",
            "vector_db": "disconnected",
            "embeddings": "unavailable",
            "collection": self.config.collection_name
        }

        if self.vector_client:
            try:
                # Get collection info
                from qdrant_client.http import models
                collection_info = self.vector_client.get_collection(self.config.collection_name)
                stats["vector_db"] = "connected"
                stats["vector_count"] = collection_info.points_count
            except:
                pass

        if self.embedding_model:
            stats["embeddings"] = "available"

        return stats

# Global instance for easy import
memory_service = UnifiedMemoryService()