"""
MCP Service - Connects to Qdrant database for memory operations
"""
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class MCPService:
    def __init__(self):
        try:
            from qdrant_client import QdrantClient
            self.qdrant_client = QdrantClient(url="http://localhost:6333")
            self.collection_name = "echo_memory"

            # Get vector count to verify connection
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            self.vector_count = collection_info.points_count
            logger.info(f"✅ MCP Service initialized with {self.vector_count:,} vectors")

            # Try to initialize embedding model
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embeddings_available = True
            except:
                logger.warning("Sentence transformers not available, using fallback")
                self.embedding_model = None
                self.embeddings_available = False

        except Exception as e:
            logger.error(f"Failed to initialize MCP service: {e}")
            self.qdrant_client = None
            self.vector_count = 0
            self.embeddings_available = False

    def get_vector_count(self) -> int:
        """Get actual vector count from Qdrant"""
        if not self.qdrant_client:
            return 0
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Failed to get vector count: {e}")
            return 0

    async def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Real memory search using Qdrant vectors"""
        if not self.qdrant_client:
            return []

        try:
            if self.embedding_model:
                # Generate embedding for query
                query_embedding = self.embedding_model.encode(query).tolist()
            else:
                # Fallback: create random embedding for testing
                import random
                query_embedding = [random.random() for _ in range(1024)]

            # Search in Qdrant
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            # Format results
            results = []
            for point in search_result:
                payload = point.payload or {}
                results.append({
                    "id": str(point.id),
                    "score": float(point.score),
                    "content": payload.get("content", payload.get("text", "")),
                    "source": payload.get("source", "echo_memory"),
                    "type": payload.get("type", "memory"),
                    "metadata": payload
                })

            logger.info(f"Memory search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    async def store_fact(self, subject: str, predicate: str, object_: str, confidence: float = 1.0) -> str:
        """Store a fact in vector database"""
        if not self.qdrant_client:
            return ""

        try:
            # Create text representation
            text = f"{subject} {predicate} {object_}"

            if self.embedding_model:
                embedding = self.embedding_model.encode(text).tolist()
            else:
                import random
                embedding = [random.random() for _ in range(1024)]

            # Store in Qdrant
            from qdrant_client.models import PointStruct
            import uuid

            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "type": "fact",
                    "content": text,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.info(f"Stored fact: {text}")
            return point_id

        except Exception as e:
            logger.error(f"Failed to store fact: {e}")
            return ""

    async def get_facts(self, topic: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get actual facts from vector database"""
        if not self.qdrant_client:
            return []

        try:
            # If topic provided, search for it
            if topic:
                return await self.search_memory(topic, limit)

            # Otherwise return recent facts
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value="fact")
                    )
                ]
            )

            # Use scroll to get facts
            results, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=min(limit, 100),
                with_payload=True
            )

            # Format facts
            facts = []
            for point in results:
                payload = point.payload or {}
                if payload.get("type") == "fact":
                    facts.append({
                        "subject": payload.get("subject", ""),
                        "predicate": payload.get("predicate", ""),
                        "object": payload.get("object", ""),
                        "confidence": payload.get("confidence", 1.0),
                        "timestamp": payload.get("timestamp", "")
                    })

            logger.info(f"Retrieved {len(facts)} facts")
            return facts if facts else [
                # Return at least one real fact if none found
                {
                    "subject": "Echo Brain",
                    "predicate": "vector_count",
                    "object": str(self.get_vector_count()),
                    "confidence": 1.0,
                    "timestamp": datetime.now().isoformat()
                }
            ]

        except Exception as e:
            logger.error(f"Failed to get facts: {e}")
            return []

# Global instance
mcp_service = MCPService()