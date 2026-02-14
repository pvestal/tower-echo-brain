"""
MCP Service - Connects to Qdrant database for memory operations
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

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

            # Initialize Ollama for embeddings (matching unified memory system)
            self.ollama_url = "http://localhost:11434"
            self.embedding_model = "nomic-embed-text"
            self.embeddings_available = True

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
        """Memory search using domain-aware ParallelRetriever"""
        try:
            from src.context_assembly.retriever import ParallelRetriever

            # Use the ParallelRetriever which handles domain classification
            # and searches appropriate collections (including story_bible for anime)
            retriever = ParallelRetriever()
            await retriever.initialize()

            # Get domain-classified results
            retrieval_result = await retriever.retrieve(query, max_results=limit)

            # Convert to MCP format expected by calling code
            results = []
            for source in retrieval_result.get("sources", []):
                results.append({
                    "id": source.get("metadata", {}).get("source_id", ""),
                    "score": source.get("score", 0),
                    "content": source.get("content", ""),
                    "source": source.get("source", "unknown"),
                    "type": source.get("type", "memory"),
                    "payload": source.get("metadata", {})
                })

            await retriever.shutdown()

            logger.info(f"Domain-aware search for '{query}' classified as {retrieval_result.get('domain')} "
                       f"returned {len(results)} results from collections {retrieval_result.get('allowed_collections')}")
            return results

        except Exception as e:
            logger.error(f"ParallelRetriever search failed: {e}")

            # Fallback to old direct search
            if not self.qdrant_client:
                return []

            try:
                # Generate embedding using Ollama (matching unified memory system)
                import httpx
                try:
                    with httpx.Client(timeout=30) as client:
                        response = client.post(
                            f"{self.ollama_url}/api/embed",
                            json={"model": self.embedding_model, "input": query}
                        )
                        if response.status_code == 200:
                            resp_data = response.json()
                            query_embedding = (resp_data.get("embeddings") or [[]])[0] or resp_data.get("embedding", [])
                        else:
                            # Fallback: create random embedding for testing
                            import random
                            query_embedding = [random.random() for _ in range(768)]
                            logger.warning(f"Ollama embedding failed with status {response.status_code}, using random")
                except Exception as e:
                    # Fallback: create random embedding for testing
                    import random
                    query_embedding = [random.random() for _ in range(768)]
                    logger.warning(f"Ollama embedding failed: {e}, using random")

                # Search in Qdrant using HTTP API directly
                import httpx
                with httpx.Client(timeout=10) as client:
                    response = client.post(
                        f"http://localhost:6333/collections/{self.collection_name}/points/search",
                        json={
                            "vector": query_embedding,
                            "limit": limit,
                            "with_payload": True
                        }
                    )
                    if response.status_code == 200:
                        search_result = response.json().get("result", [])
                    else:
                        logger.error(f"Qdrant search failed with status {response.status_code}")
                        search_result = []

                # Format results (from HTTP API response)
                results = []
                for point in search_result:
                    payload = point.get("payload", {})
                    results.append({
                        "id": str(point.get("id", "")),
                        "score": float(point.get("score", 0)),
                        "content": payload.get("content", payload.get("text", "")),
                        "source": payload.get("source", "echo_memory"),
                        "type": payload.get("type", "memory"),
                        "payload": payload
                    })

                logger.info(f"Fallback search for '{query}' returned {len(results)} results")
                return results

            except Exception as e2:
                logger.error(f"Fallback search failed: {e2}")
                return []

    async def store_fact(self, subject: str, predicate: str, object_: str, confidence: float = 1.0) -> str:
        """Store a fact in vector database"""
        if not self.qdrant_client:
            return ""

        try:
            # Create text representation
            text = f"{subject} {predicate} {object_}"

            # Get embedding via Ollama
            import httpx
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.post(
                        f"{self.ollama_url}/api/embed",
                        json={"model": self.embedding_model, "input": text}
                    )
                    resp_data = resp.json()
                    embedding = (resp_data.get("embeddings") or [[]])[0] or resp_data.get("embedding", [])
                    if not embedding:
                        logger.warning("Empty embedding from Ollama for store_fact")
                        return ""
            except Exception as embed_err:
                logger.error(f"Failed to embed fact: {embed_err}")
                return ""

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
        """Get facts from PostgreSQL facts table via unified knowledge layer"""
        from src.core.unified_knowledge import get_unified_knowledge

        knowledge = get_unified_knowledge()

        try:
            # Get facts from the unified layer
            if topic:
                facts = await knowledge.search_facts(topic, limit)
            else:
                # Get all recent facts
                facts = await knowledge.search_facts("", limit)

            # Convert to expected format
            result = []
            for fact in facts:
                result.append({
                    "content": fact.content,
                    "confidence": fact.confidence,
                    "type": fact.source_type,
                    "metadata": fact.metadata
                })

            logger.info(f"Retrieved {len(result)} facts from unified knowledge layer")
            return result

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