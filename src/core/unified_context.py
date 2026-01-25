"""
Unified Context Provider for Echo Brain
Single source of truth for all context retrieval across agents
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncpg
from qdrant_client import QdrantClient
import httpx

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.qdrant_memory import QdrantMemory

logger = logging.getLogger(__name__)


class UnifiedContextProvider:
    """Single source of truth for all context retrieval"""

    def __init__(self):
        self.qdrant = None
        self.pg_conn = None
        self.ollama_url = "http://localhost:11434"
        self.embedding_model = "nomic-embed-text:latest"
        self._initialized = False
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

    async def initialize(self):
        """Initialize all connections"""
        if self._initialized:
            return

        try:
            # Initialize Qdrant
            self.qdrant = QdrantMemory(collection_name="echo_memory_768")

            # Initialize PostgreSQL connection
            self.pg_conn = await asyncpg.connect(
                host="localhost",
                port=5432,
                database="echo_brain",
                user="patrick",
                password=os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
            )

            self._initialized = True
            logger.info("UnifiedContextProvider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize UnifiedContextProvider: {e}")
            raise

    async def get_context(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Returns combined context from all sources:
        - memories: Relevant vectors from Qdrant (echo_memory_768)
        - facts: Relevant facts from PostgreSQL
        - recent_conversations: Last 24h of conversation summaries
        """
        await self.initialize()

        # Check cache first
        cache_key = f"context:{query}:{limit}"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                logger.debug(f"Returning cached context for query: {query}")
                return cached_result

        try:
            # Gather context from all sources in parallel
            memories_task = self.search_memories(query, limit)
            facts_task = self.get_facts(query)
            recent_task = self.get_recent_context(24)

            memories, facts, recent_conversations = await asyncio.gather(
                memories_task, facts_task, recent_task
            )

            result = {
                "query": query,
                "memories": memories,
                "facts": facts,
                "recent_conversations": recent_conversations,
                "context_summary": self._summarize_context(memories, facts, recent_conversations)
            }

            # Cache the result
            self._cache[cache_key] = (datetime.now(), result)

            logger.info(f"Retrieved context for query '{query}': {len(memories)} memories, {len(facts)} facts, {len(recent_conversations)} recent conversations")
            return result

        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            # Return partial context on error
            return {
                "query": query,
                "memories": [],
                "facts": [],
                "recent_conversations": [],
                "error": str(e)
            }

    async def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Semantic search against Qdrant"""
        try:
            results = await self.qdrant.search_memories(query, limit=limit)
            return [
                {
                    "text": r.get("text", ""),
                    "score": r.get("score", 0),
                    "metadata": r.get("metadata", {}),
                    "source": "qdrant"
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    async def get_facts(self, topic: str) -> List[Dict[str, Any]]:
        """Get facts related to topic"""
        await self.initialize()

        try:
            # Search facts table for relevant information
            facts = await self.pg_conn.fetch("""
                SELECT subject, predicate, object, confidence, created_at
                FROM facts
                WHERE subject ILIKE $1 OR predicate ILIKE $1 OR object ILIKE $1
                ORDER BY confidence DESC, created_at DESC
                LIMIT 20
            """, f"%{topic}%")

            return [
                {
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "confidence": float(row.get("confidence", 1.0)),
                    "source": "facts_db",  # Default source since column doesn't exist
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None
                }
                for row in facts
            ]
        except Exception as e:
            logger.error(f"Failed to get facts: {e}")
            return []

    async def get_recent_context(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        await self.initialize()

        try:
            cutoff = datetime.now() - timedelta(hours=hours)

            # Get recent conversations
            conversations = await self.pg_conn.fetch("""
                SELECT summary, key_topics, timestamp, metadata
                FROM echo_conversations
                WHERE timestamp > $1
                ORDER BY timestamp DESC
                LIMIT 10
            """, cutoff)

            return [
                {
                    "summary": row["summary"],
                    "key_topics": row.get("key_topics", []),
                    "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                    "metadata": row.get("metadata", {}),
                    "source": "recent_conversations"
                }
                for row in conversations
            ]
        except Exception as e:
            logger.error(f"Failed to get recent context: {e}")
            return []

    async def store_fact(self, subject: str, predicate: str, obj: str,
                        confidence: float = 1.0, source: str = "agent") -> bool:
        """Store a new fact in the database"""
        await self.initialize()

        try:
            await self.pg_conn.execute("""
                INSERT INTO facts (subject, predicate, object, confidence, source, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (subject, predicate, object)
                DO UPDATE SET confidence = $4, source = $5, created_at = NOW()
            """, subject, predicate, obj, confidence, source)

            logger.info(f"Stored fact: {subject} - {predicate} - {obj}")
            return True
        except Exception as e:
            logger.error(f"Failed to store fact: {e}")
            return False

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama"""
        try:
            # Truncate text if too long
            if len(text) > 1500:
                text = text[:1500]

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("embedding", [])
                else:
                    logger.error(f"Ollama embedding failed: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def _summarize_context(self, memories: List[Dict], facts: List[Dict],
                          recent: List[Dict]) -> str:
        """Create a summary of the retrieved context"""
        summary_parts = []

        if memories:
            summary_parts.append(f"Found {len(memories)} relevant memories")

        if facts:
            summary_parts.append(f"Found {len(facts)} related facts")

        if recent:
            summary_parts.append(f"Found {len(recent)} recent conversations")

        return ". ".join(summary_parts) if summary_parts else "No context found"

    async def cleanup(self):
        """Cleanup connections"""
        if self.pg_conn:
            await self.pg_conn.close()
        self._initialized = False
        logger.info("UnifiedContextProvider cleaned up")


# Singleton instance
_context_provider = None

def get_context_provider() -> UnifiedContextProvider:
    """Get or create the singleton context provider"""
    global _context_provider
    if _context_provider is None:
        _context_provider = UnifiedContextProvider()
    return _context_provider