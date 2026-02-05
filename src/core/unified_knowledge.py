"""
Unified Knowledge Layer for Echo Brain
Connects all knowledge sources into a single coherent interface
"""
import asyncpg
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from qdrant_client import QdrantClient
import httpx

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeSource:
    """Represents a piece of knowledge with its source and confidence"""
    content: str
    source_type: str  # 'fact', 'memory', 'conversation', 'core'
    confidence: float
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None


class UnifiedKnowledgeLayer:
    """
    Central knowledge retrieval that combines:
    1. PostgreSQL facts table (6,129 facts)
    2. Qdrant vector memory (24,657 vectors)
    3. PostgreSQL conversations (13,630 messages)
    4. Core system facts (hardcoded essentials)
    """

    # Core facts that Echo Brain should ALWAYS know
    CORE_FACTS = {
        # Service Configuration
        "echo_brain_port": {
            "subject": "Echo Brain",
            "predicate": "runs on port",
            "object": "8309",
            "confidence": 1.0
        },
        "echo_brain_api": {
            "subject": "Echo Brain API",
            "predicate": "base URL",
            "object": "http://localhost:8309/api/echo",
            "confidence": 1.0
        },
        "echo_brain_health": {
            "subject": "Echo Brain",
            "predicate": "health endpoint",
            "object": "/health returns service status",
            "confidence": 1.0
        },
        "echo_brain_mcp": {
            "subject": "Echo Brain MCP",
            "predicate": "endpoint",
            "object": "/mcp for Model Context Protocol integration",
            "confidence": 1.0
        },

        # Database Configuration
        "echo_brain_database": {
            "subject": "Echo Brain",
            "predicate": "uses database",
            "object": "PostgreSQL database 'echo_brain' with user 'patrick'",
            "confidence": 1.0
        },
        "echo_brain_facts_count": {
            "subject": "Echo Brain facts table",
            "predicate": "contains",
            "object": "6,129 facts stored in PostgreSQL",
            "confidence": 1.0
        },
        "echo_brain_conversations": {
            "subject": "Echo Brain conversations",
            "predicate": "contains",
            "object": "13,630 messages from Claude conversations",
            "confidence": 1.0
        },

        # Vector Storage
        "echo_brain_vectors": {
            "subject": "Echo Brain",
            "predicate": "vector storage",
            "object": "Qdrant on port 6333 with collection 'echo_memory' (24,657+ vectors)",
            "confidence": 1.0
        },
        "echo_brain_embeddings": {
            "subject": "Echo Brain",
            "predicate": "embedding model",
            "object": "mxbai-embed-large:latest (1024 dimensions)",
            "confidence": 1.0
        },

        # LLM Models
        "echo_brain_models": {
            "subject": "Echo Brain",
            "predicate": "LLM models",
            "object": "Ollama models: mistral:7b, deepseek-r1:8b, gemma2:9b",
            "confidence": 1.0
        },
        "echo_brain_ollama_port": {
            "subject": "Ollama",
            "predicate": "runs on port",
            "object": "11434",
            "confidence": 1.0
        },

        # Frontend & Access
        "echo_brain_frontend": {
            "subject": "Echo Brain Console",
            "predicate": "served at",
            "object": "/echo-brain/ via nginx proxy",
            "confidence": 1.0
        },
        "echo_brain_frontend_build": {
            "subject": "Echo Brain frontend",
            "predicate": "built with",
            "object": "Vue 3 + TypeScript + Vite",
            "confidence": 1.0
        },

        # Purpose & Architecture
        "echo_brain_purpose": {
            "subject": "Echo Brain",
            "predicate": "purpose",
            "object": "Personal AI assistant using 13k+ conversations and 24k+ memory vectors for contextual responses",
            "confidence": 1.0
        },
        "echo_brain_llm_models": {
            "subject": "Echo Brain",
            "predicate": "uses LLM models",
            "object": "Ollama models including mistral:7b for general queries, deepseek-r1:8b for code/reasoning, gemma2:9b for analysis",
            "confidence": 1.0
        },
        "echo_brain_vector_count": {
            "subject": "Echo Brain",
            "predicate": "vector count",
            "object": "24,657 vectors in Qdrant collection echo_memory (as of Feb 2026)",
            "confidence": 1.0
        },
        "echo_brain_fact_count": {
            "subject": "Echo Brain",
            "predicate": "fact count", 
            "object": "6,129 facts in PostgreSQL facts table",
            "confidence": 1.0
        },
        "echo_brain_conversation_count": {
            "subject": "Echo Brain",
            "predicate": "conversation count",
            "object": "13,630 conversation messages indexed from Claude chat history",
            "confidence": 1.0
        },
        "echo_brain_architecture": {
            "subject": "Echo Brain",
            "predicate": "architecture",
            "object": "FastAPI backend, Vue3 frontend, PostgreSQL + Qdrant storage, Ollama LLMs",
            "confidence": 1.0
        },
        "echo_brain_unified_layer": {
            "subject": "Echo Brain",
            "predicate": "knowledge layer",
            "object": "UnifiedKnowledgeLayer combines facts, vectors, and conversations for intelligent responses",
            "confidence": 1.0
        },

        # Key Endpoints
        "echo_brain_ask_endpoint": {
            "subject": "Echo Brain /ask",
            "predicate": "function",
            "object": "Main Q&A endpoint using unified knowledge layer",
            "confidence": 1.0
        },
        "echo_brain_intelligence_endpoint": {
            "subject": "Echo Brain /intelligence/think",
            "predicate": "function",
            "object": "Multi-stage reasoning with memory and facts",
            "confidence": 1.0
        },
        "echo_brain_memory_search": {
            "subject": "Echo Brain /memory/search",
            "predicate": "function",
            "object": "Search Qdrant vectors using semantic similarity",
            "confidence": 1.0
        },

        # System Information
        "echo_brain_systemd": {
            "subject": "Echo Brain",
            "predicate": "service name",
            "object": "tower-echo-brain.service",
            "confidence": 1.0
        },
        "echo_brain_location": {
            "subject": "Echo Brain",
            "predicate": "installation path",
            "object": "/opt/tower-echo-brain",
            "confidence": 1.0
        },
        "echo_brain_github": {
            "subject": "Echo Brain",
            "predicate": "GitHub repository",
            "object": "github.com/pvestal/tower-echo-brain",
            "confidence": 1.0
        }
    }

    def __init__(
        self,
        db_config: Dict[str, str] = None,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "echo_memory"
    ):
        self.db_config = db_config or {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": "RP78eIrW7cI2jYvL5akt1yurE"
        }
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self._pool = None

    async def get_db_pool(self):
        """Get or create database connection pool"""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=2,
                max_size=10,
                timeout=10
            )
        return self._pool

    async def close(self):
        """Clean up connections"""
        if self._pool:
            await self._pool.close()

    def extract_search_terms(self, query: str) -> List[str]:
        """Extract meaningful search terms from query"""
        stop_words = {
            'what', 'is', 'the', 'how', 'does', 'do', 'why', 'when', 'where',
            'which', 'who', 'a', 'an', 'and', 'or', 'but', 'for', 'to', 'of',
            'in', 'on', 'at', 'by', 'with', 'about', 'can', 'could', 'would',
            'should', 'will', 'are', 'was', 'were', 'been', 'being', 'have'
        }

        words = query.lower().replace('?', '').replace('.', '').split()
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]
        return meaningful[:5]

    async def search_facts(self, query: str, limit: int = 5) -> List[KnowledgeSource]:
        """Search PostgreSQL facts table"""
        pool = await self.get_db_pool()
        results = []

        try:
            # First check core facts
            query_lower = query.lower()
            for fact_key, fact in self.CORE_FACTS.items():
                if any(term in query_lower for term in [
                    fact['subject'].lower(),
                    fact['predicate'].lower(),
                    fact['object'].lower()
                ]):
                    results.append(KnowledgeSource(
                        content=f"{fact['subject']} {fact['predicate']} {fact['object']}",
                        source_type='core',
                        confidence=1.0,
                        metadata={'fact_key': fact_key}
                    ))

            # Search database facts
            search_terms = self.extract_search_terms(query)
            if search_terms:
                async with pool.acquire() as conn:
                    # Build query to search facts
                    conditions = []
                    params = []
                    for i, term in enumerate(search_terms, 1):
                        conditions.append(f"""
                            (subject ILIKE ${i} OR
                             predicate ILIKE ${i} OR
                             object ILIKE ${i})
                        """)
                        params.append(f'%{term}%')

                    query_sql = f"""
                        SELECT subject, predicate, object, confidence, created_at
                        FROM facts
                        WHERE {' OR '.join(conditions)}
                        ORDER BY confidence DESC, created_at DESC
                        LIMIT ${len(params) + 1}
                    """

                    rows = await conn.fetch(query_sql, *params, limit)

                    for row in rows:
                        results.append(KnowledgeSource(
                            content=f"{row['subject']} {row['predicate']} {row['object']}",
                            source_type='fact',
                            confidence=row['confidence'],
                            metadata={'created_at': row['created_at'].isoformat() if row['created_at'] else None},
                            timestamp=row['created_at']
                        ))

            logger.info(f"Found {len(results)} facts for query: {query}")

        except Exception as e:
            logger.error(f"Error searching facts: {e}")

        return results[:limit]

    async def search_vectors(self, query: str, limit: int = 5) -> List[KnowledgeSource]:
        """Search Qdrant vector memory"""
        results = []

        try:
            # Generate embedding for query
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": "mxbai-embed-large:latest", "prompt": query}
                )
                embedding = response.json().get("embedding", [])

            if embedding:
                # Search Qdrant using HTTP API directly (like MCP service)
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.post(
                        f"http://localhost:6333/collections/{self.collection_name}/points/search",
                        json={
                            "vector": embedding,
                            "limit": limit,
                            "with_payload": True
                        }
                    )
                    if response.status_code == 200:
                        search_result = response.json().get("result", [])
                    else:
                        logger.error(f"Qdrant search failed with status {response.status_code}")
                        search_result = []

                for point in search_result:
                    payload = point.get("payload", {})
                    results.append(KnowledgeSource(
                        content=payload.get('content', payload.get('text', ''))[:300],
                        source_type=payload.get('type', 'memory'),
                        confidence=float(point.get('score', 0)),
                        metadata={
                            'source': payload.get('source', ''),
                            'qdrant_id': str(point.get('id', ''))
                        }
                    ))

            logger.info(f"Found {len(results)} vectors for query: {query}")

        except Exception as e:
            logger.error(f"Error searching vectors: {e}")

        return results

    async def search_conversations(self, query: str, limit: int = 5) -> List[KnowledgeSource]:
        """Search PostgreSQL conversations with semantic terms"""
        pool = await self.get_db_pool()
        results = []

        try:
            search_terms = self.extract_search_terms(query)

            if search_terms:
                async with pool.acquire() as conn:
                    # Build OR query for multiple terms
                    conditions = []
                    params = []
                    for i, term in enumerate(search_terms[:3], 1):
                        conditions.append(f"content ILIKE ${i}")
                        params.append(f'%{term}%')

                    query_sql = f"""
                        SELECT conversation_id, role, content, created_at
                        FROM claude_conversations
                        WHERE {' OR '.join(conditions)}
                        ORDER BY created_at DESC
                        LIMIT ${len(params) + 1}
                    """

                    rows = await conn.fetch(query_sql, *params, limit)

                    for row in rows:
                        results.append(KnowledgeSource(
                            content=row['content'][:300],
                            source_type='conversation',
                            confidence=0.7,  # Default confidence for conversations
                            metadata={
                                'conversation_id': row['conversation_id'],
                                'role': row['role']
                            },
                            timestamp=row['created_at']
                        ))

            logger.info(f"Found {len(results)} conversations for query: {query}")

        except Exception as e:
            logger.error(f"Error searching conversations: {e}")

        return results

    async def get_context(
        self,
        query: str,
        max_facts: int = 5,
        max_vectors: int = 3,
        max_conversations: int = 3
    ) -> Dict[str, Any]:
        """
        Get unified context from all knowledge sources
        Returns organized context for LLM prompting
        """
        # Run all searches in parallel
        facts_task = self.search_facts(query, max_facts)
        vectors_task = self.search_vectors(query, max_vectors)
        conversations_task = self.search_conversations(query, max_conversations)

        facts, vectors, conversations = await asyncio.gather(
            facts_task, vectors_task, conversations_task
        )

        # Organize results
        context = {
            "query": query,
            "facts": facts,
            "vectors": vectors,
            "conversations": conversations,
            "total_sources": len(facts) + len(vectors) + len(conversations),
            "timestamp": datetime.now().isoformat()
        }

        # Build prompt context
        prompt_parts = []

        if facts:
            prompt_parts.append("=== Known Facts ===")
            for fact in facts:
                prompt_parts.append(f"• {fact.content} [confidence: {fact.confidence:.2f}]")

        if vectors:
            prompt_parts.append("\n=== Relevant Memory ===")
            for vec in vectors:
                prompt_parts.append(f"• {vec.content} [score: {vec.confidence:.2f}]")

        if conversations:
            prompt_parts.append("\n=== Related Conversations ===")
            for conv in conversations:
                role = conv.metadata.get('role', 'unknown')
                prompt_parts.append(f"• [{role}]: {conv.content}")

        context["prompt_context"] = "\n".join(prompt_parts)

        return context

    def format_for_llm(self, context: Dict[str, Any], question: str) -> str:
        """Format unified context for LLM consumption"""
        return f"""You are Echo Brain, a personal AI assistant with access to the following knowledge:

{context['prompt_context']}

Based on this context, answer the following question accurately and directly:
Question: {question}

If the context contains specific facts about the question (especially from Known Facts), use them in your answer."""


# Singleton instance
_unified_knowledge = None

def get_unified_knowledge() -> UnifiedKnowledgeLayer:
    """Get or create singleton instance"""
    global _unified_knowledge
    if not _unified_knowledge:
        _unified_knowledge = UnifiedKnowledgeLayer()
    return _unified_knowledge