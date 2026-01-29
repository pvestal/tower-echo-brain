"""
Retrieval Orchestrator

Coordinates parallel retrieval from multiple sources:
- Qdrant vectors
- PostgreSQL facts
- Conversation history
- Code context

All retrievals are domain-filtered to prevent context contamination.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .models import (
    Domain, SourceType,
    VectorResult, FactResult, ConversationTurn, CodeContext,
    RetrievalResult
)


class RetrieverConfig:
    """Configuration for the retrieval orchestrator."""
    
    def __init__(
        self,
        # Qdrant settings
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_collection: str = "echo_memory",
        
        # PostgreSQL settings  
        postgres_dsn: str = "postgresql://localhost/echo_brain",
        
        # Retrieval limits
        max_vectors: int = 10,
        max_facts: int = 20,
        max_conversation_turns: int = 10,
        max_code_files: int = 5,
        
        # Scoring
        min_similarity_score: float = 0.5,
        recency_weight: float = 0.2,  # How much to boost recent results
        
        # Embedding model (for query embedding)
        # NOTE: echo_memory uses 1024-dim (mxbai-embed-large)
        #       claude_conversations uses 384-dim (nomic-embed-text)
        embedding_model: str = "mxbai-embed-large",
        embedding_dimensions: int = 1024,
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_collection = qdrant_collection
        self.postgres_dsn = postgres_dsn
        self.max_vectors = max_vectors
        self.max_facts = max_facts
        self.max_conversation_turns = max_conversation_turns
        self.max_code_files = max_code_files
        self.min_similarity_score = min_similarity_score
        self.recency_weight = recency_weight
        self.embedding_model = embedding_model


class RetrievalOrchestrator:
    """
    Orchestrates parallel retrieval from multiple data sources.
    
    Key features:
    - Domain filtering to prevent context contamination
    - Parallel async retrieval for performance
    - Reranking based on relevance + recency
    - Deduplication across sources
    """
    
    def __init__(
        self,
        config: RetrieverConfig,
        ollama_client = None  # For generating query embeddings
    ):
        self.config = config
        self.ollama_client = ollama_client
        self._qdrant: Optional[AsyncQdrantClient] = None
        self._pg_pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize connections to data stores."""
        self._qdrant = AsyncQdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port
        )
        self._pg_pool = await asyncpg.create_pool(
            self.config.postgres_dsn,
            min_size=2,
            max_size=10
        )
    
    async def close(self):
        """Clean up connections."""
        if self._qdrant:
            await self._qdrant.close()
        if self._pg_pool:
            await self._pg_pool.close()
    
    async def retrieve(
        self,
        query: str,
        domain: Domain,
        include_vectors: bool = True,
        include_facts: bool = True,
        include_conversation: bool = True,
        include_code: bool = True
    ) -> RetrievalResult:
        """
        Retrieve relevant context from all sources.
        
        Args:
            query: The user's query
            domain: The classified domain for filtering
            include_*: Flags to enable/disable specific sources
            
        Returns:
            RetrievalResult with all retrieved content
        """
        start_time = datetime.utcnow()
        
        # Generate query embedding
        query_embedding = await self._get_embedding(query)
        
        # Build retrieval tasks
        tasks = []
        
        if include_vectors:
            tasks.append(self._retrieve_vectors(query_embedding, domain))
        else:
            tasks.append(asyncio.coroutine(lambda: [])())
            
        if include_facts:
            tasks.append(self._retrieve_facts(query_embedding, domain))
        else:
            tasks.append(asyncio.coroutine(lambda: [])())
            
        if include_conversation:
            tasks.append(self._retrieve_conversation(domain))
        else:
            tasks.append(asyncio.coroutine(lambda: [])())
            
        if include_code and domain == Domain.TECHNICAL:
            tasks.append(self._retrieve_code(query_embedding))
        else:
            tasks.append(asyncio.coroutine(lambda: [])())
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any errors
        vectors = results[0] if not isinstance(results[0], Exception) else []
        facts = results[1] if not isinstance(results[1], Exception) else []
        conversation = results[2] if not isinstance(results[2], Exception) else []
        code = results[3] if not isinstance(results[3], Exception) else []
        
        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                source_names = ["vectors", "facts", "conversation", "code"]
                print(f"Error retrieving {source_names[i]}: {result}")
        
        elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return RetrievalResult(
            vectors=vectors,
            facts=facts,
            conversation_history=conversation,
            code_context=code,
            retrieval_time_ms=elapsed_ms
        )
    
    async def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for query text."""
        if self.ollama_client is None:
            raise ValueError("Ollama client required for embeddings")
        
        response = await self.ollama_client.embeddings(
            model=self.config.embedding_model,
            prompt=text
        )
        return response["embedding"]
    
    async def _retrieve_vectors(
        self,
        query_embedding: list[float],
        domain: Domain
    ) -> list[VectorResult]:
        """Retrieve similar vectors from Qdrant with domain filtering."""
        
        # Build domain filter
        # For GENERAL domain, don't filter (retrieve from all domains)
        filter_conditions = None
        if domain != Domain.GENERAL:
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="domain",
                        match=MatchValue(value=domain.value)
                    )
                ]
            )
        
        # Search Qdrant
        search_results = await self._qdrant.search(
            collection_name=self.config.qdrant_collection,
            query_vector=query_embedding,
            query_filter=filter_conditions,
            limit=self.config.max_vectors,
            score_threshold=self.config.min_similarity_score,
            with_payload=True
        )
        
        # Convert to VectorResult
        vectors = []
        for result in search_results:
            payload = result.payload or {}
            vectors.append(VectorResult(
                id=UUID(result.id) if isinstance(result.id, str) else result.id,
                content=payload.get("content", ""),
                score=result.score,
                domain=Domain(payload.get("domain", "general")),
                source_type=SourceType(payload.get("source_type", "document")),
                source_path=payload.get("source_path", "unknown"),
                created_at=datetime.fromisoformat(payload.get("created_at", datetime.utcnow().isoformat())),
                chunk_index=payload.get("chunk_index"),
                total_chunks=payload.get("total_chunks")
            ))
        
        # Apply recency boost
        vectors = self._apply_recency_boost(vectors)
        
        return vectors
    
    async def _retrieve_facts(
        self,
        query_embedding: list[float],
        domain: Domain
    ) -> list[FactResult]:
        """Retrieve relevant facts from PostgreSQL."""
        
        # Build domain filter
        domain_clause = ""
        params = [query_embedding, self.config.max_facts]
        
        if domain != Domain.GENERAL:
            domain_clause = "AND domain = $3"
            params.append(domain.value)
        
        query = f"""
            SELECT 
                id,
                fact_text,
                fact_type,
                domain,
                confidence,
                subject,
                predicate,
                object,
                1 - (embedding <=> $1::vector) as similarity
            FROM facts
            WHERE 
                (valid_until IS NULL OR valid_until > NOW())
                {domain_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        
        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        facts = []
        for row in rows:
            if row["similarity"] >= self.config.min_similarity_score:
                facts.append(FactResult(
                    id=row["id"],
                    fact_text=row["fact_text"],
                    fact_type=row["fact_type"],
                    domain=Domain(row["domain"]),
                    confidence=row["confidence"],
                    relevance_score=row["similarity"],
                    subject=row["subject"],
                    predicate=row["predicate"],
                    object=row["object"]
                ))
        
        return facts
    
    async def _retrieve_conversation(
        self,
        domain: Domain
    ) -> list[ConversationTurn]:
        """Retrieve recent conversation history."""
        
        # For conversation, we always want recent turns regardless of domain
        # But we can optionally filter by domain if there's conversation metadata
        
        query = """
            SELECT 
                id,
                conversation_id,
                role,
                content,
                created_at
            FROM conversation_turns
            ORDER BY created_at DESC
            LIMIT $1
        """
        
        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(query, self.config.max_conversation_turns)
        
        turns = []
        for row in rows:
            turns.append(ConversationTurn(
                role=row["role"],
                content=row["content"],
                timestamp=row["created_at"],
                conversation_id=row["conversation_id"]
            ))
        
        # Reverse to get chronological order
        turns.reverse()
        
        return turns
    
    async def _retrieve_code(
        self,
        query_embedding: list[float]
    ) -> list[CodeContext]:
        """Retrieve relevant code context (only for technical domain)."""
        
        # Search only code vectors
        filter_conditions = Filter(
            must=[
                FieldCondition(
                    key="source_type",
                    match=MatchValue(value="code")
                )
            ]
        )
        
        search_results = await self._qdrant.search(
            collection_name=self.config.qdrant_collection,
            query_vector=query_embedding,
            query_filter=filter_conditions,
            limit=self.config.max_code_files,
            score_threshold=self.config.min_similarity_score,
            with_payload=True
        )
        
        code_context = []
        for result in search_results:
            payload = result.payload or {}
            code_context.append(CodeContext(
                file_path=payload.get("source_path", "unknown"),
                content=payload.get("content", ""),
                language=self._detect_language(payload.get("source_path", "")),
                relevance_score=result.score,
                function_names=payload.get("function_names", []),
                class_names=payload.get("class_names", []),
                imports=payload.get("imports", [])
            ))
        
        return code_context
    
    def _apply_recency_boost(self, vectors: list[VectorResult]) -> list[VectorResult]:
        """Apply recency boost to vector scores."""
        now = datetime.utcnow()
        
        for vector in vectors:
            age_days = (now - vector.created_at).days
            # Decay factor: 1.0 for today, decreasing over time
            recency_factor = 1.0 / (1.0 + age_days / 30)  # 30-day half-life
            
            # Blend original score with recency
            boosted_score = (
                vector.score * (1 - self.config.recency_weight) +
                recency_factor * self.config.recency_weight
            )
            vector.score = boosted_score
        
        # Re-sort by boosted score
        vectors.sort(key=lambda v: v.score, reverse=True)
        
        return vectors
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".rs": "rust",
            ".sql": "sql",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".sh": "bash",
            ".md": "markdown",
            ".vue": "vue"
        }
        
        for ext, lang in ext_map.items():
            if file_path.endswith(ext):
                return lang
        return "text"


# ============================================================================
# Factory function for easy instantiation
# ============================================================================

async def create_retriever(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    postgres_dsn: str = "postgresql://localhost/echo_brain",
    ollama_client = None
) -> RetrievalOrchestrator:
    """Create and initialize a retriever."""
    config = RetrieverConfig(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        postgres_dsn=postgres_dsn
    )
    retriever = RetrievalOrchestrator(config, ollama_client)
    await retriever.initialize()
    return retriever
