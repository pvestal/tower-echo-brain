"""
LAYER 1: Context Retrieval & Assembly

This layer answers: "What does Echo Brain know that's relevant to this query?"

It searches THREE sources:
1. Qdrant vector search (semantic similarity)
2. PostgreSQL full-text search (keyword matching)
3. Facts table (structured knowledge triples)

Results are ranked, deduplicated, and assembled into a context package.
"""
import time
import logging
import httpx
import asyncpg
from typing import List, Tuple

from .models import ContextPackage, ContextSource, QueryIntent

logger = logging.getLogger("echo.pipeline.context")

# CONFIGURATION - single source of truth
QDRANT_URL = "http://localhost:6333"
QDRANT_THRESHOLD = 0.3              # LOWERED from 0.7 - was missing relevant content
QDRANT_LIMIT = 10
PG_DSN = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"  # 1024D model that matches echo_memory collection
MAX_CONTEXT_TOKENS = 4000            # Don't overwhelm the reasoning model


class ContextLayer:
    """
    Multi-source context retrieval engine.

    Usage:
        layer = ContextLayer()
        await layer.initialize()
        package = await layer.retrieve("What truck does Patrick drive?")
    """

    def __init__(self):
        self.pg_pool: asyncpg.Pool = None
        self.http_client: httpx.AsyncClient = None

    async def initialize(self):
        """Create connection pools. Call once at startup."""
        self.pg_pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=10)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("ContextLayer initialized: PG pool + HTTP client ready")

    async def shutdown(self):
        """Clean shutdown."""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.http_client:
            await self.http_client.aclose()

    async def retrieve(self, query: str) -> ContextPackage:
        """
        Main entry point. Classifies intent, searches all sources, assembles context.
        """
        start = time.time()

        # Step 1: Classify intent
        intent = self._classify_intent(query)
        logger.info(f"Query intent: {intent.value} | Query: {query[:80]}")

        # Step 2: Search all relevant sources in parallel
        sources: List[ContextSource] = []
        total_searched = 0

        # Get embedding for vector search
        embedding = await self._get_embedding(query)

        if embedding:
            # Search echo_memory collection (the only one that exists)
            results = await self._search_qdrant(embedding, "echo_memory")
            sources.extend(results)
            total_searched += 1

        # Search PostgreSQL full-text selectively based on intent
        # Skip for simple factual queries to avoid contamination
        if intent not in (QueryIntent.FACTUAL, QueryIntent.CONVERSATIONAL) or len(query.split()) > 5:
            pg_results = await self._search_postgresql(query)
            sources.extend(pg_results)
            total_searched += 1

        # Search facts table (always for personal/factual queries)
        if intent in (QueryIntent.PERSONAL, QueryIntent.FACTUAL, QueryIntent.CONVERSATIONAL):
            fact_results = await self._search_facts(query)
            sources.extend(fact_results)
            total_searched += 1

        # Step 3: Rank, deduplicate, and trim
        sources = self._rank_and_deduplicate(sources)
        sources = self._trim_to_token_budget(sources)

        # Step 4: Assemble context string
        assembled = self._assemble_context(sources, intent)

        latency = int((time.time() - start) * 1000)
        logger.info(
            f"Context retrieved: {len(sources)} sources from {total_searched} searches in {latency}ms"
        )

        return ContextPackage(
            query=query,
            intent=intent,
            sources=sources,
            assembled_context=assembled,
            total_sources_searched=total_searched,
            total_sources_found=len(sources),
            context_token_estimate=len(assembled.split()),  # Rough estimate
            retrieval_latency_ms=latency,
        )

    def _classify_intent(self, query: str) -> QueryIntent:
        """
        Rule-based intent classification. Fast, deterministic, no LLM needed.
        Upgrade to LLM classification later if rules get unwieldy.
        """
        q = query.lower().strip()

        # Check in priority order - more specific signals first

        # Creative / anime queries - check FIRST to catch anime content
        creative_signals = [
            "anime", "scene", "narrat", "story", "character",
            "comfyui", "generate", "create an image", "goblin slayer",
            "tokyo debt", "episode", "description for"
        ]
        if any(signal in q for signal in creative_signals):
            return QueryIntent.CREATIVE

        # Personal queries - about Patrick, his projects, his stuff
        personal_signals = [
            "my ", "patrick", "i have", "i own", "i drive", "my truck",
            "my rv", "my server", "echo brain",
            "what do you know about me", "do you remember",
        ]
        if any(signal in q for signal in personal_signals):
            return QueryIntent.PERSONAL

        # Special case: "What X does Y have?" when Y is Tower/server
        if ("tower" in q or "server" in q) and ("what" in q or "which" in q):
            return QueryIntent.FACTUAL

        # Reasoning queries - CHECK BEFORE CODING since some overlap
        # Specific patterns for comparisons and architectural decisions
        if "postgresql over mongodb" in q or "mongodb over postgresql" in q:
            return QueryIntent.REASONING
        if ("rest vs graphql" in q or "graphql vs rest" in q or
            "rest versus graphql" in q or "graphql versus rest" in q):
            return QueryIntent.REASONING
        if "why would you choose" in q or "why should i use" in q:
            return QueryIntent.REASONING
        if "compare" in q and ("api" in q or "database" in q or "framework" in q):
            return QueryIntent.REASONING

        reasoning_signals = [
            "why would", "why should", "how does", "explain", "analyze",
            "what are the pros", "trade-off", "should i", "best approach",
            "design", "architect", "strategy", " vs ", " over "
        ]
        if any(signal in q for signal in reasoning_signals):
            return QueryIntent.REASONING

        # Coding queries - include "show me how" and config patterns
        coding_signals = [
            "code", "function", "bug", "error", "python", "typescript",
            "javascript", "endpoint", "sql", "git",
            "debug", "implement", "refactor", "script", "import",
            "class ", "def ", "async ", "const ", "let ", "csv", "json",
            "show me how", "configured", "lora training", "hello world",
            "write a", "write hello"
        ]
        if any(signal in q for signal in coding_signals):
            return QueryIntent.CODING

        # Factual queries
        factual_signals = [
            "what is", "who is", "when did", "where is", "how many",
            "define", "list", "name", "what gpu", "what cpu"
        ]
        if any(signal in q for signal in factual_signals):
            return QueryIntent.FACTUAL

        return QueryIntent.CONVERSATIONAL

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from Ollama."""
        try:
            resp = await self.http_client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            embedding = resp.json().get("embedding", [])
            if not embedding:
                logger.error("Empty embedding returned from Ollama")
                return []
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

    async def _search_qdrant(
        self, embedding: List[float], collection: str
    ) -> List[ContextSource]:
        """Search a single Qdrant collection."""
        try:
            resp = await self.http_client.post(
                f"{QDRANT_URL}/collections/{collection}/points/search",
                json={
                    "vector": embedding,
                    "limit": QDRANT_LIMIT,
                    "score_threshold": QDRANT_THRESHOLD,
                    "with_payload": True,
                },
            )
            resp.raise_for_status()
            results = resp.json().get("result", [])

            sources = []
            for r in results:
                payload = r.get("payload", {})
                text = payload.get("text", payload.get("content", ""))
                if not text:
                    continue
                sources.append(
                    ContextSource(
                        text=text[:2000],  # Cap individual source length
                        source_type="qdrant_vectors",
                        collection=collection,
                        relevance_score=r.get("score", 0.0),
                        metadata={
                            k: v
                            for k, v in payload.items()
                            if k not in ("text", "content", "embedding")
                        },
                    )
                )

            logger.info(f"Qdrant/{collection}: {len(sources)} results (threshold {QDRANT_THRESHOLD})")
            return sources

        except Exception as e:
            logger.error(f"Qdrant search failed for {collection}: {e}")
            return []

    async def _search_postgresql(self, query: str) -> List[ContextSource]:
        """Full-text search against PostgreSQL conversations."""
        try:
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT conversation_id, role, content,
                           ts_rank(to_tsvector('english', content),
                                   plainto_tsquery('english', $1)) as rank
                    FROM claude_conversations
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                       OR content ILIKE '%' || $1 || '%'
                    ORDER BY rank DESC
                    LIMIT 5
                    """,
                    query,
                )

            sources = []
            for row in rows:
                sources.append(
                    ContextSource(
                        text=row["content"][:2000],
                        source_type="postgresql_fts",
                        collection="claude_conversations",
                        relevance_score=float(row["rank"]) if row["rank"] else 0.1,
                        metadata={
                            "conversation_id": row["conversation_id"],
                            "role": row["role"],
                        },
                    )
                )

            logger.info(f"PostgreSQL FTS: {len(sources)} results")
            return sources

        except Exception as e:
            logger.error(f"PostgreSQL search failed: {e}")
            return []

    async def _search_facts(self, query: str) -> List[ContextSource]:
        """Search structured facts table."""
        try:
            async with self.pg_pool.acquire() as conn:
                # Check if facts table exists and has the expected schema
                table_exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='facts')"
                )
                if not table_exists:
                    logger.warning("Facts table does not exist")
                    return []

                rows = await conn.fetch(
                    """
                    SELECT subject, predicate, object, confidence
                    FROM facts
                    WHERE subject ILIKE '%' || $1 || '%'
                       OR predicate ILIKE '%' || $1 || '%'
                       OR object ILIKE '%' || $1 || '%'
                    ORDER BY confidence DESC
                    LIMIT 10
                    """,
                    query,
                )

            sources = []
            for row in rows:
                fact_text = f"{row['subject']} {row['predicate']} {row['object']}"
                sources.append(
                    ContextSource(
                        text=fact_text,
                        source_type="facts_table",
                        collection="facts",
                        relevance_score=float(row["confidence"]),
                        metadata={
                            "subject": row["subject"],
                            "predicate": row["predicate"],
                            "object": row["object"],
                        },
                    )
                )

            logger.info(f"Facts table: {len(sources)} results")
            return sources

        except Exception as e:
            logger.error(f"Facts search failed: {e}")
            return []

    def _rank_and_deduplicate(self, sources: List[ContextSource]) -> List[ContextSource]:
        """Remove near-duplicates, sort by relevance."""
        seen_texts = set()
        unique = []
        for s in sources:
            # Simple dedup: first 100 chars
            key = s.text[:100].lower().strip()
            if key not in seen_texts:
                seen_texts.add(key)
                unique.append(s)

        # Sort by relevance score descending
        unique.sort(key=lambda x: x.relevance_score, reverse=True)
        return unique

    def _trim_to_token_budget(self, sources: List[ContextSource]) -> List[ContextSource]:
        """Keep sources within token budget."""
        total_tokens = 0
        trimmed = []
        for s in sources:
            est_tokens = len(s.text.split())
            if total_tokens + est_tokens > MAX_CONTEXT_TOKENS:
                break
            trimmed.append(s)
            total_tokens += est_tokens
        return trimmed

    def _assemble_context(self, sources: List[ContextSource], intent: QueryIntent) -> str:
        """Format sources into a context string for prompt injection."""
        if not sources:
            return ""

        sections = []
        for i, s in enumerate(sources, 1):
            label = f"[Source {i}: {s.source_type}/{s.collection} (score: {s.relevance_score:.2f})]"
            sections.append(f"{label}\n{s.text}")

        header = f"RELEVANT CONTEXT ({len(sources)} sources, intent: {intent.value}):"
        return f"{header}\n\n" + "\n\n---\n\n".join(sections)