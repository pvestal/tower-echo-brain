"""
Parallel Context Retriever - Fetches from multiple sources based on domain
Uses domain classification to prevent cross-contamination
"""
import os
import asyncio
import logging
import httpx
import asyncpg
from typing import List, Dict, Any, Optional
from datetime import datetime

from .classifier import DomainClassifier, Domain

logger = logging.getLogger("echo.context_assembly.retriever")


class ParallelRetriever:
    """
    Retrieves context from multiple sources in parallel.
    Enforces domain boundaries to prevent contamination.
    """

    def __init__(self):
        self.classifier = DomainClassifier()
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.qdrant_url = "http://localhost:6333"
        self.ollama_url = "http://localhost:11434"
        self.embedding_model = "mxbai-embed-large"  # 1024D
        # Use DATABASE_URL if available, otherwise use DB_PASSWORD with fallback
        self.pg_dsn = os.getenv('DATABASE_URL',
            f"postgresql://patrick:{os.getenv('DB_PASSWORD', 'WL12Ow4cuhEAWcO3Iaw7d2J7JEV8Hklr')}@localhost/echo_brain")

    async def initialize(self):
        """Initialize connection pools"""
        if not self.pg_pool:
            self.pg_pool = await asyncpg.create_pool(
                self.pg_dsn, min_size=2, max_size=10
            )
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("ParallelRetriever initialized")

    async def shutdown(self):
        """Clean shutdown"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.http_client:
            await self.http_client.aclose()

    async def retrieve(
        self,
        query: str,
        max_results: int = 10,
        override_domain: Optional[Domain] = None
    ) -> Dict[str, Any]:
        """
        Main retrieval method. Classifies query and fetches from appropriate sources.

        Args:
            query: The user's query
            max_results: Maximum results to return
            override_domain: Force a specific domain (for testing)

        Returns:
            Dictionary with classified domain and retrieved sources
        """
        start_time = datetime.now()

        # Step 1: Classify query domain
        if override_domain:
            domains = [(override_domain, 1.0)]
        else:
            domains = self.classifier.classify(query)

        primary_domain = domains[0][0] if domains else Domain.GENERAL
        confidence = domains[0][1] if domains else 0.5

        logger.info(f"Query classified as {primary_domain} (confidence: {confidence:.2f})")

        # Step 2: Get allowed sources for this domain
        allowed_sources = self.classifier.get_allowed_sources(primary_domain)

        # Step 3: Build parallel retrieval tasks
        tasks = []

        # Get embedding for vector search (if needed)
        if allowed_sources.get("qdrant_collections"):
            embedding = await self._get_embedding(query)
            if embedding:
                for collection in allowed_sources["qdrant_collections"]:
                    if await self._collection_exists(collection):
                        tasks.append(self._search_qdrant(
                            query, embedding, collection, primary_domain
                        ))

        # Search PostgreSQL tables
        for table in allowed_sources.get("pg_tables", []):
            if "." in table:  # Schema.table format
                schema, table_name = table.split(".")
                tasks.append(self._search_postgresql(
                    query, table_name, schema, primary_domain
                ))
            else:
                tasks.append(self._search_postgresql(
                    query, table, "public", primary_domain
                ))

        # Search facts (always, but filtered by domain)
        tasks.append(self._search_facts(query, primary_domain))

        # Step 4: Execute all searches in parallel
        if tasks:
            all_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Flatten results and handle errors
            sources = []
            for result in all_results:
                if isinstance(result, Exception):
                    logger.error(f"Search task failed: {result}")
                elif isinstance(result, list):
                    sources.extend(result)
        else:
            sources = []

        # Step 5: Apply domain filtering to results
        filtered_sources = self.classifier.filter_results_by_domain(sources, primary_domain)

        # Step 6: Sort by relevance
        filtered_sources.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Step 7: Limit to max_results
        final_sources = filtered_sources[:max_results]

        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        return {
            "query": query,
            "domain": primary_domain.value,
            "domain_confidence": confidence,
            "sources": final_sources,
            "total_found": len(filtered_sources),
            "total_returned": len(final_sources),
            "retrieval_ms": elapsed_ms,
            "allowed_collections": allowed_sources.get("qdrant_collections", []),
            "allowed_tables": allowed_sources.get("pg_tables", [])
        }

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector from Ollama"""
        try:
            resp = await self.http_client.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text}
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    async def _collection_exists(self, collection: str) -> bool:
        """Check if a Qdrant collection exists"""
        try:
            resp = await self.http_client.get(
                f"{self.qdrant_url}/collections/{collection}"
            )
            return resp.status_code == 200
        except:
            return False

    async def _search_qdrant(
        self,
        query: str,
        embedding: List[float],
        collection: str,
        domain: Domain
    ) -> List[Dict]:
        """Search a Qdrant collection"""
        try:
            # Domain-specific thresholds
            sources = self.classifier.get_allowed_sources(domain)
            min_score = sources.get("min_score", 0.3)

            resp = await self.http_client.post(
                f"{self.qdrant_url}/collections/{collection}/points/search",
                json={
                    "vector": embedding,
                    "limit": 20,  # Get more, filter later
                    "score_threshold": min_score,
                    "with_payload": True
                }
            )
            resp.raise_for_status()

            results = []
            for point in resp.json().get("result", []):
                payload = point.get("payload", {})
                content = payload.get("text", payload.get("content", ""))

                if content:
                    results.append({
                        "type": "vector",
                        "source": f"qdrant/{collection}",
                        "content": content[:1000],  # Truncate
                        "score": point.get("score", 0),
                        "metadata": {
                            k: v for k, v in payload.items()
                            if k not in ("text", "content", "embedding")
                        }
                    })

            logger.info(f"Qdrant/{collection}: {len(results)} results for domain {domain}")
            return results

        except Exception as e:
            logger.error(f"Qdrant search failed for {collection}: {e}")
            return []

    async def _search_postgresql(
        self,
        query: str,
        table: str,
        schema: str,
        domain: Domain
    ) -> List[Dict]:
        """Search PostgreSQL with full-text search"""
        try:
            # Simple safety check for known tables
            safe_tables = [
                "claude_conversations", "facts", "service_health",
                "transactions", "financial_data", "system_logs"
            ]
            if table not in safe_tables:
                logger.warning(f"Skipping unknown table: {schema}.{table}")
                return []

            async with self.pg_pool.acquire() as conn:
                # Build appropriate query based on table
                if table == "claude_conversations":
                    rows = await conn.fetch("""
                        SELECT conversation_id, role, content,
                               ts_rank(to_tsvector('english', content),
                                      plainto_tsquery('english', $1)) as rank
                        FROM claude_conversations
                        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                        ORDER BY rank DESC
                        LIMIT 10
                    """, query)

                    results = []
                    for row in rows:
                        results.append({
                            "type": "conversation",
                            "source": f"pg/{table}",
                            "content": row["content"][:1000],
                            "score": float(row["rank"]) if row["rank"] else 0.1,
                            "metadata": {
                                "conversation_id": row["conversation_id"],
                                "role": row["role"]
                            }
                        })
                    return results


                else:
                    logger.info(f"Table {table} not implemented yet")
                    return []

        except Exception as e:
            logger.error(f"PostgreSQL search failed for {table}: {e}")
            return []

    async def _search_facts(self, query: str, domain: Domain) -> List[Dict]:
        """Search facts table with domain filtering"""
        try:
            async with self.pg_pool.acquire() as conn:
                # First check if table exists
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='facts')"
                )
                if not exists:
                    return []

                # Extract meaningful words from query (tokenization)
                stop_words = {'what', 'does', 'how', 'the', 'are', 'is', 'for', 'with', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'from'}
                words = [w.lower() for w in query.split()
                        if len(w) > 2 and w.lower() not in stop_words]

                if not words:
                    # Fallback to original query if no meaningful words
                    words = [query]

                # Build query with individual word matching
                conditions = []
                params = []
                for i, word in enumerate(words):
                    conditions.append(
                        f"(subject ILIKE '%' || ${i+1} || '%' OR predicate ILIKE '%' || ${i+1} || '%' OR object ILIKE '%' || ${i+1} || '%')"
                    )
                    params.append(word)

                where_clause = " OR ".join(conditions)

                rows = await conn.fetch(f"""
                    SELECT subject, predicate, object, confidence
                    FROM facts
                    WHERE {where_clause}
                    ORDER BY confidence DESC
                    LIMIT 20
                """, *params)

                # Apply domain filter
                sources = self.classifier.get_allowed_sources(domain)
                fact_filter = sources.get("facts_filter", lambda f: True)

                results = []
                for row in rows:
                    fact_text = f"{row['subject']} {row['predicate']} {row['object']}"

                    # Apply domain-specific filter
                    if fact_filter(fact_text.lower()):
                        results.append({
                            "type": "fact",
                            "source": "facts",
                            "content": fact_text,
                            "score": float(row["confidence"]),
                            "metadata": {
                                "subject": row["subject"],
                                "predicate": row["predicate"],
                                "object": row["object"]
                            }
                        })

                logger.info(f"Facts: {len(results)} domain-filtered results")
                return results

        except Exception as e:
            logger.error(f"Facts search failed: {e}")
            return []