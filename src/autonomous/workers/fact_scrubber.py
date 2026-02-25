"""
FactScrubber Worker (HMLR Pipeline)
Verifies extracted facts against their source vectors using cosine similarity
and optional LLM verification. Demotes low-quality facts.
"""
import logging
import os
import httpx
import asyncpg
from pathlib import Path

logger = logging.getLogger("echo.workers.fact_scrubber")

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
COLLECTION = "echo_memory"
MAX_FACTS_PER_CYCLE = 50
GPU_BUSY_THRESHOLD = 80  # Skip LLM if GPU > 80% busy


class FactScrubber:
    """Verifies facts against source vectors. Part of the HMLR pipeline."""

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable required")

    async def _ensure_columns(self, conn):
        """Add verified_at and verification_source columns if missing."""
        await conn.execute("""
            DO $$ BEGIN
                ALTER TABLE facts ADD COLUMN IF NOT EXISTS verified_at
                    TIMESTAMP WITH TIME ZONE;
                ALTER TABLE facts ADD COLUMN IF NOT EXISTS verification_source
                    VARCHAR(50);
            EXCEPTION WHEN others THEN NULL;
            END $$;
        """)

    async def _get_embedding(self, text: str) -> list:
        """Get embedding from Ollama."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": "nomic-embed-text", "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            return (data.get("embeddings") or [[]])[0] or data.get("embedding", [])

    async def _get_source_vector(self, vector_id: str) -> list:
        """Retrieve vector from Qdrant by ID."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points",
                json={"ids": [vector_id], "with_vector": True},
            )
            if resp.status_code == 200:
                pts = resp.json().get("result", [])
                if pts:
                    return pts[0].get("vector", [])
        return []

    def _cosine_similarity(self, a: list, b: list) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _gpu_busy(self) -> bool:
        """Check if GPU is too busy for LLM calls."""
        try:
            for card_dir in Path("/sys/class/drm").glob("card*/device"):
                busy_file = card_dir / "gpu_busy_percent"
                if busy_file.exists():
                    pct = int(busy_file.read_text().strip())
                    if pct > GPU_BUSY_THRESHOLD:
                        return True
        except Exception:
            pass
        return False

    async def _get_graph_support(self, conn, subject: str, predicate: str, obj: str) -> float:
        """Check if graph-connected facts corroborate this fact.

        Returns a support score:
          > 0 means corroborating facts found (boosts confidence)
          0 means no graph data available
          < 0 means contradicting facts found (weakens confidence)
        """
        try:
            from src.core.graph_engine import get_graph_engine
            engine = get_graph_engine()
            await engine._ensure_loaded()
            if not engine._graph:
                return 0.0

            # Find entities connected to this fact's subject
            connected = engine.get_connected_entities(subject, max_hops=1)
            if not connected:
                return 0.0

            # Check if any connected entities have facts with same predicate
            connected_entities = [c["entity"] for c in connected[:10]]
            support_score = 0.0

            for entity in connected_entities:
                rows = await conn.fetch("""
                    SELECT object, confidence FROM facts
                    WHERE LOWER(subject) = $1 AND LOWER(predicate) = $2
                    LIMIT 3
                """, entity.lower(), predicate.lower())

                for row in rows:
                    if row["object"].lower().strip() == obj.lower().strip():
                        support_score += float(row["confidence"]) * 0.1
                    # Different object for same predicate on connected entity
                    # isn't necessarily contradicting (different entity)

            return min(support_score, 0.3)  # Cap bonus
        except Exception:
            return 0.0

    async def _llm_verify(self, fact_text: str, source_text: str) -> bool:
        """Ask LLM whether the fact is supported by the source text."""
        prompt = (
            f"Is the following fact supported by the source text? "
            f"Answer only YES or NO.\n\n"
            f"Fact: {fact_text}\n\n"
            f"Source: {source_text[:1500]}\n\n"
            f"Answer:"
        )
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": "gemma2:9b",
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.1,
                    },
                )
                resp.raise_for_status()
                answer = resp.json().get("response", "").strip().upper()
                return answer.startswith("YES")
        except Exception as e:
            logger.warning(f"LLM verify failed: {e}")
            return True  # Default to keeping the fact on LLM failure

    async def run_cycle(self):
        verified = 0
        demoted = 0

        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                await self._ensure_columns(conn)

                # Get unverified facts, prioritized by most-accessed first
                rows = await conn.fetch("""
                    SELECT id, subject, predicate, object, confidence,
                           source_vector_id,
                           COALESCE(fact_text, subject || ' ' || predicate || ' ' || object) as fact_text
                    FROM facts
                    WHERE verified_at IS NULL
                    ORDER BY COALESCE(access_count, 0) DESC
                    LIMIT $1
                """, MAX_FACTS_PER_CYCLE)

                if not rows:
                    logger.info("FactScrubber: no unverified facts to process")
                    return

                logger.info(f"FactScrubber: processing {len(rows)} unverified facts")

                for row in rows:
                    fact_id = row["id"]
                    fact_text = row["fact_text"]
                    source_vid = row.get("source_vector_id")
                    old_conf = float(row["confidence"])

                    if not source_vid:
                        # No source vector — can't verify, mark as verified with note
                        await conn.execute("""
                            UPDATE facts SET verified_at = NOW(),
                                verification_source = 'no_source_vector'
                            WHERE id = $1
                        """, fact_id)
                        verified += 1
                        continue

                    # Get fact embedding
                    fact_emb = await self._get_embedding(fact_text)
                    if not fact_emb:
                        continue

                    # Get source vector
                    source_vec = await self._get_source_vector(source_vid)
                    if not source_vec:
                        await conn.execute("""
                            UPDATE facts SET verified_at = NOW(),
                                verification_source = 'source_missing'
                            WHERE id = $1
                        """, fact_id)
                        verified += 1
                        continue

                    sim = self._cosine_similarity(fact_emb, source_vec)

                    # Get graph corroboration (shifts thresholds)
                    graph_support = await self._get_graph_support(
                        conn, row["subject"], row["predicate"], row["object"]
                    )
                    # Graph-corroborated facts get more lenient thresholds
                    auto_demote_threshold = 0.5 - graph_support
                    borderline_threshold = 0.7 - graph_support

                    if sim < auto_demote_threshold:
                        # Auto-demote: very low similarity
                        new_conf = max(0.2, min(0.3, old_conf * 0.4))
                        await conn.execute("""
                            UPDATE facts SET confidence = $1, verified_at = NOW(),
                                verification_source = 'auto_demoted'
                            WHERE id = $2
                        """, new_conf, fact_id)
                        demoted += 1

                    elif sim < borderline_threshold and not self._gpu_busy():
                        # Borderline — use LLM
                        # Get source content for LLM
                        async with httpx.AsyncClient(timeout=10) as client:
                            resp = await client.post(
                                f"{QDRANT_URL}/collections/{COLLECTION}/points",
                                json={"ids": [source_vid], "with_payload": True},
                            )
                            source_text = ""
                            if resp.status_code == 200:
                                pts = resp.json().get("result", [])
                                if pts:
                                    p = pts[0].get("payload", {})
                                    source_text = p.get("content", p.get("text", ""))

                        if source_text:
                            supported = await self._llm_verify(fact_text, source_text)
                            if supported:
                                await conn.execute("""
                                    UPDATE facts SET verified_at = NOW(),
                                        verification_source = 'llm_verified'
                                    WHERE id = $1
                                """, fact_id)
                                verified += 1
                            else:
                                new_conf = max(0.2, min(0.4, old_conf * 0.5))
                                await conn.execute("""
                                    UPDATE facts SET confidence = $1, verified_at = NOW(),
                                        verification_source = 'llm_demoted'
                                    WHERE id = $2
                                """, new_conf, fact_id)
                                demoted += 1
                        else:
                            # No source text, borderline sim — mark checked
                            await conn.execute("""
                                UPDATE facts SET verified_at = NOW(),
                                    verification_source = 'cosine_check'
                                WHERE id = $1
                            """, fact_id)
                            verified += 1
                    else:
                        # High similarity — fact is well-supported
                        await conn.execute("""
                            UPDATE facts SET verified_at = NOW(),
                                verification_source = 'cosine_check'
                            WHERE id = $1
                        """, fact_id)
                        verified += 1

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"FactScrubber error: {e}")

        logger.info(f"FactScrubber cycle: verified={verified}, demoted={demoted}")
