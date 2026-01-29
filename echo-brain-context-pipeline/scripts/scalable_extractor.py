#!/usr/bin/env python3
"""
Scalable Fact Extraction Engine

Designed to handle 300k+ vectors with:
- Priority-based extraction (technical/code first)
- Checkpoint/resume capability
- Content filtering (skip junk)
- Realistic progress tracking
- Batch processing with backpressure

Usage:
    # Full extraction (will take weeks)
    python scalable_extractor.py --collection echo_memory
    
    # Quick test (100 vectors)
    python scalable_extractor.py --collection echo_memory --limit 100
    
    # Resume from checkpoint
    python scalable_extractor.py --collection echo_memory --resume
    
    # High-priority only
    python scalable_extractor.py --collection echo_memory --priority-only
"""

import argparse
import asyncio
import hashlib
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

import asyncpg
import httpx
from qdrant_client import QdrantClient


@dataclass
class ExtractionConfig:
    """Configuration for extraction run."""
    # Database
    postgres_dsn: str = "postgresql://patrick@localhost:5432/echo_brain"
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection: str = "echo_memory"
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    extraction_model: str = "qwen2.5:7b"  # Use 7b for speed
    embedding_model_1024: str = "mxbai-embed-large"
    embedding_model_384: str = "nomic-embed-text"
    
    # Processing
    batch_size: int = 50
    checkpoint_interval: int = 100
    max_retries: int = 2
    timeout_seconds: int = 120
    
    # Filtering
    min_content_length: int = 50  # Skip very short content
    skip_duplicate_threshold: float = 0.95  # Skip near-duplicates
    
    # Limits
    max_vectors: Optional[int] = None  # None = all
    priority_only: bool = False  # Only extract high-priority (technical/code)


# Domain classification keywords (fast local classification)
DOMAIN_KEYWORDS = {
    "technical": {
        "postgresql", "postgres", "qdrant", "ollama", "fastapi", "docker",
        "python", "async", "api", "endpoint", "database", "server",
        "error", "debug", "exception", "function", "class", "import",
        "echo brain", "tower", "microservice", "vector", "embedding"
    },
    "code": {
        "def ", "class ", "import ", "from ", "async def", "await ",
        "return ", "if __name__", "try:", "except:", "```python",
        "function(", "const ", "let ", "var ", ".py", ".ts", ".js"
    },
    "anime": {
        "lora", "checkpoint", "comfyui", "stable diffusion", "sdxl",
        "tokyo debt desire", "cyberpunk goblin slayer", "character",
        "episode", "scene", "framepack", "animatediff"
    },
    "personal": {
        "victron", "tundra", "rv", "sundowner", "trailblazer",
        "lifepo4", "solar", "inverter", "camping"
    }
}


EXTRACTION_PROMPT = """Extract factual information from this content. Return JSON array of facts.

Each fact should have:
- fact_text: Clear standalone statement
- fact_type: One of [entity, relationship, event, preference, technical, temporal]
- subject: What/who the fact is about
- predicate: The relationship/action
- object: The target/value
- confidence: 0.0-1.0

Focus on: technical configs, decisions, problems/solutions, preferences.
Skip: filler text, greetings, incomplete thoughts.

Content:
---
{content}
---

Return ONLY valid JSON array. No markdown, no explanation."""


class OllamaClient:
    """Async Ollama client with timeout handling."""
    
    def __init__(self, host: str, timeout: int = 120):
        self.host = host
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def generate(self, model: str, prompt: str, options: dict = None) -> dict:
        response = await self.client.post(
            f"{self.host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "options": options or {"temperature": 0.1, "num_predict": 2000},
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def embeddings(self, model: str, prompt: str) -> list[float]:
        response = await self.client.post(
            f"{self.host}/api/embeddings",
            json={"model": model, "prompt": prompt}
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    async def close(self):
        await self.client.aclose()


class ScalableExtractor:
    """
    Scalable fact extraction with priority queuing and checkpointing.
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.ollama: Optional[OllamaClient] = None
        self.qdrant: Optional[QdrantClient] = None
        self.pool: Optional[asyncpg.Pool] = None
        self.running = True
        
        # Stats
        self.stats = {
            "started_at": None,
            "vectors_processed": 0,
            "facts_extracted": 0,
            "errors": 0,
            "skipped_short": 0,
            "skipped_duplicate": 0
        }
    
    async def initialize(self):
        """Initialize all connections."""
        self.ollama = OllamaClient(self.config.ollama_host, self.config.timeout_seconds)
        self.qdrant = QdrantClient(host=self.config.qdrant_host, port=self.config.qdrant_port)
        self.pool = await asyncpg.create_pool(self.config.postgres_dsn, min_size=2, max_size=10)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal - save checkpoint and exit gracefully."""
        print("\n⚠️  Shutdown requested, saving checkpoint...")
        self.running = False
    
    async def close(self):
        """Clean up resources."""
        if self.ollama:
            await self.ollama.close()
        if self.pool:
            await self.pool.close()
    
    async def run(self, resume: bool = False):
        """
        Main extraction loop.
        """
        self.stats["started_at"] = datetime.utcnow()
        print("=" * 60)
        print("SCALABLE FACT EXTRACTION ENGINE")
        print("=" * 60)
        print(f"Collection: {self.config.collection}")
        print(f"Model: {self.config.extraction_model}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Checkpoint interval: {self.config.checkpoint_interval}")
        print()
        
        # Get or create checkpoint
        checkpoint = await self._get_checkpoint() if resume else None
        start_offset = checkpoint["last_offset"] if checkpoint else 0
        
        if checkpoint:
            print(f"📍 Resuming from offset {start_offset}")
            self.stats["vectors_processed"] = checkpoint.get("vectors_processed", 0)
            self.stats["facts_extracted"] = checkpoint.get("facts_extracted", 0)
        
        # Get collection info
        collection_info = self.qdrant.get_collection(self.config.collection)
        total_vectors = collection_info.points_count
        
        print(f"📊 Total vectors: {total_vectors:,}")
        
        if self.config.max_vectors:
            total_vectors = min(total_vectors, self.config.max_vectors)
            print(f"📊 Processing limit: {total_vectors:,}")
        
        # Estimate time
        vectors_per_minute = 3  # Conservative estimate with qwen2.5:7b
        remaining = total_vectors - start_offset
        eta_hours = remaining / vectors_per_minute / 60
        print(f"⏱️  Estimated time: {eta_hours:.1f} hours ({eta_hours/24:.1f} days)")
        print()
        
        # Phase 1: Backfill tracking if needed
        await self._ensure_tracking_populated()
        
        # Phase 2: Process in batches
        offset = start_offset
        last_checkpoint = offset
        
        while self.running and offset < total_vectors:
            batch_start = time.time()
            
            # Get batch of pending extractions
            batch = await self._get_priority_batch(self.config.batch_size)
            
            if not batch:
                print("✅ No more pending extractions!")
                break
            
            # Process batch
            for record in batch:
                if not self.running:
                    break
                
                try:
                    facts = await self._extract_single(record)
                    self.stats["vectors_processed"] += 1
                    self.stats["facts_extracted"] += len(facts)
                except Exception as e:
                    self.stats["errors"] += 1
                    print(f"❌ Error: {e}")
            
            offset += len(batch)
            
            # Checkpoint
            if offset - last_checkpoint >= self.config.checkpoint_interval:
                await self._save_checkpoint(offset)
                last_checkpoint = offset
                self._print_progress(offset, total_vectors, batch_start)
        
        # Final checkpoint
        await self._save_checkpoint(offset)
        self._print_summary()
    
    async def _ensure_tracking_populated(self):
        """Ensure ingestion_tracking is populated from Qdrant."""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(f"""
                SELECT COUNT(*) FROM ingestion_tracking 
                WHERE qdrant_collection = $1
            """, self.config.collection)
        
        if count > 0:
            print(f"✅ Tracking table has {count:,} records for {self.config.collection}")
            return
        
        print(f"📥 Populating tracking from {self.config.collection}...")
        await self._backfill_tracking()
    
    async def _backfill_tracking(self):
        """Backfill tracking table from Qdrant."""
        offset = None
        inserted = 0
        
        while True:
            # Scroll Qdrant
            points, next_offset = self.qdrant.scroll(
                collection_name=self.config.collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
            
            # Batch insert
            async with self.pool.acquire() as conn:
                for point in points:
                    payload = point.payload or {}
                    content = payload.get("content", payload.get("text", ""))
                    
                    # Quick domain classification
                    domain = self._classify_domain(content)
                    
                    # Calculate priority
                    content_length = len(content) if content else 0
                    priority = self._calculate_priority(domain, content_length)
                    
                    source_hash = hashlib.sha256(
                        f"{self.config.collection}:{point.id}".encode()
                    ).hexdigest()
                    
                    try:
                        tracking_id = await conn.fetchval("""
                            INSERT INTO ingestion_tracking (
                                source_type, source_path, source_hash,
                                qdrant_collection, vector_id, vector_dimensions,
                                domain, content_length, priority_score,
                                vectorized_at
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                            ON CONFLICT (source_hash) DO UPDATE SET updated_at = NOW()
                            RETURNING id
                        """,
                            payload.get("source_type", "unknown"),
                            payload.get("source_path", payload.get("source", f"qdrant:{point.id}")),
                            source_hash,
                            self.config.collection,
                            str(point.id),
                            1024 if self.config.collection == "echo_memory" else 384,
                            domain,
                            content_length,
                            priority
                        )
                        
                        # Store content
                        if content and tracking_id:
                            await conn.execute("""
                                INSERT INTO vector_content (tracking_id, content, content_hash)
                                VALUES ($1, $2, $3)
                                ON CONFLICT (tracking_id, chunk_index) DO NOTHING
                            """,
                                tracking_id,
                                content,
                                hashlib.sha256(content.encode()).hexdigest()
                            )
                        
                        inserted += 1
                    except Exception as e:
                        print(f"Insert error: {e}")
            
            if inserted % 1000 == 0:
                print(f"  Backfilled {inserted:,} records...")
            
            offset = next_offset
            if offset is None:
                break
        
        print(f"✅ Backfilled {inserted:,} tracking records")
    
    def _classify_domain(self, content: str) -> str:
        """Fast local domain classification."""
        if not content:
            return "general"
        
        content_lower = content.lower()
        
        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            scores[domain] = score
        
        if max(scores.values()) == 0:
            return "general"
        
        return max(scores, key=scores.get)
    
    def _calculate_priority(self, domain: str, content_length: int) -> float:
        """Calculate extraction priority."""
        domain_weights = {
            "technical": 1.0,
            "code": 0.95,
            "personal": 0.8,
            "anime": 0.6,
            "general": 0.5
        }
        
        domain_score = domain_weights.get(domain, 0.5)
        length_score = min(content_length / 2000, 1.0)
        
        return domain_score * 0.6 + length_score * 0.4
    
    async def _get_priority_batch(self, limit: int) -> list[dict]:
        """Get batch of highest priority unextracted vectors."""
        domain_filter = ""
        if self.config.priority_only:
            domain_filter = "AND domain IN ('technical', 'code')"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT 
                    it.id,
                    it.vector_id,
                    it.domain,
                    it.content_length,
                    vc.content
                FROM ingestion_tracking it
                JOIN vector_content vc ON vc.tracking_id = it.id
                WHERE it.fact_extracted = FALSE
                AND it.qdrant_collection = $1
                AND it.content_length >= $2
                {domain_filter}
                ORDER BY it.priority_score DESC, it.created_at
                LIMIT $3
            """, self.config.collection, self.config.min_content_length, limit)
            
            return [dict(r) for r in rows]
    
    async def _extract_single(self, record: dict) -> list[dict]:
        """Extract facts from a single record."""
        content = record["content"]
        
        # Skip short content
        if len(content) < self.config.min_content_length:
            self.stats["skipped_short"] += 1
            await self._mark_extracted(record["id"], 0)
            return []
        
        # Extract facts using LLM
        prompt = EXTRACTION_PROMPT.format(content=content[:6000])  # Limit input
        
        try:
            response = await self.ollama.generate(
                model=self.config.extraction_model,
                prompt=prompt
            )
            
            # Parse response
            facts_data = self._parse_json(response["response"])
            
            # Store facts
            stored = 0
            for fd in facts_data:
                if await self._store_fact(record["id"], fd, record["domain"]):
                    stored += 1
            
            # Mark as extracted
            await self._mark_extracted(record["id"], stored)
            
            return facts_data
            
        except Exception as e:
            await self._mark_error(record["id"], str(e))
            raise
    
    def _parse_json(self, text: str) -> list[dict]:
        """Parse JSON from LLM response."""
        text = text.strip()
        
        # Remove markdown
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        
        # Find JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        
        if start >= 0 and end > start:
            text = text[start:end]
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return []
    
    async def _store_fact(self, source_id, fact_data: dict, domain: str) -> bool:
        """Store a single fact."""
        fact_text = fact_data.get("fact_text", "")
        if len(fact_text) < 10:
            return False
        
        # Generate embedding
        try:
            embedding = await self.ollama.embeddings(
                self.config.embedding_model_1024,
                fact_text
            )
        except Exception:
            embedding = None
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO facts (
                    id, source_embedding_id, subject, predicate, object, confidence,
                    qdrant_point_id, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
            """,
                uuid4(),
                source_id,
                fact_data.get("subject"),
                fact_data.get("predicate"),
                fact_data.get("object"),
                float(fact_data.get("confidence", 0.8)),
                f"point_{uuid4().hex[:8]}"
            )
        
        return True
    
    async def _mark_extracted(self, tracking_id, facts_count: int):
        """Mark a vector as having facts extracted."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE ingestion_tracking SET
                    fact_extracted = TRUE,
                    fact_extracted_at = NOW(),
                    facts_count = $2,
                    updated_at = NOW()
                WHERE id = $1
            """, tracking_id, facts_count)
    
    async def _mark_error(self, tracking_id, error: str):
        """Mark extraction error."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE ingestion_tracking SET
                    extraction_error = $2,
                    extraction_attempts = extraction_attempts + 1,
                    updated_at = NOW()
                WHERE id = $1
            """, tracking_id, error[:500])
    
    async def _get_checkpoint(self) -> Optional[dict]:
        """Get last checkpoint."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM extraction_checkpoints
                WHERE collection_name = $1 AND checkpoint_type = 'batch'
            """, self.config.collection)
            return dict(row) if row else None
    
    async def _save_checkpoint(self, offset: int):
        """Save checkpoint."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO extraction_checkpoints (
                    collection_name, checkpoint_type, last_offset,
                    vectors_processed, facts_extracted, errors,
                    model_used, batch_size, last_checkpoint_at
                ) VALUES ($1, 'batch', $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (collection_name, checkpoint_type) DO UPDATE SET
                    last_offset = $2,
                    vectors_processed = $3,
                    facts_extracted = $4,
                    errors = $5,
                    last_checkpoint_at = NOW()
            """,
                self.config.collection,
                offset,
                self.stats["vectors_processed"],
                self.stats["facts_extracted"],
                self.stats["errors"],
                self.config.extraction_model,
                self.config.batch_size
            )
    
    def _print_progress(self, current: int, total: int, batch_start: float):
        """Print progress update."""
        elapsed = time.time() - batch_start
        pct = (current / total) * 100 if total > 0 else 0
        
        # Calculate ETA
        if self.stats["vectors_processed"] > 0:
            total_elapsed = (datetime.utcnow() - self.stats["started_at"]).total_seconds()
            rate = self.stats["vectors_processed"] / total_elapsed
            remaining = total - current
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "calculating..."
        
        print(
            f"[{current:,}/{total:,}] ({pct:.1f}%) | "
            f"Facts: {self.stats['facts_extracted']:,} | "
            f"Errors: {self.stats['errors']} | "
            f"ETA: {eta_str}"
        )
    
    def _print_summary(self):
        """Print final summary."""
        elapsed = datetime.utcnow() - self.stats["started_at"]
        
        print()
        print("=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Vectors processed: {self.stats['vectors_processed']:,}")
        print(f"Facts extracted: {self.stats['facts_extracted']:,}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Skipped (short): {self.stats['skipped_short']}")
        print(f"Elapsed time: {elapsed}")
        
        if self.stats["vectors_processed"] > 0:
            avg_facts = self.stats["facts_extracted"] / self.stats["vectors_processed"]
            rate = self.stats["vectors_processed"] / elapsed.total_seconds() * 60
            print(f"Average facts/vector: {avg_facts:.1f}")
            print(f"Rate: {rate:.1f} vectors/minute")


async def main():
    parser = argparse.ArgumentParser(description="Scalable fact extraction")
    parser.add_argument("--collection", default="echo_memory", help="Qdrant collection")
    parser.add_argument("--model", default="qwen2.5:7b", help="Extraction model")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--limit", type=int, help="Max vectors to process")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--priority-only", action="store_true", help="Only technical/code")
    parser.add_argument("--postgres", default="postgresql://patrick@localhost:5432/echo_brain")
    
    args = parser.parse_args()
    
    config = ExtractionConfig(
        postgres_dsn=args.postgres,
        collection=args.collection,
        extraction_model=args.model,
        batch_size=args.batch_size,
        max_vectors=args.limit,
        priority_only=args.priority_only
    )
    
    extractor = ScalableExtractor(config)
    
    try:
        await extractor.initialize()
        await extractor.run(resume=args.resume)
    finally:
        await extractor.close()


if __name__ == "__main__":
    asyncio.run(main())
