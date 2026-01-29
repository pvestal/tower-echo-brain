"""
Fact Extractor

Extracts structured facts from vector content using LLM.
This is the critical component for going from 6% to 100% fact coverage.

Fact types:
- ENTITY: A thing that exists (person, service, tool, file)
- RELATIONSHIP: How things relate to each other
- EVENT: Something that happened (install, fix, decision)
- PREFERENCE: User preferences/settings
- TECHNICAL: Technical facts (configs, versions, architectures)
- TEMPORAL: Time-bound facts (deadlines, schedules)
"""

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import asyncpg

from ..context_assembly.models import Domain, FactType, Fact, SourceType


# Extraction prompt template
EXTRACTION_PROMPT = """Extract all factual information from the following content. 
Return a JSON array of facts, where each fact has:
- fact_text: A clear, standalone statement of the fact
- fact_type: One of [entity, relationship, event, preference, technical, temporal]
- subject: What/who the fact is about
- predicate: The relationship or action
- object: The target or value
- confidence: 0.0-1.0 how certain this fact is

Focus on:
1. Technical configurations and settings
2. Decisions that were made and why
3. Problems encountered and solutions
4. Relationships between systems/components
5. User preferences and requirements
6. Time-sensitive information (versions, dates)

Content domain: {domain}

Content:
---
{content}
---

Return ONLY valid JSON array. No explanation or markdown."""


DOMAIN_CLASSIFICATION_PROMPT = """Classify this content into exactly one domain:
- technical: Code, servers, databases, APIs, debugging, infrastructure
- anime: LoRA, ComfyUI, image/video generation, creative projects
- personal: RV, vehicles, personal items, schedule, preferences
- general: Everything else

Content:
---
{content}
---

Respond with only the domain name, nothing else."""


class FactExtractor:
    """
    Extracts structured facts from content using LLM.
    
    Optimized for:
    - Batch processing of large vector collections
    - RTX 3060 12GB VRAM constraint (uses smaller models)
    - Resumable extraction (tracks progress in database)
    """
    
    def __init__(
        self,
        ollama_client,
        postgres_dsn: str,
        extraction_model: str = "qwen2.5:14b",  # Good extraction, fits in 12GB Q4
        classification_model: str = "qwen2.5:7b",  # Fast classification
        batch_size: int = 10,
        max_retries: int = 3
    ):
        self.ollama = ollama_client
        self.postgres_dsn = postgres_dsn
        self.extraction_model = extraction_model
        self.classification_model = classification_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize database connection pool."""
        self._pool = await asyncpg.create_pool(
            self.postgres_dsn,
            min_size=2,
            max_size=10
        )
    
    async def close(self):
        """Clean up resources."""
        if self._pool:
            await self._pool.close()
    
    async def extract_all_pending(self, progress_callback=None) -> dict:
        """
        Extract facts from all vectors that haven't been processed.
        
        This is the main entry point for backfill operations.
        
        Args:
            progress_callback: Optional async function(processed, total, current_source)
            
        Returns:
            Summary statistics
        """
        stats = {
            "total_processed": 0,
            "facts_extracted": 0,
            "errors": 0,
            "start_time": datetime.utcnow().isoformat()
        }
        
        # Get count of pending
        total_pending = await self._count_pending()
        print(f"Found {total_pending} vectors pending fact extraction")
        
        while True:
            # Get batch of pending vectors
            batch = await self._get_pending_batch()
            
            if not batch:
                break  # All done
            
            for record in batch:
                try:
                    # Extract facts from this vector
                    facts = await self.extract_from_content(
                        content=record["content"],
                        source_id=record["id"],
                        source_type=SourceType(record["source_type"]),
                        source_path=record["source_path"]
                    )
                    
                    # Store facts
                    await self._store_facts(facts, record["id"])
                    
                    # Mark as processed
                    await self._mark_processed(record["id"], len(facts))
                    
                    stats["total_processed"] += 1
                    stats["facts_extracted"] += len(facts)
                    
                    if progress_callback:
                        await progress_callback(
                            stats["total_processed"],
                            total_pending,
                            record["source_path"]
                        )
                    
                except Exception as e:
                    print(f"Error extracting from {record['source_path']}: {e}")
                    stats["errors"] += 1
                    # Mark as failed but continue
                    await self._mark_failed(record["id"], str(e))
        
        stats["end_time"] = datetime.utcnow().isoformat()
        return stats
    
    async def extract_from_content(
        self,
        content: str,
        source_id: UUID,
        source_type: SourceType,
        source_path: str
    ) -> list[Fact]:
        """
        Extract facts from a single piece of content.
        
        Args:
            content: The text content to extract from
            source_id: ID of the source in ingestion_tracking
            source_type: Type of source (document, code, etc.)
            source_path: Path to the original source
            
        Returns:
            List of extracted facts
        """
        # First, classify the domain
        domain = await self._classify_domain(content)
        
        # Extract facts using LLM
        prompt = EXTRACTION_PROMPT.format(
            domain=domain.value,
            content=content[:8000]  # Limit content size
        )
        
        for attempt in range(self.max_retries):
            try:
                response = await self.ollama.generate(
                    model=self.extraction_model,
                    prompt=prompt,
                    options={
                        "temperature": 0.1,
                        "num_predict": 2000
                    }
                )
                
                # Parse JSON response
                facts_data = self._parse_json_response(response["response"])
                
                # Convert to Fact objects
                facts = []
                for fd in facts_data:
                    fact = Fact(
                        id=uuid4(),
                        source_id=source_id,
                        fact_text=fd.get("fact_text", ""),
                        fact_type=self._parse_fact_type(fd.get("fact_type", "entity")),
                        confidence=float(fd.get("confidence", 0.8)),
                        domain=domain,
                        subject=fd.get("subject"),
                        predicate=fd.get("predicate"),
                        object=fd.get("object")
                    )
                    
                    # Validate fact has substance
                    if len(fact.fact_text) > 10:
                        facts.append(fact)
                
                return facts
                
            except json.JSONDecodeError as e:
                print(f"JSON parse error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return []  # Give up after retries
            except Exception as e:
                print(f"Extraction error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        return []
    
    async def _classify_domain(self, content: str) -> Domain:
        """Classify content into a domain."""
        prompt = DOMAIN_CLASSIFICATION_PROMPT.format(content=content[:2000])
        
        try:
            response = await self.ollama.generate(
                model=self.classification_model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 20}
            )
            
            domain_text = response["response"].strip().lower()
            
            domain_map = {
                "technical": Domain.TECHNICAL,
                "anime": Domain.ANIME,
                "personal": Domain.PERSONAL,
                "general": Domain.GENERAL
            }
            
            return domain_map.get(domain_text, Domain.GENERAL)
        except Exception:
            return Domain.GENERAL
    
    def _parse_json_response(self, response: str) -> list[dict]:
        """Parse JSON from LLM response, handling common issues."""
        # Clean up response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        # Try to find JSON array
        start_idx = response.find("[")
        end_idx = response.rfind("]") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            response = response[start_idx:end_idx]
        
        return json.loads(response)
    
    def _parse_fact_type(self, type_str: str) -> FactType:
        """Parse fact type string to enum."""
        type_map = {
            "entity": FactType.ENTITY,
            "relationship": FactType.RELATIONSHIP,
            "event": FactType.EVENT,
            "preference": FactType.PREFERENCE,
            "technical": FactType.TECHNICAL,
            "temporal": FactType.TEMPORAL
        }
        return type_map.get(type_str.lower(), FactType.ENTITY)
    
    async def _count_pending(self) -> int:
        """Count vectors pending fact extraction."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT COUNT(*) as count 
                FROM ingestion_tracking 
                WHERE fact_extracted = FALSE
                AND vector_id IS NOT NULL
            """)
            return row["count"]
    
    async def _get_pending_batch(self) -> list[dict]:
        """Get a batch of vectors pending extraction."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    it.id,
                    it.source_type,
                    it.source_path,
                    it.domain,
                    vc.content
                FROM ingestion_tracking it
                JOIN vector_content vc ON vc.tracking_id = it.id
                WHERE it.fact_extracted = FALSE
                AND it.vector_id IS NOT NULL
                ORDER BY it.created_at
                LIMIT $1
            """, self.batch_size)
            
            return [dict(row) for row in rows]
    
    async def _store_facts(self, facts: list[Fact], source_id: UUID):
        """Store extracted facts in database."""
        if not facts:
            return
        
        async with self._pool.acquire() as conn:
            # Generate embeddings for facts (for semantic search)
            for fact in facts:
                try:
                    embedding_response = await self.ollama.embeddings(
                        model="nomic-embed-text",
                        prompt=fact.fact_text
                    )
                    fact.embedding = embedding_response["embedding"]
                except Exception as e:
                    print(f"Embedding error for fact: {e}")
                    fact.embedding = None
            
            # Batch insert
            await conn.executemany("""
                INSERT INTO facts (
                    id, source_id, fact_text, fact_type, confidence,
                    domain, subject, predicate, object, embedding, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                )
            """, [
                (
                    fact.id,
                    fact.source_id,
                    fact.fact_text,
                    fact.fact_type.value,
                    fact.confidence,
                    fact.domain.value,
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    fact.embedding,
                    datetime.utcnow()
                )
                for fact in facts
            ])
    
    async def _mark_processed(self, source_id: UUID, facts_count: int):
        """Mark a source as having facts extracted."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE ingestion_tracking
                SET 
                    fact_extracted = TRUE,
                    fact_extracted_at = NOW(),
                    facts_count = $2,
                    updated_at = NOW()
                WHERE id = $1
            """, source_id, facts_count)
    
    async def _mark_failed(self, source_id: UUID, error: str):
        """Mark a source as failed extraction."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE ingestion_tracking
                SET 
                    extraction_error = $2,
                    updated_at = NOW()
                WHERE id = $1
            """, source_id, error[:500])


# ============================================================================
# Batch extraction script
# ============================================================================

async def run_full_extraction(
    postgres_dsn: str,
    ollama_host: str = "http://localhost:11434"
):
    """
    Run full fact extraction on all pending vectors.
    
    This is the main script for going from 6% to 100% coverage.
    """
    import httpx
    
    # Simple Ollama client
    class OllamaClient:
        def __init__(self, host: str):
            self.host = host
            self.client = httpx.AsyncClient(timeout=120.0)
        
        async def generate(self, model: str, prompt: str, options: dict = None):
            response = await self.client.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": prompt, "options": options or {}, "stream": False}
            )
            return response.json()
        
        async def embeddings(self, model: str, prompt: str):
            response = await self.client.post(
                f"{self.host}/api/embeddings",
                json={"model": model, "prompt": prompt}
            )
            return response.json()
    
    ollama = OllamaClient(ollama_host)
    extractor = FactExtractor(ollama, postgres_dsn)
    
    await extractor.initialize()
    
    async def progress(processed, total, source):
        pct = (processed / total) * 100 if total > 0 else 0
        print(f"[{processed}/{total}] ({pct:.1f}%) - {source}")
    
    try:
        stats = await extractor.extract_all_pending(progress_callback=progress)
        print("\n" + "=" * 50)
        print("EXTRACTION COMPLETE")
        print("=" * 50)
        print(f"Processed: {stats['total_processed']}")
        print(f"Facts extracted: {stats['facts_extracted']}")
        print(f"Errors: {stats['errors']}")
        print(f"Duration: {stats['start_time']} to {stats['end_time']}")
    finally:
        await extractor.close()


if __name__ == "__main__":
    import sys
    
    dsn = sys.argv[1] if len(sys.argv) > 1 else "postgresql://localhost/echo_brain"
    asyncio.run(run_full_extraction(dsn))
