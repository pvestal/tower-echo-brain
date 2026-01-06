#!/usr/bin/env python3
"""
COMPLETE Fact Extraction - Processes ALL Qdrant vectors.

Features:
- Processes ALL vectors, not samples
- Tracks progress in PostgreSQL
- Resumes from where it left off
- Uses optimized models per content type
- Handles errors gracefully
"""
import asyncio
import sys
import json
import hashlib
import re
from datetime import datetime
from uuid import uuid4
from typing import List, Dict, Optional

sys.path.insert(0, '/opt/tower-echo-brain')

import asyncpg
import httpx
from src.services.embedding_service import create_embedding_service

# Config
DATABASE_URL = "postgresql://patrick:***REMOVED***@localhost/tower_consolidated"
OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"

# Models optimized for each content type
MODELS = {
    "conversations": "gemma2:9b",
    "documents": "gemma2:9b",
    "code": "deepseek-coder-v2:16b",
    "facts": None  # Skip - already facts
}

# Extraction prompts tuned for maximum fact yield
EXTRACTION_PROMPT = """You are a fact extraction system. Extract ALL concrete facts from this text.

A fact is a triple: (subject, predicate, object)
- Subject: A person, system, device, project, or thing
- Predicate: A relationship or action (drives, owns, has, uses, runs, is, contains, etc.)
- Object: The specific value (include model numbers, versions, dates, names)

IMPORTANT: Extract EVERYTHING specific, including:
- Vehicle info: make, model, year, edition (e.g., "2022 Toyota Tundra 1794 Edition")
- Hardware specs: CPU, RAM, GPU, storage with exact specs
- Software: versions, ports, configurations
- People: names, roles, relationships
- Locations: addresses, paths, URLs
- Dates: when things happened or were created
- Projects: names, status, technologies used
- Preferences: likes, dislikes, choices made

Text to analyze:
---
{text}
---

Return ONLY a JSON array of objects. Each object must have exactly these keys:
- "subject": string (the thing being described)
- "predicate": string (the relationship/action)
- "object": string (the specific value/detail)

If the text contains no extractable facts, return: []

Examples of good extractions:
[
  {{"subject": "Patrick", "predicate": "drives", "object": "2022 Toyota Tundra 1794 Edition"}},
  {{"subject": "Patrick", "predicate": "owns RV", "object": "2021 Sundowner Trailblazer toy hauler"}},
  {{"subject": "Tower server", "predicate": "has CPU", "object": "AMD Ryzen 9 7950X 24-core"}},
  {{"subject": "Tower server", "predicate": "has RAM", "object": "96GB DDR5"}},
  {{"subject": "Echo Brain", "predicate": "runs on port", "object": "8309"}},
  {{"subject": "Qdrant", "predicate": "uses dimensions", "object": "1536"}}
]

JSON array:"""


class CompleteFactExtractor:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.embedding_service = None
        self.client: Optional[httpx.AsyncClient] = None
        self.stats = {
            "total_vectors": 0,
            "processed": 0,
            "skipped_already_done": 0,
            "skipped_no_content": 0,
            "facts_extracted": 0,
            "facts_stored": 0,
            "duplicates": 0,
            "errors": 0
        }

    async def init(self):
        self.pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        self.embedding_service = await create_embedding_service()
        self.client = httpx.AsyncClient(timeout=180.0)
        print("✓ Services initialized")

    async def close(self):
        if self.client:
            await self.client.aclose()
        if self.pool:
            await self.pool.close()

    async def get_all_points(self, collection: str) -> List[Dict]:
        """Fetch ALL points from a Qdrant collection with pagination."""
        all_points = []
        offset = None

        while True:
            body = {"limit": 100, "with_payload": True, "with_vector": False}
            if offset:
                body["offset"] = offset

            resp = await self.client.post(
                f"{QDRANT_URL}/collections/{collection}/points/scroll",
                json=body
            )
            data = resp.json()
            points = data.get("result", {}).get("points", [])

            if not points:
                break

            all_points.extend(points)
            offset = data.get("result", {}).get("next_page_offset")

            if not offset:
                break

        return all_points

    async def is_processed(self, collection: str, point_id: str, content_hash: str) -> bool:
        """Check if already processed (and content hasn't changed)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT content_hash, status FROM fact_extraction_log
                WHERE source_collection = $1 AND source_point_id = $2
            """, collection, str(point_id))

            if not row:
                return False
            if row['status'] == 'failed':
                return False  # Retry failures
            if row['content_hash'] != content_hash:
                return False  # Content changed
            return True

    async def log_extraction(self, collection: str, point_id: str, content_hash: str,
                            model: str, facts_count: int, status: str, error: str = None):
        """Record extraction attempt."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO fact_extraction_log
                (source_collection, source_point_id, content_hash, extraction_model, facts_extracted, status, error_message)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (source_collection, source_point_id) DO UPDATE SET
                    content_hash = $3, extraction_model = $4, facts_extracted = $5,
                    status = $6, error_message = $7, processed_at = NOW()
            """, collection, str(point_id), content_hash, model, facts_count, status, error)

    async def extract_with_llm(self, text: str, model: str) -> List[Dict]:
        """Call LLM to extract facts."""
        try:
            prompt = EXTRACTION_PROMPT.format(text=text[:8000])

            resp = await self.client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 4000}
                }
            )

            result = resp.json().get("response", "")

            # Find JSON array in response
            match = re.search(r'\[[\s\S]*\]', result)
            if match:
                try:
                    facts = json.loads(match.group())
                    # Validate structure
                    valid = []
                    for f in facts:
                        if isinstance(f, dict) and all(k in f for k in ['subject', 'predicate', 'object']):
                            if f['subject'] and f['predicate'] and f['object']:
                                if len(str(f['object'])) > 1:  # Skip empty/trivial
                                    valid.append(f)
                    return valid
                except json.JSONDecodeError:
                    pass
            return []
        except Exception as e:
            print(f"      LLM error: {e}")
            return []

    async def store_fact(self, fact: Dict, collection: str, point_id: str) -> bool:
        """Store fact in PostgreSQL and Qdrant."""
        try:
            subject = str(fact['subject'])[:200]
            predicate = str(fact['predicate'])[:200]
            obj = str(fact['object'])[:500]

            # Check duplicate
            async with self.pool.acquire() as conn:
                exists = await conn.fetchval(
                    "SELECT 1 FROM facts WHERE subject=$1 AND predicate=$2 AND object=$3",
                    subject, predicate, obj
                )
                if exists:
                    self.stats["duplicates"] += 1
                    return False

            # Create embedding
            fact_text = f"{subject} {predicate} {obj}"
            embedding = await self.embedding_service.embed_single(fact_text)

            # Store in PostgreSQL (without foreign key for conversations)
            fact_id = str(uuid4())
            async with self.pool.acquire() as conn:
                if collection == "conversations":
                    await conn.execute("""
                        INSERT INTO facts (id, subject, predicate, object, confidence, qdrant_point_id)
                        VALUES ($1, $2, $3, $4, 0.85, $5)
                    """, fact_id, subject, predicate, obj, fact_id)
                else:
                    await conn.execute("""
                        INSERT INTO facts (id, subject, predicate, object, source_document_id, confidence, qdrant_point_id)
                        VALUES ($1, $2, $3, $4, $5, 0.85, $6)
                    """, fact_id, subject, predicate, obj, str(point_id), fact_id)

            # Store in Qdrant
            await self.client.put(
                f"{QDRANT_URL}/collections/facts/points",
                json={
                    "points": [{
                        "id": fact_id,
                        "vector": embedding,
                        "payload": {
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "fact_text": fact_text,
                            "source_collection": collection,
                            "source_point_id": str(point_id)
                        }
                    }]
                }
            )

            self.stats["facts_stored"] += 1
            return True
        except Exception as e:
            print(f"      Store error: {e}")
            return False

    async def process_collection(self, collection: str):
        """Process all vectors in a collection."""
        model = MODELS.get(collection)
        if not model:
            print(f"\n⏭️  Skipping {collection} (no model configured)")
            return

        print(f"\n{'='*60}")
        print(f"📚 COLLECTION: {collection}")
        print(f"   Model: {model}")
        print(f"{'='*60}")

        points = await self.get_all_points(collection)
        total = len(points)
        print(f"   Total vectors: {total}")
        self.stats["total_vectors"] += total

        for i, point in enumerate(points):
            point_id = point.get("id", "")
            payload = point.get("payload", {})

            # Get text content
            text = payload.get("text", "") or payload.get("content", "") or ""
            if len(text) < 30:
                self.stats["skipped_no_content"] += 1
                continue

            content_hash = hashlib.sha256(text.encode()).hexdigest()

            # Check if already done
            if await self.is_processed(collection, point_id, content_hash):
                self.stats["skipped_already_done"] += 1
                continue

            # Progress
            if (i + 1) % 25 == 0 or i == 0:
                print(f"   [{i+1}/{total}] Processed: {self.stats['processed']} | Facts: {self.stats['facts_stored']} | Skipped: {self.stats['skipped_already_done']}")

            # Extract facts
            try:
                facts = await self.extract_with_llm(text, model)
                self.stats["processed"] += 1
                self.stats["facts_extracted"] += len(facts)

                # Store facts
                stored = 0
                for fact in facts:
                    if await self.store_fact(fact, collection, point_id):
                        stored += 1

                # Log success
                await self.log_extraction(collection, point_id, content_hash, model, len(facts), "completed")

                # Brief pause to avoid overwhelming Ollama
                await asyncio.sleep(0.2)

            except Exception as e:
                self.stats["errors"] += 1
                await self.log_extraction(collection, point_id, content_hash, model, 0, "failed", str(e))

        print(f"   ✓ {collection} complete")

    async def run(self, collections: List[str] = None):
        """Run extraction on all collections."""
        collections = collections or ["conversations", "documents", "code"]

        print(f"\n{'#'*60}")
        print(f"# COMPLETE FACT EXTRACTION")
        print(f"# Started: {datetime.now().isoformat()}")
        print(f"# Collections: {collections}")
        print(f"{'#'*60}")

        await self.init()

        start_time = datetime.now()

        for collection in collections:
            await self.process_collection(collection)

        await self.close()

        # Final report
        duration = (datetime.now() - start_time).total_seconds() / 60

        print(f"\n{'#'*60}")
        print(f"# EXTRACTION COMPLETE")
        print(f"# Duration: {duration:.1f} minutes")
        print(f"{'#'*60}")
        print(f"Total vectors scanned: {self.stats['total_vectors']}")
        print(f"Processed this run: {self.stats['processed']}")
        print(f"Skipped (already done): {self.stats['skipped_already_done']}")
        print(f"Skipped (no content): {self.stats['skipped_no_content']}")
        print(f"Facts extracted: {self.stats['facts_extracted']}")
        print(f"Facts stored (new): {self.stats['facts_stored']}")
        print(f"Duplicates skipped: {self.stats['duplicates']}")
        print(f"Errors: {self.stats['errors']}")

        # Show coverage
        print(f"\n=== Coverage by Collection ===")
        pool = await asyncpg.create_pool(DATABASE_URL)
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM extraction_coverage ORDER BY source_collection")
            for row in rows:
                print(f"  {row['source_collection']}: {row['processed']} processed, {row['facts_found']} facts")
        await pool.close()

        # Final facts count
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{QDRANT_URL}/collections/facts")
            total_facts = resp.json()["result"]["points_count"]
            print(f"\n📊 TOTAL FACTS IN SYSTEM: {total_facts}")


async def main():
    collections = sys.argv[1:] if len(sys.argv) > 1 else None
    extractor = CompleteFactExtractor()
    await extractor.run(collections)

if __name__ == "__main__":
    asyncio.run(main())