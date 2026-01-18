#!/usr/bin/env python3
"""
Extract structured facts from Echo Brain content.

Reads from: Qdrant conversations/documents collections
Extracts: (subject, predicate, object) triples
Stores: PostgreSQL facts table + Qdrant facts collection
"""
import asyncio
import sys
import json
import re
from datetime import datetime
from uuid import uuid4
from typing import Optional

sys.path.insert(0, '/opt/tower-echo-brain')

import asyncpg
import httpx
from src.services.embedding_service import create_embedding_service

# Config
DATABASE_URL = "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/tower_consolidated"
OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
EXTRACTION_MODEL = "gemma2:9b"  # Fast and good at following instructions

# Extraction prompt template
EXTRACTION_PROMPT = """Extract factual information from this text as structured triples.

Rules:
1. Only extract CONCRETE facts, not opinions or speculation
2. Focus on facts about: Patrick, Tower server, Echo Brain, RV/vehicle, family, projects
3. Each fact should be: (subject, predicate, object)
4. Be specific - include model numbers, dates, versions when mentioned
5. Return JSON array of objects with keys: subject, predicate, object

Text to analyze:
---
{text}
---

Return ONLY a JSON array, no explanation. Example format:
[
  {{"subject": "Patrick", "predicate": "drives", "object": "2022 Toyota Tundra 1794 Edition"}},
  {{"subject": "Tower server", "predicate": "has GPU", "object": "NVIDIA RTX 3060 12GB"}}
]

If no facts can be extracted, return: []

JSON:"""


async def extract_facts_from_text(text: str, client: httpx.AsyncClient) -> list[dict]:
    """Use LLM to extract facts from text."""
    try:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": EXTRACTION_MODEL,
                "prompt": EXTRACTION_PROMPT.format(text=text[:4000]),  # Truncate for context
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temp for consistent extraction
                    "num_predict": 1000
                }
            },
            timeout=60.0
        )

        result = response.json().get("response", "")

        # Parse JSON from response
        # Try to find JSON array in response
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            facts = json.loads(json_match.group())
            # Validate structure
            valid_facts = []
            for f in facts:
                if isinstance(f, dict) and all(k in f for k in ['subject', 'predicate', 'object']):
                    valid_facts.append(f)
            return valid_facts
        return []

    except Exception as e:
        print(f"  Extraction error: {e}")
        return []


async def get_content_samples(collection: str, limit: int = 100) -> list[dict]:
    """Get content samples from Qdrant collection."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{QDRANT_URL}/collections/{collection}/points/scroll",
            json={
                "limit": limit,
                "with_payload": True,
                "with_vector": False
            }
        )
        data = response.json()
        return data.get("result", {}).get("points", [])


async def store_fact(
    pool: asyncpg.Pool,
    embedding_service,
    fact: dict,
    source_collection: str,
    source_id: str
) -> bool:
    """Store a fact in PostgreSQL and Qdrant."""
    try:
        # Create fact text for embedding
        fact_text = f"{fact['subject']} {fact['predicate']} {fact['object']}"

        # Check for duplicate
        async with pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT id FROM facts WHERE subject = $1 AND predicate = $2 AND object = $3",
                fact['subject'], fact['predicate'], fact['object']
            )
            if existing:
                return False  # Skip duplicate

        # Generate embedding
        embedding = await embedding_service.embed_single(fact_text)

        # Store in PostgreSQL (use source_conversation_id for conversations)
        fact_id = str(uuid4())
        async with pool.acquire() as conn:
            if source_collection == "conversations":
                # Use source_conversation_id for conversation facts
                await conn.execute("""
                    INSERT INTO facts (id, subject, predicate, object, source_conversation_id, confidence, qdrant_point_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (subject, predicate, object) DO NOTHING
                """, fact_id, fact['subject'], fact['predicate'], fact['object'],
                    source_id, 0.8, fact_id)
            else:
                # For documents, check if the document exists first
                doc_exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM documents WHERE id::text = $1)",
                    source_id
                )
                if doc_exists:
                    await conn.execute("""
                        INSERT INTO facts (id, subject, predicate, object, source_document_id, confidence, qdrant_point_id)
                        VALUES ($1, $2, $3, $4, $5::uuid, $6, $7)
                        ON CONFLICT (subject, predicate, object) DO NOTHING
                    """, fact_id, fact['subject'], fact['predicate'], fact['object'],
                        source_id, 0.8, fact_id)
                else:
                    # Store without foreign key reference
                    await conn.execute("""
                        INSERT INTO facts (id, subject, predicate, object, confidence, qdrant_point_id)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (subject, predicate, object) DO NOTHING
                    """, fact_id, fact['subject'], fact['predicate'], fact['object'],
                        0.8, fact_id)

        # Store in Qdrant
        async with httpx.AsyncClient() as client:
            await client.put(
                f"{QDRANT_URL}/collections/facts/points",
                json={
                    "points": [{
                        "id": fact_id,
                        "vector": embedding,
                        "payload": {
                            "subject": fact['subject'],
                            "predicate": fact['predicate'],
                            "object": fact['object'],
                            "fact_text": fact_text,
                            "source_collection": source_collection,
                            "source_id": source_id
                        }
                    }]
                }
            )

        return True

    except Exception as e:
        print(f"  Store error: {e}")
        return False


async def main():
    print(f"=== Fact Extraction Started: {datetime.now().isoformat()} ===\n")

    # Initialize services
    embedding_service = await create_embedding_service()
    pool = await asyncpg.create_pool(DATABASE_URL)

    stats = {"processed": 0, "facts_found": 0, "facts_stored": 0, "duplicates": 0}

    async with httpx.AsyncClient(timeout=120.0) as client:

        # Process conversations (highest value for personal facts)
        print("=== Processing Conversations ===")
        conv_points = await get_content_samples("conversations", limit=200)
        print(f"Found {len(conv_points)} conversation chunks\n")

        for i, point in enumerate(conv_points):
            text = point.get("payload", {}).get("text_preview", "")
            if not text:
                text = point.get("payload", {}).get("text", "")
            if len(text) < 50:  # Skip very short content
                continue

            print(f"Processing conversation {i+1}/{len(conv_points)}...", end=" ")

            facts = await extract_facts_from_text(text, client)
            stats["processed"] += 1

            if facts:
                print(f"Found {len(facts)} facts")
                for fact in facts:
                    stats["facts_found"] += 1
                    stored = await store_fact(
                        pool, embedding_service, fact,
                        "conversations", str(point.get("id", ""))
                    )
                    if stored:
                        stats["facts_stored"] += 1
                        obj_preview = fact['object'][:30] + "..." if len(fact['object']) > 30 else fact['object']
                        print(f"    ✅ ({fact['subject']}, {fact['predicate']}, {obj_preview})")
                    else:
                        stats["duplicates"] += 1
            else:
                print("No facts")

            # Rate limit
            await asyncio.sleep(0.5)

        # Process documents (KB articles)
        print("\n=== Processing Documents ===")
        doc_points = await get_content_samples("documents", limit=100)
        print(f"Found {len(doc_points)} document chunks\n")

        for i, point in enumerate(doc_points):
            text = point.get("payload", {}).get("text", "")
            if len(text) < 100:
                continue

            print(f"Processing document {i+1}/{len(doc_points)}...", end=" ")

            facts = await extract_facts_from_text(text, client)
            stats["processed"] += 1

            if facts:
                print(f"Found {len(facts)} facts")
                for fact in facts:
                    stats["facts_found"] += 1
                    stored = await store_fact(
                        pool, embedding_service, fact,
                        "documents", str(point.get("id", ""))
                    )
                    if stored:
                        stats["facts_stored"] += 1
                        obj_preview = fact['object'][:30] + "..." if len(fact['object']) > 30 else fact['object']
                        print(f"    ✅ ({fact['subject']}, {fact['predicate']}, {obj_preview})")
                    else:
                        stats["duplicates"] += 1
            else:
                print("No facts")

            await asyncio.sleep(0.5)

    await pool.close()
    await embedding_service.close()

    # Summary
    print(f"\n{'='*60}")
    print("FACT EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Content processed: {stats['processed']}")
    print(f"Facts found: {stats['facts_found']}")
    print(f"Facts stored: {stats['facts_stored']}")
    print(f"Duplicates skipped: {stats['duplicates']}")

    # Verify
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{QDRANT_URL}/collections/facts")
        count = resp.json()["result"]["points_count"]
        print(f"\nQdrant facts collection: {count} vectors")

    # Show sample facts
    print("\n=== Sample Facts Stored ===")
    async with asyncpg.create_pool(DATABASE_URL) as pool:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT subject, predicate, object FROM facts ORDER BY created_at DESC LIMIT 10"
            )
            for row in rows:
                obj_preview = row['object'][:40] + "..." if len(row['object']) > 40 else row['object']
                print(f"  ({row['subject']}, {row['predicate']}, {obj_preview})")


if __name__ == "__main__":
    asyncio.run(main())