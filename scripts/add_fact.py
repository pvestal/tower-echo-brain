#!/usr/bin/env python3
"""
Manually add important facts that extraction might miss.

Usage:
  python add_fact.py "Patrick" "drives" "2022 Toyota Tundra 1794 Edition"
"""
import asyncio
import sys
from uuid import uuid4

sys.path.insert(0, '/opt/tower-echo-brain')

import asyncpg
import httpx
from src.services.embedding_service import create_embedding_service

DATABASE_URL = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain"
QDRANT_URL = "http://localhost:6333"

async def add_fact(subject: str, predicate: str, obj: str):
    print(f"\nAdding: ({subject}, {predicate}, {obj})")

    pool = await asyncpg.create_pool(DATABASE_URL)
    embedding_service = await create_embedding_service()

    # Check duplicate
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT 1 FROM facts WHERE subject=$1 AND predicate=$2 AND object=$3",
            subject, predicate, obj
        )
        if exists:
            print("  ⚠️ Already exists")
            await pool.close()
            return

    # Embed
    fact_text = f"{subject} {predicate} {obj}"
    embedding = await embedding_service.embed_single(fact_text)

    # Store PostgreSQL
    fact_id = str(uuid4())
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO facts (id, subject, predicate, object, confidence, qdrant_point_id)
            VALUES ($1, $2, $3, $4, 1.0, $5)
        """, fact_id, subject, predicate, obj, fact_id)

    # Store Qdrant
    async with httpx.AsyncClient() as client:
        await client.put(
            f"{QDRANT_URL}/collections/facts/points",
            json={"points": [{
                "id": fact_id,
                "vector": embedding,
                "payload": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "fact_text": fact_text,
                    "source_collection": "manual",
                    "confidence": 1.0
                }
            }]}
        )

    await pool.close()
    await embedding_service.close()
    print(f"  ✅ Added: {fact_id}")

async def main():
    if len(sys.argv) == 4:
        await add_fact(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "--core":
        # Add core facts about Patrick
        core_facts = [
            ("Patrick", "drives", "2022 Toyota Tundra 1794 Edition"),
            ("Patrick", "owns RV", "2021 Sundowner Trailblazer toy hauler"),
            ("Tower server", "has CPU", "AMD Ryzen 9 24-core"),
            ("Tower server", "has RAM", "96GB DDR5"),
            ("Tower server", "has GPU", "NVIDIA RTX 3060 12GB"),
            ("Tower server", "has GPU", "AMD RX 9070 16GB"),
            ("Echo Brain", "runs on port", "8309"),
            ("Echo Brain", "uses embeddings", "OpenAI text-embedding-3-small 1536D"),
            ("Echo Brain", "database", "echo_brain"),
            ("Qdrant", "runs on port", "6333"),
        ]
        for s, p, o in core_facts:
            await add_fact(s, p, o)
    else:
        print("Usage:")
        print('  python add_fact.py "subject" "predicate" "object"')
        print('  python add_fact.py --core  # Add core facts about Patrick')

if __name__ == "__main__":
    asyncio.run(main())