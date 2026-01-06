#!/usr/bin/env python3
"""Query facts about a subject."""
import asyncio
import sys
sys.path.insert(0, '/opt/tower-echo-brain')

import asyncpg
import httpx
from src.services.embedding_service import create_embedding_service

DATABASE_URL = "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/tower_consolidated"
QDRANT_URL = "http://localhost:6333"

async def query_facts(query: str):
    """Find facts related to a query."""
    print(f"\n🔍 Query: '{query}'\n")

    # Method 1: Direct PostgreSQL search
    print("=== PostgreSQL Facts (exact match) ===")
    pool = await asyncpg.create_pool(DATABASE_URL)
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT subject, predicate, object
            FROM facts
            WHERE subject ILIKE $1 OR object ILIKE $1 OR predicate ILIKE $1
            ORDER BY created_at DESC
            LIMIT 10
        """, f"%{query}%")

        if rows:
            for row in rows:
                obj_preview = row['object'][:60] + "..." if len(row['object']) > 60 else row['object']
                print(f"  • {row['subject']} → {row['predicate']} → {obj_preview}")
        else:
            print("  No exact matches")
    await pool.close()

    # Method 2: Semantic search via Qdrant
    print("\n=== Qdrant Facts (semantic search) ===")
    embedding_service = await create_embedding_service()
    query_vector = await embedding_service.embed_single(query)
    await embedding_service.close()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{QDRANT_URL}/collections/facts/points/search",
            json={
                "vector": query_vector,
                "limit": 10,
                "with_payload": True
            }
        )
        results = response.json().get("result", [])

        if results:
            for r in results:
                score = r.get("score", 0)
                p = r.get("payload", {})
                obj_preview = p.get('object', '')[:40] + "..." if len(p.get('object', '')) > 40 else p.get('object', '')
                print(f"  {score:.3f} | {p.get('subject')} → {p.get('predicate')} → {obj_preview}")
        else:
            print("  No semantic matches")

    # Count total facts
    pool = await asyncpg.create_pool(DATABASE_URL)
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM facts")
    await pool.close()
    print(f"\n📊 Total facts in database: {total}")

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Patrick"
    asyncio.run(query_facts(query))