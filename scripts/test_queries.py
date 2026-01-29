#!/usr/bin/env python3
"""Test Echo Brain with real queries."""
import asyncio
import sys
sys.path.insert(0, '/opt/tower-echo-brain')

from src.services.vector_search import get_vector_search

QUERIES = [
    # Personal knowledge
    ("What truck does Patrick drive?", ["documents", "conversations", "facts"]),
    ("RV electrical system Victron", ["documents", "conversations"]),
    ("anime generation workflow", ["documents", "code"]),

    # Code search
    ("embedding service OpenAI", ["code"]),
    ("Qdrant collection create", ["code"]),
    ("FastAPI router endpoint", ["code"]),

    # Conversation history
    ("Echo Brain architecture discussion", ["conversations"]),
    ("what we decided about embeddings", ["conversations"]),

    # KB articles
    ("ComfyUI workflow", ["documents"]),
    ("Tower server setup", ["documents"]),
]

async def test_queries():
    vs = await get_vector_search()

    print("=" * 70)
    print("ECHO BRAIN REAL QUERY TESTS")
    print("=" * 70)

    for query, collections in QUERIES:
        print(f"\n🔍 Query: \"{query}\"")
        print(f"   Collections: {collections}")

        try:
            result = await vs.search(query, collections=collections, limit=3)

            if result.results:
                print(f"   ✅ Found {len(result.results)} results:")
                for r in result.results:
                    title = r.payload.get('title') or r.payload.get('entity_name') or r.payload.get('file', 'N/A')
                    print(f"      {r.score:.3f} | {r.collection:15} | {str(title)[:40]}")
            else:
                print("   ⚠️  No results found")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Summary stats
    print("\n" + "=" * 70)
    print("COLLECTION STATS")
    print("=" * 70)

    import httpx
    async with httpx.AsyncClient() as client:
        for coll in ["documents", "conversations", "facts", "code"]:
            try:
                resp = await client.get(f"http://localhost:6333/collections/{coll}")
                count = resp.json()["result"]["points_count"]
                print(f"  {coll:15}: {count:,} vectors")
            except:
                print(f"  {coll:15}: ERROR")

if __name__ == "__main__":
    asyncio.run(test_queries())