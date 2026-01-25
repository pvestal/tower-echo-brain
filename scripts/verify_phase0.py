#!/usr/bin/env python3
"""
Verify Phase 0 completion.
Checks all services, databases, and data sources.
"""
import asyncio
import httpx
import sys
sys.path.insert(0, '/opt/tower-echo-brain')

async def main():
    print("=" * 60)
    print("PHASE 0 VERIFICATION")
    print("=" * 60)

    results = []

    # Test 1: OpenAI embeddings
    print("\n[1/7] Testing OpenAI embeddings...")
    try:
        from src.services.embedding_service import create_embedding_service
        service = await create_embedding_service()
        emb = await service.embed_single("test")
        await service.close()
        if len(emb) == 1536:
            print("  ✅ OpenAI embeddings: 1536 dimensions")
            results.append(True)
        else:
            print(f"  ❌ Wrong dimensions: {len(emb)}")
            results.append(False)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append(False)

    # Test 2: Qdrant collections exist with correct dimensions
    print("\n[2/7] Testing Qdrant collections...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:6333/collections")
            collections = [c["name"] for c in resp.json()["result"]["collections"]]
            expected = ["documents", "conversations", "facts", "code"]

            all_correct = True
            for coll in expected:
                if coll not in collections:
                    print(f"  ❌ Missing collection: {coll}")
                    all_correct = False
                else:
                    # Check dimensions
                    resp = await client.get(f"http://localhost:6333/collections/{coll}")
                    size = resp.json()["result"]["config"]["params"]["vectors"]["size"]
                    if size != 1536:
                        print(f"  ❌ {coll}: wrong dimensions ({size} instead of 1536)")
                        all_correct = False
                    else:
                        count = resp.json()["result"]["points_count"]
                        print(f"  ✅ {coll}: 1536D, {count} vectors")

            results.append(all_correct)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append(False)

    # Test 3: PostgreSQL schema in tower_consolidated
    print("\n[3/7] Testing PostgreSQL schema...")
    try:
        import asyncpg
        conn = await asyncpg.connect(
            "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain"
        )
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        table_names = [t["tablename"] for t in tables]
        expected = ["documents", "document_chunks", "facts", "calendar_events"]

        missing = [t for t in expected if t not in table_names]
        if missing:
            print(f"  ❌ Missing tables: {missing}")
            results.append(False)
        else:
            print(f"  ✅ All required tables exist")
            results.append(True)
        await conn.close()
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append(False)

    # Test 4: KB articles accessible from knowledge_base
    print("\n[4/7] Testing KB articles source...")
    try:
        import asyncpg
        conn = await asyncpg.connect(
            "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/knowledge_base"
        )
        count = await conn.fetchval("SELECT COUNT(*) FROM articles")
        print(f"  ✅ knowledge_base.articles: {count} articles")
        results.append(count > 0)
        await conn.close()
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append(False)

    # Test 5: KB articles imported to Qdrant
    print("\n[5/7] Testing KB articles in Qdrant...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:6333/collections/documents")
            count = resp.json()["result"]["points_count"]
            if count >= 400:
                print(f"  ✅ documents collection: {count} vectors (KB imported)")
                results.append(True)
            else:
                print(f"  ⚠️ documents collection: only {count} vectors (expected 400+)")
                results.append(False)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append(False)

    # Test 6: Embedding cache working
    print("\n[6/7] Testing embedding cache...")
    try:
        import asyncpg
        conn = await asyncpg.connect(
            "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain"
        )
        count = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
        print(f"  ✅ Embedding cache: {count} cached embeddings")
        results.append(count > 0)
        await conn.close()
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append(False)

    # Test 7: End-to-end search
    print("\n[7/7] Testing end-to-end search...")
    try:
        from src.services.vector_search import get_vector_search
        vs = await get_vector_search()
        result = await vs.search("Echo Brain architecture", limit=3)
        if result.results:
            print(f"  ✅ Search returned {len(result.results)} results")
            print(f"     Top result: {result.results[0].payload.get('title', 'N/A')[:50]}")
            results.append(True)
        else:
            print("  ⚠️ Search returned no results")
            results.append(False)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append(False)

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"PHASE 0 RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("✅ PHASE 0 COMPLETE - Ready for Phase 1")
    elif passed >= 5:
        print("⚠️ PHASE 0 MOSTLY COMPLETE - Review failing tests")
    else:
        print("❌ PHASE 0 INCOMPLETE - Fix failing tests before proceeding")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())