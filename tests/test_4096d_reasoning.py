#!/usr/bin/env python3
"""
Test 4096D Enhanced Reasoning Capabilities
Verifies that Echo Brain is using the upgraded collections
"""

import asyncio
import httpx
import json
from datetime import datetime
from qdrant_client import QdrantClient
import psycopg2

async def test_echo_brain_api():
    """Test Echo Brain API with 4096D reasoning."""
    print("\n1. Testing Echo Brain API...")

    test_queries = [
        "What improvements have been made to Echo Brain's memory system?",
        "Explain the benefits of 4096D embeddings versus 768D",
        "How many Tower files are mapped in the knowledge graph?"
    ]

    async with httpx.AsyncClient(timeout=30) as client:
        for query in test_queries:
            print(f"\n   Query: {query}")
            try:
                response = await client.post(
                    "http://localhost:8309/api/echo/query",
                    json={
                        "query": query,
                        "conversation_id": "4096d_test"
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✓ Response received (length: {len(data.get('response', ''))} chars)")
                else:
                    print(f"   ✗ Error: Status {response.status_code}")
            except Exception as e:
                print(f"   ✗ Error: {e}")

def test_qdrant_collections():
    """Test Qdrant 4096D collections."""
    print("\n2. Testing Qdrant 4096D Collections...")

    client = QdrantClient(host="localhost", port=6333)

    collections_4096d = [
        "claude_conversations_4096d",
        "unified_media_memory_4096d",
        "agent_memories_4096d",
        "learning_facts_4096d",
        "echo_real_knowledge_4096d"
    ]

    for collection in collections_4096d:
        try:
            info = client.get_collection(collection)
            vector_size = info.config.params.vectors.size
            points_count = info.points_count

            if vector_size == 4096:
                print(f"   ✓ {collection}: {points_count} points @ 4096D")
            else:
                print(f"   ✗ {collection}: Wrong dimensions ({vector_size}D)")
        except Exception as e:
            print(f"   ✗ {collection}: {e}")

def test_postgresql_config():
    """Test PostgreSQL 4096D configuration."""
    print("\n3. Testing PostgreSQL Configuration...")

    try:
        conn = psycopg2.connect(
            host="localhost",
            database="echo_brain",
            user="patrick",
            password="tower_echo_brain_secret_key_2025"
        )
        cursor = conn.cursor()

        # Check 4096D configuration
        cursor.execute("""
            SELECT value FROM echo_configuration
            WHERE key = '4096d_intelligence'
        """)
        result = cursor.fetchone()
        if result:
            config = result[0]
            print(f"   ✓ 4096D config found")
            print(f"     - Vector dimensions: {config.get('vector_dimensions')}")
            print(f"     - Collections mapped: {len(config.get('collections', {}))}")
            print(f"     - Activated at: {config.get('activated_at')[:19]}")
        else:
            print("   ✗ 4096D configuration not found")

        # Check knowledge graph
        cursor.execute("SELECT COUNT(*) FROM tower_knowledge_graph")
        kg_count = cursor.fetchone()[0]
        print(f"   ✓ Knowledge graph: {kg_count} nodes")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"   ✗ PostgreSQL error: {e}")

def test_redis_flags():
    """Test Redis 4096D activation flags."""
    print("\n4. Testing Redis Flags...")

    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)

        active = r.hget('echo:config', '4096d_active')
        dimensions = r.hget('echo:config', '4096d_dimensions')
        activated_at = r.hget('echo:config', '4096d_activated_at')

        if active == 'true':
            print(f"   ✓ 4096D active: {active}")
            print(f"   ✓ Dimensions: {dimensions}")
            print(f"   ✓ Activated: {activated_at[:19]}")
        else:
            print("   ✗ 4096D not active in Redis")
    except Exception as e:
        print(f"   ✗ Redis error: {e}")

async def main():
    print("="*60)
    print("4096D REASONING CAPABILITY TEST")
    print("="*60)

    # Run tests
    await test_echo_brain_api()
    test_qdrant_collections()
    test_postgresql_config()
    test_redis_flags()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())