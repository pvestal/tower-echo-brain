#!/usr/bin/env python3
"""
Health check script for Echo Brain MCP Server.
Verifies all dependencies and connections are working.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
import httpx
from qdrant_client import QdrantClient


async def check_postgresql():
    """Check PostgreSQL connection and required tables."""
    try:
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            database="echo_brain",
            user="patrick",
            password=os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        )

        # Check if facts table exists
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name IN ('facts', 'echo_conversations', 'learning_items')
        """)

        table_names = [row['table_name'] for row in tables]

        await conn.close()

        return {
            "status": "healthy",
            "available_tables": table_names,
            "required_tables": ["facts"],
            "optional_tables": ["echo_conversations", "learning_items"]
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_qdrant():
    """Check Qdrant connection and available collections."""
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=10)

        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        return {
            "status": "healthy",
            "collections": collection_names,
            "target_collections": ["echo_memory_768", "claude_conversations"]
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_ollama():
    """Check Ollama connection and embedding capability."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test embeddings endpoint
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "nomic-embed-text:latest",
                    "prompt": "health check test"
                }
            )

            if response.status_code == 200:
                data = response.json()
                embedding = data.get('embedding', [])

                return {
                    "status": "healthy",
                    "model": "nomic-embed-text:latest",
                    "embedding_dimension": len(embedding)
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}: {response.text[:200]}"
                }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def main():
    """Run complete health check."""
    print("🏥 Echo Brain MCP Server Health Check\n")

    health_report = {
        "timestamp": asyncio.get_event_loop().time(),
        "overall_status": "healthy",
        "components": {}
    }

    # Check PostgreSQL
    print("🗄️  Checking PostgreSQL...")
    pg_health = await check_postgresql()
    health_report["components"]["postgresql"] = pg_health

    if pg_health["status"] == "healthy":
        print(f"   ✅ Connected, tables: {pg_health['available_tables']}")
    else:
        print(f"   ❌ Failed: {pg_health['error']}")
        health_report["overall_status"] = "unhealthy"

    # Check Qdrant
    print("\n🔍 Checking Qdrant...")
    qdrant_health = await check_qdrant()
    health_report["components"]["qdrant"] = qdrant_health

    if qdrant_health["status"] == "healthy":
        print(f"   ✅ Connected, collections: {qdrant_health['collections']}")
    else:
        print(f"   ❌ Failed: {qdrant_health['error']}")
        health_report["overall_status"] = "unhealthy"

    # Check Ollama
    print("\n🧠 Checking Ollama...")
    ollama_health = await check_ollama()
    health_report["components"]["ollama"] = ollama_health

    if ollama_health["status"] == "healthy":
        print(f"   ✅ Connected, dimension: {ollama_health['embedding_dimension']}")
    else:
        print(f"   ❌ Failed: {ollama_health['error']}")
        health_report["overall_status"] = "unhealthy"

    # Summary
    print(f"\n📋 Overall Status: {'🟢 HEALTHY' if health_report['overall_status'] == 'healthy' else '🔴 UNHEALTHY'}")

    # Output JSON for programmatic use
    if "--json" in sys.argv:
        print("\n" + json.dumps(health_report, indent=2))

    return 0 if health_report["overall_status"] == "healthy" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))