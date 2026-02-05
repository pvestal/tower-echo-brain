#!/usr/bin/env python3
"""
Import KB Articles from knowledge_base database into Qdrant.

Source: PostgreSQL knowledge_base.articles (437 articles)
Target: Qdrant documents collection (1536D OpenAI embeddings)

Also syncs metadata to tower_consolidated for SSOT.
"""
import asyncio
import os
from datetime import datetime
from uuid import uuid4
import hashlib
import json

import asyncpg

# Add parent to path for imports
import sys
sys.path.insert(0, '/opt/tower-echo-brain')

from src.services.embedding_service import create_embedding_service
from src.services.vector_search import get_vector_search

# Database connections
KB_DATABASE_URL = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/knowledge_base"
ECHO_DATABASE_URL = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain"

BATCH_SIZE = 20  # Articles per batch for embedding

async def import_kb_articles():
    """Import all KB articles with OpenAI embeddings."""
    print(f"=== KB Article Import Started: {datetime.now().isoformat()} ===\n")

    # Initialize services
    embedding_service = await create_embedding_service()
    vector_search = await get_vector_search()

    # Connect to both databases
    kb_pool = await asyncpg.create_pool(KB_DATABASE_URL)
    echo_pool = await asyncpg.create_pool(ECHO_DATABASE_URL)

    # Get all articles from knowledge_base
    async with kb_pool.acquire() as conn:
        articles = await conn.fetch("""
            SELECT id, title, content, category, tags, created_at, updated_at
            FROM articles
            WHERE content IS NOT NULL AND content != ''
            ORDER BY id
        """)

    print(f"Found {len(articles)} articles to import\n")

    stats = {"imported": 0, "skipped": 0, "errors": []}

    # Process in batches
    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(articles) + BATCH_SIZE - 1)//BATCH_SIZE}...")

        docs_to_add = []

        for article in batch:
            try:
                doc_id = str(uuid4())

                # Combine title and content for better semantic search
                # Truncate to avoid exceeding OpenAI token limits (8192 tokens ~= 32k chars)
                content = article['content']
                if len(content) > 30000:
                    content = content[:30000] + "... [truncated]"
                text = f"{article['title']}\n\n{content}"
                content_hash = hashlib.sha256(text.encode()).hexdigest()

                # Store metadata in tower_consolidated.documents
                async with echo_pool.acquire() as conn:
                    # Check if already exists
                    existing = await conn.fetchrow("""
                        SELECT id FROM documents
                        WHERE source_type = 'knowledge_base' AND source_id = $1
                    """, str(article['id']))

                    if existing:
                        doc_id = str(existing['id'])
                        # Update
                        await conn.execute("""
                            UPDATE documents SET
                                content_hash = $2,
                                source_modified_at = $3,
                                last_sync_at = NOW()
                            WHERE id = $1
                        """,
                            existing['id'],
                            content_hash,
                            article['updated_at']
                        )
                    else:
                        # Insert
                        await conn.execute("""
                            INSERT INTO documents
                            (id, source_type, source_id, title, content_type,
                             content_hash, source_created_at, source_modified_at, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                            doc_id,
                            'knowledge_base',
                            str(article['id']),
                            article['title'],
                            'text/plain',
                            content_hash,
                            article['created_at'],
                            article['updated_at'],
                            json.dumps({
                                'category': article['category'],
                                'tags': article['tags'],
                                'kb_article_id': article['id']
                            })
                        )

                # Prepare for batch vector insertion
                docs_to_add.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": {
                        "document_id": doc_id,
                        "source_type": "knowledge_base",
                        "source_id": str(article['id']),
                        "title": article['title'],
                        "category": article['category'] or "uncategorized"
                    }
                })

                stats["imported"] += 1

            except Exception as e:
                stats["errors"].append(f"Article {article['id']}: {str(e)}")
                print(f"  Error with article {article['id']}: {e}")

        # Batch insert vectors to Qdrant
        if docs_to_add:
            try:
                await vector_search.add_documents_batch("documents", docs_to_add)
                print(f"  ✅ Batch completed: {len(docs_to_add)} articles")
            except Exception as e:
                stats["errors"].append(f"Batch vector insert failed: {str(e)}")
                print(f"  ❌ Batch vector insert failed: {e}")

        # Rate limit protection
        await asyncio.sleep(0.5)

    # Close connections
    await kb_pool.close()
    await echo_pool.close()
    await embedding_service.close()

    # Print summary
    print(f"\n=== Import Complete ===")
    print(f"Imported: {stats['imported']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {len(stats['errors'])}")

    if stats['errors']:
        print("\nFirst 5 errors:")
        for err in stats['errors'][:5]:
            print(f"  - {err}")

    # Verify
    print("\n=== Verification ===")
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:6333/collections/documents")
        count = resp.json()["result"]["points_count"]
        print(f"Qdrant documents collection: {count} vectors")

    return stats

if __name__ == "__main__":
    asyncio.run(import_kb_articles())