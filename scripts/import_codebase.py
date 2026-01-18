#!/usr/bin/env python3
"""
Import codebase index into Qdrant code collection.
Source: tower_consolidated.codebase_index (5,980 entries)
Target: Qdrant code collection (1536D OpenAI embeddings)
"""
import asyncio
import sys
import json
from datetime import datetime
from uuid import uuid4
import hashlib

sys.path.insert(0, '/opt/tower-echo-brain')

import asyncpg
from src.services.embedding_service import create_embedding_service
from src.services.vector_search import get_vector_search

DATABASE_URL = "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/tower_consolidated"
BATCH_SIZE = 50  # Larger batches for efficiency

async def import_codebase():
    print(f"=== Codebase Import Started: {datetime.now().isoformat()} ===\n")

    embedding_service = await create_embedding_service()
    vector_search = await get_vector_search()

    pool = await asyncpg.create_pool(DATABASE_URL)

    async with pool.acquire() as conn:
        entries = await conn.fetch("""
            SELECT id, file_path, entity_type, entity_name,
                   signature, docstring, source_code, line_number
            FROM codebase_index
            WHERE source_code IS NOT NULL AND source_code != ''
            ORDER BY id
        """)

    print(f"Found {len(entries)} codebase entries to import\n")

    stats = {"imported": 0, "skipped": 0, "errors": []}

    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(entries) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Processing batch {batch_num}/{total_batches}...")

        docs_to_add = []

        for entry in batch:
            try:
                # Create searchable text combining all relevant fields
                parts = [
                    f"File: {entry['file_path']}",
                    f"Type: {entry['entity_type']}",
                    f"Name: {entry['entity_name']}"
                ]
                if entry['signature']:
                    parts.append(f"Signature: {entry['signature']}")
                if entry['docstring']:
                    parts.append(f"Docstring: {entry['docstring']}")

                # Truncate source code if too long
                code = entry['source_code'] or ""
                if len(code) > 8000:
                    code = code[:8000] + "... [truncated]"
                if code:
                    parts.append(f"Code:\n{code}")

                text = "\n".join(parts)

                # Prepare for batch insertion
                doc_id = str(uuid4())
                docs_to_add.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": {
                        "file_path": entry['file_path'],
                        "entity_type": entry['entity_type'],
                        "entity_name": entry['entity_name'],
                        "signature": entry['signature'] or "",
                        "line_number": entry['line_number'] or 0,
                        "text_preview": text[:500]  # Store truncated for display
                    }
                })
                stats["imported"] += 1

            except Exception as e:
                stats["errors"].append(f"{entry['entity_name']}: {str(e)}")

        # Batch insert to Qdrant
        if docs_to_add:
            try:
                await vector_search.add_documents_batch("code", docs_to_add)
                print(f"  ✅ Batch {batch_num} complete: {len(docs_to_add)} entries")
            except Exception as e:
                stats["errors"].append(f"Batch {batch_num} vector insert: {str(e)}")
                print(f"  ❌ Batch error: {e}")

        await asyncio.sleep(0.3)  # Rate limit

    await pool.close()
    await embedding_service.close()

    print(f"\n=== Import Complete ===")
    print(f"Imported: {stats['imported']}")
    print(f"Errors: {len(stats['errors'])}")

    if stats['errors'][:5]:
        print("\nFirst 5 errors:")
        for err in stats['errors'][:5]:
            print(f"  - {err}")

    # Verify
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:6333/collections/code")
        count = resp.json()["result"]["points_count"]
        print(f"\n✅ Qdrant code collection: {count} vectors")

if __name__ == "__main__":
    asyncio.run(import_codebase())