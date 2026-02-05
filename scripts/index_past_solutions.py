#!/usr/bin/env python3
"""
Index past_solutions (verified fixes) into Qdrant for semantic search.
These are critical debugging patterns that have been proven to work.
"""
import asyncio
import sys
from datetime import datetime
from uuid import uuid4

sys.path.insert(0, '/opt/tower-echo-brain')

import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.services.embedding_service import create_embedding_service

DATABASE_URL = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain"
QDRANT_URL = "http://192.168.50.135:6333"

async def index_past_solutions():
    """Index all verified past solutions into semantic search."""
    print(f"=== Past Solutions Indexing Started: {datetime.now().isoformat()} ===\n")

    # Initialize services
    embedding_service = await create_embedding_service()
    qdrant = QdrantClient(url=QDRANT_URL)
    pool = await asyncpg.create_pool(DATABASE_URL)

    # Get all verified past solutions
    async with pool.acquire() as conn:
        solutions = await conn.fetch("""
            SELECT
                id,
                problem_description,
                solution_applied,
                files_modified,
                verification_command,
                verified_working,
                created_at,
                tags
            FROM past_solutions
            WHERE verified_working = TRUE
            ORDER BY created_at DESC
        """)

    print(f"Found {len(solutions)} verified solutions to index\n")

    if solutions:
        batch = []
        for i, solution in enumerate(solutions, 1):
            print(f"Processing solution {i}/{len(solutions)}: {solution['problem_description'][:50]}...")

            # Create comprehensive searchable text
            text = f"""
PROBLEM: {solution['problem_description']}

SOLUTION: {solution['solution_applied']}

FILES MODIFIED: {', '.join(solution['files_modified']) if solution['files_modified'] else 'No files specified'}

VERIFICATION COMMAND: {solution['verification_command'] or 'Manual verification'}

TAGS: {', '.join(solution['tags']) if solution['tags'] else 'None'}

CREATED: {solution['created_at']}
"""

            # Generate embedding
            embedding = await embedding_service.embed_single(text)

            # Create point for Qdrant
            point = PointStruct(
                id=str(uuid4()),  # Always use UUID for Qdrant
                vector=embedding,
                payload={
                    'id': solution['id'],
                    'problem': solution['problem_description'],
                    'solution': solution['solution_applied'],
                    'files_modified': solution['files_modified'],
                    'verification_command': solution['verification_command'],
                    'tags': solution['tags'],
                    'created_at': solution['created_at'].isoformat() if solution['created_at'] else None,
                    'source': 'past_solutions',
                    'type': 'verified_fix',
                    'searchable_text': text[:1000]  # Store first 1000 chars for reference
                }
            )
            batch.append(point)

            # Upload in batches of 10
            if len(batch) >= 10 or i == len(solutions):
                try:
                    qdrant.upsert(collection_name="documents", points=batch)
                    print(f"  ✅ Uploaded batch of {len(batch)} solutions")
                    batch = []
                except Exception as e:
                    print(f"  ❌ Error uploading batch: {e}")
                    # Try to create collection if it doesn't exist
                    try:
                        from qdrant_client.models import VectorParams, Distance
                        qdrant.create_collection(
                            collection_name="documents",
                            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                        )
                        print("  📦 Created documents collection")
                        qdrant.upsert(collection_name="documents", points=batch)
                        print(f"  ✅ Uploaded batch after creating collection")
                        batch = []
                    except:
                        print(f"  ❌ Failed to create collection or upload")

    # Also add to a dedicated solutions collection for focused search
    print("\n📦 Creating dedicated 'solutions' collection...")
    try:
        from qdrant_client.models import VectorParams, Distance
        try:
            qdrant.delete_collection("solutions")
        except:
            pass

        qdrant.create_collection(
            collection_name="solutions",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

        # Re-index into solutions collection
        batch = []
        for i, solution in enumerate(solutions, 1):
            text = f"""
{solution['problem_description']}
{solution['solution_applied']}
{', '.join(solution['files_modified']) if solution['files_modified'] else ''}
{', '.join(solution['tags']) if solution['tags'] else ''}
"""
            embedding = await embedding_service.generate(text)

            point = PointStruct(
                id=str(uuid4()),  # Always use UUID for Qdrant
                vector=embedding,
                payload={
                    'problem': solution['problem_description'],
                    'solution': solution['solution_applied'],
                    'files_modified': solution['files_modified'],
                    'tags': solution['tags'],
                    'verified': True
                }
            )
            batch.append(point)

        qdrant.upsert(collection_name="solutions", points=batch)
        print(f"  ✅ Created solutions collection with {len(solutions)} verified fixes")
    except Exception as e:
        print(f"  ❌ Error creating solutions collection: {e}")

    # Verify indexing
    print("\n=== Verification ===")
    try:
        docs_info = qdrant.get_collection("documents")
        print(f"  documents collection: {docs_info.points_count} total points")

        solutions_info = qdrant.get_collection("solutions")
        print(f"  solutions collection: {solutions_info.points_count} verified fixes")
    except Exception as e:
        print(f"  ❌ Verification error: {e}")

    # Test search
    print("\n🔍 Testing search...")
    test_query = "error database connection failed"
    try:
        test_embedding = await embedding_service.embed_single(test_query)
        results = qdrant.search(
            collection_name="solutions",
            query_vector=test_embedding,
            limit=3
        )

        print(f"  Query: '{test_query}'")
        if results:
            print(f"  ✅ Found {len(results)} relevant solutions:")
            for r in results:
                print(f"    - {r.payload.get('problem', 'Unknown')[:60]}... (score: {r.score:.3f})")
        else:
            print("  ⚠️ No results found")
    except Exception as e:
        print(f"  ❌ Search test error: {e}")

    await pool.close()
    print(f"\n=== Indexing Complete: {datetime.now().isoformat()} ===")

if __name__ == "__main__":
    asyncio.run(index_past_solutions())