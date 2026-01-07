#!/usr/bin/env python3
"""
Index all missing content into Qdrant for comprehensive semantic search.
Includes: codebase_index, past_solutions, claude conversations, anime data
"""
import asyncio
import sys
import json
from datetime import datetime
from uuid import uuid4

sys.path.insert(0, '/opt/tower-echo-brain')

import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from src.services.embedding_service import create_embedding_service

DATABASE_URL = "postgresql://patrick:***REMOVED***@localhost/tower_consolidated"
QDRANT_URL = "http://***REMOVED***:6333"

async def index_missing_content():
    """Index all missing content into semantic search."""
    print(f"=== Comprehensive Content Indexing Started: {datetime.now().isoformat()} ===\n")

    # Initialize services
    embedding_service = await create_embedding_service()
    qdrant = QdrantClient(url=QDRANT_URL)
    pool = await asyncpg.create_pool(DATABASE_URL)

    # 1. Index codebase_index (source code)
    print("📁 Indexing Codebase (5,980 files)...")
    async with pool.acquire() as conn:
        code_files = await conn.fetch("""
            SELECT file_path, content, language, file_type, last_updated
            FROM codebase_index
            WHERE content IS NOT NULL AND content != ''
            LIMIT 1000
        """)

    if code_files:
        # Ensure code collection exists with correct dimensions
        try:
            qdrant.delete_collection("code")
        except:
            pass
        qdrant.create_collection(
            collection_name="code",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

        batch = []
        for i, file in enumerate(code_files):
            if i % 20 == 0 and i > 0:
                print(f"  Processing code batch {i//20}...")

            # Create searchable text
            text = f"File: {file['file_path']}\nLanguage: {file['language']}\n{file['content'][:2000]}"
            embedding = await embedding_service.generate(text)

            batch.append(PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    'file_path': file['file_path'],
                    'language': file['language'],
                    'content': file['content'][:2000],
                    'file_type': file['file_type'],
                    'source': 'codebase_index'
                }
            ))

            if len(batch) >= 20:
                qdrant.upsert(collection_name="code", points=batch)
                batch = []

        if batch:
            qdrant.upsert(collection_name="code", points=batch)
        print(f"  ✅ Indexed {len(code_files)} code files\n")

    # 2. Index past_solutions (verified fixes)
    print("🔧 Indexing Past Solutions (19 verified fixes)...")
    async with pool.acquire() as conn:
        solutions = await conn.fetch("""
            SELECT problem_description, solution_applied, root_cause,
                   verification_steps, created_at
            FROM past_solutions
            WHERE verified_working = TRUE
        """)

    if solutions:
        # Add to documents collection
        batch = []
        for solution in solutions:
            text = f"""Problem: {solution['problem_description']}
Solution: {solution['solution_applied']}
Root Cause: {solution['root_cause']}
Verification: {solution['verification_steps']}"""

            embedding = await embedding_service.generate(text)
            batch.append(PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    'problem': solution['problem_description'],
                    'solution': solution['solution_applied'],
                    'root_cause': solution['root_cause'],
                    'source': 'past_solutions',
                    'type': 'verified_fix'
                }
            ))

        qdrant.upsert(collection_name="documents", points=batch)
        print(f"  ✅ Indexed {len(solutions)} past solutions\n")

    # 3. Index Claude conversations from filesystem
    print("📝 Indexing Claude Conversations...")
    import os
    claude_dir = "/home/patrick/.claude/conversations"
    if os.path.exists(claude_dir):
        conv_count = 0
        batch = []
        for filename in os.listdir(claude_dir)[:50]:  # Limit to recent 50
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(claude_dir, filename), 'r') as f:
                        conv = json.load(f)

                    # Create searchable summary
                    messages = conv.get('messages', [])
                    if messages:
                        text = f"Claude Conversation {conv.get('id', 'unknown')}\n"
                        for msg in messages[-5:]:  # Last 5 messages
                            role = msg.get('role', 'unknown')
                            content = msg.get('content', '')[:500]
                            text += f"{role}: {content}\n"

                        embedding = await embedding_service.generate(text)
                        batch.append(PointStruct(
                            id=str(uuid4()),
                            vector=embedding,
                            payload={
                                'conversation_id': conv.get('id'),
                                'timestamp': conv.get('timestamp'),
                                'summary': text[:500],
                                'source': 'claude_conversations'
                            }
                        ))
                        conv_count += 1

                        if len(batch) >= 20:
                            qdrant.upsert(collection_name="conversations", points=batch)
                            batch = []
                except Exception as e:
                    print(f"    Error processing {filename}: {e}")

        if batch:
            qdrant.upsert(collection_name="conversations", points=batch)
        print(f"  ✅ Indexed {conv_count} Claude conversations\n")

    # 4. Index README files and docs
    print("📚 Indexing Documentation...")
    doc_count = 0
    batch = []

    # Find all README files
    for service_dir in ['/opt/tower-echo-brain', '/opt/tower-kb', '/opt/tower-dashboard']:
        if os.path.exists(service_dir):
            readme_path = os.path.join(service_dir, 'README.md')
            if os.path.exists(readme_path):
                with open(readme_path, 'r') as f:
                    content = f.read()

                text = f"README for {os.path.basename(service_dir)}\n{content[:3000]}"
                embedding = await embedding_service.generate(text)

                batch.append(PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload={
                        'title': f"{os.path.basename(service_dir)} README",
                        'content': content[:3000],
                        'path': readme_path,
                        'source': 'documentation',
                        'type': 'readme'
                    }
                ))
                doc_count += 1

    if batch:
        qdrant.upsert(collection_name="documents", points=batch)
    print(f"  ✅ Indexed {doc_count} documentation files\n")

    # 5. Verify all collections
    print("=== Verification ===")
    collections = ['documents', 'code', 'conversations', 'facts', 'echo_memories']
    for coll in collections:
        try:
            info = qdrant.get_collection(coll)
            print(f"  {coll}: {info.points_count} points")
        except:
            print(f"  {coll}: Not found")

    await pool.close()
    print(f"\n=== Indexing Complete: {datetime.now().isoformat()} ===")

if __name__ == "__main__":
    asyncio.run(index_missing_content())