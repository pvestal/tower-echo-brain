#!/usr/bin/env python3
"""
Index Claude Conversations for Echo Brain Memory System
Processes Claude conversation files and adds them to Qdrant for knowledge retrieval
"""
import os
import sys
import asyncio
import hashlib
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# Add Echo Brain to path
sys.path.insert(0, '/opt/tower-echo-brain')

async def main():
    """Index Claude conversations into Qdrant"""
    print("🔍 Starting Claude Conversations Indexing...")

    # Initialize Qdrant client
    client = QdrantClient(host="localhost", port=6333)

    # Initialize embedding service
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "embedding_service",
            "/opt/tower-echo-brain/src/services/embedding_service.py"
        )
        embed_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(embed_module)

        embedder = embed_module.EmbeddingService()
        await embedder.initialize()
        print("✅ Embedding service initialized")
    except Exception as e:
        print(f"❌ Failed to initialize embedding service: {e}")
        return

    # Check if conversations collection exists
    try:
        collection_info = client.get_collection("conversations")
        print(f"✅ Conversations collection exists with {collection_info.points_count} points")
    except Exception as e:
        print(f"⚠️ Creating conversations collection: {e}")
        client.create_collection(
            collection_name="conversations",
            vectors_config=VectorParams(size=embedder.dimensions, distance=Distance.COSINE)
        )

    # Find Claude conversation files
    conversations_dir = Path("/home/patrick/.claude/conversations")
    if not conversations_dir.exists():
        print(f"❌ Conversations directory not found: {conversations_dir}")
        return

    # Find both MD and JSON conversation files
    md_files = list(conversations_dir.glob("*.md"))
    json_files = list(conversations_dir.glob("*.json"))
    conversation_files = md_files + json_files
    print(f"📁 Found {len(md_files)} MD files and {len(json_files)} JSON files ({len(conversation_files)} total)")

    # Get existing conversation IDs to avoid duplicates
    existing_ids = set()
    try:
        # Note: In production, you'd paginate through this
        points = client.scroll(collection_name="conversations", limit=10000)[0]
        existing_ids = {point.payload.get("file_path") for point in points if point.payload}
        print(f"📋 Found {len(existing_ids)} already indexed conversations")
    except Exception as e:
        print(f"⚠️ Could not check existing conversations: {e}")

    # Process conversations in batches
    batch_size = 50
    points_to_add = []
    processed_count = 0
    skipped_count = 0

    for i, conversation_file in enumerate(conversation_files):
        # Skip if already indexed
        if str(conversation_file) in existing_ids:
            skipped_count += 1
            continue

        try:
            # Read and process conversation content based on file type
            if conversation_file.suffix == '.json':
                # Process JSON conversation file
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # Extract content from JSON structure
                if isinstance(json_data, dict):
                    content = ""
                    title = json_data.get('title', conversation_file.stem)

                    # Try to extract conversation messages
                    if 'messages' in json_data:
                        for msg in json_data['messages']:
                            if isinstance(msg, dict):
                                role = msg.get('role', 'unknown')
                                text = msg.get('content', '')
                                content += f"{role}: {text}\n\n"
                    elif 'conversation' in json_data:
                        content = str(json_data['conversation'])
                    else:
                        content = str(json_data)
                else:
                    content = str(json_data)
                    title = conversation_file.stem

            else:
                # Process markdown file
                content = conversation_file.read_text(encoding='utf-8')
                title = conversation_file.stem.replace('-', ' ').title()

            # Skip empty files
            if len(content.strip()) < 100:
                continue

            # Create embedding
            embedding = await embedder.embed_single(content)

            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": content[:2000],  # Truncate for storage
                    "full_content": content,
                    "title": title,
                    "file_path": str(conversation_file),
                    "file_name": conversation_file.name,
                    "type": "claude_conversation",
                    "date": conversation_file.stat().st_mtime,
                    "size": len(content)
                }
            )

            points_to_add.append(point)
            processed_count += 1

            # Process batch
            if len(points_to_add) >= batch_size:
                client.upsert(
                    collection_name="conversations",
                    points=points_to_add
                )
                print(f"📝 Indexed batch of {len(points_to_add)} conversations ({processed_count}/{len(conversation_files)})")
                points_to_add = []

        except Exception as e:
            print(f"❌ Error processing {conversation_file}: {e}")
            continue

        # Progress update
        if (i + 1) % 100 == 0:
            print(f"🔄 Progress: {i + 1}/{len(conversation_files)} files processed")

    # Process final batch
    if points_to_add:
        client.upsert(
            collection_name="conversations",
            points=points_to_add
        )
        print(f"📝 Indexed final batch of {len(points_to_add)} conversations")

    # Final summary
    collection_info = client.get_collection("conversations")
    print(f"✅ Indexing complete!")
    print(f"   • Processed: {processed_count} new conversations")
    print(f"   • Skipped: {skipped_count} existing conversations")
    print(f"   • Total in collection: {collection_info.points_count}")

    await embedder.close()

if __name__ == "__main__":
    asyncio.run(main())