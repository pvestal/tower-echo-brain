#!/usr/bin/env python3
"""
Import Claude conversation history into Qdrant.
Source: /home/patrick/.claude/**/*.jsonl (139 files)
Target: Qdrant conversations collection (1536D)
"""
import asyncio
import sys
import json
import os
from datetime import datetime
from uuid import uuid4
from pathlib import Path

sys.path.insert(0, '/opt/tower-echo-brain')

import asyncpg
from src.services.embedding_service import create_embedding_service
from src.services.vector_search import get_vector_search

CLAUDE_DIR = Path("/home/patrick/.claude")
DATABASE_URL = "postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain"
BATCH_SIZE = 20

async def import_conversations():
    print(f"=== Conversation Import Started: {datetime.now().isoformat()} ===\n")

    embedding_service = await create_embedding_service()
    vector_search = await get_vector_search()
    pool = await asyncpg.create_pool(DATABASE_URL)

    # Find all JSONL files
    jsonl_files = list(CLAUDE_DIR.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files\n")

    stats = {"files": 0, "messages": 0, "errors": []}

    for file_idx, jsonl_path in enumerate(jsonl_files):
        try:
            print(f"[{file_idx+1}/{len(jsonl_files)}] Processing: {jsonl_path.name}...")

            messages = []
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            msg = json.loads(line)
                            messages.append(msg)
                        except json.JSONDecodeError:
                            continue

            if not messages:
                continue

            # Create conversation record
            conv_id = str(uuid4())
            project_name = jsonl_path.parent.name if jsonl_path.parent != CLAUDE_DIR else "general"

            # Process messages in batches
            docs_to_add = []
            for msg_idx, msg in enumerate(messages):
                try:
                    # Extract text based on message structure
                    text_parts = []

                    # Handle different message types
                    msg_type = msg.get('type', 'unknown')

                    if msg_type == 'summary':
                        text_parts.append(f"Summary: {msg.get('summary', '')}")
                    elif 'message' in msg:
                        message = msg['message']
                        if isinstance(message, dict):
                            role = message.get('role', 'unknown')
                            content = message.get('content', '')

                            # Handle content array
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict):
                                        if item.get('type') == 'text':
                                            text_parts.append(f"[{role}]: {item.get('text', '')}")
                                        elif item.get('type') == 'tool_use':
                                            text_parts.append(f"[tool]: {item.get('name', 'unknown')}")
                                    else:
                                        text_parts.append(str(item))
                            else:
                                text_parts.append(f"[{role}]: {content}")
                        else:
                            text_parts.append(str(message))
                    else:
                        # Generic handling
                        for key, value in msg.items():
                            if key not in ['leafUuid', 'uuid', 'timestamp']:
                                text_parts.append(f"{key}: {str(value)[:500]}")

                    text = "\n".join(text_parts)

                    # Skip empty messages
                    if not text or len(text) < 10:
                        continue

                    # Truncate long messages
                    if len(text) > 8000:
                        text = text[:8000] + "... [truncated]"

                    # Prepare for batch insertion
                    msg_id = str(uuid4())
                    timestamp = msg.get('timestamp', msg.get('created_at', datetime.now().isoformat()))

                    docs_to_add.append({
                        "id": msg_id,
                        "text": text,
                        "metadata": {
                            "conversation_id": conv_id,
                            "project": project_name,
                            "file": jsonl_path.name,
                            "message_type": msg_type,
                            "message_index": msg_idx,
                            "text_preview": text[:500],
                            "timestamp": str(timestamp)
                        }
                    })
                    stats["messages"] += 1

                except Exception as e:
                    stats["errors"].append(f"{jsonl_path.name} msg {msg_idx}: {str(e)}")

            # Batch insert to Qdrant
            if docs_to_add:
                try:
                    # Process in chunks to avoid overloading
                    for i in range(0, len(docs_to_add), BATCH_SIZE):
                        batch = docs_to_add[i:i+BATCH_SIZE]
                        await vector_search.add_documents_batch("conversations", batch)
                        await asyncio.sleep(0.2)  # Rate limit

                    print(f"  ✅ {len(docs_to_add)} messages indexed")
                except Exception as e:
                    stats["errors"].append(f"{jsonl_path.name} batch insert: {str(e)}")
                    print(f"  ❌ Batch error: {e}")

            stats["files"] += 1

        except Exception as e:
            stats["errors"].append(f"{jsonl_path.name}: {str(e)}")
            print(f"  ❌ Error: {e}")

    await pool.close()
    await embedding_service.close()

    print(f"\n=== Import Complete ===")
    print(f"Files processed: {stats['files']}")
    print(f"Messages imported: {stats['messages']}")
    print(f"Errors: {len(stats['errors'])}")

    if stats['errors'][:5]:
        print("\nFirst 5 errors:")
        for err in stats['errors'][:5]:
            print(f"  - {err}")

    # Verify
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:6333/collections/conversations")
        count = resp.json()["result"]["points_count"]
        print(f"\n✅ Qdrant conversations collection: {count} vectors")

if __name__ == "__main__":
    asyncio.run(import_conversations())