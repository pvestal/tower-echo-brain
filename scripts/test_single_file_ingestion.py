#!/usr/bin/env python3
"""
Test ingestion of a single conversation file.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import hashlib
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, "/opt/tower-echo-brain")

import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import httpx

# Configuration
DB_HOST = "localhost"
DB_NAME = "echo_brain"
DB_USER = "patrick"
DB_PASS = os.environ.get("DB_PASSWORD", "")
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"

async def get_embedding(text: str, session: httpx.AsyncClient) -> list:
    """Get embedding from Ollama"""
    try:
        response = await session.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("embedding", [])
    except Exception as e:
        logger.error(f"Embedding error: {e}")
    return []

async def test_single_file():
    """Test processing a single file"""

    # Use the most recent conversation file
    test_file = Path("/home/patrick/.claude/projects/-home-patrick-Documents/dc7b8f39-dd84-4f7e-874b-2965c72eec2d.jsonl")

    logger.info(f"Testing with file: {test_file}")

    # Connect to services
    db_conn = await asyncpg.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

    qdrant_client = QdrantClient(url=QDRANT_URL)

    async with httpx.AsyncClient() as session:
        vectors_added = 0
        messages_processed = 0

        file_hash = hashlib.md5(str(test_file).encode()).hexdigest()

        with open(test_file, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                if line_num > 20:  # Only process first 20 lines for testing
                    break

                try:
                    msg = json.loads(line)

                    # Skip summary entries and non-message entries
                    if msg.get('type') in ['summary'] or 'message' not in msg:
                        logger.debug(f"Skipping line {line_num}: {msg.get('type', 'no-type')}")
                        continue

                    # Extract message content from proper structure
                    message = msg['message']
                    role = message.get('role', 'unknown')
                    content = ""

                    if isinstance(message.get('content'), list):
                        # Handle structured content (Claude format)
                        for item in message['content']:
                            if isinstance(item, dict):
                                if item.get('type') == 'text':
                                    content += item.get('text', '')
                                elif item.get('type') == 'tool_result':
                                    content += f"\n[Tool Result: {item.get('content', '')}]\n"
                    elif isinstance(message.get('content'), str):
                        content = message['content']
                    else:
                        # Fallback - sometimes content is directly in the message
                        content = str(message.get('content', ''))

                    if content:
                        messages_processed += 1
                        logger.info(f"Processing message {line_num}: [{role}] {content[:100]}...")

                        # Create chunks for long messages
                        if len(content) > 100:  # Lower threshold for testing
                            chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
                            logger.info(f"Creating {len(chunks)} chunks for message {line_num}")

                            for chunk_idx, chunk in enumerate(chunks):
                                logger.debug(f"Getting embedding for chunk {chunk_idx}: {chunk[:50]}...")
                                embedding = await get_embedding(chunk, session)

                                if embedding:
                                    logger.info(f"Got embedding with {len(embedding)} dimensions")
                                    # Generate integer ID from hash
                                    point_id = int(hashlib.md5(f"{file_hash}_{line_num}_{chunk_idx}".encode()).hexdigest()[:8], 16)

                                    point = PointStruct(
                                        id=point_id,
                                        vector=embedding,
                                        payload={
                                            "type": "conversation",
                                            "role": role,
                                            "content": chunk,
                                            "file_path": str(test_file),
                                            "line_number": line_num,
                                            "chunk_index": chunk_idx,
                                            "ingested_at": datetime.now().isoformat(),
                                            "test_run": True
                                        }
                                    )

                                    try:
                                        qdrant_client.upsert(
                                            collection_name="echo_memory",
                                            points=[point]
                                        )
                                        vectors_added += 1
                                        logger.info(f"Successfully added vector {vectors_added}")
                                    except Exception as e:
                                        logger.error(f"Qdrant upsert error: {e}")
                                else:
                                    logger.warning(f"No embedding returned for chunk {chunk_idx}")
                        else:
                            logger.debug(f"Content too short ({len(content)} chars), skipping embedding")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue

        logger.info(f"Test complete: processed {messages_processed} messages, added {vectors_added} vectors")

    await db_conn.close()

if __name__ == "__main__":
    asyncio.run(test_single_file())