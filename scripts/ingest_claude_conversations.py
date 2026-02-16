#!/usr/bin/env python3
"""
Ingest Claude conversation JSONL files from ~/.claude/projects/ into Echo Brain.
Automatically finds and processes all conversation files.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import hashlib
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
DB_PASS = os.environ.get("DB_PASSWORD")
if not DB_PASS:
    raise ValueError("DB_PASSWORD environment variable is required")
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
CLAUDE_CONVERSATIONS_DIR = Path.home() / ".claude" / "projects"

def is_readable_text(text: str) -> bool:
    """Filter out base64, binary, dense file paths, and other non-readable content."""
    if not text or len(text) < 10:
        return False
    # Check space ratio — readable text has spaces between words
    space_ratio = text.count(' ') / len(text)
    if space_ratio < 0.02:
        return False
    # Check alphanumeric density — base64/binary is nearly all alnum with no spaces
    sample = text[:200]
    if sample:
        alnum_count = sum(1 for c in sample if c.isalnum())
        if alnum_count / len(sample) > 0.85:
            return False
    # Check for base64 patterns (long strings without spaces)
    if re.search(r'[A-Za-z0-9+/=]{100,}', text[:500]):
        return False
    return True


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

async def process_jsonl_file(filepath: Path, db_conn, qdrant_client, session):
    """Process a single JSONL conversation file"""

    # Check if already processed
    file_hash = hashlib.md5(str(filepath).encode()).hexdigest()
    existing = await db_conn.fetchrow(
        "SELECT id FROM claude_conversations WHERE file_path = $1",
        str(filepath)
    )

    if existing:
        logger.info(f"Skipping already processed: {filepath.name}")
        return 0

    logger.info(f"Processing: {filepath}")

    vectors_added = 0
    conversation_content = []

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    msg = json.loads(line)

                    # Skip summary entries and non-message entries
                    if msg.get('type') in ['summary'] or 'message' not in msg:
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

                    if content and is_readable_text(content):
                        conversation_content.append(f"[{role}]: {content}")

                        # Create chunks for long messages
                        if len(content) > 1000:  # Production threshold
                            chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
                            logger.debug(f"Processing {len(chunks)} chunks for message {line_num}")
                            for chunk_idx, chunk in enumerate(chunks):
                                if not is_readable_text(chunk):
                                    continue
                                logger.debug(f"Getting embedding for chunk {chunk_idx}: {chunk[:50]}...")
                                embedding = await get_embedding(chunk, session)
                                if embedding:
                                    logger.debug(f"Got embedding with {len(embedding)} dimensions")
                                    # Generate integer ID from hash
                                    point_id = int(hashlib.md5(f"{file_hash}_{line_num}_{chunk_idx}".encode()).hexdigest()[:8], 16)

                                    payload = {
                                            "type": "conversation",
                                            "role": role,
                                            "content": chunk,
                                            "file_path": str(filepath),
                                            "line_number": line_num,
                                            "chunk_index": chunk_idx,
                                            "ingested_at": datetime.now().isoformat(),
                                            "confidence": 0.8,
                                            "last_accessed": datetime.now().isoformat(),
                                            "access_count": 0,
                                        }

                                    # Dedup check
                                    try:
                                        from src.core.dedup import check_duplicate, merge_metadata, bump_existing_point
                                        import asyncio
                                        dup = asyncio.get_event_loop().run_until_complete(
                                            check_duplicate(embedding)
                                        ) if not asyncio.get_event_loop().is_running() else None
                                        # In async context, dedup is best-effort
                                        if dup:
                                            logger.info(f"Dedup: skipping duplicate (score={dup['score']:.3f})")
                                            continue
                                    except Exception:
                                        pass  # dedup is non-blocking

                                    point = PointStruct(
                                        id=point_id,
                                        vector=embedding,
                                        payload=payload,
                                    )

                                    try:
                                        qdrant_client.upsert(
                                            collection_name="echo_memory",
                                            points=[point]
                                        )
                                        vectors_added += 1
                                    except Exception as e:
                                        logger.error(f"Qdrant upsert error: {e}")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in {filepath.name} line {line_num}: {e}")
                    continue

        # Store in PostgreSQL
        if conversation_content:
            conversation_text = "\n".join(conversation_content[:100])  # First 100 messages

            await db_conn.execute("""
                INSERT INTO claude_conversations (file_path, content, created_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (file_path) DO UPDATE
                SET content = $2, created_at = NOW()
            """, str(filepath), conversation_text)

            logger.info(f"Added {vectors_added} vectors from {filepath.name}")

    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")

    return vectors_added

async def main():
    """Main ingestion function"""

    # Connect to services
    logger.info("Connecting to services...")

    db_conn = await asyncpg.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

    qdrant_client = QdrantClient(url=QDRANT_URL)

    async with httpx.AsyncClient() as session:
        # Find all JSONL files
        jsonl_files = list(CLAUDE_CONVERSATIONS_DIR.glob("**/*.jsonl"))
        logger.info(f"Found {len(jsonl_files)} conversation files")

        total_vectors = 0
        processed_files = 0

        # Process each file
        for filepath in jsonl_files:
            try:
                vectors = await process_jsonl_file(filepath, db_conn, qdrant_client, session)
                total_vectors += vectors
                if vectors > 0:
                    processed_files += 1
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                continue

        logger.info(f"Ingestion complete: {processed_files} files, {total_vectors} vectors added")

        # Summary statistics
        result = await db_conn.fetchrow("""
            SELECT
                COUNT(*) as total_conversations,
                COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END) as new_conversations
            FROM claude_conversations
        """)

        logger.info(f"Database now has {result['total_conversations']} conversations ({result['new_conversations']} new)")

    await db_conn.close()

if __name__ == "__main__":
    asyncio.run(main())