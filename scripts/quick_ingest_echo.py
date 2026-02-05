#!/usr/bin/env python3
"""
Quick ingestion of Echo Brain's own code and documentation
Focused ingestion to get Echo Brain understanding itself
"""

import asyncio
import httpx
from pathlib import Path
import logging
from typing import List, Dict
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBED_URL = "http://localhost:11434/api/embed"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "echo_memory"
MODEL = "mxbai-embed-large:latest"

async def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            EMBED_URL,
            json={"model": MODEL, "input": text}
        )
        return response.json()["embeddings"][0]

async def add_to_qdrant(vectors: List[Dict]):
    """Add vectors to Qdrant"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points",
            json={"points": vectors}
        )
        if response.status_code == 200:
            logger.info(f"Added {len(vectors)} vectors to Qdrant")
        else:
            logger.error(f"Failed to add vectors: {response.text}")

async def ingest_echo_brain():
    """Ingest Echo Brain code and docs"""
    vectors = []
    files_processed = 0

    # 1. Core Echo Brain Python files
    logger.info("Ingesting Echo Brain core code...")
    echo_path = Path("/opt/tower-echo-brain/src")

    # Key files to prioritize
    priority_files = [
        "main.py",
        "api/endpoints/echo_main_router.py",
        "intelligence/system_model.py",
        "intelligence/code_intelligence.py",
        "intelligence/learner.py",
        "reasoning/chain.py",
        "reasoning/semantic_router.py",
        "models/kb_models.py"
    ]

    for rel_path in priority_files:
        file_path = echo_path / rel_path
        if not file_path.exists():
            continue

        try:
            content = file_path.read_text()
            if len(content) > 100:  # Skip tiny files
                # Create meaningful chunks (by function/class)
                chunks = split_code_intelligently(content, str(file_path))

                for chunk in chunks:
                    vec_id = hashlib.md5(chunk.encode()).hexdigest()
                    embedding = await get_embedding(chunk)

                    vectors.append({
                        "id": vec_id,
                        "vector": embedding,
                        "payload": {
                            "content": chunk[:2000],  # Truncate for payload
                            "source": str(file_path),
                            "type": "code",
                            "project": "echo-brain",
                            "language": "python"
                        }
                    })

                files_processed += 1
                logger.info(f"  Processed {file_path.name} ({len(chunks)} chunks)")

                # Add batch when we have enough
                if len(vectors) >= 20:
                    await add_to_qdrant(vectors)
                    vectors = []

        except Exception as e:
            logger.error(f"  Error processing {file_path}: {e}")

    # 2. Database schema
    logger.info("\nIngesting database schemas...")
    schema_info = """
    Echo Brain PostgreSQL Schema:

    Database: echo_brain
    Main tables:
    - conversations: Stores conversation history
    - vector_content: Metadata for vectors
    - facts: Extracted facts from conversations
    - code_changes: Tracked code modifications
    - system_metrics: System performance metrics
    - ingestion_tracking: Track what's been ingested

    Qdrant Collection: echo_memory (1024 dimensions)
    - Types: code, documentation, schema, article, fact
    """

    vec_id = hashlib.md5(schema_info.encode()).hexdigest()
    embedding = await get_embedding(schema_info)
    vectors.append({
        "id": vec_id,
        "vector": embedding,
        "payload": {
            "content": schema_info,
            "source": "database_schema",
            "type": "schema",
            "project": "echo-brain"
        }
    })

    # 3. API documentation
    logger.info("Ingesting API structure...")
    api_docs = """
    Echo Brain API Endpoints:

    Main Routes (/api/echo/):
    - /health - System health and metrics
    - /status - Alias for health
    - /ask - Main question answering endpoint
    - /brain - Brain activity visualization
    - /intelligence/diagnose - System diagnosis
    - /intelligence/code/search - Code search
    - /intelligence/explain - Code explanation
    - /memory/search - Vector memory search

    MCP Endpoints (/mcp):
    - search_memory - Semantic search
    - get_facts - Retrieve facts

    Port: 8309
    """

    vec_id = hashlib.md5(api_docs.encode()).hexdigest()
    embedding = await get_embedding(api_docs)
    vectors.append({
        "id": vec_id,
        "vector": embedding,
        "payload": {
            "content": api_docs,
            "source": "api_documentation",
            "type": "documentation",
            "project": "echo-brain"
        }
    })

    # Add remaining vectors
    if vectors:
        await add_to_qdrant(vectors)

    logger.info(f"\n✅ Ingestion complete! Processed {files_processed} files")
    logger.info(f"Ready for fact extraction next")

def split_code_intelligently(content: str, filename: str) -> List[str]:
    """Split code into meaningful chunks"""
    chunks = []
    lines = content.split('\n')

    current_chunk = []
    current_size = 0
    in_class = False
    in_function = False

    for line in lines:
        # Detect class/function boundaries
        if line.startswith('class '):
            if current_chunk and current_size > 100:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [f"# From {filename}"]
                current_size = len(current_chunk[0])
            in_class = True

        elif line.startswith('def ') or line.startswith('async def '):
            if not in_class and current_chunk and current_size > 100:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [f"# From {filename}"]
                current_size = len(current_chunk[0])
            in_function = True

        current_chunk.append(line)
        current_size += len(line)

        # Keep chunks reasonable size
        if current_size > 3000:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [f"# From {filename}"]
            current_size = len(current_chunk[0])

    # Add final chunk
    if current_chunk and current_size > 100:
        chunks.append('\n'.join(current_chunk))

    return chunks if chunks else [content[:3000]]  # Fallback to simple truncation

async def main():
    logger.info("=== ECHO BRAIN QUICK INGESTION ===")
    logger.info("Focus: Echo Brain understanding itself\n")

    await ingest_echo_brain()

    # Check final count
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{QDRANT_URL}/collections/{COLLECTION}")
        count = resp.json()["result"]["points_count"]
        logger.info(f"\nTotal vectors after ingestion: {count:,}")

if __name__ == "__main__":
    asyncio.run(main())