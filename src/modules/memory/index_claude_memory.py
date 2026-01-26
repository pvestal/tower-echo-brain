#!/usr/bin/env python3
"""
Index Claude conversation histories into Echo Brain's memory.
This gives Echo access to all our development context and decisions.
"""

import os
import json
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
import psycopg2
import psycopg2.extras
from qdrant_client import QdrantClient
from src.api.models import Distance, VectorParams, PointStruct
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeMemoryIndexer:
    """Index Claude conversations into Echo's memory systems."""

    def __init__(self):
        # Database connection
        self.db_conn = psycopg2.connect(
            host="localhost",
            database="echo_brain",
            user="patrick",
            password=os.getenv("TOWER_DB_PASSWORD", os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"))
        )

        # Qdrant vector database
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.collection_name = "claude_conversations"

        # Ollama for embeddings
        self.ollama_url = "http://localhost:11434"
        self.embedding_model = "nomic-embed-text:latest"

        # Claude conversations path
        self.conversations_path = Path.home() / ".claude" / "conversations"

    async def create_collection(self):
        """Create Qdrant collection for Claude conversations."""
        try:
            # Check if collection exists
            collections = self.qdrant.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                logger.info(f"Collection {self.collection_name} already exists")
                return

            # Create new collection with 768 dimensions (nomic-embed-text)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")

    async def get_embedding(self, text: str):
        """Get embedding vector for text using Ollama."""
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": text},
                    timeout=30
                )
                if resp.status_code == 200:
                    return resp.json()["embedding"]
            except Exception as e:
                logger.error(f"Embedding error: {e}")
        return None

    def extract_conversation_metadata(self, filepath: Path):
        """Extract metadata from conversation file."""
        metadata = {
            "filename": filepath.name,
            "date": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
            "type": "unknown",
            "service": "unknown",
            "topics": []
        }

        # Extract info from filename
        name = filepath.stem.lower()

        # Identify service
        services = ["echo", "anime", "auth", "kb", "dashboard", "plaid", "telegram"]
        for service in services:
            if service in name:
                metadata["service"] = service
                break

        # Identify type
        if "implementation" in name:
            metadata["type"] = "implementation"
        elif "fix" in name or "debug" in name:
            metadata["type"] = "debugging"
        elif "design" in name or "architecture" in name:
            metadata["type"] = "architecture"
        elif "test" in name:
            metadata["type"] = "testing"

        return metadata

    async def index_conversation(self, filepath: Path):
        """Index a single conversation file."""
        try:
            # Read file content
            if filepath.suffix == ".json":
                with open(filepath, 'r') as f:
                    content = json.load(f)
                    if isinstance(content, dict):
                        text = json.dumps(content, indent=2)
                    else:
                        text = str(content)
            else:  # .md or other text files
                with open(filepath, 'r') as f:
                    text = f.read()

            # Skip if too short
            if len(text) < 100:
                return

            # Create summary (first 500 chars)
            summary = text[:500] if len(text) > 500 else text

            # Get embedding
            embedding = await self.get_embedding(summary)
            if not embedding:
                return

            # Extract metadata
            metadata = self.extract_conversation_metadata(filepath)

            # Generate unique ID
            file_id = hashlib.md5(str(filepath).encode()).hexdigest()

            # Store in Qdrant
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=file_id,
                        vector=embedding,
                        payload={
                            **metadata,
                            "path": str(filepath),
                            "summary": summary,
                            "size": len(text)
                        }
                    )
                ]
            )

            # Store reference in PostgreSQL with all required fields
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO echo_unified_interactions
                (conversation_id, user_id, query, response, model_used,
                 processing_time, escalation_path, metadata, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT DO NOTHING
            """, (
                f"claude_{file_id}",
                "patrick",
                f"Claude conversation: {filepath.name}",
                summary[:500],
                "claude",
                0.0,  # processing_time
                psycopg2.extras.Json([]),  # escalation_path
                psycopg2.extras.Json(metadata)
            ))
            self.db_conn.commit()
            cursor.close()

            logger.info(f"✓ Indexed: {filepath.name}")
            return True

        except Exception as e:
            logger.error(f"Error indexing {filepath}: {e}")
            return False

    async def index_all_conversations(self, limit=100):
        """Index all Claude conversations."""
        # Create collection
        await self.create_collection()

        # Get conversation files
        files = list(self.conversations_path.glob("*.json"))
        files.extend(list(self.conversations_path.glob("*.md")))

        logger.info(f"Found {len(files)} conversation files")

        # Index files (limited for initial test)
        indexed = 0
        for filepath in files[:limit]:
            if await self.index_conversation(filepath):
                indexed += 1

        logger.info(f"Indexed {indexed}/{min(len(files), limit)} conversations")

        # Update stats
        info = self.qdrant.get_collection(self.collection_name)
        logger.info(f"Collection now has {info.points_count} vectors")

    async def search_conversations(self, query: str, limit=5):
        """Search Claude conversations for relevant context."""
        # Get query embedding
        embedding = await self.get_embedding(query)
        if not embedding:
            return []

        # Search Qdrant using the correct method
        from src.api.models import PointIdsList, QueryVector, QueryRequest
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit
        ).points

        return [
            {
                "file": r.payload["path"],
                "service": r.payload["service"],
                "type": r.payload["type"],
                "date": r.payload["date"],
                "score": r.score,
                "summary": r.payload["summary"]
            }
            for r in results
        ]

async def main():
    """Main indexing function."""
    indexer = ClaudeMemoryIndexer()

    print("\n🧠 CLAUDE MEMORY INTEGRATION")
    print("=" * 60)

    # Index conversations
    print("\nIndexing Claude conversations...")
    await indexer.index_all_conversations(limit=50)  # Start with 50

    # Test search
    print("\nTesting search...")
    query = "anime production system architecture"
    results = await indexer.search_conversations(query)

    print(f"\nSearch results for: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {Path(r['file']).name}")
        print(f"   Type: {r['type']}, Service: {r['service']}")
        print(f"   Score: {r['score']:.3f}")
        print(f"   Summary: {r['summary'][:100]}...")

    print("\n✅ Claude memory integration complete!")

if __name__ == "__main__":
    asyncio.run(main())