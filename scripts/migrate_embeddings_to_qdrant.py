#!/usr/bin/env python3
"""
Migration script to sync embedding_cache from PostgreSQL to Qdrant
Transfers 53,962 embeddings to echo_memories collection
"""

import asyncio
import asyncpg
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingMigrator:
    def __init__(self):
        # PostgreSQL connection
        self.pg_host = "localhost"
        self.pg_port = 5432
        self.pg_db = "echo_brain"
        self.pg_user = "patrick"
        self.pg_password = os.environ.get("ECHO_BRAIN_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")

        # Qdrant connection
        self.qdrant_client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=30
        )
        # Use a new collection for these embeddings with correct dimension
        self.collection_name = "embedding_cache_vectors"
        self.vector_dimension = 1536  # Actual dimension from embedding_cache

        self.batch_size = 100
        self.total_processed = 0
        self.total_failed = 0

    async def connect_postgres(self) -> asyncpg.Connection:
        """Connect to PostgreSQL database"""
        return await asyncpg.connect(
            host=self.pg_host,
            port=self.pg_port,
            database=self.pg_db,
            user=self.pg_user,
            password=self.pg_password
        )

    def ensure_collection_exists(self):
        """Ensure Qdrant collection exists with correct configuration"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension,
                    distance=Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection {self.collection_name} already exists")

    async def fetch_embeddings(self, conn: asyncpg.Connection, offset: int) -> List[Dict[str, Any]]:
        """Fetch a batch of embeddings from PostgreSQL"""
        query = """
            SELECT
                id,
                text_hash,
                text,
                embedding,
                model,
                dimensions,
                created_at,
                accessed_at
            FROM embedding_cache
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """

        rows = await conn.fetch(query, self.batch_size, offset)
        embeddings = []

        for row in rows:
            embeddings.append({
                'id': str(row['id']),
                'text': row['text'],
                'embedding': list(row['embedding']),  # Convert from array to list
                'metadata': {
                    'text_hash': row['text_hash'],
                    'dimensions': row['dimensions'],
                    'accessed_at': row['accessed_at'].isoformat() if row['accessed_at'] else None
                },
                'model': row['model'],
                'created_at': row['created_at'].isoformat() if row['created_at'] else None
            })

        return embeddings

    def prepare_points(self, embeddings: List[Dict[str, Any]]) -> List[PointStruct]:
        """Convert embeddings to Qdrant points"""
        points = []

        for emb in embeddings:
            # Prepare metadata
            metadata = {
                'text': emb['text'][:1000],  # Truncate very long text
                'source': 'embedding_cache_migration',
                'model': emb['model'] or 'unknown',
                'migrated_at': datetime.now().isoformat(),
                'original_id': emb['id'],
                'created_at': emb['created_at']
            }

            # Add any existing metadata
            if emb['metadata']:
                metadata.update(emb['metadata'])

            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),  # Generate new UUID for Qdrant
                vector=emb['embedding'],
                payload=metadata
            )
            points.append(point)

        return points

    async def migrate(self):
        """Run the migration"""
        logger.info("Starting embedding migration from PostgreSQL to Qdrant")

        # Ensure collection exists
        self.ensure_collection_exists()

        # Get current count in Qdrant
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        initial_count = collection_info.points_count
        logger.info(f"Initial Qdrant vectors: {initial_count}")

        # Connect to PostgreSQL
        conn = await self.connect_postgres()

        try:
            # Get total count
            count_result = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
            logger.info(f"Total embeddings to migrate: {count_result}")

            # Process in batches
            offset = 0
            while True:
                # Fetch batch
                embeddings = await self.fetch_embeddings(conn, offset)
                if not embeddings:
                    break

                try:
                    # Convert to points
                    points = self.prepare_points(embeddings)

                    # Upload to Qdrant
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )

                    self.total_processed += len(embeddings)
                    logger.info(f"Migrated batch: {offset} - {offset + len(embeddings)} "
                               f"({self.total_processed}/{count_result} total)")

                except Exception as e:
                    logger.error(f"Failed to migrate batch at offset {offset}: {e}")
                    self.total_failed += len(embeddings)

                offset += self.batch_size

                # Small delay to avoid overload
                await asyncio.sleep(0.1)

        finally:
            await conn.close()

        # Get final count
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        final_count = collection_info.points_count

        logger.info("=" * 60)
        logger.info("Migration completed!")
        logger.info(f"Total processed: {self.total_processed}")
        logger.info(f"Total failed: {self.total_failed}")
        logger.info(f"Qdrant vectors before: {initial_count}")
        logger.info(f"Qdrant vectors after: {final_count}")
        logger.info(f"New vectors added: {final_count - initial_count}")
        logger.info("=" * 60)

async def main():
    """Main entry point"""
    migrator = EmbeddingMigrator()
    await migrator.migrate()

if __name__ == "__main__":
    asyncio.run(main())