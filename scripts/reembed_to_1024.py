#!/usr/bin/env python3
"""
Re-embed all records from embedding_cache using mxbai-embed-large (1024-dim)
Stores results in Qdrant collection "echo_memory"
Includes progress tracking, resume capability, and time estimation
"""

import asyncio
import asyncpg
import json
import logging
import sys
import os
import time
import signal
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import httpx
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Setup logging to file
log_dir = Path('/var/log/tower')
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'reembedding_1024.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReembeddingProcessor:
    def __init__(self):
        # PostgreSQL settings
        self.pg_host = "localhost"
        self.pg_port = 5432
        self.pg_db = "echo_brain"
        self.pg_user = "patrick"
        self.pg_password = os.environ.get("ECHO_BRAIN_DB_PASSWORD", os.getenv("TOWER_DB_PASSWORD", os.getenv("DB_PASSWORD", "")))

        # Qdrant settings
        self.qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)
        self.collection_name = "echo_memory"  # Standard collection with 1024 dimensions
        self.vector_dimension = 1024  # mxbai-embed-large dimensions

        # Ollama settings
        self.ollama_url = "http://localhost:11434"
        self.embedding_model = "mxbai-embed-large:latest"  # 1024-dim model

        # Processing settings
        self.batch_size = 50  # Process 50 records at a time
        self.progress_update_interval = 100  # Update progress every 100 records

        # Statistics
        self.start_time = None
        self.total_records = 0
        self.processed_records = 0
        self.failed_records = 0
        self.skipped_records = 0

        # Graceful shutdown
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Shutdown requested (signal {signum})")
        self.shutdown_requested = True

    async def connect_postgres(self) -> asyncpg.Connection:
        """Connect to PostgreSQL"""
        return await asyncpg.connect(
            host=self.pg_host,
            port=self.pg_port,
            database=self.pg_db,
            user=self.pg_user,
            password=self.pg_password
        )

    async def create_progress_table(self, conn: asyncpg.Connection):
        """Create progress tracking table if it doesn't exist"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS reembedding_progress (
                id SERIAL PRIMARY KEY,
                embedding_cache_id INTEGER UNIQUE NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                processed_at TIMESTAMP,
                error_message TEXT,
                new_vector_id VARCHAR(64),
                processing_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for faster queries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reembedding_status
            ON reembedding_progress(status)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reembedding_cache_id
            ON reembedding_progress(embedding_cache_id)
        """)

        logger.info("Progress tracking table ready")

    def ensure_collection_exists(self):
        """Ensure Qdrant collection exists with correct dimensions"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name} with {self.vector_dimension} dimensions")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension,
                    distance=Distance.COSINE
                )
            )
        else:
            # Verify dimensions
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            existing_dim = collection_info.config.params.vectors.size
            if existing_dim != self.vector_dimension:
                logger.error(f"Collection {self.collection_name} exists with {existing_dim} dimensions, expected {self.vector_dimension}")
                raise ValueError(f"Dimension mismatch: collection has {existing_dim}, expected {self.vector_dimension}")
            logger.info(f"Collection {self.collection_name} already exists with correct dimensions")

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama with mxbai-embed-large"""
        # Clean and truncate text for Ollama stability
        MAX_TEXT_LENGTH = 1500  # Safe limit based on Ollama testing

        # Clean problematic characters
        text = text.replace('\x00', '')  # Remove null bytes
        text = ''.join(char for char in text if ord(char) < 65536)  # Remove high unicode

        if len(text) > MAX_TEXT_LENGTH:
            # Simple truncation - just take the first part
            text = text[:MAX_TEXT_LENGTH]
            logger.debug(f"Truncated text to {len(text)} chars")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get('embedding')
                    if embedding and len(embedding) == self.vector_dimension:
                        return embedding
                    else:
                        logger.error(f"Invalid embedding dimension: got {len(embedding) if embedding else 0}, expected {self.vector_dimension}")
                        return None
                else:
                    logger.error(f"Embedding generation failed: {response.status_code} for text length {len(text)}")
                    return None

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    async def get_unprocessed_batch(self, conn: asyncpg.Connection) -> List[Dict[str, Any]]:
        """Get a batch of unprocessed records"""
        query = """
            SELECT
                ec.id,
                ec.text,
                ec.text_hash,
                ec.model as original_model,
                ec.created_at,
                ec.dimensions as original_dimensions
            FROM embedding_cache ec
            LEFT JOIN reembedding_progress rp
                ON ec.id = rp.embedding_cache_id
            WHERE rp.status IS NULL OR rp.status = 'failed'
            ORDER BY ec.id
            LIMIT $1
        """

        rows = await conn.fetch(query, self.batch_size)
        return [dict(row) for row in rows]

    async def process_batch(self, conn: asyncpg.Connection, batch: List[Dict[str, Any]]):
        """Process a batch of records"""
        for record in batch:
            if self.shutdown_requested:
                logger.info("Shutdown requested, stopping batch processing")
                break

            start_time = time.time()
            cache_id = record['id']
            text = record['text']

            try:
                # Generate new embedding
                embedding = await self.generate_embedding(text)

                if embedding and len(embedding) == self.vector_dimension:
                    # Store in Qdrant
                    vector_id = str(uuid.uuid4())
                    point = PointStruct(
                        id=vector_id,
                        vector=embedding,
                        payload={
                            'text': text[:1000],  # Truncate long text
                            'original_cache_id': cache_id,
                            'text_hash': record['text_hash'],
                            'original_model': record['original_model'],
                            'original_dimensions': record['original_dimensions'],
                            'reembedded_model': self.embedding_model,
                            'reembedded_dimensions': self.vector_dimension,
                            'reembedded_at': datetime.now().isoformat(),
                            'created_at': record['created_at'].isoformat() if record['created_at'] else None
                        }
                    )

                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=[point]
                    )

                    # Update progress
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    await conn.execute("""
                        INSERT INTO reembedding_progress
                            (embedding_cache_id, status, processed_at, new_vector_id, processing_time_ms)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (embedding_cache_id)
                        DO UPDATE SET
                            status = $2,
                            processed_at = $3,
                            new_vector_id = $4,
                            processing_time_ms = $5
                    """, cache_id, 'completed', datetime.now(), vector_id, processing_time_ms)

                    self.processed_records += 1

                else:
                    # Failed to generate embedding
                    await conn.execute("""
                        INSERT INTO reembedding_progress
                            (embedding_cache_id, status, processed_at, error_message)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (embedding_cache_id)
                        DO UPDATE SET
                            status = $2,
                            processed_at = $3,
                            error_message = $4
                    """, cache_id, 'failed', datetime.now(), 'Failed to generate embedding')

                    self.failed_records += 1

            except Exception as e:
                logger.error(f"Error processing record {cache_id}: {e}")
                await conn.execute("""
                    INSERT INTO reembedding_progress
                        (embedding_cache_id, status, processed_at, error_message)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (embedding_cache_id)
                    DO UPDATE SET
                        status = $2,
                        processed_at = $3,
                        error_message = $4
                """, cache_id, 'failed', datetime.now(), str(e))

                self.failed_records += 1

            # Log progress periodically
            if (self.processed_records + self.failed_records) % self.progress_update_interval == 0:
                await self.log_progress(conn)

    async def log_progress(self, conn: asyncpg.Connection):
        """Log current progress and estimate remaining time"""
        # Get current stats
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'completed') as completed,
                COUNT(*) FILTER (WHERE status = 'failed') as failed,
                COUNT(*) FILTER (WHERE status IS NULL) as pending,
                AVG(processing_time_ms) FILTER (WHERE status = 'completed') as avg_time_ms
            FROM (
                SELECT ec.id, rp.status, rp.processing_time_ms
                FROM embedding_cache ec
                LEFT JOIN reembedding_progress rp ON ec.id = rp.embedding_cache_id
            ) t
        """)

        completed = stats['completed'] or 0
        failed = stats['failed'] or 0
        pending = self.total_records - completed - failed
        avg_time_ms = stats['avg_time_ms'] or 1000

        # Calculate time estimates
        elapsed_time = time.time() - self.start_time
        if completed > 0:
            rate = completed / elapsed_time  # records per second
            if rate > 0:
                remaining_seconds = pending / rate
                eta = datetime.now() + timedelta(seconds=remaining_seconds)

                logger.info(f"""
===== PROGRESS UPDATE (1024-dim) =====
Model: {self.embedding_model}
Collection: {self.collection_name}
Dimensions: {self.vector_dimension}
Total: {self.total_records:,}
Completed: {completed:,} ({completed/self.total_records*100:.1f}%)
Failed: {failed:,}
Pending: {pending:,}
Rate: {rate:.1f} records/sec
Avg time per record: {avg_time_ms:.0f}ms
Elapsed: {timedelta(seconds=int(elapsed_time))}
Estimated remaining: {timedelta(seconds=int(remaining_seconds))}
ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}
======================================
""")
        else:
            logger.info(f"Starting processing... {completed} completed so far")

    async def reset_progress(self, conn: asyncpg.Connection):
        """Reset all progress to re-process everything"""
        logger.warning("Resetting all progress - will re-process all records")
        await conn.execute("DELETE FROM reembedding_progress")
        logger.info("Progress reset complete")

    async def get_status(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """Get current processing status"""
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE status = 'completed') as completed,
                COUNT(*) FILTER (WHERE status = 'failed') as failed,
                COUNT(*) FILTER (WHERE status = 'pending') as pending,
                AVG(processing_time_ms) FILTER (WHERE status = 'completed') as avg_time_ms,
                MIN(processed_at) FILTER (WHERE status = 'completed') as first_processed,
                MAX(processed_at) FILTER (WHERE status = 'completed') as last_processed
            FROM (
                SELECT ec.id,
                       COALESCE(rp.status, 'pending') as status,
                       rp.processing_time_ms,
                       rp.processed_at
                FROM embedding_cache ec
                LEFT JOIN reembedding_progress rp ON ec.id = rp.embedding_cache_id
            ) t
        """)

        return {
            'model': self.embedding_model,
            'collection': self.collection_name,
            'dimensions': self.vector_dimension,
            'total': self.total_records,
            'completed': stats['completed'] or 0,
            'failed': stats['failed'] or 0,
            'pending': stats['pending'] or self.total_records,
            'avg_processing_time_ms': float(stats['avg_time_ms']) if stats['avg_time_ms'] else 0,
            'first_processed': stats['first_processed'].isoformat() if stats['first_processed'] else None,
            'last_processed': stats['last_processed'].isoformat() if stats['last_processed'] else None,
            'is_running': not self.shutdown_requested
        }

    async def run_test_batch(self, conn: asyncpg.Connection) -> float:
        """Run a test batch of 10 records to estimate time"""
        logger.info(f"Running test batch of 10 records with {self.embedding_model}...")

        # Get 10 test records
        test_batch = await conn.fetch("""
            SELECT id, text, text_hash, model, created_at, dimensions
            FROM embedding_cache
            LIMIT 10
        """)

        start_time = time.time()
        successful = 0

        for row in test_batch:
            embedding = await self.generate_embedding(row['text'])
            if embedding and len(embedding) == self.vector_dimension:
                successful += 1
                logger.info(f"✓ Generated {self.vector_dimension}-dim embedding successfully")
            else:
                logger.warning(f"✗ Failed to generate embedding")
            await asyncio.sleep(0.1)  # Small delay between requests

        elapsed = time.time() - start_time
        avg_time = elapsed / 10

        logger.info(f"""
===== TEST BATCH RESULTS (1024-dim) =====
Model: {self.embedding_model}
Collection: {self.collection_name}
Dimensions: {self.vector_dimension}
Records tested: 10
Successful: {successful}
Failed: {10 - successful}
Total time: {elapsed:.2f} seconds
Average per record: {avg_time:.2f} seconds
Estimated time for {self.total_records:,} records: {timedelta(seconds=int(avg_time * self.total_records))}
==========================================
""")

        return avg_time

    async def run(self, test_mode: bool = False, reset: bool = False):
        """Main processing loop"""
        logger.info(f"Starting reembedding processor for {self.vector_dimension} dimensions...")
        logger.info(f"Model: {self.embedding_model}")
        logger.info(f"Collection: {self.collection_name}")

        conn = await self.connect_postgres()

        try:
            # Setup
            await self.create_progress_table(conn)

            if reset:
                await self.reset_progress(conn)

            self.ensure_collection_exists()

            # Get total count
            self.total_records = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
            logger.info(f"Total records to process: {self.total_records:,}")

            if test_mode:
                # Run test batch and exit
                await self.run_test_batch(conn)
                return

            self.start_time = time.time()

            # Process in batches
            while not self.shutdown_requested:
                batch = await self.get_unprocessed_batch(conn)

                if not batch:
                    logger.info("No more records to process")
                    break

                logger.info(f"Processing batch of {len(batch)} records...")
                await self.process_batch(conn, batch)

                # Small delay between batches
                await asyncio.sleep(0.5)

            # Final progress report
            await self.log_progress(conn)

            # Final status
            status = await self.get_status(conn)
            logger.info(f"""
===== FINAL RESULTS (1024-dim) =====
Model: {status['model']}
Collection: {status['collection']}
Dimensions: {status['dimensions']}
Total processed: {status['completed']:,}
Failed: {status['failed']:,}
Pending: {status['pending']:,}
Total time: {timedelta(seconds=int(time.time() - self.start_time))}
=====================================
""")

        finally:
            await conn.close()

        logger.info("Reembedding processor finished")

async def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Re-embed vectors to 1024 dimensions using mxbai-embed-large')
    parser.add_argument('--test', action='store_true', help='Run test batch only')
    parser.add_argument('--status', action='store_true', help='Show current status and exit')
    parser.add_argument('--reset', action='store_true', help='Reset progress and start from beginning')
    args = parser.parse_args()

    processor = ReembeddingProcessor()

    if args.status:
        # Just show status and exit
        conn = await processor.connect_postgres()
        try:
            await processor.create_progress_table(conn)
            processor.total_records = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
            status = await processor.get_status(conn)
            print(json.dumps(status, indent=2))
        finally:
            await conn.close()
    else:
        await processor.run(test_mode=args.test, reset=args.reset)

if __name__ == "__main__":
    asyncio.run(main())