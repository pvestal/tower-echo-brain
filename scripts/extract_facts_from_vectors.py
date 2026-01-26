#!/usr/bin/env python3
"""
Extract facts from Qdrant echo_memory_768 collection vectors.
Processes vectors, extracts structured facts, stores in PostgreSQL.
Resumable with progress tracking.
"""

import asyncio
import asyncpg
import json
import logging
import sys
import os
import time
import signal
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4
import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

# Setup logging
log_dir = Path('/var/log/tower')
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'fact_extraction.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FactExtractor:
    def __init__(self):
        # PostgreSQL settings
        self.pg_host = "localhost"
        self.pg_port = 5432
        self.pg_db = "echo_brain"
        self.pg_user = "patrick"
        self.pg_password = os.environ.get("ECHO_BRAIN_DB_PASSWORD", os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"))

        # Qdrant settings
        self.qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)
        self.source_collection = "echo_memory_768"  # Will read from this collection

        # Ollama settings for fact extraction
        self.ollama_url = "http://localhost:11434"
        self.extraction_model = "gemma2:9b"  # Fast and good at following instructions

        # Processing settings
        self.batch_size = 20  # Process 20 vectors at a time
        self.min_text_length = 50  # Skip very short texts
        self.max_text_length = 10000  # Truncate very long texts for extraction

        # Statistics
        self.start_time = None
        self.total_vectors = 0
        self.processed_vectors = 0
        self.facts_extracted = 0
        self.failed_vectors = 0

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

    async def extract_facts_from_text(self, text: str, client: httpx.AsyncClient) -> List[Dict[str, str]]:
        """Use LLM to extract facts from text."""
        # Truncate if too long
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "..."

        prompt = f"""Extract factual information from this text as structured triples.

Rules:
1. Only extract CONCRETE facts, not opinions or speculation
2. Focus on facts about: Patrick, Tower server, Echo Brain, projects, technical details, family, locations
3. Each fact should be: (subject, predicate, object)
4. Be specific - include model numbers, dates, versions, quantities when mentioned
5. Return JSON array of objects with keys: subject, predicate, object

Text to analyze:
---
{text}
---

Return ONLY a JSON array, no explanation. Example format:
[
  {{"subject": "Patrick", "predicate": "owns", "object": "2022 Toyota Tundra 1794 Edition"}},
  {{"subject": "Tower server", "predicate": "runs", "object": "Ubuntu 24.04 LTS"}},
  {{"subject": "Echo Brain", "predicate": "uses", "object": "PostgreSQL database"}}
]

If no facts can be extracted, return: []

JSON:"""

        try:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.extraction_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,  # Lower temperature for more consistent extraction
                    "format": "json"
                },
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()

                # Clean up response
                response_text = re.sub(r'^```json\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)

                try:
                    facts = json.loads(response_text)
                    if isinstance(facts, list):
                        # Validate structure
                        valid_facts = []
                        for fact in facts:
                            if (isinstance(fact, dict) and
                                'subject' in fact and
                                'predicate' in fact and
                                'object' in fact):
                                # Clean and validate
                                subject = str(fact['subject']).strip()[:200]
                                predicate = str(fact['predicate']).strip()[:200]
                                obj = str(fact['object']).strip()

                                if subject and predicate and obj:
                                    valid_facts.append({
                                        'subject': subject,
                                        'predicate': predicate,
                                        'object': obj
                                    })

                        return valid_facts
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")

            return []

        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

    async def get_unprocessed_batch(self) -> List[Dict[str, Any]]:
        """Get a batch of unprocessed vectors from Qdrant"""
        try:
            # Get total count
            collection_info = self.qdrant_client.get_collection(self.source_collection)
            self.total_vectors = collection_info.points_count

            # Scroll through vectors to find unprocessed ones
            # We'll check against fact_extraction_log to see what's been processed
            conn = await self.connect_postgres()
            try:
                # Get already processed point IDs
                processed_ids = await conn.fetch("""
                    SELECT source_point_id
                    FROM fact_extraction_log
                    WHERE source_collection = $1
                """, self.source_collection)

                processed_set = {row['source_point_id'] for row in processed_ids}

                # Scroll through Qdrant to find unprocessed vectors
                offset = None
                unprocessed = []

                while len(unprocessed) < self.batch_size:
                    response = self.qdrant_client.scroll(
                        collection_name=self.source_collection,
                        scroll_filter=None,
                        limit=100,
                        offset=offset
                    )

                    if not response[0]:  # No more points
                        break

                    for point in response[0]:
                        if str(point.id) not in processed_set:
                            # Get text from payload
                            text = point.payload.get('text', '')
                            if len(text) >= self.min_text_length:
                                unprocessed.append({
                                    'id': str(point.id),
                                    'text': text,
                                    'metadata': point.payload
                                })

                            if len(unprocessed) >= self.batch_size:
                                break

                    offset = response[1]  # Next offset

                    if offset is None:  # Reached end of collection
                        break

                return unprocessed

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Failed to get unprocessed batch: {e}")
            return []

    async def store_facts(self, conn: asyncpg.Connection, facts: List[Dict[str, str]],
                          source_point_id: str) -> int:
        """Store facts in PostgreSQL"""
        stored_count = 0

        for fact in facts:
            try:
                # Check if fact already exists (unique constraint)
                exists = await conn.fetchval("""
                    SELECT 1 FROM facts
                    WHERE subject = $1 AND predicate = $2 AND object = $3
                """, fact['subject'], fact['predicate'], fact['object'])

                if not exists:
                    await conn.execute("""
                        INSERT INTO facts
                        (id, subject, predicate, object, qdrant_point_id, confidence, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                        str(uuid4()),
                        fact['subject'],
                        fact['predicate'],
                        fact['object'],
                        source_point_id,
                        0.8,  # Default confidence for LLM extraction
                        datetime.now()
                    )
                    stored_count += 1

            except Exception as e:
                logger.warning(f"Failed to store fact: {e}")

        return stored_count

    async def process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of vectors"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            conn = await self.connect_postgres()

            try:
                for vector_data in batch:
                    if self.shutdown_requested:
                        logger.info("Shutdown requested, stopping batch processing")
                        break

                    point_id = vector_data['id']
                    text = vector_data['text']

                    try:
                        # Extract facts
                        facts = await self.extract_facts_from_text(text, client)

                        # Store facts
                        stored_count = 0
                        if facts:
                            stored_count = await self.store_facts(conn, facts, point_id)
                            self.facts_extracted += stored_count

                        # Log extraction
                        content_hash = hashlib.sha256(text.encode()).hexdigest()
                        await conn.execute("""
                            INSERT INTO fact_extraction_log
                            (id, source_collection, source_point_id, content_hash,
                             extraction_model, facts_extracted, processed_at, status)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (source_collection, source_point_id)
                            DO UPDATE SET
                                facts_extracted = $6,
                                processed_at = $7,
                                status = $8
                        """,
                            str(uuid4()),
                            self.source_collection,
                            point_id,
                            content_hash,
                            self.extraction_model,
                            stored_count,
                            datetime.now(),
                            'completed'
                        )

                        self.processed_vectors += 1
                        logger.info(f"Processed vector {point_id}: {stored_count} facts extracted")

                    except Exception as e:
                        logger.error(f"Error processing vector {point_id}: {e}")
                        # Log failure
                        await conn.execute("""
                            INSERT INTO fact_extraction_log
                            (id, source_collection, source_point_id, extraction_model,
                             facts_extracted, processed_at, status, error_message)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (source_collection, source_point_id)
                            DO UPDATE SET
                                status = $7,
                                error_message = $8,
                                processed_at = $6
                        """,
                            str(uuid4()),
                            self.source_collection,
                            point_id,
                            self.extraction_model,
                            0,
                            datetime.now(),
                            'failed',
                            str(e)
                        )
                        self.failed_vectors += 1

                    # Log progress periodically
                    if (self.processed_vectors + self.failed_vectors) % 100 == 0:
                        await self.log_progress(conn)

            finally:
                await conn.close()

    async def log_progress(self, conn: asyncpg.Connection):
        """Log extraction progress"""
        # Get statistics
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as total_processed,
                SUM(facts_extracted) as total_facts,
                COUNT(*) FILTER (WHERE status = 'completed') as successful,
                COUNT(*) FILTER (WHERE status = 'failed') as failed
            FROM fact_extraction_log
            WHERE source_collection = $1
        """, self.source_collection)

        elapsed = time.time() - self.start_time
        rate = self.processed_vectors / elapsed if elapsed > 0 else 0

        # Estimate remaining time
        remaining = self.total_vectors - stats['total_processed']
        eta_seconds = remaining / rate if rate > 0 else 0
        eta = datetime.now() + timedelta(seconds=eta_seconds)

        logger.info(f"""
===== EXTRACTION PROGRESS =====
Source: {self.source_collection}
Total vectors: {self.total_vectors:,}
Processed: {stats['total_processed']:,} ({stats['total_processed']/self.total_vectors*100:.1f}%)
Successful: {stats['successful']:,}
Failed: {stats['failed']:,}
Facts extracted: {stats['total_facts']:,}
Rate: {rate:.1f} vectors/sec
Elapsed: {timedelta(seconds=int(elapsed))}
ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}
================================
""")

    async def run(self):
        """Main extraction loop"""
        logger.info(f"Starting fact extraction from {self.source_collection}")

        self.start_time = time.time()

        while not self.shutdown_requested:
            # Get batch of unprocessed vectors
            batch = await self.get_unprocessed_batch()

            if not batch:
                logger.info("No more vectors to process")
                break

            logger.info(f"Processing batch of {len(batch)} vectors...")
            await self.process_batch(batch)

            # Small delay between batches
            await asyncio.sleep(1.0)

        # Final report
        conn = await self.connect_postgres()
        try:
            await self.log_progress(conn)
        finally:
            await conn.close()

        logger.info("Fact extraction completed")

async def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Extract facts from Qdrant vectors')
    parser.add_argument('--collection', default='echo_memory_768', help='Source collection name')
    parser.add_argument('--status', action='store_true', help='Show status and exit')
    args = parser.parse_args()

    if args.status:
        # Show extraction status
        extractor = FactExtractor()
        conn = await extractor.connect_postgres()
        try:
            stats = await conn.fetchrow("""
                SELECT
                    source_collection,
                    COUNT(*) as vectors_processed,
                    SUM(facts_extracted) as total_facts,
                    MAX(processed_at) as last_run,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed
                FROM fact_extraction_log
                WHERE source_collection = $1
                GROUP BY source_collection
            """, args.collection)

            if stats:
                print(f"""
Fact Extraction Status for {args.collection}:
  Vectors processed: {stats['vectors_processed']:,}
  Facts extracted: {stats['total_facts']:,}
  Failed: {stats['failed']:,}
  Last run: {stats['last_run']}
""")
            else:
                print(f"No extraction history for {args.collection}")

        finally:
            await conn.close()
    else:
        extractor = FactExtractor()
        if args.collection:
            extractor.source_collection = args.collection
        await extractor.run()

if __name__ == "__main__":
    asyncio.run(main())