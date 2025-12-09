#!/usr/bin/env python3
"""
Index ALL Claude conversations into Echo's memory in batches.
Run this in the background to process all 12,248 files.
"""

import asyncio
import sys
import os
sys.path.append('/opt/tower-echo-brain')
os.chdir('/opt/tower-echo-brain')

from index_claude_memory import ClaudeMemoryIndexer
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/tower-echo-brain/logs/claude_indexing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def index_all():
    """Index all Claude conversations in batches."""

    indexer = ClaudeMemoryIndexer()

    # Create collection
    await indexer.create_collection()

    # Get all conversation files
    files = list(indexer.conversations_path.glob("*.json"))
    files.extend(list(indexer.conversations_path.glob("*.md")))

    total = len(files)
    logger.info(f"Starting to index {total} Claude conversation files")

    indexed = 0
    failed = 0
    batch_size = 100

    start_time = datetime.now()

    # Process in batches
    for i in range(0, total, batch_size):
        batch = files[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}: files {i+1}-{min(i+batch_size, total)} of {total}")

        for filepath in batch:
            try:
                # Reset transaction if needed
                indexer.db_conn.rollback()

                if await indexer.index_conversation(filepath):
                    indexed += 1
                else:
                    failed += 1

                if indexed % 100 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = indexed / elapsed if elapsed > 0 else 0
                    eta = (total - indexed) / rate if rate > 0 else 0
                    logger.info(f"Progress: {indexed}/{total} indexed ({indexed*100/total:.1f}%), "
                               f"Rate: {rate:.1f}/sec, ETA: {eta/60:.1f} min")

            except Exception as e:
                logger.error(f"Error with {filepath.name}: {e}")
                failed += 1
                indexer.db_conn.rollback()

        # Small delay between batches
        await asyncio.sleep(1)

    # Final stats
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"""
    ========================================
    CLAUDE MEMORY INDEXING COMPLETE
    ========================================
    Total files: {total}
    Successfully indexed: {indexed}
    Failed: {failed}
    Time taken: {elapsed/60:.1f} minutes
    Average rate: {indexed/elapsed:.1f} files/sec

    Collection stats:
    """)

    try:
        info = indexer.qdrant.get_collection(indexer.collection_name)
        logger.info(f"  Vectors in Qdrant: {info.points_count}")
    except:
        pass

    logger.info("Claude memory integration complete!")

if __name__ == "__main__":
    asyncio.run(index_all())