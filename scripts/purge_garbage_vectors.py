#!/usr/bin/env python3
"""
Purge garbage vectors from Qdrant and PostgreSQL
Keeps only useful content types: documentation, code, schema, article
"""

import asyncio
import httpx
import asyncpg
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
DB_CONFIG = {
    "host": "localhost",
    "database": "echo_brain",
    "user": "patrick",
    "password": os.getenv("DB_PASSWORD", "")
}

# Keep these types
KEEP_TYPES = {"documentation", "code", "schema", "article", "fact", "project", "api"}

async def analyze_vectors():
    """Analyze current vector distribution"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get total count
        resp = await client.get(f"{QDRANT_URL}/collections/echo_memory")
        data = resp.json()
        total = data["result"]["points_count"]
        logger.info(f"Total vectors in Qdrant: {total:,}")

        # Sample vectors to understand distribution
        all_types = {}
        offset = None
        checked = 0

        while checked < min(total, 10000):  # Sample up to 10k
            payload = {"limit": 100, "with_payload": ["type", "content_type", "source"]}
            if offset:
                payload["offset"] = offset

            resp = await client.post(f"{QDRANT_URL}/collections/echo_memory/points/scroll", json=payload)
            batch = resp.json()

            if not batch["result"]["points"]:
                break

            for point in batch["result"]["points"]:
                vec_type = point["payload"].get("type") or point["payload"].get("content_type") or "null"
                all_types[vec_type] = all_types.get(vec_type, 0) + 1
                checked += 1

            if batch["result"].get("next_page_offset"):
                offset = batch["result"]["next_page_offset"]
            else:
                break

        logger.info(f"\nVector type distribution (sample of {checked}):")
        for vec_type, count in sorted(all_types.items(), key=lambda x: -x[1]):
            pct = (count / checked) * 100
            keep = "✓ KEEP" if vec_type in KEEP_TYPES else "✗ DELETE"
            logger.info(f"  {vec_type:30} {count:6} ({pct:5.1f}%) {keep}")

        # Estimate garbage
        garbage_count = sum(count for t, count in all_types.items() if t not in KEEP_TYPES)
        garbage_pct = (garbage_count / checked) * 100 if checked > 0 else 0

        logger.info(f"\nEstimated garbage: {garbage_pct:.1f}% ({int(total * garbage_pct / 100):,} vectors)")
        return total, garbage_pct

async def purge_qdrant_vectors():
    """Delete garbage vectors from Qdrant"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Build filter for deletion
        delete_filter = {
            "must_not": [
                {
                    "key": "type",
                    "match": {
                        "any": list(KEEP_TYPES)
                    }
                }
            ]
        }

        logger.info("\nDeleting vectors where type NOT IN " + str(KEEP_TYPES))

        # Delete in batches
        total_deleted = 0
        while True:
            # Find vectors to delete
            resp = await client.post(
                f"{QDRANT_URL}/collections/echo_memory/points/scroll",
                json={
                    "filter": delete_filter,
                    "limit": 100,
                    "with_payload": False
                }
            )
            batch = resp.json()

            if not batch["result"]["points"]:
                break

            # Delete this batch
            point_ids = [p["id"] for p in batch["result"]["points"]]
            delete_resp = await client.post(
                f"{QDRANT_URL}/collections/echo_memory/points/delete",
                json={"points": point_ids}
            )

            if delete_resp.status_code == 200:
                total_deleted += len(point_ids)
                logger.info(f"  Deleted batch of {len(point_ids)} vectors (total: {total_deleted})")
            else:
                logger.error(f"  Failed to delete batch: {delete_resp.text}")
                break

        logger.info(f"\nTotal vectors deleted from Qdrant: {total_deleted:,}")
        return total_deleted

async def clean_postgresql():
    """Clean PostgreSQL vector_content table"""
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        # Count before
        count_before = await conn.fetchval("SELECT COUNT(*) FROM vector_content")
        logger.info(f"\nPostgreSQL vector_content before: {count_before:,}")

        # Delete all with null content_type (all of them apparently)
        deleted = await conn.execute(
            "DELETE FROM vector_content WHERE content_type IS NULL"
        )
        logger.info(f"Deleted {deleted} rows with NULL content_type")

        # Count after
        count_after = await conn.fetchval("SELECT COUNT(*) FROM vector_content")
        logger.info(f"PostgreSQL vector_content after: {count_after:,}")

    finally:
        await conn.close()

async def main():
    logger.info("=== VECTOR GARBAGE PURGE ===")
    logger.info(f"Started at {datetime.now()}")

    # 1. Analyze current state
    total, garbage_pct = await analyze_vectors()

    if garbage_pct > 90:
        logger.warning("\n⚠️  Over 90% garbage detected!")
        logger.info("Proceeding with purge...")

        # 2. Purge Qdrant
        deleted = await purge_qdrant_vectors()

        # 3. Clean PostgreSQL
        await clean_postgresql()

        # 4. Check final state
        logger.info("\n=== FINAL STATE ===")
        await analyze_vectors()
    else:
        logger.info("\nGarbage level acceptable, no purge needed")

    logger.info(f"\nCompleted at {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(main())