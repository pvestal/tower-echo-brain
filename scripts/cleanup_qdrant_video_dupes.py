#!/usr/bin/env python3
"""
One-time cleanup: remove orphaned duplicate video points from Qdrant.

The queue_reanalysis method used to clear qdrant_point_id in Postgres
without deleting the old Qdrant point, causing duplicates on re-ingest.

This script:
1. Gets all valid video qdrant_point_ids from Postgres
2. Scrolls all type=video points in Qdrant
3. Deletes any Qdrant point whose ID is NOT in the Postgres set
"""
import asyncio
import os

import asyncpg
import httpx

DB_URL = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "echo_memory")

DRY_RUN = os.getenv("DRY_RUN", "1") == "1"


async def main():
    # Step 1: Get all valid qdrant_point_ids from Postgres
    conn = await asyncpg.connect(DB_URL)
    try:
        rows = await conn.fetch(
            "SELECT qdrant_point_id FROM photos WHERE qdrant_point_id IS NOT NULL"
        )
        valid_ids = {r["qdrant_point_id"] for r in rows}
        print(f"Postgres: {len(valid_ids)} valid qdrant_point_ids")
    finally:
        await conn.close()

    # Step 2: Scroll all type=video points in Qdrant
    orphaned_ids = []
    offset = None
    total_video_points = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            body = {
                "filter": {"must": [{"key": "type", "match": {"value": "video"}}]},
                "limit": 500,
                "with_payload": ["filename"],
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset

            resp = await client.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                json=body
            )
            data = resp.json()
            points = data["result"]["points"]
            if not points:
                break

            total_video_points += len(points)

            for point in points:
                pid = point["id"]
                if pid not in valid_ids:
                    orphaned_ids.append(pid)

            offset = data["result"].get("next_page_offset")
            if offset is None:
                break

        print(f"Qdrant: {total_video_points} total video points")
        print(f"Orphaned: {len(orphaned_ids)} points to delete")

        if not orphaned_ids:
            print("Nothing to clean up!")
            return

        if DRY_RUN:
            print(f"\n[DRY RUN] Would delete {len(orphaned_ids)} orphaned video points.")
            print("Run with DRY_RUN=0 to actually delete.")
            return

        # Step 3: Delete orphaned points in batches
        deleted = 0
        batch_size = 500
        for i in range(0, len(orphaned_ids), batch_size):
            batch = orphaned_ids[i:i + batch_size]
            resp = await client.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/delete",
                json={"points": batch}
            )
            if resp.status_code in (200, 201):
                deleted += len(batch)
                print(f"  Deleted batch {i // batch_size + 1}: {len(batch)} points")
            else:
                print(f"  ERROR deleting batch: {resp.status_code} {resp.text[:200]}")

        print(f"\nDone. Deleted {deleted} orphaned video points from Qdrant.")


if __name__ == "__main__":
    asyncio.run(main())
