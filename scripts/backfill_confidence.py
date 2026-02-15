#!/usr/bin/env python3
"""
One-time backfill: sets confidence=0.7, last_accessed=<ingested_at>, access_count=0
on all vectors in echo_memory that are missing the confidence field.

Run after deploying v0.6.0:
    cd /opt/tower-echo-brain && source venv/bin/activate
    python scripts/backfill_confidence.py
"""
import asyncio
import logging
import httpx
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
COLLECTION = "echo_memory"
BATCH_SIZE = 100
DEFAULT_CONFIDENCE = 0.7


async def backfill():
    updated = 0
    skipped = 0

    async with httpx.AsyncClient(timeout=30) as client:
        offset = None
        while True:
            body = {
                "limit": BATCH_SIZE,
                "with_payload": True,
                "with_vector": False,
            }
            if offset:
                body["offset"] = offset

            # Also try filtering for points missing confidence
            # Qdrant doesn't have a native "field missing" filter,
            # so we scroll everything and check client-side
            resp = await client.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                json=body,
            )
            resp.raise_for_status()
            data = resp.json().get("result", {})
            points = data.get("points", [])
            offset = data.get("next_page_offset")

            if not points:
                break

            for point in points:
                payload = point.get("payload", {})

                # Skip if already has confidence field
                if "confidence" in payload:
                    skipped += 1
                    continue

                # Backfill fields
                ingested_at = (
                    payload.get("ingested_at")
                    or payload.get("timestamp")
                    or payload.get("extracted_at")
                    or datetime.now().isoformat()
                )
                new_fields = {
                    "confidence": DEFAULT_CONFIDENCE,
                    "last_accessed": ingested_at,
                    "access_count": 0,
                }

                await client.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/payload",
                    json={
                        "payload": new_fields,
                        "points": [point["id"]],
                    },
                )
                updated += 1

                if updated % 500 == 0:
                    logger.info(f"Progress: {updated} updated, {skipped} skipped")

            if not offset:
                break

    logger.info(f"Backfill complete: {updated} updated, {skipped} already had confidence")


if __name__ == "__main__":
    asyncio.run(backfill())
