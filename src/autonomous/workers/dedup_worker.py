"""
Background Dedup Worker — Scans existing vectors for near-duplicates.
Keeps the vector with higher access_count, merges metadata, deletes the other.
Runs every 6 hours.
"""
import logging
import httpx
from src.core.dedup import merge_metadata

logger = logging.getLogger("echo.workers.dedup")

QDRANT_URL = "http://localhost:6333"
COLLECTION = "echo_memory"
SIMILARITY_THRESHOLD = 0.98
BATCH_SIZE = 50


class DedupWorker:
    """Scans the collection for near-duplicate vectors and merges them."""

    async def run_cycle(self):
        merged_count = 0
        scanned = 0

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Scroll through vectors in batches (with vectors for comparison)
                offset = None
                body = {
                    "limit": BATCH_SIZE,
                    "with_payload": True,
                    "with_vector": True,
                }
                if offset:
                    body["offset"] = offset

                resp = await client.post(
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json().get("result", {})
                points = data.get("points", [])

                deleted_ids = set()

                for point in points:
                    pid = point["id"]
                    if pid in deleted_ids:
                        continue
                    scanned += 1

                    vector = point.get("vector")
                    if not vector:
                        continue

                    # Search for similar vectors (excluding self)
                    search_resp = await client.post(
                        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                        json={
                            "vector": vector,
                            "limit": 5,
                            "score_threshold": SIMILARITY_THRESHOLD,
                            "with_payload": True,
                            "filter": {
                                "must_not": [
                                    {"has_id": [pid]}
                                ]
                            },
                        },
                    )
                    search_resp.raise_for_status()
                    matches = search_resp.json().get("result", [])

                    for match in matches:
                        match_id = match["id"]
                        if match_id in deleted_ids:
                            continue

                        my_access = int(point.get("payload", {}).get("access_count", 0))
                        their_access = int(match.get("payload", {}).get("access_count", 0))

                        # Keep the one with higher access_count
                        if my_access >= their_access:
                            keeper_id = pid
                            keeper_payload = point.get("payload", {})
                            loser_id = match_id
                            loser_payload = match.get("payload", {})
                        else:
                            keeper_id = match_id
                            keeper_payload = match.get("payload", {})
                            loser_id = pid
                            loser_payload = point.get("payload", {})

                        # Merge metadata into keeper
                        merged = merge_metadata(keeper_payload, loser_payload)
                        await client.post(
                            f"{QDRANT_URL}/collections/{COLLECTION}/points/payload",
                            json={
                                "payload": merged,
                                "points": [keeper_id],
                            },
                        )

                        # Delete loser
                        await client.post(
                            f"{QDRANT_URL}/collections/{COLLECTION}/points/delete",
                            json={"points": [loser_id]},
                        )
                        deleted_ids.add(loser_id)
                        merged_count += 1

                        logger.debug(f"Merged {loser_id} into {keeper_id} (score={match['score']:.4f})")

        except Exception as e:
            logger.error(f"Dedup worker error: {e}")

        logger.info(f"Dedup cycle: scanned={scanned}, merged={merged_count}")
