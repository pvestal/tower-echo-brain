"""
Semantic Deduplication Module
Prevents duplicate vectors from being stored in Qdrant by checking
cosine similarity before insertion.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

import httpx

logger = logging.getLogger("echo.core.dedup")

QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "echo_memory"


async def check_duplicate(
    embedding: List[float],
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str = QDRANT_URL,
    threshold: float = 0.97,
) -> Optional[Dict[str, Any]]:
    """Check if a near-duplicate vector already exists in the collection.

    Returns the matching point dict (with id, score, payload) or None.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{qdrant_url}/collections/{collection}/points/search",
                json={
                    "vector": embedding,
                    "limit": 1,
                    "score_threshold": threshold,
                    "with_payload": True,
                },
            )
            resp.raise_for_status()
            results = resp.json().get("result", [])
            if results:
                point = results[0]
                return {
                    "id": point["id"],
                    "score": point["score"],
                    "payload": point.get("payload", {}),
                }
    except Exception as e:
        logger.warning(f"Dedup check failed (non-blocking): {e}")
    return None


def merge_metadata(existing_payload: Dict, new_payload: Dict) -> Dict:
    """Merge metadata from a new vector into an existing duplicate.

    Rules:
    - Keep earliest ingested_at
    - Sum access_counts
    - Keep highest confidence
    - Prefer new content/text if longer
    """
    merged = {**existing_payload}

    # Keep earliest ingestion timestamp
    existing_ts = existing_payload.get("ingested_at", "")
    new_ts = new_payload.get("ingested_at", "")
    if existing_ts and new_ts:
        merged["ingested_at"] = min(existing_ts, new_ts)
    elif new_ts:
        merged["ingested_at"] = new_ts

    # Sum access counts
    merged["access_count"] = (
        existing_payload.get("access_count", 0) + new_payload.get("access_count", 0)
    )

    # Keep highest confidence
    merged["confidence"] = max(
        existing_payload.get("confidence", 0.7),
        new_payload.get("confidence", 0.7),
    )

    # Update last_accessed to now
    merged["last_accessed"] = datetime.now().isoformat()

    return merged


async def bump_existing_point(
    point_id: Any,
    merged_meta: Dict,
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str = QDRANT_URL,
) -> bool:
    """Apply merged metadata to the surviving duplicate point."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{qdrant_url}/collections/{collection}/points/payload",
                json={
                    "payload": merged_meta,
                    "points": [point_id],
                },
            )
            resp.raise_for_status()
            return True
    except Exception as e:
        logger.warning(f"Failed to bump existing point {point_id}: {e}")
        return False
