"""
Decay Worker — Applies logarithmic confidence decay to stale, low-access vectors.
Runs daily. Vectors with access_count > 5 are exempt (usage-validated).
"""
import math
import logging
import httpx
from datetime import datetime, timezone

logger = logging.getLogger("echo.workers.decay")

QDRANT_URL = "http://localhost:6333"
COLLECTION = "echo_memory"
HALFLIFE_DAYS = 90
CONFIDENCE_FLOOR = 0.2
ACCESS_EXEMPT_THRESHOLD = 5
BATCH_SIZE = 100


class DecayWorker:
    """Periodically decays confidence on stale, unused vectors."""

    async def run_cycle(self):
        updated = 0
        skipped = 0
        now = datetime.now(timezone.utc)

        try:
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
                        access_count = int(payload.get("access_count", 0))

                        # Exempt usage-validated vectors
                        if access_count > ACCESS_EXEMPT_THRESHOLD:
                            skipped += 1
                            continue

                        # Need a timestamp to compute age
                        ts_str = (
                            payload.get("last_accessed")
                            or payload.get("ingested_at")
                            or payload.get("timestamp")
                        )
                        if not ts_str:
                            continue

                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                        except (ValueError, TypeError):
                            continue

                        age_days = (now - ts).total_seconds() / 86400
                        if age_days < 30:
                            # Skip recent vectors
                            continue

                        old_conf = float(payload.get("confidence", 0.7))
                        new_conf = old_conf * (1.0 / (1.0 + math.log1p(age_days / HALFLIFE_DAYS)))
                        new_conf = max(CONFIDENCE_FLOOR, round(new_conf, 4))

                        if abs(new_conf - old_conf) < 0.01:
                            continue  # No meaningful change

                        # Update confidence
                        await client.post(
                            f"{QDRANT_URL}/collections/{COLLECTION}/points/payload",
                            json={
                                "payload": {"confidence": new_conf},
                                "points": [point["id"]],
                            },
                        )
                        updated += 1

                    if not offset:
                        break

        except Exception as e:
            logger.error(f"Decay worker error: {e}")

        logger.info(f"Decay cycle: updated={updated}, exempt={skipped}")
