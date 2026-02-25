"""
Decay Worker — Applies logarithmic confidence decay to stale, low-access vectors.
Runs daily. Vectors with access_count > 5 are exempt (usage-validated).
"""
import math
import logging
import httpx
import networkx as nx
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

    def __init__(self):
        self._graph_hub_entities: set = set()
        self._graph_loaded = False

    async def _load_graph_hubs(self):
        """Cache hub entities (top 10% by centrality) to shield from decay."""
        if self._graph_loaded:
            return
        try:
            from src.core.graph_engine import get_graph_engine
            engine = get_graph_engine()
            await engine._ensure_loaded()
            if engine._graph and engine._graph.number_of_nodes() > 0:
                centrality = nx.degree_centrality(engine._graph)
                if centrality:
                    threshold = sorted(centrality.values(), reverse=True)[
                        max(0, len(centrality) // 10)
                    ]
                    self._graph_hub_entities = {
                        node for node, score in centrality.items()
                        if score >= threshold
                    }
                    logger.info(f"Decay worker: shielding {len(self._graph_hub_entities)} hub entities")
        except Exception as e:
            logger.debug(f"Decay worker: graph hubs unavailable: {e}")
        self._graph_loaded = True

    def _is_hub_content(self, payload: dict) -> bool:
        """Check if vector content mentions a hub entity."""
        if not self._graph_hub_entities:
            return False
        text = (payload.get("text", "") + " " + payload.get("content", "")).lower()
        return any(hub in text for hub in self._graph_hub_entities)

    async def run_cycle(self):
        updated = 0
        skipped = 0
        hub_shielded = 0
        now = datetime.now(timezone.utc)

        await self._load_graph_hubs()

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

                        # Exempt vectors about hub entities (graph-important)
                        if self._is_hub_content(payload):
                            hub_shielded += 1
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

        logger.info(f"Decay cycle: updated={updated}, exempt={skipped}, hub_shielded={hub_shielded}")
