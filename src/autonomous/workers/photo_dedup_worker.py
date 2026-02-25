"""
Photo & Video Memory Worker — Echo Brain
Autonomous worker that incrementally processes the unified media pipeline:
1. Scan local photos (if incomplete)
1b. Scan Takeout media (if Takeout root exists and not yet scanned)
2. Fetch cloud metadata (if token available, with backoff on repeated failures)
3. Run dedup matching
4a. Analyze photos with vision analysis
4b. Analyze videos with vision analysis
4c. Run face detection on analyzed photos
5. Ingest analyzed media to Qdrant
"""
import logging
from datetime import datetime

import asyncpg

from src.services.photo_dedup_service import PhotoDedupService, DB_URL, TAKEOUT_PHOTOS_ROOT

logger = logging.getLogger(__name__)

# Skip cloud fetch if the last N runs all returned 0 items
CLOUD_FETCH_BACKOFF_THRESHOLD = 5


class PhotoDedupWorker:
    """Autonomous worker for photo/video memory pipeline — 30 min interval."""

    def __init__(self):
        self.service = PhotoDedupService()
        self._takeout_scanned = False

    async def _should_skip_cloud_fetch(self) -> bool:
        """Check if recent cloud_fetch runs all returned 0 — skip if so."""
        try:
            conn = await asyncpg.connect(DB_URL)
            try:
                recent = await conn.fetch("""
                    SELECT items_processed FROM photo_dedup_runs
                    WHERE run_type = 'cloud_fetch'
                    ORDER BY id DESC
                    LIMIT $1
                """, CLOUD_FETCH_BACKOFF_THRESHOLD)
            finally:
                await conn.close()

            if len(recent) < CLOUD_FETCH_BACKOFF_THRESHOLD:
                return False  # not enough history to judge

            return all(r["items_processed"] == 0 for r in recent)
        except Exception as e:
            logger.warning(f"[PhotoDedupWorker] Backoff check failed: {e}")
            return False

    async def _is_takeout_scanned(self) -> bool:
        """Check if any takeout_scan run has completed."""
        if self._takeout_scanned:
            return True
        try:
            conn = await asyncpg.connect(DB_URL)
            try:
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM photo_dedup_runs
                    WHERE run_type = 'takeout_scan' AND finished_at IS NOT NULL
                """)
                if count and count > 0:
                    self._takeout_scanned = True
                    return True
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"[PhotoDedupWorker] Takeout scan check failed: {e}")
        return False

    async def run_cycle(self):
        """Main worker cycle — runs all applicable steps each cycle."""
        logger.info(f"[PhotoDedupWorker] Starting cycle at {datetime.now().isoformat()}")

        try:
            stats = await self.service.get_stats()
        except Exception as e:
            logger.error(f"[PhotoDedupWorker] Failed to get stats: {e}")
            return

        # Step 1: Local scan if not done
        if not stats.get("scan_complete"):
            logger.info("[PhotoDedupWorker] Running local scan...")
            try:
                result = await self.service.scan_local_photos(batch_size=500)
                logger.info(f"[PhotoDedupWorker] Local scan: {result}")
            except Exception as e:
                logger.error(f"[PhotoDedupWorker] Local scan error: {e}")
            return  # one major step per cycle

        # Step 1b: Takeout scan if not yet done and directory exists
        if not await self._is_takeout_scanned() and TAKEOUT_PHOTOS_ROOT.exists():
            logger.info("[PhotoDedupWorker] Running Takeout media scan...")
            try:
                result = await self.service.scan_takeout_media(batch_size=500)
                logger.info(f"[PhotoDedupWorker] Takeout scan: {result}")
                self._takeout_scanned = True
            except Exception as e:
                logger.error(f"[PhotoDedupWorker] Takeout scan error: {e}")
            return  # one major step per cycle

        # Step 2: Cloud fetch if token available and not yet fetched
        # Cloud fetch failure does NOT block steps 3-5
        if not stats.get("cloud_fetched"):
            if await self._should_skip_cloud_fetch():
                logger.warning(
                    f"[PhotoDedupWorker] Skipping cloud fetch — last {CLOUD_FETCH_BACKOFF_THRESHOLD} "
                    "runs all returned 0 items. Google OAuth likely needs re-auth."
                )
            else:
                oauth = await self.service.get_oauth_status()
                if oauth.get("has_token"):
                    logger.info("[PhotoDedupWorker] Fetching cloud metadata...")
                    try:
                        result = await self.service.fetch_cloud_metadata()
                        logger.info(f"[PhotoDedupWorker] Cloud fetch: {result}")
                    except Exception as e:
                        logger.error(f"[PhotoDedupWorker] Cloud fetch error: {e}")
                else:
                    logger.info("[PhotoDedupWorker] No Google token — skipping cloud fetch")

        # Step 3: Run matching if cloud data available and unmatched photos exist
        if stats.get("cloud_fetched") and stats.get("unmatched_local", 0) > 0:
            total = stats.get("local_photos", 0)
            unmatched = stats.get("unmatched_local", 0)
            if total > 0 and unmatched > total * 0.5:
                logger.info("[PhotoDedupWorker] Running dedup matching...")
                try:
                    result = await self.service.run_dedup_matching()
                    logger.info(f"[PhotoDedupWorker] Matching: {result}")
                except Exception as e:
                    logger.error(f"[PhotoDedupWorker] Matching error: {e}")

        # Step 4a: Analyze PHOTOS with vision analysis (500 per cycle)
        pending_analysis = stats.get("local_photos", 0) - stats.get("analyzed", 0) - stats.get("sha256_dupes", 0)
        if pending_analysis > 0:
            batch = min(500, pending_analysis)
            logger.info(f"[PhotoDedupWorker] Analyzing {batch} photos with vision analysis...")
            try:
                result = await self.service.analyze_photos_batch(
                    batch_size=500, media_type="photo")
                logger.info(f"[PhotoDedupWorker] Photo vision analysis: {result}")
            except Exception as e:
                logger.error(f"[PhotoDedupWorker] Photo vision analysis error: {e}")

        # Step 4b: Analyze VIDEOS with vision analysis (20 per cycle — slower)
        videos_pending = stats.get("videos_count", 0)
        if videos_pending > 0:
            logger.info(f"[PhotoDedupWorker] Analyzing up to 20 videos with vision analysis...")
            try:
                result = await self.service.analyze_photos_batch(
                    batch_size=20, media_type="video")
                logger.info(f"[PhotoDedupWorker] Video vision analysis: {result}")
            except Exception as e:
                logger.error(f"[PhotoDedupWorker] Video vision analysis error: {e}")

        # Step 4c: Face detection on analyzed photos (200 per cycle)
        try:
            result = await self.service.detect_faces_batch(batch_size=200)
            if result.get("processed", 0) > 0:
                logger.info(f"[PhotoDedupWorker] Face detection: {result}")

                # Re-cluster if we detected new faces
                if result.get("faces_detected", 0) > 0:
                    cluster_result = await self.service.cluster_faces()
                    logger.info(f"[PhotoDedupWorker] Face clustering: {cluster_result}")
        except Exception as e:
            logger.error(f"[PhotoDedupWorker] Face detection error: {e}")

        # Step 5: Ingest to Qdrant
        # Re-fetch stats to pick up media just analyzed in steps 4a-4c
        try:
            stats = await self.service.get_stats()
        except Exception:
            pass  # use stale stats if refresh fails
        pending_ingest = stats.get("analyzed", 0) - stats.get("embedded_in_qdrant", 0)
        if pending_ingest > 0:
            batch = min(500, pending_ingest)
            logger.info(f"[PhotoDedupWorker] Ingesting {batch} media to Qdrant...")
            try:
                result = await self.service.ingest_to_qdrant(batch_size=500)
                logger.info(f"[PhotoDedupWorker] Qdrant: {result}")
            except Exception as e:
                logger.error(f"[PhotoDedupWorker] Qdrant ingest error: {e}")

        logger.info("[PhotoDedupWorker] Cycle complete")
