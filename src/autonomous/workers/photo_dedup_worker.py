"""
Photo Dedup Worker — Echo Brain
Autonomous worker that incrementally processes the photo dedup pipeline:
1. Scan local photos (if incomplete)
2. Fetch cloud metadata (if token available)
3. Run dedup matching
4. Analyze next batch with LLaVA
5. Ingest analyzed photos to Qdrant
"""
import logging
from datetime import datetime

from src.services.photo_dedup_service import PhotoDedupService

logger = logging.getLogger(__name__)


class PhotoDedupWorker:
    """Autonomous worker for photo dedup pipeline — 120 min interval."""

    def __init__(self):
        self.service = PhotoDedupService()

    async def run_cycle(self):
        """Main worker cycle — run the next step in the pipeline."""
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

        # Step 2: Cloud fetch if token available and not yet fetched
        if not stats.get("cloud_fetched"):
            oauth = await self.service.get_oauth_status()
            if oauth.get("has_token"):
                logger.info("[PhotoDedupWorker] Fetching cloud metadata...")
                try:
                    result = await self.service.fetch_cloud_metadata()
                    logger.info(f"[PhotoDedupWorker] Cloud fetch: {result}")
                except Exception as e:
                    logger.error(f"[PhotoDedupWorker] Cloud fetch error: {e}")
                return
            else:
                logger.info("[PhotoDedupWorker] No Google token — skipping cloud fetch")

        # Step 3: Run matching if cloud data available and unmatched photos exist
        if stats.get("cloud_fetched") and stats.get("unmatched_local", 0) > 0:
            # Only run matching if we haven't done it recently
            # (unmatched_local > total_local * 0.8 suggests it hasn't been run)
            total = stats.get("local_photos", 0)
            unmatched = stats.get("unmatched_local", 0)
            if total > 0 and unmatched > total * 0.5:
                logger.info("[PhotoDedupWorker] Running dedup matching...")
                try:
                    result = await self.service.run_dedup_matching()
                    logger.info(f"[PhotoDedupWorker] Matching: {result}")
                except Exception as e:
                    logger.error(f"[PhotoDedupWorker] Matching error: {e}")
                return

        # Step 4: Analyze photos with LLaVA (100 per cycle)
        pending_analysis = stats.get("local_photos", 0) - stats.get("analyzed", 0) - stats.get("sha256_dupes", 0)
        if pending_analysis > 0:
            logger.info(f"[PhotoDedupWorker] Analyzing {min(100, pending_analysis)} photos with LLaVA...")
            try:
                result = await self.service.analyze_photos_batch(batch_size=100)
                logger.info(f"[PhotoDedupWorker] LLaVA: {result}")
            except Exception as e:
                logger.error(f"[PhotoDedupWorker] LLaVA error: {e}")
            # Continue to step 5 — ingest what was already analyzed

        # Step 5: Ingest to Qdrant
        pending_ingest = stats.get("analyzed", 0) - stats.get("embedded_in_qdrant", 0)
        if pending_ingest > 0:
            logger.info(f"[PhotoDedupWorker] Ingesting {min(100, pending_ingest)} photos to Qdrant...")
            try:
                result = await self.service.ingest_to_qdrant(batch_size=100)
                logger.info(f"[PhotoDedupWorker] Qdrant: {result}")
            except Exception as e:
                logger.error(f"[PhotoDedupWorker] Qdrant ingest error: {e}")

        logger.info("[PhotoDedupWorker] Cycle complete")
