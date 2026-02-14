"""
Google Ingest Worker — Echo Brain
Autonomous worker that periodically ingests Google data
(Calendar, Gmail, Drive) into Qdrant.
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class GoogleIngestWorker:
    """Autonomous worker for Google data ingestion — 60 min interval."""

    async def run_cycle(self):
        logger.info(
            f"[GoogleIngestWorker] Starting cycle at {datetime.now().isoformat()}"
        )

        from src.services.google_data_ingestor import GoogleDataIngestor

        ingestor = GoogleDataIngestor()

        # 1. Calendar events — lightweight, always run
        try:
            cal_result = await ingestor.ingest_calendar_events()
            logger.info(f"[GoogleIngestWorker] Calendar: {cal_result}")
        except Exception as e:
            logger.error(f"[GoogleIngestWorker] Calendar error: {e}")

        # 2. Email — incremental, only last day for scheduled runs
        try:
            email_result = await ingestor.ingest_gmail_messages(
                max_messages=200, query="newer_than:1d"
            )
            logger.info(f"[GoogleIngestWorker] Email: {email_result}")
        except Exception as e:
            logger.error(f"[GoogleIngestWorker] Email error: {e}")

        # 3. Drive — metadata only, if scope allows
        try:
            drive_result = await ingestor.ingest_drive_files(max_files=500)
            logger.info(f"[GoogleIngestWorker] Drive: {drive_result}")
        except Exception as e:
            logger.error(f"[GoogleIngestWorker] Drive error: {e}")

        logger.info("[GoogleIngestWorker] Cycle complete")
