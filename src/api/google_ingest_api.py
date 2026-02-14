"""
Google Data Ingestion API — Echo Brain
Trigger ingestion of Calendar, Gmail, and Drive data into Qdrant.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/google/ingest", tags=["google_ingest"])


def _get_ingestor():
    from src.services.google_data_ingestor import GoogleDataIngestor
    return GoogleDataIngestor()


@router.post("/calendar")
async def ingest_calendar(months_back: int = 3, months_forward: int = 3):
    """Trigger calendar event ingestion from all calendars."""
    ingestor = _get_ingestor()
    result = await ingestor.ingest_calendar_events(
        months_back=months_back, months_forward=months_forward
    )
    return result


@router.post("/email")
async def ingest_email(
    max_messages: int = 500, query: Optional[str] = "newer_than:90d"
):
    """Trigger Gmail message ingestion (metadata + snippet)."""
    ingestor = _get_ingestor()
    result = await ingestor.ingest_gmail_messages(
        max_messages=max_messages, query=query
    )
    return result


@router.post("/drive")
async def ingest_drive(max_files: int = 1000):
    """Trigger Drive file metadata ingestion.

    NOTE: drive.file scope only sees app-created files.
    Re-auth with drive.readonly for full read access.
    """
    ingestor = _get_ingestor()
    result = await ingestor.ingest_drive_files(max_files=max_files)
    return result


@router.post("/all")
async def ingest_all():
    """Trigger ingestion for all three Google data sources."""
    ingestor = _get_ingestor()
    calendar = await ingestor.ingest_calendar_events()
    email = await ingestor.ingest_gmail_messages()
    drive = await ingestor.ingest_drive_files()
    return {
        "calendar": calendar,
        "email": email,
        "drive": drive,
    }


@router.get("/stats")
async def ingestion_stats():
    """Get ingestion counts by source type."""
    ingestor = _get_ingestor()
    return await ingestor.get_ingestion_stats()
