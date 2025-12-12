#!/usr/bin/env python3
"""
Google Takeout Processing Status Routes
"""
from fastapi import APIRouter
import asyncpg
import logging

router = APIRouter(prefix="/api/echo", tags=["takeout"])
logger = logging.getLogger(__name__)

@router.get("/takeout/progress")
async def get_takeout_progress():
    """Get Google Takeout processing progress"""
    try:
        # Connect to echo_brain database
        conn = await asyncpg.connect(
            'postgresql://patrick:tower_echo_brain_secret_key_2025@localhost/echo_brain'
        )

        try:
            # Get processing progress from view
            progress = await conn.fetchrow('SELECT * FROM takeout_processing_progress')

            if progress:
                return {
                    "files_processed": progress['files_processed'] or 0,
                    "bytes_processed": progress['bytes_processed'] or 0,
                    "total_insights": progress['total_insights'] or 0,
                    "people_discovered": progress['people_discovered'] or 0,
                    "locations_discovered": progress['locations_discovered'] or 0,
                    "events_identified": progress['events_identified'] or 0,
                    "last_processed_at": str(progress['last_processed_at']) if progress['last_processed_at'] else None,
                    "processing_percentage": ((progress['bytes_processed'] or 0) / 786000000000) * 100,  # 786GB total
                    "status": "active" if progress['files_processed'] else "not_started"
                }
            else:
                return {
                    "files_processed": 0,
                    "bytes_processed": 0,
                    "total_insights": 0,
                    "people_discovered": 0,
                    "locations_discovered": 0,
                    "events_identified": 0,
                    "last_processed_at": None,
                    "processing_percentage": 0,
                    "status": "not_started"
                }
        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Takeout progress error: {e}")
        return {"error": str(e), "status": "error"}
