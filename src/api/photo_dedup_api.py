"""
Photo Dedup API Router — Echo Brain
Endpoints for managing personal photo dedup and Echo Brain ingestion.
"""
import logging
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.services.photo_dedup_service import PhotoDedupService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/photos", tags=["photos"])
service = PhotoDedupService()


@router.get("/oauth/status")
async def oauth_status():
    """Check if Google OAuth token exists, return login URL if not."""
    return await service.get_oauth_status()


@router.post("/scan/local")
async def scan_local(batch_size: int = Query(500, ge=1, le=5000)):
    """Trigger local photo scan of ~/Pictures/."""
    try:
        result = await service.scan_local_photos(batch_size=batch_size)
        return result
    except Exception as e:
        logger.error(f"Local scan error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/scan/cloud")
async def scan_cloud():
    """Trigger Google Photos cloud metadata fetch."""
    try:
        result = await service.fetch_cloud_metadata()
        if "error" in result:
            return JSONResponse(status_code=400, content=result)
        return result
    except Exception as e:
        logger.error(f"Cloud scan error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/dedup/run")
async def dedup_run():
    """Trigger dedup matching across local and cloud photos."""
    try:
        result = await service.run_dedup_matching()
        return result
    except Exception as e:
        logger.error(f"Dedup run error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/dedup/report")
async def dedup_report():
    """Get dedup report with match breakdown, by-year stats, top dupe groups."""
    try:
        return await service.generate_report()
    except Exception as e:
        logger.error(f"Report error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/analyze")
async def analyze_batch(batch_size: int = Query(50, ge=1, le=500)):
    """Trigger LLaVA batch analysis on un-analyzed photos."""
    try:
        result = await service.analyze_photos_batch(batch_size=batch_size)
        return result
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/ingest")
async def ingest_to_qdrant(batch_size: int = Query(100, ge=1, le=1000)):
    """Embed analyzed photos and store in Qdrant echo_memory."""
    try:
        result = await service.ingest_to_qdrant(batch_size=batch_size)
        return result
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/stats")
async def stats():
    """Overall stats: local count, cloud count, matched, analyzed, embedded."""
    try:
        return await service.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
