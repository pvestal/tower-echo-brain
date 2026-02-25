"""
Photo & Video Memory API Router — Echo Brain
Endpoints for managing personal media dedup, search, face detection, and Echo Brain ingestion.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.services.photo_dedup_service import PhotoDedupService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/photos", tags=["photos"])
service = PhotoDedupService()


class SearchRequest(BaseModel):
    query: str
    media_type: Optional[str] = None
    year: Optional[str] = None
    category: Optional[str] = None
    person: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit: int = 20


class NameClusterRequest(BaseModel):
    name: str


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


@router.post("/scan/takeout")
async def scan_takeout(batch_size: int = Query(500, ge=1, le=5000)):
    """Trigger Google Takeout media scan (photos + videos)."""
    try:
        result = await service.scan_takeout_media(batch_size=batch_size)
        if "error" in result:
            return JSONResponse(status_code=400, content=result)
        return result
    except Exception as e:
        logger.error(f"Takeout scan error: {e}")
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
async def analyze_batch(batch_size: int = Query(50, ge=1, le=500),
                        media_type: Optional[str] = Query(None)):
    """Trigger vision batch analysis on un-analyzed media."""
    try:
        result = await service.analyze_photos_batch(batch_size=batch_size,
                                                     media_type=media_type)
        return result
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/ingest")
async def ingest_to_qdrant(batch_size: int = Query(100, ge=1, le=1000)):
    """Embed analyzed media and store in Qdrant echo_memory."""
    try:
        result = await service.ingest_to_qdrant(batch_size=batch_size)
        return result
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/stats")
async def stats():
    """Overall stats: local count, cloud count, matched, analyzed, embedded, faces."""
    try:
        return await service.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/search")
async def search_media(request: SearchRequest):
    """Semantic search for photos and videos."""
    try:
        results = await service.search_media(
            query=request.query,
            media_type=request.media_type,
            year=request.year,
            category=request.category,
            person=request.person,
            date_from=request.date_from,
            date_to=request.date_to,
            limit=request.limit,
        )
        return {"results": results, "count": len(results), "query": request.query}
    except Exception as e:
        logger.error(f"Search error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/people")
async def list_people():
    """List face clusters (detected people) with counts and sample photos."""
    try:
        clusters = await service.get_face_clusters()
        return {"people": clusters, "count": len(clusters)}
    except Exception as e:
        logger.error(f"People list error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/people/{cluster_id}/name")
async def name_person(cluster_id: int, request: NameClusterRequest):
    """Assign a name to a face cluster."""
    try:
        result = await service.name_face_cluster(cluster_id, request.name)
        if "error" in result:
            return JSONResponse(status_code=404, content=result)
        return result
    except Exception as e:
        logger.error(f"Name cluster error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/faces/detect")
async def detect_faces(batch_size: int = Query(100, ge=1, le=500)):
    """Run face detection on analyzed photos."""
    try:
        result = await service.detect_faces_batch(batch_size=batch_size)
        if "error" in result:
            return JSONResponse(status_code=500, content=result)
        return result
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/faces/cluster")
async def cluster_faces(distance_threshold: float = Query(0.4, ge=0.1, le=1.0)):
    """Cluster detected faces by similarity."""
    try:
        result = await service.cluster_faces(distance_threshold=distance_threshold)
        return result
    except Exception as e:
        logger.error(f"Face clustering error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
