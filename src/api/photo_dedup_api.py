"""
Photo & Video Memory API Router — Echo Brain
Endpoints for managing personal media dedup, search, face detection, and Echo Brain ingestion.
"""
import logging
import os
from io import BytesIO
from typing import List, Optional

import asyncpg
from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from PIL import Image

from src.services.photo_dedup_service import PhotoDedupService
from src.services.google_contacts_face_matcher import GoogleContactsFaceMatcher

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


class MergeClusterRequest(BaseModel):
    cluster_ids: List[int]
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
    """Google Photos Library API was deprecated on 2025-03-31. Use Takeout import instead."""
    return JSONResponse(status_code=410, content={
        "error": "Google Photos Library API deprecated 2025-03-31",
        "alternative": "Use POST /api/photos/scan/takeout with a Google Takeout export"
    })


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


@router.post("/reanalyze/queue")
async def queue_reanalysis(batch_size: int = Query(500, ge=1, le=5000),
                            media_type: Optional[str] = Query(None),
                            year: Optional[str] = Query(None)):
    """Queue already-analyzed media for re-analysis with improved prompts.
    Clears their analysis so the worker re-processes them gradually.
    Use to upgrade old single-sentence descriptions to rich multi-sentence ones.
    """
    try:
        result = await service.queue_reanalysis(
            batch_size=batch_size, media_type=media_type, year=year)
        return result
    except Exception as e:
        logger.error(f"Queue reanalysis error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/reanalyze/stats")
async def reanalysis_stats():
    """Show how many media items are pending re-analysis vs done."""
    try:
        return await service.reanalysis_stats()
    except Exception as e:
        logger.error(f"Reanalysis stats error: {e}")
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


@router.post("/analyze/clinical")
async def analyze_clinical(file_paths: List[str] = None,
                           batch_size: int = Query(10, ge=1, le=50),
                           media_type: Optional[str] = Query(None)):
    """Re-analyze specific media with clinical/uncensored prompts.
    Describes ALL content including nudity, sexual acts.
    Pass file_paths in request body to re-scan specific files,
    or omit to scan next un-analyzed batch.
    """
    try:
        result = await service.analyze_photos_batch(
            batch_size=batch_size,
            media_type=media_type,
            mode="clinical",
            file_paths=file_paths,
        )
        return result
    except Exception as e:
        logger.error(f"Clinical analyze error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/reindex")
async def reindex_qdrant(file_paths: List[str]):
    """Re-embed and replace Qdrant points for specific files.
    Use after clinical re-scan to update search descriptions.
    Deletes old point, creates new one with updated content.
    """
    try:
        result = await service.reindex_to_qdrant(file_paths=file_paths)
        return result
    except Exception as e:
        logger.error(f"Reindex error: {e}")
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


@router.post("/dedup/backfill-phash")
async def backfill_phash(batch_size: int = Query(500, ge=1, le=10000)):
    """Compute perceptual hashes for photos missing them."""
    try:
        result = await service.backfill_perceptual_hashes(batch_size=batch_size)
        return result
    except Exception as e:
        logger.error(f"Phash backfill error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/dedup/purge-dupe-faces")
async def purge_dupe_faces():
    """Remove face records from photos marked as SHA256 or phash duplicates."""
    try:
        result = await service.purge_duplicate_faces()
        return result
    except Exception as e:
        logger.error(f"Purge dupe faces error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/faces/detect-videos")
async def detect_faces_videos(batch_size: int = Query(20, ge=1, le=100)):
    """Run face detection on analyzed videos using keyframe extraction."""
    try:
        result = await service.detect_faces_videos_batch(batch_size=batch_size)
        if "error" in result:
            return JSONResponse(status_code=500, content=result)
        return result
    except Exception as e:
        logger.error(f"Video face detection error: {e}")
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


# ------------------------------------------------------------------
# Photo thumbnail serving
# ------------------------------------------------------------------

# Allowed root directories for photo serving (security: prevent path traversal)
_ALLOWED_ROOTS = [
    "/home/patrick/Pictures",
    "/mnt/10TB2",
    "/mnt/1tb-storage",
]


@router.get("/thumb/{photo_id}")
async def serve_thumbnail(
    photo_id: int = Path(..., ge=1),
    size: int = Query(256, ge=64, le=1024),
):
    """Serve a photo thumbnail by photo ID. Resizes to `size`px on longest edge."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://echo:echo_secure_password_123@localhost/echo_brain",
    )
    conn = await asyncpg.connect(db_url)
    try:
        row = await conn.fetchrow(
            "SELECT file_path FROM photos WHERE id = $1", photo_id
        )
        if not row:
            return JSONResponse(status_code=404, content={"error": "Photo not found"})

        file_path = row["file_path"]

        # Security: ensure file is under an allowed root
        resolved = os.path.realpath(file_path)
        if not any(resolved.startswith(root) for root in _ALLOWED_ROOTS):
            return JSONResponse(status_code=403, content={"error": "Path not allowed"})

        if not os.path.isfile(resolved):
            return JSONResponse(status_code=404, content={"error": "File missing on disk"})

        # Generate thumbnail
        img = Image.open(resolved)
        img.thumbnail((size, size), Image.LANCZOS)

        buf = BytesIO()
        fmt = "JPEG"
        if resolved.lower().endswith(".png"):
            fmt = "PNG"
        img.save(buf, format=fmt, quality=80)
        buf.seek(0)

        media_type = "image/jpeg" if fmt == "JPEG" else "image/png"
        return Response(
            content=buf.getvalue(),
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except Exception as e:
        logger.error(f"Thumbnail error for photo {photo_id}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


@router.get("/people/review")
async def people_for_review(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    unnamed_only: bool = Query(True),
    min_photos: int = Query(5, ge=1),
):
    """Return face clusters for the naming review UI, paginated, with sample photo IDs."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://echo:echo_secure_password_123@localhost/echo_brain",
    )
    conn = await asyncpg.connect(db_url)
    try:
        name_filter = "AND fc.cluster_name IS NULL" if unnamed_only else ""
        offset = (page - 1) * per_page

        total = await conn.fetchval(f"""
            SELECT COUNT(*) FROM face_clusters fc
            WHERE fc.photo_count >= $1 {name_filter}
        """, min_photos)

        clusters = await conn.fetch(f"""
            SELECT fc.id, fc.cluster_name, fc.photo_count, fc.sample_photo_ids,
                   fc.created_at
            FROM face_clusters fc
            WHERE fc.photo_count >= $1 {name_filter}
            ORDER BY fc.photo_count DESC
            LIMIT $2 OFFSET $3
        """, min_photos, per_page, offset)

        results = []
        for c in clusters:
            sample_ids = c["sample_photo_ids"] or []
            results.append({
                "id": c["id"],
                "name": c["cluster_name"],
                "photo_count": c["photo_count"],
                "sample_photo_ids": list(sample_ids),
                "created_at": c["created_at"].isoformat() if c["created_at"] else None,
            })

        return {
            "clusters": results,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page if total else 0,
        }
    finally:
        await conn.close()


@router.post("/people/{cluster_id}/skip")
async def skip_cluster(cluster_id: int):
    """Mark a cluster as skipped (name = '__skipped__') so it doesn't show in review."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://echo:echo_secure_password_123@localhost/echo_brain",
    )
    conn = await asyncpg.connect(db_url)
    try:
        result = await conn.execute(
            "UPDATE face_clusters SET cluster_name = '__skipped__' WHERE id = $1",
            cluster_id,
        )
        if "UPDATE 0" in result:
            return JSONResponse(status_code=404, content={"error": "Cluster not found"})
        return {"cluster_id": cluster_id, "status": "skipped"}
    finally:
        await conn.close()


@router.post("/people/merge")
async def merge_clusters(request: MergeClusterRequest):
    cluster_ids = request.cluster_ids
    name = request.name
    """Merge multiple clusters into one, keeping the largest and reassigning faces."""
    if len(cluster_ids) < 2:
        return JSONResponse(status_code=400, content={"error": "Need at least 2 cluster IDs"})

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://echo:echo_secure_password_123@localhost/echo_brain",
    )
    conn = await asyncpg.connect(db_url)
    try:
        # Find the largest cluster to keep
        rows = await conn.fetch("""
            SELECT id, photo_count FROM face_clusters
            WHERE id = ANY($1::int[])
            ORDER BY photo_count DESC
        """, cluster_ids)

        if len(rows) < 2:
            return JSONResponse(status_code=404, content={"error": "Clusters not found"})

        keep_id = rows[0]["id"]
        merge_ids = [r["id"] for r in rows[1:]]

        # Reassign faces to the kept cluster
        await conn.execute("""
            UPDATE photo_faces SET cluster_id = $1
            WHERE cluster_id = ANY($2::int[])
        """, keep_id, merge_ids)

        # Update photo count on kept cluster
        new_count = await conn.fetchval(
            "SELECT COUNT(DISTINCT photo_id) FROM photo_faces WHERE cluster_id = $1",
            keep_id,
        )
        await conn.execute(
            "UPDATE face_clusters SET cluster_name = $2, photo_count = $3 WHERE id = $1",
            keep_id, name, new_count,
        )

        # Delete merged clusters
        await conn.execute(
            "DELETE FROM face_clusters WHERE id = ANY($1::int[])", merge_ids
        )

        return {
            "kept_cluster_id": keep_id,
            "merged_count": len(merge_ids),
            "new_photo_count": new_count,
            "name": name,
        }
    finally:
        await conn.close()


# ------------------------------------------------------------------
# Google Contacts → Face Cluster matching
# ------------------------------------------------------------------

_matcher = GoogleContactsFaceMatcher()


class ConfirmMatchRequest(BaseModel):
    contact_id: int
    cluster_id: int
    name: str


@router.post("/contacts/sync-and-match")
async def contacts_sync_and_match():
    """Pull Google Contacts, compute face embeddings, match to clusters.
    Auto-names high-confidence matches. Returns stats + review queue."""
    try:
        result = await _matcher.match_contacts_to_clusters()
        if "error" in result:
            return JSONResponse(status_code=400, content=result)
        return result
    except Exception as e:
        logger.error(f"Contact sync error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/contacts/review")
async def contacts_review_queue():
    """Get contacts with uncertain face cluster matches for manual review."""
    try:
        queue = await _matcher.get_review_queue()
        return {"review": queue, "count": len(queue)}
    except Exception as e:
        logger.error(f"Review queue error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/contacts/confirm")
async def contacts_confirm_match(request: ConfirmMatchRequest):
    """Confirm a contact→cluster match and name the cluster."""
    try:
        return await _matcher.confirm_match(
            request.contact_id, request.cluster_id, request.name
        )
    except Exception as e:
        logger.error(f"Confirm match error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/contacts/reject/{contact_id}")
async def contacts_reject_match(contact_id: int):
    """Reject a suggested contact→cluster match."""
    try:
        return await _matcher.reject_match(contact_id)
    except Exception as e:
        logger.error(f"Reject match error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/contacts/stats")
async def contacts_stats():
    """Get current contact sync and matching stats."""
    try:
        return await _matcher.get_stats()
    except Exception as e:
        logger.error(f"Contacts stats error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------------------
# Media browsing (photos + videos, local + cloud)
# ------------------------------------------------------------------

@router.get("/browse")
async def browse_media(
    page: int = Query(1, ge=1),
    per_page: int = Query(40, ge=1, le=100),
    media_type: Optional[str] = Query(None, description="photo or video"),
    source: Optional[str] = Query(None, description="local, takeout, or all"),
    year: Optional[str] = Query(None),
    sort: str = Query("newest", description="newest, oldest, or largest"),
    has_description: Optional[bool] = Query(None),
    in_qdrant: Optional[bool] = Query(None),
):
    """Browse local media with filters. Returns paginated results with thumbnails."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://echo:echo_secure_password_123@localhost/echo_brain",
    )
    conn = await asyncpg.connect(db_url)
    try:
        conditions = ["1=1"]
        params: list = []
        idx = 1

        if media_type:
            conditions.append(f"media_type = ${idx}")
            params.append(media_type)
            idx += 1
        if source and source != "all":
            if source == "local":
                conditions.append(f"(source_root = 'local' OR source_root IS NULL)")
            else:
                conditions.append(f"source_root = ${idx}")
                params.append(source)
                idx += 1
        if year:
            conditions.append(f"year_folder = ${idx}")
            params.append(year)
            idx += 1
        if has_description is True:
            conditions.append("llava_description IS NOT NULL")
        elif has_description is False:
            conditions.append("llava_description IS NULL")
        if in_qdrant is True:
            conditions.append("qdrant_point_id IS NOT NULL")
        elif in_qdrant is False:
            conditions.append("qdrant_point_id IS NULL")

        where = " AND ".join(conditions)

        order = "created_at DESC"
        if sort == "oldest":
            order = "created_at ASC"
        elif sort == "largest":
            order = "file_size DESC NULLS LAST"

        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM photos WHERE {where}", *params
        )

        offset = (page - 1) * per_page
        rows = await conn.fetch(f"""
            SELECT id, file_path, filename, media_type, source_root, year_folder,
                   file_size, width, height, created_at, sha256,
                   llava_description, qdrant_point_id, match_type,
                   face_count, audio_transcript
            FROM photos
            WHERE {where}
            ORDER BY {order}
            LIMIT ${idx} OFFSET ${idx + 1}
        """, *params, per_page, offset)

        items = []
        for r in rows:
            items.append({
                "id": r["id"],
                "filename": r["filename"],
                "media_type": r["media_type"] or "photo",
                "source": r["source_root"] or "local",
                "year": r["year_folder"],
                "size_mb": round(r["file_size"] / (1024 * 1024), 2) if r["file_size"] else None,
                "width": r["width"],
                "height": r["height"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "description": r["llava_description"],
                "in_qdrant": r["qdrant_point_id"] is not None,
                "match_type": r["match_type"],
                "face_count": r["face_count"],
                "has_audio": r["audio_transcript"] is not None,
                "thumb_url": f"/api/photos/thumb/{r['id']}?size=280",
            })

        return {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page if total else 0,
        }
    finally:
        await conn.close()


@router.get("/browse/cloud")
async def browse_cloud_media():
    """Google Photos Library API was deprecated on 2025-03-31. Use Takeout import instead."""
    return JSONResponse(status_code=410, content={
        "error": "Google Photos Library API deprecated 2025-03-31",
        "alternative": "All cloud photos are available via Takeout import at /api/photos/browse/local"
    })


@router.get("/runs")
async def recent_runs(limit: int = Query(20, ge=1, le=50)):
    """Get recent dedup pipeline runs (scans, cloud fetch, analysis, ingestion)."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://echo:echo_secure_password_123@localhost/echo_brain",
    )
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch("""
            SELECT id, run_type, started_at, finished_at,
                   items_processed, items_new, items_skipped, items_error
            FROM photo_dedup_runs
            ORDER BY id DESC
            LIMIT $1
        """, limit)
        return {"runs": [dict(r) for r in rows]}
    finally:
        await conn.close()
