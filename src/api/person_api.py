"""
Person Identity API Router — Echo Brain
Endpoints for managing person identities, merge suggestions, and timelines.
"""
import logging
import os
from typing import List, Optional

import asyncpg
from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.services.person_id_service import PersonIDService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/persons", tags=["persons"])

DB_URL = os.getenv("DATABASE_URL",
                    "postgresql://echo:echo_secure_password_123@localhost/echo_brain")

_service = PersonIDService()


class NameRequest(BaseModel):
    name: str


class MergeRequest(BaseModel):
    source_id: int
    target_id: int


class SplitRequest(BaseModel):
    cluster_ids: List[int]


# ------------------------------------------------------------------
# Person CRUD
# ------------------------------------------------------------------

@router.get("")
async def list_persons(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    named_only: bool = Query(False),
):
    """List persons with face count, date range."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        return await _service.list_persons(conn, limit=limit, offset=offset,
                                           named_only=named_only)
    except Exception as e:
        logger.error(f"List persons error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


@router.get("/suggestions")
async def get_suggestions(limit: int = Query(20, ge=1, le=100)):
    """Get auto-merge suggestions for review."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        suggestions = await _service.get_merge_suggestions(conn, limit=limit)
        return {"suggestions": suggestions, "count": len(suggestions)}
    except Exception as e:
        logger.error(f"Suggestions error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


@router.post("/suggestions/{suggestion_id}/accept")
async def accept_suggestion(suggestion_id: int = Path(..., ge=1)):
    """Accept a merge suggestion."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        result = await _service.accept_suggestion(conn, suggestion_id)
        if "error" in result:
            return JSONResponse(status_code=404, content=result)
        return result
    except Exception as e:
        logger.error(f"Accept suggestion error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


@router.post("/suggestions/{suggestion_id}/reject")
async def reject_suggestion(suggestion_id: int = Path(..., ge=1)):
    """Reject a merge suggestion."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        result = await _service.reject_suggestion(conn, suggestion_id)
        if "error" in result:
            return JSONResponse(status_code=404, content=result)
        return result
    except Exception as e:
        logger.error(f"Reject suggestion error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


@router.get("/{person_id}")
async def get_person(person_id: int = Path(..., ge=1)):
    """Person detail + clusters + samples."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        detail = await _service.get_person_detail(conn, person_id)
        if detail is None:
            return JSONResponse(status_code=404,
                                content={"error": f"Person {person_id} not found"})
        return detail
    except Exception as e:
        logger.error(f"Get person error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


@router.put("/{person_id}/name")
async def name_person(person_id: int, request: NameRequest):
    """Name person → propagates to all clusters."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        result = await _service.name_person(conn, person_id, request.name)
        if "error" in result:
            return JSONResponse(status_code=404, content=result)
        return result
    except Exception as e:
        logger.error(f"Name person error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


@router.post("/merge")
async def merge_persons(request: MergeRequest):
    """Merge two persons."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        result = await _service.merge_persons(conn, request.source_id, request.target_id)
        if "error" in result:
            return JSONResponse(status_code=404, content=result)
        return result
    except Exception as e:
        logger.error(f"Merge persons error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


@router.post("/{person_id}/split")
async def split_person(person_id: int, request: SplitRequest):
    """Detach clusters from person."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        result = await _service.split_clusters(conn, person_id, request.cluster_ids)
        if "error" in result:
            return JSONResponse(status_code=404, content=result)
        return result
    except Exception as e:
        logger.error(f"Split person error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


@router.get("/{person_id}/timeline")
async def person_timeline(person_id: int = Path(..., ge=1)):
    """Photos by year for person."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        return await _service.get_person_timeline(conn, person_id)
    except Exception as e:
        logger.error(f"Timeline error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


# ------------------------------------------------------------------
# Recluster trigger
# ------------------------------------------------------------------

@router.post("/prune-singletons")
async def prune_singletons(min_faces: int = Query(3, ge=1, le=20)):
    """Remove unnamed persons with fewer than min_faces faces."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        result = await _service.prune_singletons(conn, min_faces=min_faces)
        return result
    except Exception as e:
        logger.error(f"Prune singletons error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


@router.post("/recluster")
async def trigger_recluster():
    """Prepare (clear non-locked assignments) then full graph recluster."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        result = await _service.prepare_for_recluster(conn)
        return result
    except Exception as e:
        logger.error(f"Recluster error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()


# ------------------------------------------------------------------
# Migration endpoint (one-time)
# ------------------------------------------------------------------

@router.post("/migrate")
async def run_migration():
    """One-time migration: create Person records for existing named clusters."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await _service.ensure_schema(conn)
        result = await _service.migrate_named_clusters(conn)
        return result
    except Exception as e:
        logger.error(f"Migration error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await conn.close()
