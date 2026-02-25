#!/usr/bin/env python3
"""
Apple Music API endpoints for Echo Brain
Proxies requests through tower-auth bridge for Apple Music integration
"""
from fastapi import APIRouter, HTTPException
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/music", tags=["apple_music"])


@router.get("/status")
async def apple_music_status():
    """Check Apple Music connection status"""
    try:
        from src.integrations.tower_auth_bridge import tower_auth
        status = await tower_auth.get_apple_music_status()
        return status
    except Exception as e:
        logger.error(f"Error checking Apple Music status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_apple_music(q: str, types: str = "songs,albums,playlists", limit: int = 10):
    """Search Apple Music catalog"""
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    try:
        from src.integrations.tower_auth_bridge import tower_auth
        results = await tower_auth.search_apple_music(q, types=types, limit=limit)
        if "error" in results:
            raise HTTPException(status_code=503, detail=results["error"])
        return {"query": q, "results": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apple Music search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/playlists")
async def get_playlists():
    """Get user's Apple Music playlists (requires user authorization)"""
    try:
        from src.integrations.tower_auth_bridge import tower_auth
        result = await tower_auth.get_apple_music_playlists()
        if "error" in result:
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apple Music playlists error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/playlists/{playlist_id}/tracks")
async def get_playlist_tracks(playlist_id: str):
    """Get tracks from a playlist, enriched with 30-sec preview URLs from the catalog."""
    try:
        from src.integrations.tower_auth_bridge import tower_auth
        result = await tower_auth.get_apple_music_playlist_tracks(playlist_id)
        if "error" in result:
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apple Music playlist tracks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/catalog/{catalog_id}")
async def get_catalog_track(catalog_id: str):
    """Look up a single track in the Apple Music catalog (preview URL, metadata)."""
    try:
        from src.integrations.tower_auth_bridge import tower_auth
        result = await tower_auth.get_apple_music_catalog_track(catalog_id)
        if "error" in result:
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apple Music catalog lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
