"""
Echo Brain Memory API
Provides endpoints for unified memory system management
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging

router = APIRouter(prefix="/memory", tags=["memory"])
logger = logging.getLogger(__name__)

class MemorySearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

@router.get("/status")
async def get_memory_status() -> Dict[str, Any]:
    """Get unified memory system status"""
    try:
        from src.core.unified_memory import memory_system
        return memory_system.get_stats()
    except Exception as e:
        logger.error(f"Failed to get memory status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def trigger_ingestion() -> Dict[str, Any]:
    """Manually trigger conversation ingestion"""
    try:
        from src.core.unified_memory import memory_system
        import asyncio

        if not memory_system._running:
            await memory_system.initialize()

        # Check if already running
        if hasattr(memory_system, '_ingestion_task') and memory_system._ingestion_task and not memory_system._ingestion_task.done():
            return {
                "status": "already_running",
                "stats": memory_system.get_stats()
            }

        # Run ingestion in background
        memory_system._ingestion_task = asyncio.create_task(memory_system.ingest_conversations())

        return {
            "status": "started",
            "message": "Ingestion started in background",
            "stats": memory_system.get_stats()
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_memory(request: MemorySearchRequest) -> Dict[str, Any]:
    """Search memories using semantic similarity"""
    try:
        from src.core.unified_memory import memory_system

        if not memory_system._running:
            await memory_system.initialize()

        results = await memory_system.search_memory(request.query, request.limit)

        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def memory_health() -> Dict[str, Any]:
    """Check memory system health"""
    try:
        from src.core.unified_memory import memory_system

        health = {
            "status": "healthy" if memory_system._running else "not_initialized",
            "stats": memory_system.get_stats()
        }

        # Check for issues
        if memory_system.stats.get("errors"):
            health["status"] = "degraded"
            health["recent_errors"] = memory_system.stats["errors"][-5:]

        return health
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }