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
    """Get memory system status — delegates to /health for real data"""
    return await memory_health()

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
    """Check memory system health — queries actual Qdrant + DB status"""
    import httpx
    import asyncpg
    import os
    from datetime import datetime

    health: Dict[str, Any] = {"status": "healthy", "checked_at": datetime.now().isoformat()}
    issues = []

    # 1. Check Qdrant
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:6333/collections/echo_memory")
            if resp.status_code == 200:
                info = resp.json().get("result", {})
                health["qdrant"] = {
                    "status": info.get("status", "unknown"),
                    "vectors": info.get("points_count", 0),
                    "indexed": info.get("indexed_vectors_count", 0),
                }
            else:
                issues.append("Qdrant collection unreachable")
                health["qdrant"] = {"status": "error"}
    except Exception as e:
        issues.append(f"Qdrant connection failed: {e}")
        health["qdrant"] = {"status": "error", "error": str(e)}

    # 2. Check PostgreSQL conversation tracking
    try:
        conn = await asyncpg.connect(
            host="localhost", database="echo_brain",
            user="patrick", password=os.getenv("DB_PASSWORD", "")
        )
        try:
            conv_count = await conn.fetchval("SELECT COUNT(*) FROM claude_conversations")
            fact_count = await conn.fetchval("SELECT COUNT(*) FROM facts")
            kf_count = await conn.fetchval("SELECT COUNT(*) FROM knowledge_facts WHERE valid_until IS NULL")
            health["postgresql"] = {
                "conversations_indexed": conv_count,
                "facts": fact_count,
                "knowledge_facts_active": kf_count,
            }
        finally:
            await conn.close()
    except Exception as e:
        issues.append(f"PostgreSQL connection failed: {e}")
        health["postgresql"] = {"status": "error", "error": str(e)}

    # 3. Check Ollama embedding model
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                has_embed = any("nomic-embed" in m for m in models)
                health["embedding_model"] = {
                    "available": has_embed,
                    "model": "nomic-embed-text:latest",
                }
                if not has_embed:
                    issues.append("Embedding model nomic-embed-text not loaded in Ollama")
    except Exception as e:
        issues.append(f"Ollama connection failed: {e}")
        health["embedding_model"] = {"status": "error", "error": str(e)}

    if issues:
        health["status"] = "degraded"
        health["issues"] = issues

    return health