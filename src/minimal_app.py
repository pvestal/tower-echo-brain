#!/usr/bin/env python3
"""
Minimal Vector Search API
Only exposes POST /api/context endpoint for Qdrant search
"""

import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_memory import QdrantMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global memory instance
memory_system: Optional[QdrantMemory] = None


class VectorSearchRequest(BaseModel):
    """Request model for vector search"""
    query: str
    limit: int = 5


class VectorSearchResponse(BaseModel):
    """Response model for vector search"""
    results: List[Dict]
    count: int
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global memory_system

    # Startup
    logger.info("🚀 Starting minimal vector search API...")
    try:
        memory_system = QdrantMemory(collection_name="echo_memories")
        info = memory_system.get_collection_info()
        logger.info(f"✅ Connected to Qdrant: {info.get('vectors_count', 0)} vectors")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Qdrant: {e}")
        memory_system = None

    yield

    # Shutdown
    logger.info("🛑 Shutting down vector search API...")


# Create minimal FastAPI app
app = FastAPI(
    title="Minimal Vector Search",
    description="Single endpoint for Qdrant vector search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
async def health_check():
    """Simple health check"""
    global memory_system

    if memory_system:
        try:
            info = memory_system.get_collection_info()
            return {
                "status": "healthy",
                "vectors": info.get("vectors_count", 0),
                "collection": info.get("name", "unknown")
            }
        except:
            pass

    return {
        "status": "unhealthy",
        "error": "Qdrant not connected"
    }


@app.post("/api/context", response_model=VectorSearchResponse)
async def vector_search(request: VectorSearchRequest):
    """
    Search for relevant vectors in Qdrant

    Accepts:
    - query: The search query string
    - limit: Number of results to return (default 5)

    Returns:
    - Top matching vectors with scores and metadata
    """
    global memory_system

    if not memory_system:
        raise HTTPException(
            status_code=503,
            detail="Vector search system not available"
        )

    try:
        # Perform vector search
        results = await memory_system.search_memories(
            query=request.query,
            limit=request.limit
        )

        return VectorSearchResponse(
            results=results,
            count=len(results),
            status="success"
        )

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# Disable all other endpoints by not including any other routers
# This is the ONLY functionality exposed

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8309,
        log_level="info"
    )