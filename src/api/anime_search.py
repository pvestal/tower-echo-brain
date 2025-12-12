#!/usr/bin/env python3
"""
Anime Semantic Search API Endpoints for Echo Brain
"""

import sys
import asyncio
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import logging

# Add anime production path for imports
sys.path.append('/opt/tower-anime-production/src')

try:
    from anime_semantic_search import (
        search_similar_generations,
        get_optimal_generation_params,
        index_new_generation,
        batch_index_all
    )
    SEARCH_AVAILABLE = True
except ImportError as e:
    logging.error(f"Anime semantic search not available: {e}")
    SEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/echo/anime", tags=["anime_search"])

class SearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., description="Search query text")
    limit: int = Field(5, ge=1, le=20, description="Maximum results to return")
    min_quality: float = Field(0.0, ge=0.0, le=10.0, description="Minimum quality score filter")

class OptimalParamsRequest(BaseModel):
    """Request model for optimal parameter extraction"""
    query: str = Field(..., description="Target prompt/style description")
    min_quality: float = Field(7.0, ge=0.0, le=10.0, description="Minimum quality threshold")

class IndexRequest(BaseModel):
    """Request model for indexing a generation"""
    job_id: int = Field(..., description="Job ID to index")

@router.post("/search")
async def search_anime_generations(request: SearchRequest):
    """
    Search for similar anime generations using semantic search

    Returns similar generations with their parameters and quality scores
    """
    if not SEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Semantic search service not available")

    try:
        results = await search_similar_generations(
            query=request.query,
            limit=request.limit
        )

        return {
            "success": True,
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimal-params")
async def get_optimal_params(request: OptimalParamsRequest):
    """
    Get optimal generation parameters based on similar successful generations

    Analyzes past successes to recommend best parameters
    """
    if not SEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Semantic search service not available")

    try:
        result = await get_optimal_generation_params(request.query)

        if not result.get('success'):
            raise HTTPException(status_code=404, detail=result.get('message', 'No data found'))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimal params extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index")
async def index_generation(request: IndexRequest):
    """
    Index a completed generation for semantic search

    Adds generation to searchable knowledge base
    """
    if not SEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Semantic search service not available")

    try:
        success = await index_new_generation(request.job_id)

        if success:
            return {
                "success": True,
                "message": f"Job {request.job_id} indexed successfully",
                "job_id": request.job_id
            }
        else:
            raise HTTPException(status_code=404, detail=f"Job {request.job_id} not found or indexing failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-index")
async def batch_index_generations():
    """
    Index all existing completed generations

    One-time operation to build initial search index
    """
    if not SEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Semantic search service not available")

    try:
        result = await batch_index_all()

        if result.get('success'):
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Batch indexing failed'))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/character/{character_name}/evolution")
async def get_character_evolution(
    character_name: str,
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get character style evolution over time

    Shows how a character's generation has improved
    """
    if not SEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Semantic search service not available")

    try:
        # Search for all generations of this character
        results = await search_similar_generations(
            query=f"{character_name} anime character",
            limit=limit
        )

        # Filter to only this character
        character_results = [
            r for r in results
            if r.get('character_name', '').lower() == character_name.lower()
        ]

        # Sort by date
        character_results.sort(key=lambda x: x.get('created_at', ''))

        # Calculate evolution metrics
        if character_results:
            quality_trend = []
            for r in character_results:
                scores = r.get('quality_scores', {})
                if scores:
                    avg_score = sum(scores.values()) / len(scores)
                    quality_trend.append({
                        'job_id': r['job_id'],
                        'date': r['created_at'],
                        'quality': avg_score,
                        'scores': scores
                    })

            return {
                "success": True,
                "character": character_name,
                "generations": len(character_results),
                "quality_trend": quality_trend,
                "latest_params": character_results[-1].get('generation_params', {}) if character_results else {},
                "improvement": quality_trend[-1]['quality'] - quality_trend[0]['quality'] if len(quality_trend) > 1 else 0
            }
        else:
            raise HTTPException(status_code=404, detail=f"No generations found for character: {character_name}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Character evolution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def semantic_search_health():
    """Check if semantic search service is available"""
    return {
        "status": "healthy" if SEARCH_AVAILABLE else "unavailable",
        "service": "anime_semantic_search",
        "available": SEARCH_AVAILABLE
    }