#!/usr/bin/env python3
"""
Collaboration API Routes for Echo Brain
Exposes multi-LLM collaboration framework through REST API
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import the collaboration framework
from ..collaboration_framework import collaborate_on_query, CollaborationResult, collaborator

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/collaboration", tags=["Multi-LLM Collaboration"])

class CollaborationRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    stream: bool = False

class CollaborationResponse(BaseModel):
    query: str
    consensus: str
    confidence_score: float
    fabrication_detected: bool
    collaboration_time: float
    models_used: list
    phases_completed: list
    inquisitive_validation: Optional[str] = None
    timestamp: str

@router.post("/collaborate", response_model=CollaborationResponse)
async def multi_llm_collaboration(request: CollaborationRequest):
    """
    Execute real-time collaboration between qwen-coder and deepseek-coder models

    This endpoint orchestrates a multi-phase collaboration workflow:
    1. Initial analysis with qwen-coder
    2. Technical implementation with qwen-coder
    3. Code review with deepseek-coder (or fallback)
    4. Optimization based on review
    5. Consensus building
    6. Inquisitive validation for fabrication detection
    """
    try:
        logger.info(f"🤝 Starting collaboration for query: {request.query[:100]}...")

        # Execute collaboration
        result: CollaborationResult = await collaborate_on_query(request.query, request.context)

        # Format response
        response = CollaborationResponse(
            query=result.query,
            consensus=result.consensus,
            confidence_score=result.confidence_score,
            fabrication_detected=result.fabrication_detected,
            collaboration_time=result.collaboration_time,
            models_used=[r.model for r in result.responses],
            phases_completed=[p.value for p in result.phases_completed],
            inquisitive_validation=result.inquisitive_validation,
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"✅ Collaboration completed: {result.collaboration_time:.2f}s, confidence: {result.confidence_score:.1f}%")
        return response

    except Exception as e:
        logger.error(f"❌ Collaboration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Collaboration error: {str(e)}")

@router.post("/collaborate/stream")
async def stream_collaboration(request: CollaborationRequest):
    """
    Stream real-time collaboration progress as Server-Sent Events
    """
    if not request.stream:
        request.stream = True

    async def generate_collaboration_stream():
        """Generate SSE stream of collaboration progress"""
        try:
            # Start collaboration
            yield f"data: {{'event': 'start', 'message': 'Starting multi-LLM collaboration...', 'timestamp': '{datetime.now().isoformat()}'}}\n\n"

            # We'll need to modify the collaborator to support streaming
            # For now, provide progress updates
            phases = [
                "Initial analysis with qwen-coder",
                "Technical implementation with qwen-coder",
                "Code review with deepseek-coder",
                "Optimization based on review",
                "Building consensus",
                "Inquisitive validation"
            ]

            for i, phase in enumerate(phases, 1):
                yield f"data: {{'event': 'phase', 'phase': {i}, 'message': '{phase}', 'timestamp': '{datetime.now().isoformat()}'}}\n\n"
                await asyncio.sleep(2)  # Simulate processing time

            # Execute actual collaboration
            result = await collaborate_on_query(request.query, request.context)

            # Send final result
            final_data = {
                'event': 'complete',
                'consensus': result.consensus,
                'confidence_score': result.confidence_score,
                'fabrication_detected': result.fabrication_detected,
                'collaboration_time': result.collaboration_time,
                'timestamp': datetime.now().isoformat()
            }

            import json
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            error_data = {
                'event': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_collaboration_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@router.get("/models/status")
async def get_collaboration_models_status():
    """
    Get status of all models available for collaboration
    """
    try:
        status = {}

        # Check qwen-coder (Ollama)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "qwen2.5-coder:7b",
                        "prompt": "test",
                        "stream": False
                    }
                )
                status["qwen-coder"] = {
                    "available": response.status_code == 200,
                    "endpoint": "http://localhost:11434",
                    "model": "qwen2.5-coder:7b",
                    "role": "Technical implementation"
                }
        except Exception as e:
            status["qwen-coder"] = {
                "available": False,
                "error": str(e),
                "role": "Technical implementation"
            }

        # Check deepseek-coder (API - placeholder)
        status["deepseek-coder"] = {
            "available": False,
            "note": "API key configuration needed",
            "role": "Code review and optimization"
        }

        # Check inquisitive core
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get("http://localhost:8330/api/echo/status")
                status["inquisitive-core"] = {
                    "available": response.status_code == 200,
                    "endpoint": "http://localhost:8330",
                    "role": "Fabrication detection and validation"
                }
        except Exception as e:
            status["inquisitive-core"] = {
                "available": False,
                "error": str(e),
                "role": "Fabrication detection and validation"
            }

        return {
            "collaboration_framework": "active",
            "models": status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.get("/history")
async def get_collaboration_history(limit: int = 10):
    """
    Get recent collaboration history from database
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host='***REMOVED***',
            database='echo_brain',
            user='patrick',
            password=os.getenv('DB_PASSWORD', '***REMOVED***')
        )

        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT query, response, model_used, processing_time, timestamp, metadata
            FROM echo_unified_interactions
            WHERE model_used LIKE 'multi-llm%'
            ORDER BY timestamp DESC
            LIMIT %s
        """, (limit,))

        results = cursor.fetchall()
        conn.close()

        return {
            "collaboration_history": [dict(row) for row in results],
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to fetch collaboration history: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/test")
async def test_collaboration_framework():
    """
    Test the collaboration framework with a simple query
    """
    test_query = "Create a simple Python function that calculates the factorial of a number."

    try:
        result = await collaborate_on_query(test_query)

        return {
            "test_status": "success",
            "test_query": test_query,
            "collaboration_time": result.collaboration_time,
            "confidence_score": result.confidence_score,
            "models_involved": [r.model for r in result.responses],
            "phases_completed": [p.value for p in result.phases_completed],
            "fabrication_detected": result.fabrication_detected,
            "timestamp": datetime.now().isoformat(),
            "sample_consensus": result.consensus[:500] + "..." if len(result.consensus) > 500 else result.consensus
        }

    except Exception as e:
        logger.error(f"Collaboration test failed: {e}")
        return {
            "test_status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Add to existing Echo Brain main app
def register_collaboration_routes(app):
    """Register collaboration routes with the main FastAPI app"""
    app.include_router(router)
    logger.info("🤝 Multi-LLM collaboration routes registered")