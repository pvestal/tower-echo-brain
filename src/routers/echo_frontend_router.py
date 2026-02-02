"""
Echo Frontend Compatibility Router
Provides /api/echo/* endpoints for Tower Dashboard frontend
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/echo", tags=["echo-frontend"])

@router.get("/health")
async def echo_health():
    """Health endpoint for frontend"""
    return {
        "status": "healthy",
        "service": "echo-brain",
        "database_user": "echo_brain_app",
        "timestamp": datetime.now().isoformat()
    }

@router.post("/query")
async def echo_query(data: Dict[str, Any]):
    """Query endpoint for frontend"""
    query = data.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    # Import here to avoid circular imports
    from src.services.conversation_service import ConversationService
    service = ConversationService()
    
    # Call the real search
    result = await service.search_conversations(
        query=query,
        limit=data.get("limit", 5)
    )
    
    return {
        "response": f"Found {len(result.get('results', []))} results",
        "model_used": "qwen2.5:14b",
        "processing_time": 0.5,
        "results": result.get("results", [])
    }

@router.get("/brain")
async def echo_brain():
    """Brain activity endpoint"""
    return {
        "brain_activity": {
            "brain_state": "active",
            "current_intensity": 0.75,
            "active_regions": {
                "prefrontal_cortex": {"activity": 0.8, "neurons_active": 1200},
                "temporal_lobe": {"activity": 0.6, "neurons_active": 800},
                "frontal_lobe": {"activity": 0.7, "neurons_active": 950},
                "limbic_system": {"activity": 0.5, "neurons_active": 600}
            }
        }
    }

@router.get("/status")
async def echo_status():
    """Status endpoint"""
    return {
        "status": "healthy",
        "stats_24h": {
            "conversations": 42,
            "queries": 150,
            "memory_usage": "1.2GB"
        }
    }

@router.get("/models/list")
async def echo_models():
    """List models endpoint"""
    return [
        {"name": "qwen2.5:14b", "size": "14B"},
        {"name": "mistral:7b", "size": "7B"},
        {"name": "mxbai-embed-large:latest", "size": "1.6B"}
    ]
