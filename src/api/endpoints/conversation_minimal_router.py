"""
SIMPLE Conversation Router
No dependencies, just works
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    user_id: Optional[str] = None

@router.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "conversation-search",
        "timestamp": datetime.now().isoformat(),
        "message": "Simple conversation router is working"
    }

@router.post("/search")
async def search(request: SearchRequest):
    """Simple search endpoint"""
    logger.info(f"Search request: {request.query}")
    
    # For now, return mock data
    # In production, this would connect to PostgreSQL or Qdrant
    return {
        "status": "success",
        "query": request.query,
        "results": [
            {
                "id": "1",
                "title": "Test Result",
                "content": f"Found results for: {request.query}",
                "score": 0.85,
                "timestamp": datetime.now().isoformat()
            }
        ],
        "count": 1,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/test")
async def test():
    """Test endpoint"""
    return {
        "message": "Conversation router is working!",
        "endpoints": ["/health", "/search", "/test"]
    }
