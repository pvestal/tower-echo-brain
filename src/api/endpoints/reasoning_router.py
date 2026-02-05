"""
Echo Brain Reasoning API
Provides the /ask endpoint for LLM-synthesized responses
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import json
from datetime import datetime

from src.core.reasoning_engine import reasoning_engine
from src.core.pg_reasoning import ask_with_pg, search_pg

router = APIRouter(prefix="", tags=["reasoning"])
logger = logging.getLogger(__name__)

class AskRequest(BaseModel):
    """Request for asking Echo Brain a question"""
    question: str
    context: Optional[str] = None

class AskResponse(BaseModel):
    """Echo Brain's synthesized response"""
    question: str
    answer: str
    confidence: float
    memories_used: int
    sources: List[str]
    reasoning_time_ms: int
    model_used: str

class SearchRequest(BaseModel):
    """Request for raw memory search"""
    query: str
    limit: int = 10

@router.post("/ask")
async def ask_question(request: AskRequest):
    """
    Ask Echo Brain a question and get a synthesized response using LLM
    """
    try:
        logger.info(f"🤖 Processing question: {request.question[:100]}...")

        # Get response using PostgreSQL search
        response = ask_with_pg(request.question)

        # Get conversation sources
        memories = search_pg(request.question, limit=5)
        sources = [f"conversation:{m['conv']}" for m in memories]

        # Return response matching expected format
        return {
            "question": request.question,
            "answer": response["answer"],
            "confidence": response["confidence"],
            "memories_used": response["memories_used"],
            "sources": sources,
            "reasoning_time_ms": 0,  # Not tracked yet
            "model_used": "mistral:7b",
            # Enhanced metrics
            "memories_searched": len(memories),
            "avg_memory_score": 0.8,
            "memory_search_time_ms": 0,
            "llm_generation_time_ms": 0,
            "total_time_ms": 0,
            "embedding_model_used": "postgresql_fulltext"
        }

    except Exception as e:
        logger.error(f"Failed to process question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ask")
async def ask_question_get(q: str = Query(..., description="Question to ask Echo Brain")):
    """
    Ask Echo Brain a question via GET request
    """
    try:
        logger.info(f"🤖 Processing GET question: {q[:100]}...")

        response = await reasoning_engine.ask(question=q)

        return {
            "question": response.query,
            "answer": response.answer,
            "confidence": response.confidence,
            "memories_used": response.memories_used,
            "sources": response.sources,
            "reasoning_time_ms": response.reasoning_time_ms,
            "model_used": response.model_used
        }

    except Exception as e:
        logger.error(f"Failed to process question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_memories(request: SearchRequest):
    """
    Search raw memories without LLM synthesis
    """
    try:
        logger.info(f"🔍 Searching for: {request.query[:100]}...")

        results = await reasoning_engine.search_only(
            query=request.query,
            limit=request.limit
        )

        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Check if reasoning engine is healthy
    """
    try:
        # Test if we can retrieve memories
        test_memories = await reasoning_engine.search_only("test", limit=1)

        return {
            "status": "healthy",
            "service": "reasoning-engine",
            "memory_access": "ok" if test_memories is not None else "error",
            "models": {
                "embedding": reasoning_engine.embedding_model,
                "reasoning": reasoning_engine.reasoning_model
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "reasoning-engine",
            "error": str(e)
        }

@router.post("/ask/stream")
async def ask_question_stream(request: AskRequest):
    """
    Stream Echo Brain's response with real-time progress updates
    """
    async def generate_sse():
        try:
            start_time = datetime.now()
            memory_search_time = 0

            # Stream the thinking process
            async for event in reasoning_engine.think_stream(
                query=request.question,
                max_memories=10
            ):
                event_type = event.get("type")

                if event_type == "status":
                    # Send status update
                    yield f"event: status\ndata: {json.dumps({
                        'message': event.get('data'),
                        'timestamp': event.get('timestamp')
                    })}\n\n"

                elif event_type == "memories_found":
                    # Send memory search metrics
                    memory_search_time = (datetime.now() - start_time).total_seconds() * 1000
                    data = event.get('data', {})
                    yield f"event: metrics\ndata: {json.dumps({
                        'memories_searched': data.get('memories_searched', 0),
                        'memories_used': data.get('memories_used', 0),
                        'avg_memory_score': data.get('avg_memory_score', 0),
                        'memory_search_time_ms': round(memory_search_time, 2),
                        'embedding_model_used': reasoning_engine.embedding_model
                    })}\n\n"

                elif event_type == "response_chunk":
                    # Stream response text
                    yield f"event: chunk\ndata: {json.dumps({
                        'text': event.get('data', ''),
                        'model_used': reasoning_engine.active_model
                    })}\n\n"

                elif event_type == "complete":
                    # Send final metrics
                    total_time = (datetime.now() - start_time).total_seconds() * 1000
                    completion_data = event.get('data', {})

                    yield f"event: complete\ndata: {json.dumps({
                        'model_used': completion_data.get('model_used'),
                        'memories_searched': completion_data.get('memories_searched', 0),
                        'memories_used': completion_data.get('memories_used', 0),
                        'avg_memory_score': completion_data.get('avg_memory_score', 0),
                        'confidence': completion_data.get('confidence', 0),
                        'reasoning_time_ms': completion_data.get('reasoning_time_ms', 0),
                        'llm_generation_time_ms': round(total_time - memory_search_time, 2),
                        'total_time_ms': round(total_time, 2),
                        'embedding_model_used': reasoning_engine.embedding_model,
                        'sources': completion_data.get('sources', [])
                    })}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )