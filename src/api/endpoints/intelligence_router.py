"""
Echo Brain Intelligence API
This is where Echo Brain actually THINKS and responds intelligently
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import json
from datetime import datetime

from src.core.intelligence_engine import intelligence, KnowledgeDomain

router = APIRouter(prefix="/intelligence", tags=["intelligence"])
logger = logging.getLogger(__name__)

class ThinkRequest(BaseModel):
    """Request for Echo Brain to think about something"""
    query: str
    context: Optional[str] = None
    return_reasoning: bool = True

class ThinkResponse(BaseModel):
    """Echo Brain's intelligent response"""
    query: str
    response: str
    domain: str
    confidence: float
    reasoning_steps: Optional[List[str]]
    sources: List[str]
    memories_used: int
    thinking_time_ms: int

@router.post("/think", response_model=ThinkResponse)
async def think(request: ThinkRequest):
    """
    Ask Echo Brain to think about something and respond intelligently
    """
    start_time = datetime.now()

    try:
        # Let Echo Brain think
        thought = await intelligence.think_and_respond(
            request.query,
            request.context
        )

        # Calculate thinking time
        thinking_time = (datetime.now() - start_time).total_seconds() * 1000

        response = ThinkResponse(
            query=request.query,
            response=thought.response,
            domain=thought.domain.value,
            confidence=thought.confidence_score,
            reasoning_steps=thought.reasoning_steps if request.return_reasoning else None,
            sources=thought.sources_used,
            memories_used=len(thought.memories_retrieved),
            thinking_time_ms=int(thinking_time)
        )

        logger.info(f"🧠 Thought about '{request.query[:50]}...' - Confidence: {thought.confidence_score:.2%}")

        return response

    except Exception as e:
        logger.error(f"Thinking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-map")
async def get_knowledge_map():
    """
    Get a comprehensive map of what Echo Brain knows
    """
    try:
        coverage = await intelligence.analyze_knowledge_coverage()

        # Format for better readability
        knowledge_map = {
            "summary": {
                "total_domains": coverage["domains_analyzed"],
                "domains_with_knowledge": coverage["domains_with_knowledge"],
                "total_knowledge_points": coverage["total_memories_sampled"]
            },
            "domains": {}
        }

        for domain, stats in coverage["domain_coverage"].items():
            knowledge_map["domains"][domain] = {
                "depth": stats["knowledge_depth"],
                "confidence": f"{stats['avg_confidence']:.1%}",
                "memory_count": stats["memories_found"],
                "top_examples": [
                    ex["content"][:100] for ex in stats["sample_memories"][:2]
                ]
            }

        knowledge_map["timestamp"] = coverage["timestamp"]

        return knowledge_map

    except Exception as e:
        logger.error(f"Knowledge mapping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-knowledge")
async def compare_knowledge(topics: List[str]):
    """
    Compare Echo Brain's knowledge across multiple topics
    """
    if not topics or len(topics) > 10:
        raise HTTPException(status_code=400, detail="Provide 1-10 topics to compare")

    comparisons = {}

    for topic in topics:
        # Get memories for each topic
        thought = await intelligence.think_and_respond(topic)

        comparisons[topic] = {
            "domain": thought.domain.value,
            "confidence": thought.confidence_score,
            "memories_found": len(thought.memories_retrieved),
            "has_knowledge": thought.confidence_score > 0.5,
            "knowledge_quality": (
                "comprehensive" if thought.confidence_score > 0.7 else
                "moderate" if thought.confidence_score > 0.4 else
                "limited"
            )
        }

    # Determine strongest and weakest areas
    strongest = max(comparisons.items(), key=lambda x: x[1]["confidence"])
    weakest = min(comparisons.items(), key=lambda x: x[1]["confidence"])

    return {
        "topics_analyzed": len(topics),
        "comparisons": comparisons,
        "strongest_knowledge": {
            "topic": strongest[0],
            "confidence": strongest[1]["confidence"]
        },
        "weakest_knowledge": {
            "topic": weakest[0],
            "confidence": weakest[1]["confidence"]
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/thinking-log")
async def get_thinking_log(limit: int = 10):
    """
    Get Echo Brain's recent thinking history
    """
    recent_thoughts = intelligence.thinking_log[-limit:]

    log_entries = []
    for thought in recent_thoughts:
        log_entries.append({
            "query": thought.query,
            "domain": thought.domain.value,
            "confidence": thought.confidence_score,
            "memories_used": len(thought.memories_retrieved),
            "response_preview": thought.response[:200] + "..." if len(thought.response) > 200 else thought.response
        })

    return {
        "total_thoughts": len(intelligence.thinking_log),
        "recent_thoughts": log_entries,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/test-understanding")
async def test_understanding(topic: str, questions: List[str]):
    """
    Test Echo Brain's understanding of a topic with specific questions
    """
    if not questions or len(questions) > 5:
        raise HTTPException(status_code=400, detail="Provide 1-5 questions")

    results = {
        "topic": topic,
        "questions_tested": len(questions),
        "answers": [],
        "overall_understanding": 0.0
    }

    total_confidence = 0.0

    for question in questions:
        full_query = f"{topic}: {question}"
        thought = await intelligence.think_and_respond(full_query)

        results["answers"].append({
            "question": question,
            "answer": thought.response,
            "confidence": thought.confidence_score,
            "sources_used": len(thought.sources_used)
        })

        total_confidence += thought.confidence_score

    results["overall_understanding"] = total_confidence / len(questions)

    # Assess understanding level
    understanding = results["overall_understanding"]
    if understanding > 0.7:
        results["assessment"] = "Strong understanding"
    elif understanding > 0.5:
        results["assessment"] = "Moderate understanding"
    elif understanding > 0.3:
        results["assessment"] = "Basic understanding"
    else:
        results["assessment"] = "Limited understanding"

    return results

@router.post("/think/stream")
async def think_stream_endpoint(request: ThinkRequest):
    """
    Stream Echo Brain's thinking process with real-time updates
    """
    async def generate_sse():
        try:
            start_time = datetime.now()
            memory_search_time = 0

            async for event in intelligence.think_stream(
                request.query,
                request.context
            ):
                event_type = event.get("type")

                if event_type == "status":
                    yield f"event: status\ndata: {json.dumps({
                        'message': event.get('data'),
                        'timestamp': event.get('timestamp')
                    })}\n\n"

                elif event_type == "memories_found":
                    memory_search_time = (datetime.now() - start_time).total_seconds() * 1000
                    data = event.get('data', {})
                    yield f"event: metrics\ndata: {json.dumps({
                        'memories_searched': data.get('memories_searched', 0),
                        'memories_used': data.get('memories_used', 0),
                        'avg_memory_score': data.get('avg_memory_score', 0),
                        'memory_search_time_ms': round(memory_search_time, 2),
                        'embedding_model_used': intelligence.embedding_model
                    })}\n\n"

                elif event_type == "response_chunk":
                    yield f"event: chunk\ndata: {json.dumps({
                        'text': event.get('data', ''),
                        'model_used': intelligence.active_model
                    })}\n\n"

                elif event_type == "complete":
                    total_time = (datetime.now() - start_time).total_seconds() * 1000
                    completion_data = event.get('data', {})

                    yield f"event: complete\ndata: {json.dumps({
                        'domain': completion_data.get('domain'),
                        'model_used': completion_data.get('model_used'),
                        'confidence': completion_data.get('confidence'),
                        'memories_searched': completion_data.get('memories_searched', 0),
                        'memories_used': completion_data.get('memories_used', 0),
                        'avg_memory_score': completion_data.get('avg_memory_score', 0),
                        'memory_search_time_ms': round(memory_search_time, 2),
                        'llm_generation_time_ms': round(total_time - memory_search_time, 2),
                        'total_time_ms': round(total_time, 2),
                        'thinking_time_ms': completion_data.get('thinking_time_ms'),
                        'embedding_model_used': intelligence.embedding_model,
                        'reasoning_steps': completion_data.get('reasoning_steps', []),
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
            "X-Accel-Buffering": "no"
        }
    )