"""Deep Research API endpoints — decompose, search, evaluate, synthesize."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["deep-research"])


class ResearchRequest(BaseModel):
    question: str
    depth: str = "standard"  # quick, standard, deep


class ResearchStartResponse(BaseModel):
    job_id: str
    status: str


@router.post("", response_model=ResearchStartResponse)
async def start_research(request: ResearchRequest):
    """Start a deep research job. Returns immediately with job_id."""
    from src.services.research_engine import get_research_engine

    if request.depth not in ("quick", "standard", "deep"):
        raise HTTPException(status_code=400, detail="depth must be quick, standard, or deep")

    engine = get_research_engine()
    job = engine.start_research(request.question, request.depth)
    return ResearchStartResponse(job_id=job.id, status=job.status)


@router.get("/history")
async def research_history(limit: int = 20):
    """List past research sessions."""
    from src.services.research_engine import get_research_engine

    engine = get_research_engine()
    history = await engine.get_history(limit=min(limit, 100))
    return {"jobs": history, "count": len(history)}


@router.get("/{job_id}")
async def get_research_job(job_id: str):
    """Get a research job's status and report."""
    from src.services.research_engine import get_research_engine

    engine = get_research_engine()
    job = await engine.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = {
        "id": job.id,
        "question": job.question,
        "depth": job.depth,
        "status": job.status,
        "iterations": job.iterations,
        "sources_consulted": job.sources_consulted,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "total_time_ms": job.total_time_ms,
        "error_message": job.error_message,
        "progress": job.progress,
    }

    if job.report:
        result["report"] = {
            "answer": job.report.answer,
            "sources": [
                {"ref": s.ref, "type": s.source_type, "title": s.title,
                 "snippet": s.snippet, "url": s.url}
                for s in job.report.sources
            ],
            "sub_questions": job.report.sub_questions,
            "iterations": job.report.iterations,
            "total_sources_consulted": job.report.total_sources_consulted,
        }

    return result


@router.get("/{job_id}/stream")
async def stream_research(job_id: str):
    """SSE stream of research progress events."""
    from src.services.research_engine import get_research_engine

    engine = get_research_engine()
    job = await engine.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        async for event in engine.stream_progress(job_id):
            event_type = event.get("event", "message")
            data = json.dumps(event.get("data", {}))
            yield f"event: {event_type}\ndata: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
