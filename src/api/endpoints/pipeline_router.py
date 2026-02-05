"""
Pipeline API endpoint.
This REPLACES the broken /api/ask and /api/echo/query endpoints.
"""
from fastapi import APIRouter, Query
import sys
import os
sys.path.insert(0, '/opt/tower-echo-brain/src')

from pipeline.orchestrator import EchoBrainPipeline

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])
pipeline: EchoBrainPipeline = None


async def get_pipeline() -> EchoBrainPipeline:
    global pipeline
    if pipeline is None:
        pipeline = EchoBrainPipeline()
        await pipeline.initialize()
    return pipeline


@router.post("/query")
async def query(body: dict):
    """Process a query through the full Context → Reasoning → Narrative pipeline."""
    p = await get_pipeline()
    query_text = body.get("query", body.get("q", ""))
    debug = body.get("debug", False)

    if not query_text:
        return {"error": "No query provided", "usage": "POST with {\"query\": \"your question\"}"}

    result = await p.process(query_text, debug=debug)
    return result.model_dump()


@router.get("/query")
async def query_get(q: str = Query(...), debug: bool = False):
    """GET version for quick testing."""
    p = await get_pipeline()
    result = await p.process(q, debug=debug)
    return result.model_dump()


@router.get("/health")
async def pipeline_health():
    """Check pipeline component status."""
    p = await get_pipeline()
    return {
        "status": "operational",
        "layers": {
            "context": p.context.pg_pool is not None and not p.context.pg_pool._closed,
            "reasoning": p.reasoning.http_client is not None,
            "narrative": True,
        },
    }