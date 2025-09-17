#!/usr/bin/env python3
"""
Integration module to connect Expert Personalities with Echo Brain API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sys
import os

# Add Echo Brain path
sys.path.append('/opt/tower-echo-brain')

from echo_expert_personalities import (
    ExpertOrchestrator,
    get_model_for_expert,
    map_expert_to_brain_region
)

# Create router for expert endpoints
expert_router = APIRouter(prefix="/api/echo/experts", tags=["experts"])

# Initialize orchestrator
orchestrator = ExpertOrchestrator()

class ExpertQuery(BaseModel):
    """Request model for expert queries"""
    query: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = "patrick"

class ExpertResponse(BaseModel):
    """Response model for expert analysis"""
    expert: str
    emoji: str
    color: str
    analysis: Optional[Dict[str, Any]] = None
    suggestions: Optional[Dict[str, Any]] = None
    design: Optional[Dict[str, Any]] = None
    debugging: Optional[Dict[str, Any]] = None
    model_recommendation: str
    brain_region: str
    formatted_output: str

@expert_router.post("/query", response_model=ExpertResponse)
async def query_expert(request: ExpertQuery):
    """Query the expert system"""
    try:
        # Process with expert orchestrator
        result = orchestrator.process_query(request.query, request.context)

        # Get active expert for additional info
        active_expert = orchestrator.get_active_expert()

        response = ExpertResponse(
            expert=result["expert"],
            emoji=result["emoji"],
            color=result["color"],
            analysis=result.get("analysis"),
            suggestions=result.get("suggestions"),
            design=result.get("design"),
            debugging=result.get("debugging"),
            model_recommendation=get_model_for_expert(active_expert),
            brain_region=map_expert_to_brain_region(active_expert),
            formatted_output=result["formatted_output"]
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@expert_router.get("/list")
async def list_experts():
    """List all available expert personalities"""
    return {
        "experts": orchestrator.list_experts(),
        "total": len(orchestrator.experts)
    }

@expert_router.get("/active")
async def get_active_expert():
    """Get currently active expert"""
    expert = orchestrator.get_active_expert()

    if expert:
        return {
            "name": expert.name,
            "emoji": expert.emoji,
            "color": expert.color,
            "active": expert.active
        }

    return {"message": "No active expert"}

@expert_router.post("/select")
async def select_expert(query: str):
    """Manually select expert for a query"""
    expert = orchestrator.select_expert(query)

    return {
        "selected": expert.name,
        "emoji": expert.emoji,
        "color": expert.color,
        "model": get_model_for_expert(expert)
    }

# Integration function for main Echo Brain app
def integrate_experts_with_echo(app):
    """Add expert endpoints to Echo Brain FastAPI app"""
    app.include_router(expert_router)

    # Also enhance existing /api/echo/query endpoint
    original_query_handler = None

    # Find and wrap the original query handler
    for route in app.routes:
        if route.path == "/api/echo/query" and route.methods == {"POST"}:
            original_query_handler = route.endpoint
            break

    if original_query_handler:
        async def enhanced_query_handler(request):
            """Enhanced query handler with expert selection"""
            # Select expert based on query
            expert = orchestrator.select_expert(request.get("query", ""))

            # Add expert context to request
            if not request.get("context"):
                request["context"] = {}

            request["context"]["expert"] = {
                "name": expert.name,
                "color": expert.color,
                "emoji": expert.emoji,
                "model_override": get_model_for_expert(expert)
            }

            # Call original handler with enhanced context
            response = await original_query_handler(request)

            # Add expert info to response
            if isinstance(response, dict):
                response["expert_personality"] = {
                    "active": expert.name,
                    "visualization": {
                        "color": expert.color,
                        "brain_region": map_expert_to_brain_region(expert)
                    }
                }

            return response

        # Replace the endpoint
        route.endpoint = enhanced_query_handler

    return app

# Standalone test
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    # Create test app
    test_app = FastAPI(title="Echo Expert Test")
    test_app.include_router(expert_router)

    print("Starting Echo Expert Personalities test server...")
    print("Test endpoints:")
    print("  POST /api/echo/experts/query")
    print("  GET  /api/echo/experts/list")
    print("  GET  /api/echo/experts/active")

    uvicorn.run(test_app, host="127.0.0.1", port=8355)