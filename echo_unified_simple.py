#!/usr/bin/env python3
"""
AI Assist Unified API - Simplified version with temporal logic (no DB)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, '/opt/tower-echo-brain')

# Import modules
from temporal_reasoning import EchoTemporalInterface
from echo_self_awareness import EchoSelfAwareness, EchoCapabilityEndpoint

app = FastAPI(title="AI Assist Unified API", version="2.0.0")

# Initialize modules
temporal_interface = EchoTemporalInterface()
self_awareness = EchoSelfAwareness()
capability_endpoint = EchoCapabilityEndpoint()

# Request models
class TemporalQuery(BaseModel):
    type: str = 'validate'
    events: Optional[List[Dict]] = None
    event: Optional[Dict] = None
    timeline_id: Optional[str] = 'main'
    start_event_id: Optional[str] = None
    end_event_id: Optional[str] = None

class CapabilityTest(BaseModel):
    test_type: str = 'self_identification'
    query: Optional[str] = None

# Health endpoints
@app.get("/")
@app.get("/health")
@app.get("/api/echo/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "echo-brain-unified",
        "version": "2.0.0",
        "temporal_logic": True,
        "self_awareness": True,
        "timestamp": datetime.now().isoformat()
    }

# Self-awareness endpoints
@app.post("/api/echo/testing/capabilities")
async def test_capabilities(request: CapabilityTest):
    """Test Echo's capabilities including self-identification"""
    result = await capability_endpoint.handle_capability_request(request.dict())
    return result

@app.get("/api/echo/capabilities")
async def get_capabilities():
    return {
        "capabilities": self_awareness.capabilities,
        "endpoints": len(self_awareness.endpoints),
        "services": self_awareness.services,
        "temporal_enabled": self_awareness.temporal_capable
    }

# Temporal reasoning endpoints
@app.post("/api/echo/temporal/query")
async def temporal_query(query: TemporalQuery):
    """Process temporal logic queries"""
    result = await temporal_interface.process_temporal_query(query.dict())
    return result

@app.post("/api/echo/temporal/validate")
async def validate_temporal_consistency(events: List[Dict]):
    """Validate temporal consistency of events"""
    query = TemporalQuery(type='validate', events=events)
    result = await temporal_interface.process_temporal_query(query.dict())
    return result

@app.get("/api/echo/endpoints")
async def list_endpoints():
    """List all available endpoints"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            routes.append({
                'path': route.path,
                'methods': list(route.methods) if hasattr(route, 'methods') else []
            })
    
    return {
        "endpoints": routes,
        "count": len(routes),
        "categories": {
            "health": ["/", "/health", "/api/echo/health"],
            "self_awareness": ["/api/echo/testing/capabilities", "/api/echo/capabilities"],
            "temporal": ["/api/echo/temporal/query", "/api/echo/temporal/validate"],
            "utility": ["/api/echo/endpoints"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8309)
