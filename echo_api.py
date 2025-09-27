#!/usr/bin/env python3
"""
AI Assist Unified API - Consolidated endpoint management with temporal logic
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
from datetime import datetime

# Import new modules
from temporal_reasoning import EchoTemporalInterface, TemporalEvent, TemporalEventType
from echo_self_awareness import EchoSelfAwareness, EchoCapabilityEndpoint
from echo_temporal_db_integration import TemporalDatabaseManager, setup_temporal_database

# Import existing modules (these should exist)
import sys
sys.path.append('/opt/tower-echo-brain/src')

app = FastAPI(title="AI Assist Unified API", version="2.0.0")

# Global instances
temporal_interface = None
self_awareness = None
capability_endpoint = None
db_manager = None

# Request models
class TemporalQuery(BaseModel):
    type: str = 'validate'
    events: Optional[List[Dict]] = None
    event: Optional[Dict] = None
    timeline_id: Optional[str] = 'main'
    start_event_id: Optional[str] = None
    end_event_id: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None
    request_type: Optional[str] = 'chat'

class CapabilityTest(BaseModel):
    test_type: str = 'self_identification'
    query: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize all modules on startup"""
    global temporal_interface, self_awareness, capability_endpoint, db_manager
    
    # Initialize temporal reasoning
    temporal_interface = EchoTemporalInterface()
    
    # Initialize self-awareness
    self_awareness = EchoSelfAwareness()
    capability_endpoint = EchoCapabilityEndpoint()
    
    # Initialize database
    db_manager = await setup_temporal_database()
    
    print("✅ AI Assist Unified API initialized with temporal logic and self-awareness")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global db_manager
    if db_manager:
        await db_manager.close()

# Health endpoints
@app.get("/")
@app.get("/health")
@app.get("/api/echo/health")
async def health_check():
    """Unified health check endpoint"""
    return {
        "status": "healthy",
        "service": "echo-brain-unified",
        "version": "2.0.0",
        "temporal_logic": temporal_interface is not None,
        "self_awareness": self_awareness is not None,
        "database": db_manager is not None,
        "timestamp": datetime.now().isoformat()
    }

# Self-awareness endpoints
@app.post("/api/echo/testing/capabilities")
async def test_capabilities(request: CapabilityTest):
    """Test Echo's capabilities including self-identification"""
    result = await capability_endpoint.handle_capability_request(request.dict())
    
    # Save to database
    if db_manager and result.get('success'):
        report = result.get('capabilities', {})
        report['report_type'] = request.test_type
        report['query_context'] = request.query or ''
        report['response'] = result.get('response', '')
        report['temporal_capable'] = self_awareness.temporal_capable
        
        report_id = await db_manager.save_self_awareness_report(report)
        result['report_id'] = report_id
    
    return result

@app.get("/api/echo/self-analysis")
async def self_analysis():
    """Generate self-analysis report"""
    report = await self_awareness.generate_self_report(detailed=True)
    
    # Save to database
    if db_manager:
        report['report_type'] = 'self_analysis'
        await db_manager.save_self_awareness_report(report)
    
    return report

@app.get("/api/echo/capabilities")
async def get_capabilities():
    """Get current capabilities"""
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
    
    # Save events to database if adding
    if db_manager and query.type == 'add_event' and result.get('success'):
        await db_manager.save_temporal_event(query.event)
    
    # Save paradoxes if detected
    if db_manager and query.type == 'detect_paradox':
        for paradox in result.get('paradoxes', []):
            await db_manager.save_paradox(paradox)
    
    # Save causal chains if verified
    if db_manager and query.type == 'verify_causality' and result.get('causal_chain_exists'):
        chain = {
            'start_event_id': query.start_event_id,
            'end_event_id': query.end_event_id,
            'path': result.get('path', [])
        }
        await db_manager.save_causal_chain(chain)
    
    return result

@app.post("/api/echo/temporal/validate")
async def validate_temporal_consistency(events: List[Dict]):
    """Validate temporal consistency of events"""
    query = TemporalQuery(type='validate', events=events)
    result = await temporal_interface.process_temporal_query(query.dict())
    
    # Update timeline consistency in database
    if db_manager:
        timeline_id = events[0].get('timeline_id', 'main') if events else 'main'
        score = result.get('consistency_score', 0)
        total = result.get('total_count', 0)
        valid = result.get('valid_count', 0)
        
        await db_manager.update_timeline_consistency(timeline_id, score, total, valid)
    
    return result

@app.get("/api/echo/temporal/paradoxes")
async def get_paradoxes():
    """Get detected paradoxes"""
    if db_manager:
        paradoxes = await db_manager.get_unresolved_paradoxes()
        return {"paradoxes": paradoxes, "count": len(paradoxes)}
    
    return {"error": "Database not initialized"}

@app.get("/api/echo/temporal/analysis/{timeline_id}")
async def analyze_timeline(timeline_id: str = 'main'):
    """Analyze temporal patterns in timeline"""
    if db_manager:
        analysis = await db_manager.analyze_temporal_patterns(timeline_id)
        return analysis
    
    return {"error": "Database not initialized"}

# Core AI Assist endpoints (simplified from original)
@app.post("/api/echo/brain")
@app.post("/api/echo/chat")
async def process_message(message: ChatMessage):
    """Main AI Assist processing endpoint"""
    # Check if this is a temporal logic query
    if 'temporal' in message.message.lower() or 'timeline' in message.message.lower():
        # Route to temporal reasoning
        return {
            "response": "I can now handle temporal reasoning. Use /api/echo/temporal/query for specific temporal logic operations.",
            "temporal_capable": True
        }
    
    # Check if this is a self-identification query
    if 'capabilities' in message.message.lower() or 'what can you do' in message.message.lower():
        test = CapabilityTest(test_type='self_identification', query=message.message)
        return await test_capabilities(test)
    
    # Default response (integrate with actual Echo brain logic here)
    return {
        "response": f"Processing: {message.message}",
        "context": message.context,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/echo/stats")
async def get_stats():
    """Get AI Assist statistics"""
    stats = {
        "temporal_events": 0,
        "paradoxes_detected": 0,
        "capabilities_tracked": 0,
        "self_reports_generated": 0
    }
    
    if db_manager:
        # Get counts from database
        async with db_manager.pool.acquire() as conn:
            stats["temporal_events"] = await conn.fetchval("SELECT COUNT(*) FROM temporal_events")
            stats["paradoxes_detected"] = await conn.fetchval("SELECT COUNT(*) FROM temporal_paradoxes")
            stats["capabilities_tracked"] = await conn.fetchval("SELECT COUNT(*) FROM capability_evolution")
            stats["self_reports_generated"] = await conn.fetchval("SELECT COUNT(*) FROM self_awareness_reports")
    
    stats["timestamp"] = datetime.now().isoformat()
    return stats

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
            "self_awareness": ["/api/echo/testing/capabilities", "/api/echo/self-analysis", "/api/echo/capabilities"],
            "temporal": ["/api/echo/temporal/query", "/api/echo/temporal/validate", "/api/echo/temporal/paradoxes"],
            "core": ["/api/echo/brain", "/api/echo/chat", "/api/echo/stats"]
        }
    }

# Admin endpoints
@app.post("/api/echo/admin/cleanup")
async def cleanup_old_data(days: int = 30):
    """Clean up old temporal data"""
    if db_manager:
        result = await db_manager.cleanup_old_data(days)
        return result
    
    return {"error": "Database not initialized"}

@app.get("/api/echo/admin/capability-history")
async def capability_history(capability: Optional[str] = None):
    """Get capability evolution history"""
    if db_manager:
        history = await db_manager.get_capability_history(capability)
        return {"history": history, "count": len(history)}
    
    return {"error": "Database not initialized"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8309)
