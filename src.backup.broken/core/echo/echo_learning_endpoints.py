#!/usr/bin/env python3
"""
Echo Learning Endpoints - FastAPI Integration
============================================

FastAPI endpoints for Echo's comprehensive learning system.
These endpoints can be integrated into the main Echo FastAPI service
to provide learning capabilities via REST API.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Import learning integration
from src.core.echo.echo_learning_integration import (
    get_learning_integration, initialize_echo_learning,
    make_intelligent_decision, record_decision_outcome,
    get_learning_system_status, EchoDecisionResponse, LearningSystemStatus
)

# Import individual systems for direct access
from src.core.echo.echo_self_diagnosis import get_health_status
from bin.echo_board_manager import get_board_status
from src.core.echo.echo_outcome_tracker import get_performance_summary

logger = logging.getLogger(__name__)

# Create router
learning_router = APIRouter(prefix="/api/learning", tags=["learning"])

# Pydantic models for API
class DecisionRequest(BaseModel):
    request: str
    context: Dict[str, Any] = {}
    priority: str = "medium"
    user_context: Optional[Dict[str, Any]] = None

class DecisionOutcomeRecord(BaseModel):
    request_id: str
    outcome: str
    metrics: Dict[str, float] = {}
    user_feedback: Optional[str] = None

class LearningStatusResponse(BaseModel):
    overall_health: str
    active_processes: int
    system_components: Dict[str, str]
    recent_insights: List[str]
    recommendations: List[str]

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, Any]

# Learning System Endpoints

@learning_router.post("/initialize")
async def initialize_learning_systems():
    """Initialize all Echo learning systems"""
    try:
        success = await initialize_echo_learning()
        return {
            "success": success,
            "message": "Learning systems initialized successfully" if success else "Initialization failed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Learning system initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {e}")

@learning_router.post("/decision", response_model=Dict[str, Any])
async def make_decision(request: DecisionRequest):
    """Make an intelligent decision using Echo's learning systems"""
    try:
        response = await make_intelligent_decision(
            request=request.request,
            context=request.context,
            priority=request.priority,
            user_context=request.user_context
        )

        return {
            "request_id": response.request_id,
            "decision": response.decision_made,
            "confidence": response.confidence,
            "decision_path": response.decision_path,
            "board_consultation_used": response.board_consultation_used,
            "task_workflow_created": response.task_workflow_created,
            "learning_insights": response.learning_insights,
            "monitoring_recommendations": response.monitoring_recommendations,
            "expected_outcomes": response.expected_outcomes,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Decision making failed: {e}")
        raise HTTPException(status_code=500, detail=f"Decision failed: {e}")

@learning_router.post("/outcome")
async def record_outcome(outcome_record: DecisionOutcomeRecord):
    """Record the outcome of a decision for learning"""
    try:
        await record_decision_outcome(
            request_id=outcome_record.request_id,
            outcome=outcome_record.outcome,
            metrics=outcome_record.metrics,
            user_feedback=outcome_record.user_feedback
        )

        return {
            "success": True,
            "message": f"Outcome recorded for decision {outcome_record.request_id}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Outcome recording failed: {e}")
        raise HTTPException(status_code=500, detail=f"Outcome recording failed: {e}")

@learning_router.get("/status", response_model=Dict[str, Any])
async def get_status():
    """Get comprehensive learning system status"""
    try:
        status = await get_learning_system_status()

        return {
            "overall_health": status.overall_health,
            "active_learning_processes": status.active_learning_processes,
            "components": {
                "learning_core": status.learning_core_status,
                "self_diagnosis": status.self_diagnosis_status,
                "board_manager": status.board_manager_status,
                "task_decomposer": status.task_decomposer_status,
                "outcome_tracker": status.outcome_tracker_status
            },
            "recent_insights": status.recent_insights,
            "system_recommendations": status.system_recommendations,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}")

@learning_router.get("/health")
async def health_check():
    """Quick health check for learning systems"""
    try:
        # Get health from various systems
        health_status = await get_health_status()
        board_status = await get_board_status()
        performance_summary = await get_performance_summary()

        components = {
            "self_diagnosis": health_status.get("system_health", "unknown") if health_status else "unavailable",
            "board_manager": board_status.get("system_health", "unknown") if board_status else "unavailable",
            "outcome_tracker": "operational" if performance_summary else "unavailable"
        }

        # Determine overall status
        if all(status in ["operational", "good", "excellent"] for status in components.values()):
            overall_status = "healthy"
        elif "unavailable" in components.values():
            overall_status = "degraded"
        else:
            overall_status = "operational"

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": components
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "components": {"error": str(e)}
        }

@learning_router.get("/insights")
async def get_insights(hours: int = 24):
    """Get recent learning insights"""
    try:
        integration = get_learning_integration()
        insights = await integration.get_recent_learning_insights(hours=hours)

        return {
            "insights": [
                {
                    "source": insight["source"],
                    "insight": insight["insight"],
                    "timestamp": insight["timestamp"].isoformat()
                }
                for insight in insights
            ],
            "total_insights": len(insights),
            "period_hours": hours,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Getting insights failed: {e}")
        raise HTTPException(status_code=500, detail=f"Getting insights failed: {e}")

@learning_router.post("/optimize")
async def optimize_systems(background_tasks: BackgroundTasks):
    """Optimize learning systems performance"""
    try:
        integration = get_learning_integration()

        # Run optimization in background
        background_tasks.add_task(integration.optimize_learning_systems)

        return {
            "message": "Learning system optimization started",
            "status": "running",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"System optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

# Individual System Endpoints

@learning_router.get("/diagnosis/status")
async def get_diagnosis_status():
    """Get self-diagnosis system status"""
    try:
        status = await get_health_status()
        return {
            "status": status if status else {"message": "Diagnosis system unavailable"},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnosis status failed: {e}")

@learning_router.get("/board/status")
async def get_board_system_status():
    """Get board manager status"""
    try:
        status = await get_board_status()
        return {
            "status": status if status else {"message": "Board system unavailable"},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Board status failed: {e}")

@learning_router.get("/performance/summary")
async def get_performance_system_summary():
    """Get outcome tracker performance summary"""
    try:
        summary = await get_performance_summary()
        return {
            "summary": summary if summary else {"message": "Performance system unavailable"},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance summary failed: {e}")

# Utility endpoints

@learning_router.get("/test")
async def test_learning_system():
    """Test the learning system with a simple decision"""
    try:
        response = await make_intelligent_decision(
            request="Test the learning system functionality",
            context={"test": True, "complexity": 0.3},
            priority="low"
        )

        # Record a successful test outcome
        await record_decision_outcome(
            request_id=response.request_id,
            outcome="Test completed successfully",
            metrics={"test_success": 1.0, "response_time": 0.5},
            user_feedback="Learning system test passed"
        )

        return {
            "test_result": "success",
            "decision_id": response.request_id,
            "decision": response.decision_made,
            "confidence": response.confidence,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Learning system test failed: {e}")
        return {
            "test_result": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@learning_router.get("/")
async def learning_info():
    """Get information about the learning system"""
    return {
        "name": "Echo Learning System",
        "version": "1.0.0",
        "description": "Comprehensive learning and decision-making system for Echo",
        "components": [
            "Learning Core - Pattern recognition and decision learning",
            "Self Diagnosis - Health monitoring and auto-repair",
            "Board Manager - Director management and optimization",
            "Task Decomposer - Complex task breakdown and execution",
            "Outcome Tracker - Decision impact analysis and correlation"
        ],
        "endpoints": {
            "POST /initialize": "Initialize learning systems",
            "POST /decision": "Make intelligent decision",
            "POST /outcome": "Record decision outcome",
            "GET /status": "Get system status",
            "GET /health": "Quick health check",
            "GET /insights": "Get learning insights",
            "POST /optimize": "Optimize systems",
            "GET /test": "Test functionality"
        },
        "timestamp": datetime.now().isoformat()
    }

# Export router for integration with main Echo system
def get_learning_router() -> APIRouter:
    """Get the learning system router for integration"""
    return learning_router

# Example integration with main Echo system:
"""
To integrate with the main Echo FastAPI application, add this to echo.py:

from src.core.echo.echo_learning_endpoints import get_learning_router

# In your FastAPI app setup:
app.include_router(get_learning_router())

This will add all learning endpoints under /api/learning/
"""