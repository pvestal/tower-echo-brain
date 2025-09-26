"""
Resilient orchestration routes for Echo Brain
These endpoints use the ResilientOrchestrator with retry logic and cloud escalation
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from src.utils.helpers import tower_orchestrator as orchestrator, TowerOrchestrator
from src.orchestrators.resilient_orchestrator import TaskPriority, ComputeLocation

logger = logging.getLogger(__name__)

# Create router for resilient endpoints
resilient_router = APIRouter(prefix="/api/echo/resilient", tags=["resilient"])


@resilient_router.get("/health/comprehensive")
async def comprehensive_health_check():
    """Get comprehensive health with metrics and circuit breaker status"""
    try:
        health = await orchestrator.health_check()
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@resilient_router.get("/metrics")
async def get_system_metrics():
    """Get current system metrics"""
    try:
        metrics = orchestrator.get_current_metrics()
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@resilient_router.post("/generate/image")
async def generate_image_resilient(request: Dict[str, Any]):
    """Generate image with resilience and retry logic"""
    try:
        prompt = request.get("prompt", "cyberpunk anime scene")
        style = request.get("style", "anime")
        priority_str = request.get("priority", "medium").upper()

        # Convert priority string to enum
        priority = TaskPriority[priority_str] if priority_str in TaskPriority.__members__ else TaskPriority.MEDIUM

        result = await orchestrator.generate_image(
            prompt=prompt,
            style=style,
            priority=priority
        )

        return {
            "endpoint": "/api/echo/resilient/generate/image",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        logger.error(f"Resilient image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@resilient_router.post("/generate/voice")
async def generate_voice_resilient(request: Dict[str, Any]):
    """Generate voice with resilience and retry logic"""
    try:
        text = request.get("text", "Hello from Echo Brain")
        character = request.get("character", "echo_default")
        priority_str = request.get("priority", "medium").upper()

        priority = TaskPriority[priority_str] if priority_str in TaskPriority.__members__ else TaskPriority.MEDIUM

        result = await orchestrator.generate_voice(
            text=text,
            character=character,
            priority=priority
        )

        return {
            "endpoint": "/api/echo/resilient/generate/voice",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        logger.error(f"Resilient voice generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@resilient_router.post("/orchestrate")
async def orchestrate_complex_resilient(request: Dict[str, Any]):
    """Orchestrate complex task with intelligent routing and cloud escalation"""
    try:
        task_type = request.get("task_type", "unknown")
        requirements = request.get("requirements", {})
        priority_str = request.get("priority", "medium").upper()

        priority = TaskPriority[priority_str] if priority_str in TaskPriority.__members__ else TaskPriority.MEDIUM

        result = await orchestrator.orchestrate_complex_task(
            task_type=task_type,
            requirements=requirements,
            priority=priority
        )

        return {
            "endpoint": "/api/echo/resilient/orchestrate",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        logger.error(f"Resilient orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@resilient_router.post("/test/burst")
async def test_burst_handling(request: Dict[str, Any]):
    """Test burst load handling capability"""
    try:
        num_tasks = request.get("num_tasks", 10)
        task_type = request.get("task_type", "image")

        # Queue multiple tasks
        results = []
        for i in range(num_tasks):
            if task_type == "image":
                result = await orchestrator.generate_image(
                    prompt=f"Test scene {i}",
                    style="anime",
                    priority=TaskPriority.LOW
                )
            else:
                result = await orchestrator.generate_voice(
                    text=f"Test message {i}",
                    character="echo_default",
                    priority=TaskPriority.LOW
                )

            results.append({
                "task": i,
                "success": result.get("success", False),
                "location": result.get("compute_location", "unknown")
            })

        # Analyze distribution
        successful = sum(1 for r in results if r["success"])
        local = sum(1 for r in results if r["location"] == "local")
        cloud = sum(1 for r in results if r["location"] in ["firebase", "gcp"])

        return {
            "endpoint": "/api/echo/resilient/test/burst",
            "timestamp": datetime.now().isoformat(),
            "num_tasks": num_tasks,
            "success_rate": successful / num_tasks,
            "local_processing": local,
            "cloud_escalation": cloud,
            "results": results
        }
    except Exception as e:
        logger.error(f"Burst test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@resilient_router.get("/circuit/status")
async def get_circuit_breaker_status():
    """Get status of all circuit breakers"""
    try:
        status = {}
        for service, breaker in orchestrator.circuit_breakers.items():
            status[service] = {
                "state": breaker.state,
                "failures": breaker.failures,
                "is_open": breaker.is_open(),
                "is_half_open": breaker.is_half_open()
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "circuit_breakers": status
        }
    except Exception as e:
        logger.error(f"Failed to get circuit status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@resilient_router.post("/circuit/reset")
async def reset_circuit_breaker(request: Dict[str, Any]):
    """Reset a circuit breaker for a specific service"""
    try:
        service = request.get("service")

        if service not in orchestrator.circuit_breakers:
            raise HTTPException(status_code=404, detail=f"Service {service} not found")

        breaker = orchestrator.circuit_breakers[service]
        breaker.failures = 0
        breaker.state = "closed"

        return {
            "message": f"Circuit breaker for {service} reset",
            "service": service,
            "new_state": "closed"
        }
    except Exception as e:
        logger.error(f"Failed to reset circuit: {e}")
        raise HTTPException(status_code=500, detail=str(e))