#!/usr/bin/env python3
"""
API endpoints for circuit breaker status monitoring
Provides REST endpoints for resilience monitoring and control
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ..resilience.service_breakers import get_service_breaker_manager
from ..resilience.monitoring import get_circuit_breaker_monitor
from ..resilience.fallback_handlers import get_fallback_handler

logger = logging.getLogger(__name__)


# Pydantic models for request/response
class CircuitBreakerTestRequest(BaseModel):
    service_name: str
    operation: str
    simulate_failure: bool = False
    failure_count: int = 1


class CircuitBreakerResetRequest(BaseModel):
    service_name: str
    force: bool = False


class AlertConfigRequest(BaseModel):
    enabled: bool = True
    alert_threshold: int = 3
    email_notifications: bool = False
    webhook_url: Optional[str] = None


# Create router
router = APIRouter(prefix="/api/resilience", tags=["resilience"])

# Get global instances
service_manager = get_service_breaker_manager()
monitor = get_circuit_breaker_monitor(service_manager)
fallback_handler = get_fallback_handler()


@router.get("/health", summary="Get overall system health")
async def get_system_health():
    """Get comprehensive system health dashboard"""
    try:
        dashboard = monitor.get_system_health_dashboard()
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": dashboard
        }
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {e}")


@router.get("/services", summary="Get all service statuses")
async def get_service_statuses():
    """Get status of all monitored services"""
    try:
        service_status = service_manager.get_service_status()
        health_summary = service_manager.get_health_summary()

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": health_summary,
            "services": service_status
        }
    except Exception as e:
        logger.error(f"Error getting service statuses: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get service statuses: {e}")


@router.get("/services/{service_name}", summary="Get specific service status")
async def get_service_status(service_name: str = Path(..., description="Name of the service")):
    """Get detailed status for specific service"""
    try:
        breaker = service_manager.get_breaker(service_name)
        if not breaker:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

        metrics = breaker.get_metrics()
        health = monitor.get_service_health_metrics(service_name)

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "service": service_name,
            "circuit_breaker": metrics,
            "health_metrics": health.to_dict() if health else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting service status for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get service status: {e}")


@router.get("/circuit-breakers", summary="Get all circuit breaker states")
async def get_circuit_breakers():
    """Get detailed information about all circuit breakers"""
    try:
        breakers = service_manager.get_all_breakers()
        breaker_info = {}

        for name, breaker in breakers.items():
            breaker_info[name] = breaker.get_metrics()

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "circuit_breakers": breaker_info
        }
    except Exception as e:
        logger.error(f"Error getting circuit breakers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get circuit breakers: {e}")


@router.post("/circuit-breakers/{service_name}/reset", summary="Reset circuit breaker")
async def reset_circuit_breaker(
    service_name: str = Path(..., description="Name of the service"),
    force: bool = Query(False, description="Force reset even if service is unhealthy")
):
    """Reset circuit breaker for specific service"""
    try:
        breaker = service_manager.get_breaker(service_name)
        if not breaker:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

        # Check service health before reset (unless forced)
        if not force:
            # You could add health checks here
            pass

        # Reset the circuit breaker
        breaker.reset()

        return {
            "success": True,
            "message": f"Circuit breaker for '{service_name}' has been reset",
            "timestamp": datetime.utcnow().isoformat(),
            "forced": force
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting circuit breaker for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset circuit breaker: {e}")


@router.post("/circuit-breakers/{service_name}/force-open", summary="Force circuit breaker open")
async def force_circuit_breaker_open(service_name: str = Path(..., description="Name of the service")):
    """Force circuit breaker to open state (for testing/maintenance)"""
    try:
        success = service_manager.force_open_breaker(service_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

        return {
            "success": True,
            "message": f"Circuit breaker for '{service_name}' has been forced open",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forcing circuit breaker open for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to force circuit breaker open: {e}")


@router.post("/circuit-breakers/reset-all", summary="Reset all circuit breakers")
async def reset_all_circuit_breakers():
    """Reset all circuit breakers to closed state"""
    try:
        service_manager.reset_all_breakers()

        return {
            "success": True,
            "message": "All circuit breakers have been reset",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error resetting all circuit breakers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset all circuit breakers: {e}")


@router.post("/test/{service_name}", summary="Test service with circuit breaker")
async def test_service_with_circuit_breaker(
    service_name: str = Path(..., description="Name of the service"),
    simulate_failure: bool = Query(False, description="Simulate service failure"),
    failure_count: int = Query(1, description="Number of failures to simulate")
):
    """Test service calls through circuit breaker"""
    try:
        breaker = service_manager.get_breaker(service_name)
        if not breaker:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

        test_results = []

        for i in range(failure_count if simulate_failure else 1):
            # Simulate different operations based on service
            if service_name == "comfyui":
                if simulate_failure:
                    # Simulate ComfyUI failure
                    async def failing_comfyui_call():
                        await asyncio.sleep(0.1)
                        raise Exception("Simulated ComfyUI failure")

                    try:
                        result = await breaker.call(failing_comfyui_call)
                        test_results.append({"attempt": i + 1, "success": True, "result": result})
                    except Exception as e:
                        test_results.append({"attempt": i + 1, "success": False, "error": str(e)})
                else:
                    # Simulate successful ComfyUI call
                    async def successful_comfyui_call():
                        await asyncio.sleep(0.1)
                        return {"status": "completed", "output": "test_image.png"}

                    try:
                        result = await breaker.call(successful_comfyui_call)
                        test_results.append({"attempt": i + 1, "success": True, "result": result})
                    except Exception as e:
                        test_results.append({"attempt": i + 1, "success": False, "error": str(e)})

            elif service_name == "qdrant":
                if simulate_failure:
                    async def failing_qdrant_call():
                        await asyncio.sleep(0.05)
                        raise ConnectionError("Simulated Qdrant connection failure")

                    try:
                        result = await breaker.call(failing_qdrant_call)
                        test_results.append({"attempt": i + 1, "success": True, "result": result})
                    except Exception as e:
                        test_results.append({"attempt": i + 1, "success": False, "error": str(e)})
                else:
                    async def successful_qdrant_call():
                        await asyncio.sleep(0.05)
                        return {"results": [], "total_count": 0}

                    try:
                        result = await breaker.call(successful_qdrant_call)
                        test_results.append({"attempt": i + 1, "success": True, "result": result})
                    except Exception as e:
                        test_results.append({"attempt": i + 1, "success": False, "error": str(e)})

            elif service_name == "ollama":
                if simulate_failure:
                    async def failing_ollama_call():
                        await asyncio.sleep(0.2)
                        raise TimeoutError("Simulated Ollama timeout")

                    try:
                        result = await breaker.call(failing_ollama_call)
                        test_results.append({"attempt": i + 1, "success": True, "result": result})
                    except Exception as e:
                        test_results.append({"attempt": i + 1, "success": False, "error": str(e)})
                else:
                    async def successful_ollama_call():
                        await asyncio.sleep(0.2)
                        return {"response": "Test response from model", "model": "test"}

                    try:
                        result = await breaker.call(successful_ollama_call)
                        test_results.append({"attempt": i + 1, "success": True, "result": result})
                    except Exception as e:
                        test_results.append({"attempt": i + 1, "success": False, "error": str(e)})

            elif service_name == "postgresql":
                if simulate_failure:
                    async def failing_db_call():
                        await asyncio.sleep(0.05)
                        raise ConnectionError("Simulated database connection failure")

                    try:
                        result = await breaker.call(failing_db_call)
                        test_results.append({"attempt": i + 1, "success": True, "result": result})
                    except Exception as e:
                        test_results.append({"attempt": i + 1, "success": False, "error": str(e)})
                else:
                    async def successful_db_call():
                        await asyncio.sleep(0.05)
                        return {"rows": [], "count": 0}

                    try:
                        result = await breaker.call(successful_db_call)
                        test_results.append({"attempt": i + 1, "success": True, "result": result})
                    except Exception as e:
                        test_results.append({"attempt": i + 1, "success": False, "error": str(e)})

        # Get updated circuit breaker status
        updated_metrics = breaker.get_metrics()

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "service": service_name,
            "test_results": test_results,
            "circuit_breaker_state": updated_metrics["state"],
            "total_attempts": len(test_results),
            "successful_attempts": sum(1 for r in test_results if r["success"]),
            "failed_attempts": sum(1 for r in test_results if not r["success"])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing service {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test service: {e}")


@router.get("/fallback-cache", summary="Get fallback cache status")
async def get_fallback_cache():
    """Get status of fallback cache operations"""
    try:
        cached_ops = fallback_handler.get_cached_operations()

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "cached_operations": len(cached_ops),
            "operations_by_service": {},
            "cache_entries": [
                {
                    "cache_key": key,
                    "operation": op.get("operation", "unknown"),
                    "created_at": op.get("created_at").isoformat() if op.get("created_at") else None,
                    "service": op.get("operation", "").split("_")[0] if op.get("operation") else "unknown"
                }
                for key, op in cached_ops.items()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting fallback cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get fallback cache: {e}")


@router.post("/fallback-cache/clear", summary="Clear fallback cache")
async def clear_fallback_cache(service: Optional[str] = Query(None, description="Service to clear cache for")):
    """Clear fallback cache for specific service or all services"""
    try:
        fallback_handler.clear_cache(service)

        return {
            "success": True,
            "message": f"Fallback cache cleared{' for ' + service if service else ''}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing fallback cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear fallback cache: {e}")


@router.get("/events", summary="Get recent circuit breaker events")
async def get_recent_events(
    service: Optional[str] = Query(None, description="Filter by service name"),
    limit: int = Query(50, description="Maximum number of events to return")
):
    """Get recent circuit breaker state change events"""
    try:
        # Get events from monitor
        all_events = list(monitor.events)

        # Filter by service if specified
        if service:
            all_events = [e for e in all_events if e.service_name == service]

        # Limit results
        recent_events = all_events[-limit:]

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "events": [e.to_dict() for e in recent_events],
            "total_events": len(recent_events),
            "filtered_by_service": service
        }
    except Exception as e:
        logger.error(f"Error getting recent events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent events: {e}")


@router.get("/metrics/prometheus", summary="Get metrics in Prometheus format")
async def get_prometheus_metrics():
    """Get circuit breaker metrics in Prometheus format"""
    try:
        metrics_text = monitor.export_metrics_prometheus()
        return PlainTextResponse(
            content=metrics_text,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate Prometheus metrics: {e}")


@router.post("/monitoring/start", summary="Start circuit breaker monitoring")
async def start_monitoring(
    background_tasks: BackgroundTasks,
    interval: float = Query(30.0, description="Monitoring interval in seconds")
):
    """Start continuous circuit breaker monitoring"""
    try:
        if monitor.monitoring:
            return {
                "success": True,
                "message": "Monitoring is already running",
                "timestamp": datetime.utcnow().isoformat()
            }

        background_tasks.add_task(monitor.start_monitoring, interval)

        return {
            "success": True,
            "message": f"Circuit breaker monitoring started (interval: {interval}s)",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {e}")


@router.post("/monitoring/stop", summary="Stop circuit breaker monitoring")
async def stop_monitoring():
    """Stop continuous circuit breaker monitoring"""
    try:
        await monitor.stop_monitoring()

        return {
            "success": True,
            "message": "Circuit breaker monitoring stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {e}")


@router.get("/monitoring/status", summary="Get monitoring status")
async def get_monitoring_status():
    """Get current monitoring status"""
    try:
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_active": monitor.monitoring,
            "alert_callbacks_count": len(monitor.alert_callbacks),
            "events_stored": len(monitor.events),
            "services_tracked": len(monitor.metrics_history)
        }
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring status: {e}")


# Setup default alert callback for logging
def default_alert_callback(alert: Dict[str, Any]):
    """Default alert callback that logs alerts"""
    level = alert.get("level", "info")
    service = alert.get("service", "unknown")
    message = alert.get("message", "")

    if level == "critical":
        logger.error(f"🚨 CRITICAL ALERT - {service}: {message}")
    elif level == "warning":
        logger.warning(f"⚠️ WARNING - {service}: {message}")
    else:
        logger.info(f"ℹ️ INFO - {service}: {message}")


# Add default alert callback
monitor.add_alert_callback(default_alert_callback)