"""
Echo Control Endpoints - Direct execution and monitoring
Provides real-time visibility into what Echo is actually doing
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Create control router
control_router = APIRouter(prefix="/api/echo/control", tags=["control"])

class ServiceEndpoint(BaseModel):
    """Service endpoint definition"""
    name: str
    url: str
    port: int
    method: str = "POST"
    test_endpoint: str = "/health"
    execute_endpoint: str = "/"

class ExecutionRequest(BaseModel):
    """Direct execution request"""
    service: str
    endpoint: str
    method: str = "POST"
    payload: Optional[Dict] = None
    timeout: int = 30
    force: bool = False

class EndpointTestRequest(BaseModel):
    """Endpoint testing request"""
    endpoints: List[Dict[str, str]]
    parallel: bool = True
    timeout: int = 5

# Active executions tracking
active_executions = {}
execution_history = []
service_status = {}

# Service registry
SERVICES = {
    "comfyui": ServiceEndpoint(
        name="ComfyUI",
        url="http://127.0.0.1:8188",
        port=8188,
        test_endpoint="/queue",
        execute_endpoint="/prompt"
    ),
    "anime": ServiceEndpoint(
        name="Anime Production",
        url="http://127.0.0.1:8328",
        port=8328,
        test_endpoint="/api/health",
        execute_endpoint="/api/projects"
    ),
    "voice": ServiceEndpoint(
        name="Voice Service",
        url="http://127.0.0.1:8312",
        port=8312,
        test_endpoint="/health",
        execute_endpoint="/generate"
    ),
    "music": ServiceEndpoint(
        name="Music Service",
        url="http://127.0.0.1:8308",
        port=8308,
        test_endpoint="/health",
        execute_endpoint="/generate"
    ),
    "kb": ServiceEndpoint(
        name="Knowledge Base",
        url="http://127.0.0.1:8307",
        port=8307,
        test_endpoint="/api/health",
        execute_endpoint="/api/articles"
    ),
    "ollama": ServiceEndpoint(
        name="Ollama",
        url="http://127.0.0.1:11434",
        port=11434,
        test_endpoint="/api/tags",
        execute_endpoint="/api/generate"
    )
}

@control_router.get("/services")
async def get_services():
    """Get all registered services with current status"""
    result = {}
    for name, service in SERVICES.items():
        result[name] = {
            "name": service.name,
            "url": service.url,
            "port": service.port,
            "status": service_status.get(name, "unknown"),
            "last_check": service_status.get(f"{name}_checked", "never")
        }
    return result

@control_router.post("/test/endpoint")
async def test_endpoint(url: str, method: str = "GET", timeout: int = 5):
    """Test a specific endpoint directly"""
    start_time = time.time()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                elapsed = time.time() - start_time
                body = await response.text()

                # Try to parse as JSON
                try:
                    data = json.loads(body)
                except:
                    data = {"raw": body[:500]}  # Limit raw response

                return {
                    "success": True,
                    "url": url,
                    "status": response.status,
                    "response_time": elapsed,
                    "headers": dict(response.headers),
                    "data": data
                }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "url": url,
            "error": "timeout",
            "response_time": time.time() - start_time
        }
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": str(e),
            "response_time": time.time() - start_time
        }

@control_router.post("/test/all")
async def test_all_services():
    """Test all service endpoints in parallel"""
    tasks = []

    async def check_service(name: str, service: ServiceEndpoint):
        url = f"{service.url}{service.test_endpoint}"
        result = await test_endpoint(url)

        # Update service status
        service_status[name] = "healthy" if result["success"] else "error"
        service_status[f"{name}_checked"] = datetime.now().isoformat()

        return {
            "service": name,
            "endpoint": url,
            **result
        }

    async with asyncio.TaskGroup() as tg:
        for name, service in SERVICES.items():
            tasks.append(tg.create_task(check_service(name, service)))

    results = [await task for task in tasks]

    return {
        "timestamp": datetime.now().isoformat(),
        "services_tested": len(results),
        "healthy": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }

@control_router.post("/execute/direct")
async def execute_directly(request: ExecutionRequest):
    """Execute a command directly on a service, bypassing Echo's planning"""

    if request.service not in SERVICES:
        raise HTTPException(status_code=400, detail=f"Unknown service: {request.service}")

    service = SERVICES[request.service]
    url = f"{service.url}{request.endpoint}"
    execution_id = f"exec_{int(time.time()*1000)}"

    # Track execution
    active_executions[execution_id] = {
        "id": execution_id,
        "service": request.service,
        "endpoint": request.endpoint,
        "started": datetime.now().isoformat(),
        "status": "executing",
        "payload": request.payload
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                request.method,
                url,
                json=request.payload,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                data = await response.text()

                try:
                    result = json.loads(data)
                except:
                    result = {"raw": data[:1000]}

                # Update execution
                active_executions[execution_id]["status"] = "completed"
                active_executions[execution_id]["result"] = result
                active_executions[execution_id]["completed"] = datetime.now().isoformat()

                # Add to history
                execution_history.append({
                    **active_executions[execution_id],
                    "response_status": response.status
                })

                # Clean up
                del active_executions[execution_id]

                return {
                    "execution_id": execution_id,
                    "success": response.status == 200,
                    "status": response.status,
                    "service": request.service,
                    "endpoint": request.endpoint,
                    "result": result
                }

    except Exception as e:
        # Update execution
        active_executions[execution_id]["status"] = "failed"
        active_executions[execution_id]["error"] = str(e)
        active_executions[execution_id]["completed"] = datetime.now().isoformat()

        # Add to history
        execution_history.append(active_executions[execution_id])

        # Clean up
        del active_executions[execution_id]

        return {
            "execution_id": execution_id,
            "success": False,
            "service": request.service,
            "endpoint": request.endpoint,
            "error": str(e)
        }

@control_router.get("/executions/active")
async def get_active_executions():
    """Get all currently active executions"""
    return {
        "count": len(active_executions),
        "executions": list(active_executions.values())
    }

@control_router.get("/executions/history")
async def get_execution_history(limit: int = 100):
    """Get execution history"""
    return {
        "count": len(execution_history),
        "history": execution_history[-limit:]
    }

@control_router.post("/verify/echo-claim")
async def verify_echo_claim(query: str, echo_response: str):
    """Verify what Echo claimed to do vs what actually happened"""

    verification = {
        "query": query,
        "echo_claim": echo_response,
        "timestamp": datetime.now().isoformat(),
        "checks": []
    }

    # Check if Echo mentioned generating an image
    if "image" in echo_response.lower() or "generate" in echo_response.lower():
        # Check ComfyUI queue
        comfyui_result = await test_endpoint("http://127.0.0.1:8188/queue")
        verification["checks"].append({
            "type": "comfyui_queue",
            "result": comfyui_result
        })

        # Check recent history
        history_result = await test_endpoint("http://127.0.0.1:8188/history?max_items=5")
        verification["checks"].append({
            "type": "comfyui_history",
            "result": history_result
        })

    # Check if Echo mentioned voice
    if "voice" in echo_response.lower() or "say" in echo_response.lower():
        voice_result = await test_endpoint("http://127.0.0.1:8312/health")
        verification["checks"].append({
            "type": "voice_service",
            "result": voice_result
        })

    # Check active executions
    verification["active_executions"] = len(active_executions)

    # Determine verification result
    actual_actions = sum(1 for check in verification["checks"]
                        if check["result"].get("success"))

    verification["verdict"] = {
        "echo_claimed_action": bool(verification["checks"]),
        "actual_actions_detected": actual_actions,
        "match": actual_actions > 0 if verification["checks"] else True
    }

    return verification

@control_router.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket endpoint for real-time execution monitoring"""
    await websocket.accept()

    try:
        while True:
            # Send current state every second
            await websocket.send_json({
                "type": "status_update",
                "timestamp": datetime.now().isoformat(),
                "active_executions": list(active_executions.values()),
                "service_status": service_status,
                "queue_depth": len(active_executions)
            })

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info("Monitor WebSocket disconnected")
    except Exception as e:
        logger.error(f"Monitor WebSocket error: {e}")

@control_router.post("/force/execution")
async def force_execution(service: str, action: str, params: Dict):
    """Force immediate execution of an action on a service"""

    if service == "comfyui":
        # Build and execute ComfyUI workflow
        from src.orchestrators.resilient_orchestrator import ResilientOrchestrator
        orchestrator = ResilientOrchestrator()

        if action == "generate_image":
            prompt = params.get("prompt", "test image")
            style = params.get("style", "anime")

            result = await orchestrator.generate_image(prompt, style)

            return {
                "forced": True,
                "service": service,
                "action": action,
                "result": result
            }

    elif service == "anime":
        # Direct anime service call
        request = ExecutionRequest(
            service="anime",
            endpoint="/api/projects",
            method="POST",
            payload=params,
            force=True
        )
        return await execute_directly(request)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported forced action: {service}/{action}")

@control_router.get("/diagnostics")
async def get_diagnostics():
    """Comprehensive diagnostics of Echo's execution system"""

    # Test all services
    service_tests = await test_all_services()

    # Check orchestrator status
    from src.utils.helpers import tower_orchestrator
    orchestrator_metrics = tower_orchestrator.get_current_metrics()

    # Get database status
    from src.db.database import database
    try:
        db_status = await database.test_connection()
    except:
        db_status = {"connected": False}

    return {
        "timestamp": datetime.now().isoformat(),
        "services": service_tests,
        "orchestrator": orchestrator_metrics,
        "database": db_status,
        "active_executions": len(active_executions),
        "execution_history_count": len(execution_history),
        "health": {
            "services_healthy": service_tests["healthy"],
            "services_total": service_tests["services_tested"],
            "orchestrator_overloaded": orchestrator_metrics.get("is_overloaded", False),
            "database_connected": db_status.get("connected", False)
        }
    }