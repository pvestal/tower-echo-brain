#!/usr/bin/env python3
"""
Echo Brain Unified Service - Refactored Main Entry Point with Autonomous Task System
"""

# from modules.video_generator import VideoGenerator
from src.vision_capabilities import EchoVision
from src.api.resilient_routes import resilient_router
from src.api.system_metrics import router as system_metrics_router
from src.api.routes import router as api_router
from src.photo_comparison import router as photo_router
import requests
from agent_development_endpoints import agent_dev_router
from veteran_guardian_endpoints import veteran_router
from telegram_general_chat import general_telegram_router
from telegram_integration import telegram_router
from model_manager import (
    get_model_manager,
    ModelManagementRequest,
    ModelOperation,
)
from board_api import create_board_api
from routing.knowledge_manager import create_simple_knowledge_manager
from routing.request_logger import RequestLogger
from routing.service_registry import ServiceRegistry
from src.tasks.task_queue import Task, TaskType, TaskPriority, TaskStatus
from src.tasks import TaskQueue, BackgroundWorker, AutonomousBehaviors
from src.board_integration import BoardIntegration
from src.db.database import database
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from typing import Dict, Set
from datetime import datetime
import uuid
import time
import uvicorn
import logging
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modular components

# Board of Directors Integration

# Import task system components

# Import new enhanced modules
try:
    from src.tasks.proactive_improvements import proactive_engine
    from src.tasks.git_manager import git_manager
    from src.tasks.persona_trainer import persona_trainer
    from src.integrations.auth_manager import auth_manager

    enhanced_modules_loaded = True
    logger.info("✅ Enhanced modules loaded successfully")
except ImportError as e:
    enhanced_modules_loaded = False
    proactive_engine = None
    git_manager = None
    persona_trainer = None
    auth_manager = None
    logger.error(f"❌ Enhanced modules import failed: {e}")
    import traceback

    logger.error(traceback.format_exc())

# Import existing modules that remain external

# Log enhanced module loading status
if enhanced_modules_loaded:
    logger.info("✅ Enhanced modules loaded")
else:
    logger.warning("⚠️ Some enhanced modules not available")

# Import cognitive model selector for intelligent model selection
model_selector = None
try:
    from fixed_model_selector import ModelSelector

    model_selector = ModelSelector()
    logger.info("✅ Cognitive model selector loaded")
except ImportError:
    logger.warning("⚠️ Cognitive model selector not available")

# Initialize FastAPI application

# Ollama Integration for NVIDIA GPU

escalation_manager = None


def query_ollama(prompt: str, model: str = "qwen2.5-coder:7b"):
    """Query Ollama running on NVIDIA GPU"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json().get("response", "No response")
    except Exception as e:
        print(f"Ollama error: {e}")
        return None
    return None


app = FastAPI(
    title="Echo Brain Unified Service with Autonomous Task System",
    description="Consolidated Echo intelligence with dynamic 1B-70B parameter scaling, Board of Directors decision system, and autonomous task execution",
    version="2.0.0",
)

# Mount static files for board dashboard
try:
    app.mount(
        "/board",
        StaticFiles(directory="/opt/tower-echo-brain/frontend", html=True),
        name="board_dashboard",
    )
    logger.info("🎨 Board dashboard static files mounted at /board")
except Exception as e:
    logger.warning(f"Failed to mount board dashboard static files: {e}")

# Include modular API routes

app.include_router(api_router)
# Include system metrics routes

app.include_router(system_metrics_router)
logger.info("📊 System metrics API routes loaded")


# Include resilient orchestration routes (optional extra endpoints)

app.include_router(resilient_router)
logger.info("🔗 Echo Brain API routes loaded")

# Include external routers
app.include_router(general_telegram_router)
logger.info("📱💬 Telegram General Chat enabled (full Echo capabilities)")

# General Telegram Chat (full Echo capabilities with conversation persistence)
logger.info("📱💬 Telegram General Chat enabled (full Echo capabilities)")

# Veteran-specific Telegram integration
app.include_router(telegram_router)
logger.info("📱🎖️ Telegram Veteran Support enabled")

app.include_router(veteran_router)
logger.info("🎖️ Veteran Guardian Bot integration enabled")

app.include_router(agent_dev_router)
logger.info("🤖 Agent Development System enabled")

# Include control endpoints
try:
    from src.api.control_endpoints import control_router

    app.include_router(control_router)
    logger.info("✅ Control endpoints router included")
except Exception as e:
    logger.warning(f"⚠️ Could not load control endpoints: {e}")

# Include multi-LLM collaboration routes
try:
    from src.api.collaboration_routes import router as collaboration_router

    app.include_router(collaboration_router)
    logger.info("🤝 Multi-LLM collaboration routes included")
except Exception as e:
    logger.warning(f"⚠️ Could not load collaboration routes: {e}")

# Import and initialize Anime Story Orchestrator
# try:
#     from anime_story_orchestrator import AnimeStoryOrchestrator
#     anime_orchestrator = AnimeStoryOrchestrator()
# except Exception as e:
#     logger.warning(f"Failed to initialize Anime Story Orchestrator: {e}")

board_api = None
model_manager = None
service_registry = ServiceRegistry()
request_logger = RequestLogger()  # Fixed: RequestLogger takes no arguments

# Register dependencies for API routes
try:
    from src.api.dependencies import set_dependencies

    set_dependencies(service_registry, request_logger)
    logger.info("✅ Dependencies registered for API routes")
except Exception as e:
    logger.warning(f"Could not register dependencies: {e}")

# Initialize autonomous task system
task_queue = None
background_worker = None
autonomous_behaviors = None


# Expert System Integration
try:
    from src.experts.echo_expert_integration import (
        get_expert_integration,
        enhance_echo_decision,
    )

    EXPERT_SYSTEM_AVAILABLE = True
    logger.info("🧠 Expert System loaded successfully")
except ImportError as e:
    logger.warning(f"Expert System not available: {e}")
    EXPERT_SYSTEM_AVAILABLE = False


@app.on_event("startup")
async def startup_event():
    app.include_router(photo_router)
    """Initialize all components on startup including autonomous task system"""
    global board_api, model_manager, task_queue, background_worker, autonomous_behaviors

    logger.info("🚀 Echo Brain startup sequence initiated")

    # Start tower orchestrator background tasks
    try:
        from src.utils.helpers import tower_orchestrator

        if tower_orchestrator:
            await tower_orchestrator.start_background_tasks()
            logger.info("✅ Tower orchestrator background tasks started")
    except Exception as e:
        logger.warning(f"Could not start tower orchestrator tasks: {e}")

    # Initialize database
    await database.create_tables_if_needed()
    logger.info("✅ Database initialized")

    # Initialize conversation manager with database persistence

    # Initialize board API
    try:
        board_api = create_board_api(service_registry, request_logger)
        app.mount("/board-api", board_api, name="board_api")
        logger.info("✅ Board of Directors API mounted")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Board API: {e}")

    # Initialize model manager
    try:
        model_manager = get_model_manager(service_registry, request_logger)
        logger.info("✅ Model manager initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model manager: {e}")

    # Initialize knowledge manager
    try:
        knowledge_manager = create_simple_knowledge_manager(database.db_config)
        logger.info("✅ Knowledge manager initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize knowledge manager: {e}")

    # Initialize autonomous task system
    try:
        logger.info("🤖 Initializing autonomous task system...")

        # Create task queue
        task_queue = TaskQueue(
            redis_url="redis://localhost:6379/1",  # Use DB 1 for tasks
            db_config=database.db_config,
        )
        await task_queue.initialize()
        logger.info("✅ Task queue initialized")

        # Create background worker
        background_worker = BackgroundWorker(
            task_queue, max_concurrent_tasks=3)

        # Create autonomous behaviors
        autonomous_behaviors = AutonomousBehaviors(task_queue)

        # Start background worker and autonomous behaviors
        asyncio.create_task(background_worker.start())
        asyncio.create_task(autonomous_behaviors.start()
                            )  # DEBUG: Log if this fails

        logger.info("🧠 Echo autonomous task system active")

        # Initialize enhanced modules
        if persona_trainer:
            logger.info("🧠 Initializing persona trainer...")
            try:
                await persona_trainer.initialize()
                logger.info("✅ PersonaTrainer initialized successfully")

                # Start autonomous training loop
                asyncio.create_task(
                    persona_trainer.autonomous_self_improvement())
                logger.info(
                    "✅ Persona training system started - autonomous learning active"
                )
            except Exception as e:
                logger.error(f"❌ PersonaTrainer initialization failed: {e}")
                import traceback

                logger.error(traceback.format_exc())
        else:
            logger.warning(
                "⚠️ PersonaTrainer not available - enhanced modules not loaded"
            )

        if git_manager:
            await git_manager.initialize_all_repos()
            git_manager.enable_auto_commit(
                False
            )  # Start with manual commits for safety
            asyncio.create_task(git_manager.monitor_and_commit_loop())
            logger.info("✅ Git management system started")

        if proactive_engine:
            # Configure monitored projects
            monitored_projects = [
                "/opt/tower-echo-brain/src",
                "/opt/tower-anime-production",
                "/opt/tower-auth",
                "/opt/tower-kb",
            ]
            asyncio.create_task(
                proactive_engine.proactive_improvement_loop(monitored_projects)
            )
            logger.info("✅ Proactive improvement engine started")

        if auth_manager:
            logger.info("✅ Authentication manager loaded")

    except Exception as e:
        logger.error(f"❌ Failed to initialize autonomous task system: {e}")
        import traceback

        logger.error(traceback.format_exc())

    logger.info(
        "🧠 Echo Brain unified service with autonomous capabilities ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown of all components"""
    global background_worker, autonomous_behaviors

    logger.info("🛑 Echo Brain shutdown sequence initiated")

    # Stop autonomous components
    if background_worker:
        await background_worker.stop()

    if autonomous_behaviors:
        await autonomous_behaviors.stop()

    logger.info("✅ Echo Brain shutdown complete")


# Task Management API Endpoints
@app.get("/api/tasks/status")
async def get_task_system_status():
    """Get autonomous task system status"""
    if not task_queue or not background_worker or not autonomous_behaviors:
        raise HTTPException(
            status_code=503,
            detail="Task system not initialized")

    try:
        stats = await task_queue.get_task_stats()
        worker_status = background_worker.get_worker_status()
        behavior_stats = autonomous_behaviors.get_behavior_stats()

        return {
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "task_queue": stats,
            "background_worker": worker_status,
            "autonomous_behaviors": behavior_stats,
        }
    except Exception as e:
        logger.error(f"Error getting task system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tasks/add")
async def add_manual_task(task_data: Dict):
    """Add a manual task to the queue"""
    if not task_queue:
        raise HTTPException(
            status_code=503,
            detail="Task system not initialized")

    try:
        # Create task from provided data
        task = Task(
            id=str(uuid.uuid4()),
            name=task_data.get("name", "Manual Task"),
            task_type=TaskType(task_data.get("task_type", "monitoring")),
            priority=TaskPriority(task_data.get("priority", 3)),
            status=TaskStatus.PENDING,
            payload=task_data.get("payload", {}),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            creator="manual_api",
        )

        success = await task_queue.add_task(task)

        if success:
            return {
                "status": "added",
                "task_id": task.id,
                "message": f"Task '{task.name}' added to queue",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add task")

    except Exception as e:
        logger.error(f"Error adding manual task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks/brain-state")
async def get_brain_state():
    """Get Echo's current brain state and activity"""
    if not background_worker:
        return {"brain_state": "inactive",
                "reason": "Task system not initialized"}

    try:
        worker_status = background_worker.get_worker_status()
        current_tasks = len(worker_status.get("current_tasks", []))

        brain_state = worker_status.get("brain_state", "unknown")

        # Enhance brain state with activity details
        activity_detail = ""
        if brain_state == "processing" and current_tasks > 0:
            activity_detail = f"actively processing {current_tasks} task(s)"
        elif brain_state == "resting":
            activity_detail = "monitoring systems, ready for tasks"
        elif brain_state == "active":
            activity_detail = "autonomous behaviors running"
        elif brain_state == "initializing":
            activity_detail = "starting up autonomous systems"

        return {
            "brain_state": brain_state,
            "activity_detail": activity_detail,
            "current_tasks": current_tasks,
            "stats": worker_status.get("stats", {}),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting brain state: {e}")
        return {"brain_state": "error", "error": str(e)}


@app.post("/api/tasks/emergency")
async def trigger_emergency_response(emergency_data: Dict):
    """Trigger emergency task generation"""
    if not autonomous_behaviors:
        raise HTTPException(
            status_code=503, detail="Autonomous behaviors not initialized"
        )

    try:
        issue_type = emergency_data.get("issue_type")
        details = emergency_data.get("details", {})

        tasks = await autonomous_behaviors.generate_emergency_tasks(issue_type, details)

        return {
            "status": "emergency_response_triggered",
            "tasks_generated": len(tasks),
            "task_ids": [task.id for task in tasks],
            "issue_type": issue_type,
        }

    except Exception as e:
        logger.error(f"Error triggering emergency response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# The capability handler is now in the API routes module

# API Routes are now in the router module


@app.get("/board-dashboard", response_class=HTMLResponse)
async def board_dashboard():
    """Serve the board dashboard HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Echo Brain - Board of Directors</title></head>
    <body>
        <h1>🧠 Echo Brain - Board of Directors Dashboard</h1>
        <p>Advanced AI governance and decision tracking system with autonomous task execution</p>
        <p><a href="/board-api/docs">API Documentation</a></p>
        <p><a href="/api/tasks/status">Task System Status</a></p>
        <p><a href="/api/tasks/brain-state">Brain State</a></p>
    </body>
    </html>
    """


# This will be added to the main.py file


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            # Send acknowledgment
            response = {
                "type": "response",
                "text": f"Echo received: {data}",
                "timestamp": time.time(),
                "hemisphere": "both",
            }
            await websocket.send_json(response)

            # Send a thought
            thought = {
                "type": "thought",
                "content": f"Processing: {data}",
                "hemisphere": "left",
                "confidence": 85,
            }
            await websocket.send_json(thought)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app", host="0.0.0.0", port=8309, reload=False, access_log=True
    )


# Board-Enhanced Code Evaluation Endpoint
@app.post("/api/evaluate", response_model=Dict)
async def evaluate_code_with_board(
        request: Dict, background_tasks: BackgroundTasks):
    """
    Evaluate code through Board of Directors before execution
    """
    code = request.get("code", "")
    context = request.get("context", {})

    # Get Board evaluation
    async with BoardIntegration() as board_integration:
        board_decision = await board_integration.evaluate_with_board(code, context)

        # Determine if safe to execute
        should_execute = board_integration.should_execute(board_decision)
        reasoning = board_integration.get_board_reasoning(board_decision)

        response = {
            "code": code,
            "board_decision": board_decision.get("final_recommendation", "unknown"),
            "should_execute": should_execute,
            "reasoning": reasoning,
            "directors": board_decision.get("directors", []),
            "consensus_strength": board_decision.get("consensus_strength", 0),
        }

        if should_execute and context.get("auto_execute", False):
            # Execute code if approved and requested
            try:
                exec_globals = {}
                exec(code, exec_globals)
                response["execution_result"] = "Success"
            except Exception as e:
                response["execution_result"] = f"Error: {str(e)}"

        return response


logger.info("Board integration endpoint added at /api/evaluate")

# Vision Analysis Endpoints

vision = EchoVision()


@app.post("/api/vision/analyze")
async def analyze_image(request: dict):
    """Analyze image with LLaVA"""
    image_path = request.get("image_path")
    prompt = request.get("prompt", "Describe this image")
    result = await vision.analyze_image(image_path, prompt)
    return {"analysis": result, "model": "llava:7b"}


@app.post("/api/vision/quality-check")
async def quality_check_anime(request: dict):
    """Quality check anime frame"""
    frame_path = request.get("frame_path")
    result = await vision.quality_check_anime(frame_path)
    return {"quality_analysis": result, "frame": frame_path}


# Service Repair and Diagnosis Endpoints
@app.post("/api/service/diagnose")
async def diagnose_service(request: dict):
    """Diagnose a specific service"""
    service_name = request.get("service_name")
    port = request.get("port")

    # Basic service check
    import subprocess
    import requests

    result = {
        "service_name": service_name,
        "port": port,
        "status": "unknown",
        "issues": [],
        "repair_suggestions": [],
    }

    try:
        # Check if port is listening
        proc = subprocess.run(
            ["lsof", "-i", f":{port}"], capture_output=True, text=True
        )
        if proc.returncode == 0:
            result["status"] = "port_listening"

            # Check if service responds
            try:
                resp = requests.get(f"http://localhost:{port}/", timeout=3)
                if resp.status_code == 200:
                    result["status"] = "healthy"
                elif resp.status_code == 404:
                    result["status"] = "responding_but_broken"
                    result["issues"].append(
                        "Service responds but endpoints return 404")
                    result["repair_suggestions"].append(
                        "Check route configuration")
                else:
                    result["status"] = f"http_error_{resp.status_code}"
            except requests.exceptions.RequestException as e:
                result["status"] = "not_responding"
                result["issues"].append(f"HTTP request failed: {str(e)}")
        else:
            result["status"] = "port_not_listening"
            result["issues"].append("No process listening on port")
            result["repair_suggestions"].append("Start the service")

    except Exception as e:
        result["issues"].append(f"Diagnosis failed: {str(e)}")

    return result


@app.post("/api/service/repair")
async def repair_service(request: dict):
    """Attempt to repair a service"""
    service_name = request.get("service_name")
    port = request.get("port")
    repair_action = request.get("action", "restart")

    result = {
        "service_name": service_name,
        "port": port,
        "action": repair_action,
        "success": False,
        "message": "",
        "steps_taken": [],
    }

    try:
        if repair_action == "restart":
            # Find and restart the service
            import subprocess

            # Kill processes on the port
            proc = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True
            )
            if proc.returncode == 0 and proc.stdout.strip():
                pids = proc.stdout.strip().split("\n")
                for pid in pids:
                    subprocess.run(["kill", pid], capture_output=True)
                    result["steps_taken"].append(f"Killed process {pid}")

            # Start service based on name
            service_paths = {
                "anime": "/opt/tower-anime-production/anime_service.py",
                "3d_studio": "/opt/tower-3d-models/3d_studio_server.py",
            }

            if service_name in service_paths:
                service_path = service_paths[service_name]
                # Start the service
                subprocess.Popen(
                    ["python3", service_path], cwd=os.path.dirname(service_path)
                )
                result["steps_taken"].append(f"Started {service_path}")
                result["success"] = True
                result["message"] = f"Service {service_name} restarted"
            else:
                result["message"] = f"Don't know how to restart {service_name}"

    except Exception as e:
        result["message"] = f"Repair failed: {str(e)}"

    return result


@app.get("/api/service/status-all")
async def get_all_service_status():
    """Get status of all Tower services"""
    services = {
        "echo_brain": 8309,
        "anime_production": 8328,
        "3d_studio": 8500,
        "comfyui": 8188,
        "auth": 8088,
        "knowledge_base": 8307,
    }

    status = {}
    for name, port in services.items():
        try:
            import requests

            resp = requests.get(f"http://localhost:{port}/", timeout=2)
            status[name] = {
                "port": port,
                "status": (
                    "healthy"
                    if resp.status_code == 200
                    else f"http_{
                        resp.status_code}"
                ),
                "response_time": resp.elapsed.total_seconds(),
            }
        except Exception as e:
            status[name] = {"port": port, "status": "down", "error": str(e)}

    return status


# Video Generation Endpoint

# video_gen = VideoGenerator()


# @app.post("/api/video/generate")
# async def generate_video(request: dict):
#     """Generate anime/video from prompt"""
#     prompt = request.get("prompt", "")
#     style = request.get("style", "anime")
#     duration = request.get("duration", 30)
#
#     result = await video_gen.generate_video_from_prompt(prompt, style, duration)
#     return result


# @app.post("/api/video/batch_frames")
# async def batch_frames(request: dict):
#     """Generate batch frames for video"""
#     prompt = request.get("prompt", "")
#     style = request.get("style", "anime")
#     num_frames = request.get("num_frames", 30)
#
#     frames = await video_gen.generate_batch_frames(prompt, style, num_frames)
#     return {"frames": frames, "count": len(frames)}


# Model Management Endpoints
@app.get("/api/echo/models/list")
async def list_installed_models():
    """List all installed Ollama models"""
    try:
        if model_manager is None:
            # Fallback: directly list models using ollama
            import subprocess

            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                models = []
                for line in lines:
                    if line:
                        parts = line.split()
                        if len(parts) >= 4:
                            models.append(
                                {
                                    "name": parts[0],
                                    "size": f"{parts[2]} {parts[3]}",
                                    "modified": (
                                        " ".join(parts[4:]) if len(
                                            parts) > 4 else ""
                                    ),
                                }
                            )
                return models
            return {"error": "Failed to list models"}

        models = await model_manager.get_installed_models()
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return {"error": str(e)}


@app.post("/api/echo/models/pull/{model_name}")
async def pull_model_quick(
    model_name: str, tag: str = "latest", background_tasks: BackgroundTasks = None
):
    """Pull a specific model"""
    try:
        if model_manager is None:
            # Fallback: directly pull using ollama
            import asyncio

            model_full = f"{model_name}:{tag}"
            process = await asyncio.create_subprocess_exec(
                "ollama",
                "pull",
                model_full,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return {
                    "status": "success",
                    "model": model_full,
                    "message": "Model pulled successfully",
                }
            else:
                return {
                    "status": "error",
                    "model": model_full,
                    "error": stderr.decode(),
                }

        request = ModelManagementRequest(
            operation=ModelOperation.PULL,
            model_name=model_name,
            tag=tag,
            reason="API request",
            user_id="admin",
        )
        response = await model_manager.request_model_operation(
            request, background_tasks
        )
        return response
    except Exception as e:
        logger.error(f"Failed to pull model: {e}")
        return {"error": str(e)}


@app.delete("/api/echo/models/{model_name}")
async def remove_model(
    model_name: str, tag: str = "latest", background_tasks: BackgroundTasks = None
):
    """Remove a specific model"""
    try:
        if model_manager is None:
            # Fallback: directly remove using ollama
            import asyncio

            model_full = f"{model_name}:{tag}"
            process = await asyncio.create_subprocess_exec(
                "ollama",
                "rm",
                model_full,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return {
                    "status": "success",
                    "model": model_full,
                    "message": "Model removed successfully",
                }
            else:
                return {
                    "status": "error",
                    "model": model_full,
                    "error": stderr.decode(),
                }

        request = ModelManagementRequest(
            operation=ModelOperation.REMOVE,
            model_name=model_name,
            tag=tag,
            reason="API request",
            user_id="admin",
            force=False,
        )
        response = await model_manager.request_model_operation(
            request, background_tasks
        )
        return response
    except Exception as e:
        logger.error(f"Failed to remove model: {e}")
        return {"error": str(e)}


# Redirect to Vue app
@app.get("/dashboard")
async def serve_dashboard():
    """Redirect to Vue app"""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/static/dist/", status_code=302)


@app.get("/brain-visual", response_class=HTMLResponse)
async def serve_brain_visual():
    """Serve the Echo brain visualization"""
    brain_path = "/opt/tower-echo-brain/static/echo_brain_visual.html"
    if os.path.exists(brain_path):
        with open(brain_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="Brain visualization not found", status_code=404)


@app.get("/api/echo/brain")
async def get_brain_activity():
    """Get current brain activity for visualization"""
    # Check if there's current processing
    brain_state = await get_brain_state()
    return {
        "status": brain_state.get("state", "resting"),
        "thoughts_active": brain_state.get("processing", False),
        "hemisphere": "both",  # Could be enhanced based on query type
        "timestamp": datetime.now().isoformat(),
    }


# Mount static files
app.mount(
    "/static", StaticFiles(directory="/opt/tower-echo-brain/static"), name="static"
)


@app.get("/echo-brain/")
async def serve_vue_spa_root():
    """Serve Vue3 SPA root"""
    return FileResponse("/opt/tower-echo-brain/static/dist/index.html")


# ===== AUTONOMOUS TASK IMPLEMENTATION =====


@app.post("/api/echo/tasks/implement")
async def implement_task(request: Dict):
    """
    Autonomous task implementation endpoint - ASYNC version

    Immediately returns task_id and processes in background.
    Use /api/echo/tasks/status/{task_id} to check progress.

    Request:
    {
        "task": "make anime more cinematic",
        "service": "anime-production",
        "test": true
    }

    Returns:
        {
            "task_id": "uuid",
            "status": "queued",
            "message": "Task queued for autonomous implementation"
        }
    """
    try:
        task_description = request.get("task")
        service = request.get("service")
        test = request.get("test", True)

        if not task_description or not service:
            raise HTTPException(
                status_code=400,
                detail="Missing task or service")

        # Create task for queue
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=f"Implement: {task_description[:50]}...",
            task_type=TaskType.CODE_REFACTOR,
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING,
            payload={
                "task": task_description,
                "service": service,
                "test": test,
                "implementation_type": "autonomous",
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Queue for background processing
        await task_queue.add_task(task)

        logger.info(
            f"📋 Queued autonomous implementation task {task_id[:8]}: {task_description[:60]}"
        )

        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Task queued for autonomous implementation. Use /api/echo/tasks/status/{task_id} to check progress.",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Task queueing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/echo/tasks/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of autonomous implementation task"""
    try:
        # Check task queue
        task = await task_queue.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Check database for results
        query = """
            SELECT service, request, changes, review_score, success,
                   backup_path, test_results, created_at
            FROM code_modifications
            WHERE task_id = :task_id
            ORDER BY created_at DESC
            LIMIT 1
        """

        result = await database.fetch_one(query, {"task_id": task_id})

        response = {
            "task_id": task_id,
            "status": task.status.value,
            "name": task.name,
            "created_at": task.created_at.isoformat(),
            "details": task.details,
        }

        if result:
            response["result"] = {
                "service": result["service"],
                "task": result["request"],
                "review_score": result["review_score"],
                "success": result["success"],
                "test_results": result["test_results"],
                "completed_at": result["created_at"].isoformat(),
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
