#!/usr/bin/env python3
"""
Echo Brain Unified Service - Refactored Main Entry Point with Autonomous Task System
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import asyncio
import logging
import uvicorn
import psycopg2
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import modular components
from src.db.models import (
    QueryRequest, QueryResponse, ExecuteRequest, ExecuteResponse,
    TestRequest, VoiceNotificationRequest, VoiceStatusRequest
)
from src.db.database import database
from src.core.intelligence import intelligence_router
from src.services.conversation import conversation_manager
from src.services.testing import testing_framework
from src.utils.helpers import safe_executor, tower_orchestrator

# Import task system components
from src.tasks import TaskQueue, BackgroundWorker, AutonomousBehaviors
from src.tasks.task_queue import Task, TaskType, TaskPriority, TaskStatus

# Import existing modules that remain external
from echo_brain_thoughts import echo_brain
from directors.director_registry import DirectorRegistry
from directors.decision_tracker import DecisionTracker
from directors.feedback_system import FeedbackProcessor, UserFeedback, FeedbackType
from directors.user_preferences import UserPreferences, PreferenceType
from directors.knowledge_manager import KnowledgeManager, create_simple_knowledge_manager
from directors.sandbox_executor import SandboxExecutor, create_strict_sandbox
from board_api import create_board_api
from model_manager import (
    get_model_manager, ModelManagementRequest, ModelManagementResponse,
    ModelOperation, ModelInfo
)
from directors.auth_middleware import get_current_user
from model_decision_engine import get_decision_engine
from telegram_integration import telegram_router
from veteran_guardian_endpoints import veteran_router
from agent_development_endpoints import agent_dev_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Echo Brain Unified Service with Autonomous Task System",
    description="Consolidated Echo intelligence with dynamic 1B-70B parameter scaling, Board of Directors decision system, and autonomous task execution",
    version="2.0.0"
)

# Mount static files for board dashboard
try:
    app.mount("/board", StaticFiles(directory="/opt/tower-echo-brain/frontend", html=True), name="board_dashboard")
    logger.info("🎨 Board dashboard static files mounted at /board")
except Exception as e:
    logger.warning(f"Failed to mount board dashboard static files: {e}")

# Include modular API routes
from src.api.routes import router as api_router
app.include_router(api_router)
logger.info("🔗 Echo Brain API routes loaded")

# Include external routers
app.include_router(telegram_router)
logger.info("📱 Telegram general integration enabled")

app.include_router(veteran_router)
logger.info("🎖️ Veteran Guardian Bot integration enabled")

app.include_router(agent_dev_router)
logger.info("🤖 Agent Development System enabled")

# Import and initialize Anime Story Orchestrator
try:
    from anime_story_orchestrator import AnimeStoryOrchestrator
    anime_orchestrator = AnimeStoryOrchestrator()
    logger.info("🎌 Anime Story Orchestrator (agenticPersona) enabled")
except Exception as e:
    logger.warning(f"Failed to initialize Anime Story Orchestrator: {e}")
    anime_orchestrator = None

# Initialize global instances
board_api = None
model_manager = None
director_registry = DirectorRegistry()
decision_tracker = DecisionTracker(database.db_config)

# Initialize autonomous task system
task_queue = None
background_worker = None
autonomous_behaviors = None

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup including autonomous task system"""
    global board_api, model_manager, task_queue, background_worker, autonomous_behaviors

    logger.info("🚀 Echo Brain startup sequence initiated")

    # Initialize database
    await database.create_tables_if_needed()
    logger.info("✅ Database initialized")

    # Initialize board API
    try:
        board_api = create_board_api(director_registry, decision_tracker)
        app.mount("/board-api", board_api, name="board_api")
        logger.info("✅ Board of Directors API mounted")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Board API: {e}")

    # Initialize model manager
    try:
        model_manager = get_model_manager(director_registry, decision_tracker)
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
            db_config=database.db_config
        )
        await task_queue.initialize()
        logger.info("✅ Task queue initialized")
        
        # Create background worker
        background_worker = BackgroundWorker(task_queue, max_concurrent_tasks=3)
        
        # Create autonomous behaviors
        autonomous_behaviors = AutonomousBehaviors(task_queue)
        
        # Start background worker and autonomous behaviors
        asyncio.create_task(background_worker.start())
        asyncio.create_task(autonomous_behaviors.start())
        
        logger.info("🧠 Echo autonomous task system active")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize autonomous task system: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("🧠 Echo Brain unified service with autonomous capabilities ready")

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
        raise HTTPException(status_code=503, detail="Task system not initialized")
        
    try:
        stats = await task_queue.get_task_stats()
        worker_status = background_worker.get_worker_status()
        behavior_stats = autonomous_behaviors.get_behavior_stats()
        
        return {
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "task_queue": stats,
            "background_worker": worker_status,
            "autonomous_behaviors": behavior_stats
        }
    except Exception as e:
        logger.error(f"Error getting task system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasks/add")
async def add_manual_task(task_data: Dict):
    """Add a manual task to the queue"""
    if not task_queue:
        raise HTTPException(status_code=503, detail="Task system not initialized")
        
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
            creator="manual_api"
        )
        
        success = await task_queue.add_task(task)
        
        if success:
            return {
                "status": "added",
                "task_id": task.id,
                "message": f"Task '{task.name}' added to queue"
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
        return {"brain_state": "inactive", "reason": "Task system not initialized"}
        
    try:
        worker_status = background_worker.get_worker_status()
        current_tasks = len(worker_status.get('current_tasks', []))
        
        brain_state = worker_status.get('brain_state', 'unknown')
        
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
            "stats": worker_status.get('stats', {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting brain state: {e}")
        return {"brain_state": "error", "error": str(e)}

@app.post("/api/tasks/emergency")
async def trigger_emergency_response(emergency_data: Dict):
    """Trigger emergency task generation"""
    if not autonomous_behaviors:
        raise HTTPException(status_code=503, detail="Autonomous behaviors not initialized")
        
    try:
        issue_type = emergency_data.get("issue_type")
        details = emergency_data.get("details", {})
        
        tasks = await autonomous_behaviors.generate_emergency_tasks(issue_type, details)
        
        return {
            "status": "emergency_response_triggered",
            "tasks_generated": len(tasks),
            "task_ids": [task.id for task in tasks],
            "issue_type": issue_type
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

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="127.0.0.1",
        port=8309,
        reload=False,
        access_log=True
    )
