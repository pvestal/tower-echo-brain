"""
Anime Production Orchestration Service

FastAPI service wrapper for the comprehensive anime production workflow orchestration system.
Provides REST API endpoints for managing anime production projects from concept to delivery.

Key Features:
- Project creation and management
- Real-time progress monitoring
- Resource usage tracking
- Quality validation reporting
- Integration with AutonomousCore
- WebSocket support for real-time updates

API Endpoints:
- POST /projects - Create new anime production project
- GET /projects - List all projects
- GET /projects/{project_id} - Get project details and status
- POST /projects/{project_id}/start - Start project production
- POST /projects/{project_id}/pause - Pause project production
- GET /projects/{project_id}/progress - Get detailed progress information
- GET /system/status - Get orchestrator system status
- GET /system/resources - Get resource usage information
- WebSocket /ws/projects/{project_id} - Real-time project updates
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocket
from pydantic import BaseModel, Field
import uvicorn

# Import orchestration components
from .core import AutonomousCore
from .anime_production_orchestrator import (
    AnimeProductionOrchestrator, ProductionProject, ProductionStage,
    ProjectStatus, enhance_autonomous_core_for_anime_production
)
from .anime_task_executors import EnhancedAnimeExecutor

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses

class ProjectCreationRequest(BaseModel):
    """Request model for creating a new anime production project"""
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field("", description="Project description")
    project_type: str = Field("short_film", description="Type of project: short_film, music_video, commercial, etc.")
    priority: int = Field(5, ge=1, le=10, description="Priority level (1-10)")

    characters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Character definitions with names, descriptions, appearance"
    )
    scenes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Scene definitions with names, descriptions, characters involved"
    )
    style: Dict[str, Any] = Field(
        default_factory=dict,
        description="Style configuration including visual style, mood, technical specs"
    )
    technical_specs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Technical specifications for output format, resolution, etc."
    )
    deadline: Optional[datetime] = Field(None, description="Project deadline")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "name": "Tokyo Debt Desire - Pilot Episode",
                "description": "Pilot episode for cyberpunk anime series",
                "project_type": "short_film",
                "priority": 3,
                "characters": [
                    {
                        "name": "Mei Tanaka",
                        "description": "Protagonist, debt collector in futuristic Tokyo",
                        "appearance": "Short black hair, cybernetic arm, street clothes"
                    },
                    {
                        "name": "Ryuu",
                        "description": "Mei's AI companion",
                        "appearance": "Holographic projection, blue glow, geometric patterns"
                    }
                ],
                "scenes": [
                    {
                        "name": "Opening - Neon Streets",
                        "description": "Mei walks through rain-soaked neon-lit streets",
                        "characters": ["Mei Tanaka"],
                        "setting": "Night, urban, cyberpunk cityscape"
                    },
                    {
                        "name": "Confrontation",
                        "description": "Mei confronts a debtor in their apartment",
                        "characters": ["Mei Tanaka", "Debtor"],
                        "setting": "Small apartment, dramatic lighting"
                    }
                ],
                "style": {
                    "base_style": "cyberpunk anime",
                    "color_palette": "neon blues and purples with orange accents",
                    "mood": "noir, dramatic, futuristic",
                    "art_style": "detailed, cinematic, professional anime"
                },
                "technical_specs": {
                    "resolution": "1920x1080",
                    "frame_rate": 24,
                    "duration_target": 120,
                    "format": "mp4"
                }
            }
        }


class ProjectUpdateRequest(BaseModel):
    """Request model for updating project details"""
    name: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    deadline: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class ProjectStatusResponse(BaseModel):
    """Response model for project status"""
    project_id: str
    name: str
    status: str
    current_stage: str
    total_progress_percent: float
    stages_completed: List[str]
    stage_progress: Dict[str, float]
    active_executions: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    deadline: Optional[datetime]
    quality_issues: List[Dict[str, Any]]
    resource_usage: Dict[str, Any]


class SystemStatusResponse(BaseModel):
    """Response model for system status"""
    orchestrator_status: str
    autonomous_core_status: str
    active_projects: int
    running_executions: int
    total_projects_created: int
    system_uptime_seconds: float
    resource_utilization: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.project_subscribers: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, project_id: Optional[str] = None):
        """Accept WebSocket connection"""
        await websocket.accept()

        if project_id:
            if project_id not in self.project_subscribers:
                self.project_subscribers[project_id] = []
            self.project_subscribers[project_id].append(websocket)
        else:
            if "global" not in self.active_connections:
                self.active_connections["global"] = []
            self.active_connections["global"].append(websocket)

    def disconnect(self, websocket: WebSocket, project_id: Optional[str] = None):
        """Remove WebSocket connection"""
        if project_id and project_id in self.project_subscribers:
            if websocket in self.project_subscribers[project_id]:
                self.project_subscribers[project_id].remove(websocket)
        else:
            if "global" in self.active_connections and websocket in self.active_connections["global"]:
                self.active_connections["global"].remove(websocket)

    async def send_project_update(self, project_id: str, data: Dict[str, Any]):
        """Send update to all subscribers of a specific project"""
        if project_id in self.project_subscribers:
            disconnected = []
            for websocket in self.project_subscribers[project_id]:
                try:
                    await websocket.send_json({
                        "type": "project_update",
                        "project_id": project_id,
                        "timestamp": datetime.now().isoformat(),
                        "data": data
                    })
                except:
                    disconnected.append(websocket)

            # Remove disconnected clients
            for ws in disconnected:
                self.project_subscribers[project_id].remove(ws)

    async def send_system_update(self, data: Dict[str, Any]):
        """Send system-wide update to all global subscribers"""
        if "global" in self.active_connections:
            disconnected = []
            for websocket in self.active_connections["global"]:
                try:
                    await websocket.send_json({
                        "type": "system_update",
                        "timestamp": datetime.now().isoformat(),
                        "data": data
                    })
                except:
                    disconnected.append(websocket)

            # Remove disconnected clients
            for ws in disconnected:
                self.active_connections["global"].remove(ws)


# Initialize FastAPI app
app = FastAPI(
    title="Anime Production Orchestration Service",
    description="Comprehensive workflow orchestration for autonomous anime production",
    version="1.0.0"
)

# Global instances
orchestrator: Optional[AnimeProductionOrchestrator] = None
autonomous_core: Optional[AutonomousCore] = None
websocket_manager = WebSocketManager()


async def get_orchestrator() -> AnimeProductionOrchestrator:
    """Dependency to get orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


async def get_autonomous_core() -> AutonomousCore:
    """Dependency to get autonomous core instance"""
    global autonomous_core
    if autonomous_core is None:
        raise HTTPException(status_code=503, detail="Autonomous core not initialized")
    return autonomous_core


@app.on_event("startup")
async def startup_event():
    """Initialize orchestration system on startup"""
    global orchestrator, autonomous_core

    try:
        logger.info("Starting anime production orchestration service...")

        # Initialize AutonomousCore
        autonomous_core = AutonomousCore({
            'cycle_interval': 15,  # 15 second cycles for responsive anime production
            'max_concurrent_tasks': 5,
            'max_tasks_per_minute': 20,
            'max_tasks_per_hour': 200
        })

        # Enhance AutonomousCore with anime production capabilities
        await enhance_autonomous_core_for_anime_production(autonomous_core)

        # Replace executor with enhanced anime executor
        autonomous_core.executor = EnhancedAnimeExecutor({
            'output_base': '/mnt/1TB-storage/ComfyUI/output',
            'lora_base': '/mnt/1TB-storage/models/loras',
            'temp_base': '/tmp/anime_production'
        })

        # Start AutonomousCore
        core_started = await autonomous_core.start()
        if not core_started:
            raise Exception("Failed to start AutonomousCore")

        # Initialize orchestrator
        orchestrator = AnimeProductionOrchestrator(
            autonomous_core=autonomous_core,
            config={
                'comfyui_url': 'http://localhost:8188',
                'echo_brain_url': 'http://localhost:8309'
            }
        )

        # Start orchestrator
        orchestrator_started = await orchestrator.start()
        if not orchestrator_started:
            raise Exception("Failed to start AnimeProductionOrchestrator")

        # Start background tasks
        asyncio.create_task(progress_monitoring_task())
        asyncio.create_task(websocket_update_task())

        logger.info("Anime production orchestration service started successfully")

    except Exception as e:
        logger.error(f"Failed to start orchestration service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global orchestrator, autonomous_core

    try:
        if orchestrator:
            await orchestrator.stop()

        if autonomous_core:
            await autonomous_core.stop()

        logger.info("Anime production orchestration service stopped")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with service information"""
    return """
    <html>
        <head>
            <title>Anime Production Orchestration Service</title>
        </head>
        <body>
            <h1>Anime Production Orchestration Service</h1>
            <p>Comprehensive workflow orchestration for autonomous anime production</p>
            <h2>API Documentation</h2>
            <ul>
                <li><a href="/docs">Interactive API Documentation (Swagger UI)</a></li>
                <li><a href="/redoc">Alternative API Documentation (ReDoc)</a></li>
            </ul>
            <h2>System Status</h2>
            <ul>
                <li><a href="/system/status">System Status</a></li>
                <li><a href="/system/resources">Resource Usage</a></li>
            </ul>
        </body>
    </html>
    """


@app.post("/projects", response_model=Dict[str, str])
async def create_project(
    request: ProjectCreationRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AnimeProductionOrchestrator = Depends(get_orchestrator)
):
    """Create a new anime production project"""
    try:
        project_spec = request.dict()
        project_id = await orchestrator.create_project(project_spec)

        if project_id:
            # Add background task to send WebSocket updates
            background_tasks.add_task(
                websocket_manager.send_system_update,
                {"event": "project_created", "project_id": project_id}
            )

            return {
                "project_id": project_id,
                "status": "created",
                "message": f"Project '{request.name}' created successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create project")

    except Exception as e:
        logger.error(f"Project creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects", response_model=List[Dict[str, Any]])
async def list_projects(
    status: Optional[str] = None,
    orchestrator: AnimeProductionOrchestrator = Depends(get_orchestrator)
):
    """List all anime production projects"""
    try:
        projects = []
        for project in orchestrator.active_projects.values():
            if status is None or project.status.value == status:
                projects.append({
                    "project_id": project.id,
                    "name": project.name,
                    "status": project.status.value,
                    "current_stage": project.current_stage.value,
                    "progress_percent": project.total_progress_percent,
                    "created_at": project.created_at.isoformat(),
                    "updated_at": project.updated_at.isoformat()
                })

        return projects

    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}", response_model=ProjectStatusResponse)
async def get_project_status(
    project_id: str,
    orchestrator: AnimeProductionOrchestrator = Depends(get_orchestrator)
):
    """Get detailed project status"""
    try:
        project_status = await orchestrator.get_project_status(project_id)

        if project_status:
            return ProjectStatusResponse(**project_status)
        else:
            raise HTTPException(status_code=404, detail="Project not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{project_id}/start")
async def start_project_production(
    project_id: str,
    background_tasks: BackgroundTasks,
    orchestrator: AnimeProductionOrchestrator = Depends(get_orchestrator)
):
    """Start production for a project"""
    try:
        success = await orchestrator.start_project_production(project_id)

        if success:
            # Send WebSocket update
            background_tasks.add_task(
                websocket_manager.send_project_update,
                project_id,
                {"event": "production_started", "status": "in_progress"}
            )

            return {
                "project_id": project_id,
                "status": "started",
                "message": "Project production started successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to start project production")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start project production: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{project_id}/pause")
async def pause_project_production(
    project_id: str,
    background_tasks: BackgroundTasks,
    orchestrator: AnimeProductionOrchestrator = Depends(get_orchestrator)
):
    """Pause production for a project"""
    try:
        project = orchestrator.active_projects.get(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Update project status to paused
        project.status = ProjectStatus.ON_HOLD
        project.updated_at = datetime.now()
        await orchestrator.save_project_state(project)

        # Send WebSocket update
        background_tasks.add_task(
            websocket_manager.send_project_update,
            project_id,
            {"event": "production_paused", "status": "on_hold"}
        )

        return {
            "project_id": project_id,
            "status": "paused",
            "message": "Project production paused"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause project production: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}/progress")
async def get_project_progress(
    project_id: str,
    orchestrator: AnimeProductionOrchestrator = Depends(get_orchestrator)
):
    """Get detailed progress information for a project"""
    try:
        project_status = await orchestrator.get_project_status(project_id)

        if not project_status:
            raise HTTPException(status_code=404, detail="Project not found")

        # Add additional progress details
        progress_details = {
            **project_status,
            "stage_breakdown": {
                stage.value: {
                    "name": stage.value.replace('_', ' ').title(),
                    "progress": project_status["stage_progress"].get(stage.value, 0.0),
                    "completed": stage.value in project_status["stages_completed"]
                }
                for stage in ProductionStage
            },
            "estimated_completion": None,  # Could add estimation logic
            "performance_metrics": {
                "average_stage_time": None,  # Could calculate from history
                "quality_score_average": None  # Could aggregate quality scores
            }
        }

        return progress_details

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    orchestrator: AnimeProductionOrchestrator = Depends(get_orchestrator),
    autonomous_core: AutonomousCore = Depends(get_autonomous_core)
):
    """Get overall system status"""
    try:
        # Get autonomous core status
        core_status = await autonomous_core.get_status()

        # Get orchestrator statistics
        active_projects = len(orchestrator.active_projects)
        running_executions = len(orchestrator.active_executions)

        # Calculate resource utilization
        resource_utilization = {}
        if orchestrator.resource_manager:
            for resource_type, available in orchestrator.resource_manager.available_resources.items():
                allocated = orchestrator.resource_manager.allocated_resources.get(resource_type, 0.0)
                utilization = (allocated / available * 100) if available > 0 else 0
                resource_utilization[resource_type.value] = {
                    "available": available,
                    "allocated": allocated,
                    "utilization_percent": utilization
                }

        # Get performance metrics
        performance_metrics = autonomous_core.get_performance_metrics()

        return SystemStatusResponse(
            orchestrator_status="running",
            autonomous_core_status=core_status.state.value,
            active_projects=active_projects,
            running_executions=running_executions,
            total_projects_created=len(orchestrator.execution_history),
            system_uptime_seconds=core_status.uptime_seconds,
            resource_utilization=resource_utilization,
            performance_metrics=performance_metrics
        )

    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/resources")
async def get_resource_usage(
    orchestrator: AnimeProductionOrchestrator = Depends(get_orchestrator)
):
    """Get detailed resource usage information"""
    try:
        resource_info = {
            "resource_types": {},
            "current_allocations": {},
            "usage_history": list(orchestrator.resource_manager.usage_history)[-20:]  # Last 20 entries
        }

        # Current resource status
        for resource_type, available in orchestrator.resource_manager.available_resources.items():
            allocated = orchestrator.resource_manager.allocated_resources.get(resource_type, 0.0)
            resource_info["resource_types"][resource_type.value] = {
                "available": available,
                "allocated": allocated,
                "free": available - allocated,
                "utilization_percent": (allocated / available * 100) if available > 0 else 0
            }

        # Active allocations by stage
        for stage_id, execution in orchestrator.active_executions.items():
            resource_info["current_allocations"][stage_id] = {
                "project_id": execution.project_id,
                "stage": execution.stage.value,
                "resources": [
                    {
                        "type": req.resource_type.value,
                        "amount": req.amount,
                        "duration_minutes": req.duration_minutes
                    }
                    for req in execution.resources_required
                ]
            }

        return resource_info

    except Exception as e:
        logger.error(f"Failed to get resource usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/projects/{project_id}")
async def websocket_project_updates(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for real-time project updates"""
    await websocket_manager.connect(websocket, project_id)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now().isoformat()})
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, project_id)
    except Exception as e:
        logger.error(f"WebSocket error for project {project_id}: {e}")
        websocket_manager.disconnect(websocket, project_id)


@app.websocket("/ws/system")
async def websocket_system_updates(websocket: WebSocket):
    """WebSocket endpoint for system-wide updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now().isoformat()})
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket system error: {e}")
        websocket_manager.disconnect(websocket)


# Background tasks for real-time updates

async def progress_monitoring_task():
    """Background task to monitor project progress and send updates"""
    global orchestrator, websocket_manager

    while True:
        try:
            if orchestrator:
                for project_id, project in orchestrator.active_projects.items():
                    if project.status == ProjectStatus.IN_PROGRESS:
                        # Get current status
                        project_status = await orchestrator.get_project_status(project_id)

                        if project_status:
                            # Send progress update via WebSocket
                            await websocket_manager.send_project_update(
                                project_id,
                                {
                                    "event": "progress_update",
                                    "progress": project_status["total_progress_percent"],
                                    "current_stage": project_status["current_stage"],
                                    "stage_progress": project_status["stage_progress"]
                                }
                            )

            await asyncio.sleep(30)  # Update every 30 seconds

        except Exception as e:
            logger.error(f"Error in progress monitoring task: {e}")
            await asyncio.sleep(60)  # Wait longer on error


async def websocket_update_task():
    """Background task to send periodic system updates via WebSocket"""
    global orchestrator, autonomous_core, websocket_manager

    while True:
        try:
            if orchestrator and autonomous_core:
                # Get system status
                core_status = await autonomous_core.get_status()

                system_update = {
                    "event": "system_status",
                    "active_projects": len(orchestrator.active_projects),
                    "running_executions": len(orchestrator.active_executions),
                    "autonomous_core_state": core_status.state.value,
                    "cycles_completed": core_status.cycles_completed,
                    "tasks_executed": core_status.tasks_executed
                }

                await websocket_manager.send_system_update(system_update)

            await asyncio.sleep(60)  # Update every minute

        except Exception as e:
            logger.error(f"Error in WebSocket update task: {e}")
            await asyncio.sleep(120)  # Wait longer on error


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting Anime Production Orchestration Service")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8320,  # Dedicated port for orchestration service
        log_level="info",
        access_log=True
    )