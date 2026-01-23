#!/usr/bin/env python3
"""
Content Generation API Service
==============================

REST API service for the Content Generation Bridge.
Provides endpoints for autonomous anime content generation using Echo Brain agents and ComfyUI.

Endpoints:
- POST /api/generate/scene/{scene_id} - Generate single scene
- POST /api/generate/project/{project_id} - Generate entire project
- GET /api/status/{job_id} - Check generation status
- GET /api/health - Health check
- GET /api/capabilities - System capabilities

Author: Claude Code & Echo Brain System
Date: January 2026
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from autonomous_content_coordinator import AutonomousContentCoordinator
from content_generation_bridge import ContentGenerationBridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Content Generation Bridge API",
    description="Autonomous anime content generation using Echo Brain agents and ComfyUI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global state
generation_jobs = {}
coordinator = None
bridge = None


class GenerationRequest(BaseModel):
    """Request model for generation operations"""
    priority: Optional[str] = Field("normal", description="Priority level: low, normal, high")
    quality_preset: Optional[str] = Field("high", description="Quality preset: draft, normal, high")
    parameters: Optional[Dict] = Field(None, description="Custom generation parameters")
    notify_webhook: Optional[str] = Field(None, description="Webhook URL for completion notification")


class GenerationStatus(BaseModel):
    """Response model for generation status"""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    created_at: str
    updated_at: str
    result: Optional[Dict] = None
    error: Optional[str] = None


class SystemCapabilities(BaseModel):
    """System capabilities and configuration"""
    echo_brain_agents: List[str]
    comfyui_available: bool
    max_concurrent_jobs: int
    supported_formats: List[str]
    quality_presets: List[str]
    lora_models_count: int


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global coordinator, bridge

    logger.info("Starting Content Generation API service")

    try:
        # Initialize coordinator and bridge
        coordinator = AutonomousContentCoordinator()
        bridge = ContentGenerationBridge()

        # Test Echo Brain connection
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8309/health", timeout=5.0)
            if response.status_code == 200:
                logger.info("Echo Brain connection verified")
            else:
                logger.warning("Echo Brain connection issue")

    except Exception as e:
        logger.error(f"Startup error: {e}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Echo Brain
        import httpx
        async with httpx.AsyncClient() as client:
            echo_brain_response = await client.get("http://localhost:8309/health", timeout=5.0)
            echo_brain_healthy = echo_brain_response.status_code == 200

            comfyui_response = await client.get("http://localhost:8188/system_stats", timeout=5.0)
            comfyui_healthy = comfyui_response.status_code == 200

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "echo_brain": "healthy" if echo_brain_healthy else "unhealthy",
                "comfyui": "healthy" if comfyui_healthy else "unhealthy"
            },
            "active_jobs": len([job for job in generation_jobs.values() if job["status"] == "running"]),
            "total_jobs": len(generation_jobs)
        }

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


@app.get("/api/capabilities")
async def get_capabilities():
    """Get system capabilities"""
    try:
        # Count LoRA models
        lora_path = Path("/mnt/1TB-storage/models/loras")
        lora_count = len(list(lora_path.glob("*.safetensors"))) if lora_path.exists() else 0

        # Check Echo Brain agents
        import httpx
        agents = []
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8309/health", timeout=5.0)
                if response.status_code == 200:
                    health_data = response.json()
                    agents = [agent["name"] for agent in health_data.get("agents", [])]
        except:
            agents = ["CodingAgent", "ReasoningAgent", "NarrationAgent"]  # Default

        return SystemCapabilities(
            echo_brain_agents=agents,
            comfyui_available=True,
            max_concurrent_jobs=3,
            supported_formats=["mp4", "webm", "gif"],
            quality_presets=["draft", "normal", "high", "ultra"],
            lora_models_count=lora_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting capabilities: {e}")


@app.post("/api/generate/scene/{scene_id}")
async def generate_scene(
    scene_id: str,
    request: GenerationRequest = GenerationRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generate content for a single scene"""
    try:
        # Validate scene exists
        scene_data = bridge._get_scene_data(scene_id)
        if not scene_data:
            raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

        # Create job
        job_id = str(uuid.uuid4())
        generation_jobs[job_id] = {
            "job_id": job_id,
            "type": "scene",
            "target_id": scene_id,
            "status": "pending",
            "progress": 0.0,
            "message": "Generation job created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "request": request.dict(),
            "result": None,
            "error": None
        }

        # Start generation in background
        background_tasks.add_task(
            execute_scene_generation,
            job_id, scene_id, request
        )

        logger.info(f"Scene generation job {job_id} created for scene {scene_id}")

        return {
            "job_id": job_id,
            "scene_id": scene_id,
            "status": "pending",
            "message": "Scene generation started",
            "estimated_duration": "6-10 minutes"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating scene generation job: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting generation: {e}")


@app.post("/api/generate/project/{project_id}")
async def generate_project(
    project_id: int,
    request: GenerationRequest = GenerationRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generate content for an entire project"""
    try:
        # Validate project exists
        project_data = coordinator._get_project_data(project_id)
        if not project_data:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

        # Get scene count for estimation
        scenes = coordinator._get_project_scenes(project_id)
        if not scenes:
            raise HTTPException(status_code=400, detail=f"Project {project_id} has no scenes")

        # Create job
        job_id = str(uuid.uuid4())
        generation_jobs[job_id] = {
            "job_id": job_id,
            "type": "project",
            "target_id": project_id,
            "status": "pending",
            "progress": 0.0,
            "message": "Project generation job created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "request": request.dict(),
            "result": None,
            "error": None,
            "total_scenes": len(scenes)
        }

        # Start generation in background
        background_tasks.add_task(
            execute_project_generation,
            job_id, project_id, request
        )

        logger.info(f"Project generation job {job_id} created for project {project_id} ({len(scenes)} scenes)")

        return {
            "job_id": job_id,
            "project_id": project_id,
            "status": "pending",
            "message": "Project generation started",
            "total_scenes": len(scenes),
            "estimated_duration": f"{len(scenes) * 6}-{len(scenes) * 10} minutes"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating project generation job: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting generation: {e}")


@app.get("/api/status/{job_id}")
async def get_generation_status(job_id: str):
    """Get status of a generation job"""
    if job_id not in generation_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_data = generation_jobs[job_id]
    return GenerationStatus(**job_data)


@app.get("/api/jobs")
async def list_jobs(status: Optional[str] = None, limit: Optional[int] = 50):
    """List generation jobs with optional filtering"""
    jobs = list(generation_jobs.values())

    # Filter by status if provided
    if status:
        jobs = [job for job in jobs if job["status"] == status]

    # Sort by created_at descending
    jobs.sort(key=lambda x: x["created_at"], reverse=True)

    # Limit results
    if limit:
        jobs = jobs[:limit]

    return {
        "jobs": jobs,
        "total": len(generation_jobs),
        "filtered": len(jobs)
    }


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a generation job"""
    if job_id not in generation_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_data = generation_jobs[job_id]

    if job_data["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed job")

    job_data["status"] = "cancelled"
    job_data["updated_at"] = datetime.now().isoformat()
    job_data["message"] = "Job cancelled by user request"

    return {"message": "Job cancelled", "job_id": job_id}


@app.get("/api/scene/{scene_id}/info")
async def get_scene_info(scene_id: str):
    """Get scene information and generation history"""
    scene_data = bridge._get_scene_data(scene_id)
    if not scene_data:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    # Get generation history for this scene
    scene_jobs = [job for job in generation_jobs.values()
                  if job.get("type") == "scene" and job.get("target_id") == scene_id]

    return {
        "scene_data": scene_data,
        "generation_history": scene_jobs,
        "available_loras": _get_available_loras(scene_data.get('character_lora_mapping', {}))
    }


@app.get("/api/project/{project_id}/info")
async def get_project_info(project_id: int):
    """Get project information and generation status"""
    project_data = coordinator._get_project_data(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    scenes = coordinator._get_project_scenes(project_id)
    characters = coordinator._get_project_characters(project_id)

    # Get generation history for this project
    project_jobs = [job for job in generation_jobs.values()
                   if job.get("type") == "project" and job.get("target_id") == project_id]

    return {
        "project_data": project_data,
        "scenes": len(scenes),
        "characters": len(characters),
        "generation_history": project_jobs,
        "scenes_detail": scenes[:10],  # First 10 scenes
        "characters_detail": characters[:5]  # First 5 characters
    }


async def execute_scene_generation(job_id: str, scene_id: str, request: GenerationRequest):
    """Background task for scene generation"""
    try:
        # Update job status
        generation_jobs[job_id].update({
            "status": "running",
            "progress": 0.1,
            "message": "Starting scene analysis",
            "updated_at": datetime.now().isoformat()
        })

        # Execute scene generation
        result = await coordinator.autonomous_scene_generation(scene_id)

        # Update job with completion
        generation_jobs[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "message": "Scene generation completed successfully",
            "updated_at": datetime.now().isoformat(),
            "result": result
        })

        logger.info(f"Scene generation job {job_id} completed successfully")

        # Send webhook notification if provided
        if request.notify_webhook:
            await _send_webhook_notification(request.notify_webhook, generation_jobs[job_id])

    except Exception as e:
        logger.error(f"Scene generation job {job_id} failed: {e}")
        generation_jobs[job_id].update({
            "status": "failed",
            "message": f"Generation failed: {str(e)}",
            "updated_at": datetime.now().isoformat(),
            "error": str(e)
        })


async def execute_project_generation(job_id: str, project_id: int, request: GenerationRequest):
    """Background task for project generation"""
    try:
        # Update job status
        generation_jobs[job_id].update({
            "status": "running",
            "progress": 0.1,
            "message": "Starting project analysis",
            "updated_at": datetime.now().isoformat()
        })

        # Execute project generation with progress updates
        async def progress_callback(progress: float, message: str):
            generation_jobs[job_id].update({
                "progress": progress,
                "message": message,
                "updated_at": datetime.now().isoformat()
            })

        # Execute project generation
        result = await coordinator.autonomous_project_generation(project_id)

        # Update job with completion
        generation_jobs[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "message": "Project generation completed successfully",
            "updated_at": datetime.now().isoformat(),
            "result": result
        })

        logger.info(f"Project generation job {job_id} completed successfully")

        # Send webhook notification if provided
        if request.notify_webhook:
            await _send_webhook_notification(request.notify_webhook, generation_jobs[job_id])

    except Exception as e:
        logger.error(f"Project generation job {job_id} failed: {e}")
        generation_jobs[job_id].update({
            "status": "failed",
            "message": f"Generation failed: {str(e)}",
            "updated_at": datetime.now().isoformat(),
            "error": str(e)
        })


async def _send_webhook_notification(webhook_url: str, job_data: Dict):
    """Send webhook notification for job completion"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                json={
                    "job_id": job_data["job_id"],
                    "status": job_data["status"],
                    "type": job_data["type"],
                    "target_id": job_data["target_id"],
                    "completed_at": job_data["updated_at"],
                    "success": job_data["status"] == "completed",
                    "error": job_data.get("error")
                },
                timeout=10.0
            )
            logger.info(f"Webhook notification sent to {webhook_url}: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to send webhook notification to {webhook_url}: {e}")


def _get_available_loras(character_mapping: Dict) -> List[str]:
    """Get list of available LoRA models for characters"""
    try:
        lora_path = Path("/mnt/1TB-storage/models/loras")
        if not lora_path.exists():
            return []

        available_loras = [f.stem for f in lora_path.glob("*.safetensors")]

        # Filter to only include LoRAs mentioned in character mapping
        mapped_loras = list(character_mapping.values())
        return [lora for lora in available_loras if lora in mapped_loras]

    except Exception as e:
        logger.error(f"Error getting available LoRAs: {e}")
        return []


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "content_generation_api:app",
        host="0.0.0.0",
        port=8313,
        reload=True,
        log_level="info"
    )