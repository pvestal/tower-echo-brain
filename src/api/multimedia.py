#!/usr/bin/env python3
"""
Multimedia generation API routes for Echo Brain
"""
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from src.utils.helpers import tower_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/api/echo/multimedia/generate/image")
async def generate_image_multimedia(request: dict):
    """Generate image using ComfyUI through real orchestration"""
    logger.info(f"🎨 Image generation request: {request.get('prompt', 'No prompt')[:50]}...")

    try:
        result = await tower_orchestrator.generate_image(
            prompt=request.get('prompt', 'cyberpunk anime scene'),
            style=request.get('style', 'anime')
        )

        return {
            "endpoint": "/api/echo/multimedia/generate/image",
            "action": "image_generation",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return {"success": False, "error": str(e)}

@router.post("/api/echo/multimedia/generate/voice")
async def generate_voice_multimedia(request: dict):
    """Generate voice using Tower voice service"""
    logger.info(f"🗣️ Voice generation request: {request.get('text', 'No text')[:50]}...")

    try:
        result = await tower_orchestrator.generate_voice(
            text=request.get('text', 'Hello from Echo Brain'),
            character=request.get('character', 'echo_default')
        )

        return {
            "endpoint": "/api/echo/multimedia/generate/voice",
            "action": "voice_generation",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        logger.error(f"Voice generation failed: {e}")
        return {"success": False, "error": str(e)}

@router.post("/api/echo/multimedia/generate/music")
async def generate_music_multimedia(request: dict):
    """Generate music using Tower music service"""
    logger.info(f"🎵 Music generation request: {request.get('description', 'No description')[:50]}...")

    try:
        result = await tower_orchestrator.create_music(
            description=request.get('description', 'Epic cinematic soundtrack'),
            duration=request.get('duration', 30)
        )

        return {
            "endpoint": "/api/echo/multimedia/generate/music",
            "action": "music_generation",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        logger.error(f"Music generation failed: {e}")
        return {"success": False, "error": str(e)}

@router.post("/api/echo/multimedia/orchestrate")
async def orchestrate_multimedia_task(request: dict):
    """Orchestrate complex multimedia tasks across services"""
    task_type = request.get('task_type', 'unknown')
    description = request.get('description', 'No description')
    requirements = request.get('requirements', {})

    logger.info(f"🎬 Multimedia orchestration: {task_type} - {description[:50]}...")

    try:
        result = await tower_orchestrator.orchestrate_multimedia(
            task_type=task_type,
            description=description,
            requirements=requirements
        )

        return {
            "endpoint": "/api/echo/multimedia/orchestrate",
            "action": "multimedia_orchestration",
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        logger.error(f"Multimedia orchestration failed: {e}")
        return {"success": False, "error": str(e)}

@router.get("/api/echo/multimedia/services/status")
async def get_multimedia_services_status():
    """Get status of all multimedia services"""
    logger.info("📊 Checking multimedia services status...")

    try:
        services = ['comfyui', 'voice', 'music', 'anime']
        status_results = {}

        for service in services:
            status_results[service] = await tower_orchestrator.get_service_status(service)

        overall_health = all(result.get('success', False) for result in status_results.values())

        return {
            "endpoint": "/api/echo/multimedia/services/status",
            "overall_health": "healthy" if overall_health else "degraded",
            "timestamp": datetime.now().isoformat(),
            "services": status_results
        }
    except Exception as e:
        logger.error(f"Service status check failed: {e}")
        return {"success": False, "error": str(e)}