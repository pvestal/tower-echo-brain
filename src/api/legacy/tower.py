#!/usr/bin/env python3
"""
Tower service integration API routes for Echo Brain
"""
import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException
from src.services.testing import testing_framework

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/api/echo/tower/status")
async def get_tower_status():
    """Get Tower system status"""
    try:
        status = await testing_framework.get_tower_service_status()
        return {
            "tower_status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Tower status check failed: {e}")
        return {"error": str(e)}

@router.get("/api/echo/tower/health")
async def get_tower_health():
    """Get comprehensive Tower health summary"""
    try:
        health = await testing_framework.get_tower_health_summary()
        return health
    except Exception as e:
        logger.error(f"Tower health check failed: {e}")
        return {"error": str(e)}

@router.post("/api/echo/tower/{command}")
async def execute_tower_command(command: str, args: List[str] = []):
    """Execute Tower framework command"""
    logger.info(f"🏗️ Tower command: {command} {args}")

    try:
        result = await testing_framework.run_tower_command(command, args)
        return result
    except Exception as e:
        logger.error(f"Tower command failed: {e}")
        return {"success": False, "error": str(e)}