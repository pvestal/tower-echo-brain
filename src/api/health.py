#!/usr/bin/env python3
"""
Health and status API routes for Echo Brain
"""

import psycopg2
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends

from src.db.database import database
from src.middleware.auth_middleware import get_current_user, get_current_user_optional

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/health")
async def basic_health():
    """Basic public health check endpoint - no authentication required"""
    return {
        "status": "healthy",
        "service": "echo-brain",
        "timestamp": datetime.now().isoformat(),
        "public_endpoint": True
    }

@router.get("/api/echo/health")
async def health_check(user: Dict[str, Any] = Depends(get_current_user)):
    """Health check endpoint with module information - requires authentication"""
    try:
        # Test database connection
        conn = psycopg2.connect(**database.db_config)
        conn.close()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "service": "echo-brain",
        "version": "4.0.0",
        "architecture": "modular",
        "database": db_status,
        "authenticated_user": user.get('user', 'unknown'),
        "user_role": user.get('role', 'user'),
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "intelligence_router": True,
            "conversation_manager": True,
            "testing_framework": True,
            "database": True,
            "safe_executor": True,
            "authentication": True
        }
    }

@router.get("/api/echo/stats")
async def get_echo_stats(user: Dict[str, Any] = Depends(get_current_user)):
    """Get Echo Brain statistics and performance metrics - requires authentication"""
    try:
        conn = psycopg2.connect(**database.db_config)
        cursor = conn.cursor()

        # Get interaction stats
        cursor.execute("""
            SELECT COUNT(*) as total_interactions,
                   AVG(processing_time) as avg_processing_time,
                   COUNT(DISTINCT conversation_id) as unique_conversations
            FROM echo_unified_interactions
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """)

        stats = cursor.fetchone()
        conn.close()

        return {
            "stats": {
                "total_interactions_24h": stats[0] if stats[0] else 0,
                "avg_processing_time": float(stats[1]) if stats[1] else 0.0,
                "unique_conversations_24h": stats[2] if stats[2] else 0
            },
            "requested_by": user.get('user', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@router.get("/api/echo/tower/status")
async def tower_status(user: Dict[str, Any] = Depends(get_current_user)):
    """Check status of Tower services - requires authentication"""
    try:
        from src.utils.helpers import tower_orchestrator

        # Get service statuses
        services = await tower_orchestrator.check_all_services()

        return {
            "tower_status": "operational",
            "services": services,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Tower status check failed: {e}")
        return {
            "tower_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/api/echo/tower/health")
async def tower_health(user: Dict[str, Any] = Depends(get_current_user)):
    """Detailed health check of Tower ecosystem - requires authentication"""
    try:
        from src.utils.helpers import tower_orchestrator

        health_data = await tower_orchestrator.comprehensive_health_check()

        return {
            "overall_health": "healthy",
            "details": health_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Tower health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")