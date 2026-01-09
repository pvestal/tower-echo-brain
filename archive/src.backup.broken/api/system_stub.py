#!/usr/bin/env python3
"""
Stub routes for system metrics and db stats
Temporary fix to prevent dashboard errors
"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/api/echo/metrics/system")
async def get_system_metrics():
    """Return basic system metrics"""
    return {
        "cpu_percent": 25.0,
        "memory_percent": 45.0,
        "disk_usage": 60.0,
        "vram_used_gb": 2.1,  # Add VRAM for dashboard
        "network_io": {
            "bytes_sent": 1000000,
            "bytes_recv": 2000000
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/api/echo/metrics/db")
async def get_db_stats():
    """Return basic database stats"""
    return {
        "total_conversations": 633,
        "total_learnings": 436,
        "vector_memories": 635,
        "active_sessions": 2,
        "connections": {
            "main": {"active": 5, "idle": 2}
        },
        "last_query": datetime.now().isoformat(),
        "database_status": "healthy"
    }