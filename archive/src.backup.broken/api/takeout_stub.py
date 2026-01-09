#!/usr/bin/env python3
"""
Stub routes for takeout functionality
Temporary fix to prevent dashboard errors
"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/api/echo/takeout/progress")
@router.get("/api/takeout/progress")
async def get_takeout_progress():
    """Return empty takeout progress"""
    return {
        "status": "idle",
        "progress": 0,
        "message": "Takeout service temporarily unavailable",
        "files_processed": 0,
        "total_files": 0
    }

@router.get("/api/echo/takeout/status")
@router.get("/api/takeout/status")
async def get_takeout_status():
    """Return takeout status"""
    return {
        "status": "disabled",
        "message": "Takeout functionality is being restructured"
    }