"""
Diagnostic router for troubleshooting Echo Brain
"""
from fastapi import APIRouter
from datetime import datetime
import psutil
import os
import sys

router = APIRouter()

@router.get("/debug")
async def debug_info():
    """Comprehensive debug information"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "service": "tower-echo-brain",
        "environment": {
            "DB_USER": os.getenv('DB_USER'),
            "QDRANT_HOST": os.getenv('QDRANT_HOST'),
            "OLLAMA_URL": os.getenv('OLLAMA_URL')
        },
        "process": {
            "pid": os.getpid(),
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
    }

@router.get("/test/routes")
async def test_routes():
    """Test if routes are accessible"""
    import urllib.request
    import json
    
    base_url = "http://localhost:8309"
    routes = [
        "/health",
        "/api/conversations/health",
        "/api/echo/brain",
        "/api/system/health"
    ]
    
    results = []
    for route in routes:
        try:
            with urllib.request.urlopen(f"{base_url}{route}", timeout=2) as response:
                status = response.getcode()
                results.append({"route": route, "status": status, "error": None})
        except Exception as e:
            results.append({"route": route, "status": "error", "error": str(e)})
    
    return {"routes": results}
