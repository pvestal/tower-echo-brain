"""
Monitoring and diagnostics for Echo Brain
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import psutil
import os
import json
import subprocess
from typing import Dict, Any

router = APIRouter()

@router.get("/system/status")
async def system_status():
    """Get comprehensive system status"""
    status = {
        "timestamp": datetime.now().isoformat(),
        "service": "echo-brain",
        "version": "4.0.0"
    }
    
    # Process info
    try:
        proc = psutil.Process()
        status["process"] = {
            "pid": proc.pid,
            "cpu_percent": proc.cpu_percent(interval=0.1),
            "memory_mb": proc.memory_info().rss / 1024 / 1024,
            "threads": proc.num_threads()
        }
    except Exception as e:
        status["process_error"] = str(e)
    
    # Database status
    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "echo_brain"),
            user=os.getenv("DB_USER", "echo_brain_app"),
            password=os.getenv("DB_PASSWORD", ""),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version(), current_database(), current_user")
        db_info = cursor.fetchone()
        cursor.close()
        conn.close()
        
        status["database"] = {
            "status": "connected",
            "version": db_info[0].split(',')[0],
            "database": db_info[1],
            "user": db_info[2]
        }
    except Exception as e:
        status["database"] = {"status": "error", "error": str(e)}
    
    # Qdrant status
    try:
        import requests
        resp = requests.get("http://localhost:6333", timeout=2)
        status["qdrant"] = {
            "status": "connected" if resp.status_code == 200 else "error",
            "http_status": resp.status_code
        }
    except Exception as e:
        status["qdrant"] = {"status": "error", "error": str(e)}
    
    # Ollama status
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = resp.json().get('models', [])
            status["ollama"] = {
                "status": "connected",
                "model_count": len(models),
                "models": [m.get('name') for m in models[:3]]
            }
        else:
            status["ollama"] = {"status": "error", "http_status": resp.status_code}
    except Exception as e:
        status["ollama"] = {"status": "error", "error": str(e)}
    
    return status

@router.get("/endpoints")
async def list_endpoints():
    """List all available endpoints"""
    # This would be populated by the main app
    return {
        "endpoints": [
            "/health",
            "/api/conversations/health",
            "/api/conversations/search",
            "/api/echo/brain",
            "/api/system/status",
            "/api/monitoring/system/status"
        ],
        "timestamp": datetime.now().isoformat()
    }
