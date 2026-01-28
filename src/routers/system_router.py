#!/usr/bin/env python3
"""
Echo Brain System Router - System operations and monitoring
Handles: health checks, diagnostics, metrics, system status
"""

import os
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["System"])

# ============= Response Models =============

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: datetime
    uptime_seconds: Optional[float] = None

class MetricsResponse(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    active_connections: int
    timestamp: datetime

class DiagnosticsResponse(BaseModel):
    system: Dict
    services: Dict
    resources: Dict
    errors: List[str]

# Track service start time
SERVICE_START_TIME = datetime.now()

# ============= Health Check Endpoints =============

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    uptime = (datetime.now() - SERVICE_START_TIME).total_seconds()

    return HealthResponse(
        status="healthy",
        service="echo-brain",
        version="4.0.0",
        timestamp=datetime.now(),
        uptime_seconds=uptime
    )

@router.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes/Docker"""
    try:
        # Check if critical services are available
        checks = await _check_critical_services()

        if not checks["database"]:
            raise HTTPException(status_code=503, detail="Database not ready")

        if not checks["ollama"] and not checks["openai"]:
            raise HTTPException(status_code=503, detail="No LLM service available")

        return {"ready": True, "checks": checks}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/alive")
async def liveness_check():
    """Liveness probe - simple check that service is running"""
    return {"alive": True, "timestamp": datetime.now().isoformat()}

# ============= Metrics Endpoints =============

@router.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics():
    """Get current system metrics"""
    try:
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Disk usage
        disk = psutil.disk_usage('/')

        # GPU usage (if available)
        gpu_usage = await _get_gpu_usage()

        # Active connections
        connections = len(psutil.net_connections())

        return MetricsResponse(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage_percent=disk.percent,
            gpu_usage_percent=gpu_usage,
            active_connections=connections,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/history")
async def get_metrics_history(hours: int = 24):
    """Get historical metrics"""
    try:
        # Query historical metrics from actual database
        # For now, return empty metrics until time-series DB is implemented
        return {
            "period": f"last_{hours}_hours",
            "metrics": []
        }
    except Exception as e:
        logger.error(f"Failed to get metrics history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= Diagnostics Endpoints =============

@router.get("/diagnostics", response_model=DiagnosticsResponse)
async def run_diagnostics():
    """Run comprehensive system diagnostics"""
    try:
        errors = []

        # System info
        system_info = {
            "platform": os.uname().sysname,
            "hostname": os.uname().nodename,
            "python_version": os.sys.version,
            "process_id": os.getpid(),
            "working_directory": os.getcwd()
        }

        # Check services
        services = await _check_all_services()

        # Check resources
        resources = {
            "cpu_cores": psutil.cpu_count(),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
        }

        # Check for common issues
        if resources["available_memory_gb"] < 1:
            errors.append("Low memory available")

        if psutil.disk_usage('/').percent > 90:
            errors.append("Disk usage above 90%")

        if not services.get("database"):
            errors.append("Database connection failed")

        return DiagnosticsResponse(
            system=system_info,
            services=services,
            resources=resources,
            errors=errors
        )

    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/diagnostics/database")
async def check_database_health():
    """Check database connectivity and stats"""
    try:
        import asyncpg

        conn = await asyncpg.connect(
            host="localhost",
            database="echo_brain",
            user="patrick",
            password=os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"),
            timeout=5
        )

        # Get database stats
        version = await conn.fetchval("SELECT version()")
        db_size = await conn.fetchval(
            "SELECT pg_database_size(current_database())"
        )
        connection_count = await conn.fetchval(
            "SELECT count(*) FROM pg_stat_activity"
        )

        await conn.close()

        return {
            "status": "healthy",
            "version": version.split('\n')[0],
            "size_mb": round(db_size / (1024*1024), 2),
            "active_connections": connection_count,
            "max_connections": 100  # Default PostgreSQL max
        }

    except asyncpg.PostgresError as e:
        logger.error(f"Database check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")

@router.get("/diagnostics/services")
async def check_services_status():
    """Check status of all integrated services"""
    services = await _check_all_services()

    # Add more details for each service
    detailed_status = {}

    for service, is_up in services.items():
        detailed_status[service] = {
            "status": "online" if is_up else "offline",
            "checked_at": datetime.now().isoformat()
        }

        # Add service-specific details
        if service == "ollama" and is_up:
            detailed_status[service]["models"] = await _get_ollama_models()
        elif service == "qdrant" and is_up:
            detailed_status[service]["collections"] = await _get_qdrant_collections()

    return detailed_status

# ============= System Status Endpoints =============

@router.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        uptime = (datetime.now() - SERVICE_START_TIME).total_seconds()

        # Get current metrics
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Check services
        services = await _check_critical_services()

        return {
            "status": "operational" if all(services.values()) else "degraded",
            "uptime_seconds": uptime,
            "uptime_human": _format_uptime(uptime),
            "metrics": {
                "cpu_percent": cpu,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            },
            "services": services,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/status/logs")
async def get_recent_logs(lines: int = 100, level: Optional[str] = None):
    """Get recent log entries"""
    try:
        # This would typically read from log files or logging service
        # For now, return a message
        return {
            "message": "Log retrieval not yet implemented",
            "requested_lines": lines,
            "level_filter": level
        }
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= Helper Functions =============

async def _check_critical_services() -> Dict[str, bool]:
    """Check if critical services are available"""
    services = {}

    # Check database
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host="localhost",
            database="echo_brain",
            user="patrick",
            password=os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"),
            timeout=2
        )
        await conn.close()
        services["database"] = True
    except:
        services["database"] = False

    # Check Ollama
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=2)
            services["ollama"] = response.status_code == 200
    except:
        services["ollama"] = False

    # Check Qdrant
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:6333/health", timeout=2)
            services["qdrant"] = response.status_code == 200
    except:
        services["qdrant"] = False

    # Check ComfyUI
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8188/system_stats", timeout=2)
            services["comfyui"] = response.status_code == 200
    except:
        services["comfyui"] = False

    # Check OpenAI (if configured)
    services["openai"] = bool(os.getenv("OPENAI_API_KEY"))

    return services

async def _check_all_services() -> Dict[str, bool]:
    """Check all integrated services"""
    critical = await _check_critical_services()

    # Add additional service checks
    additional = {}

    # Check Tower Auth
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8088/health", timeout=2)
            additional["tower_auth"] = response.status_code == 200
    except:
        additional["tower_auth"] = False

    # Check Anime Production
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8328/health", timeout=2)
            additional["anime_production"] = response.status_code == 200
    except:
        additional["anime_production"] = False

    return {**critical, **additional}

async def _get_gpu_usage() -> Optional[float]:
    """Get GPU usage if available"""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return None

async def _get_ollama_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
    except:
        pass
    return []

async def _get_qdrant_collections() -> List[str]:
    """Get list of Qdrant collections"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:6333/collections", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return [c["name"] for c in data.get("result", {}).get("collections", [])]
    except:
        pass
    return []

def _format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")

    return " ".join(parts) if parts else "< 1m"