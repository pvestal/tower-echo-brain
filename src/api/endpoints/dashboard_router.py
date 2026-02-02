#!/usr/bin/env python3
"""
Dashboard API Router
Provides server-side endpoints for the Echo Brain dashboard
"""

import logging
import asyncio
import aiohttp
from typing import Dict, Any, List
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

@router.get("/services/status")
async def check_services_status():
    """Check status of all Tower services from server side"""
    services = [
        {"name": "Echo Brain", "port": 8309, "endpoint": "/health"},
        {"name": "Echo MCP", "port": 8312, "endpoint": "/health"},
        {"name": "Auth Service", "port": 8088, "endpoint": "/health"},
        {"name": "ComfyUI", "port": 8188, "endpoint": "/"},
        {"name": "Anime API", "port": 8328, "endpoint": "/health"},
        {"name": "Ollama", "port": 11434, "endpoint": "/"}
    ]

    results = []

    async with aiohttp.ClientSession() as session:
        for service in services:
            try:
                async with session.get(
                    f"http://localhost:{service['port']}{service['endpoint']}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    status = "up" if response.status == 200 else "down"
                    results.append({
                        "name": service["name"],
                        "port": service["port"],
                        "status": status,
                        "response_code": response.status
                    })
            except Exception as e:
                results.append({
                    "name": service["name"],
                    "port": service["port"],
                    "status": "down",
                    "error": str(e)
                })

    return {"services": results}

@router.post("/mcp/search")
async def proxy_mcp_search(request: dict):
    """Proxy MCP memory search requests"""
    try:
        query = request.get("query", "")
        limit = request.get("limit", 5)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8312/mcp",
                json={
                    "method": "tools/call",
                    "params": {
                        "name": "search_memory",
                        "arguments": {"query": query, "limit": limit}
                    }
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                data = await response.json()
                return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP search error: {str(e)}")

@router.post("/mcp/facts")
async def proxy_mcp_facts(request: dict):
    """Proxy MCP facts requests"""
    try:
        topic = request.get("topic", "general")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8312/mcp",
                json={
                    "method": "tools/call",
                    "params": {
                        "name": "get_facts",
                        "arguments": {"topic": topic}
                    }
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                data = await response.json()
                return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP facts error: {str(e)}")

@router.get("/models/ollama")
async def get_ollama_models():
    """Get Ollama models from server side"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:11434/api/tags",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                data = await response.json()

                # Format models for dashboard display
                models = []
                for model in data.get("models", []):
                    models.append({
                        "name": model["name"],
                        "status": "available",
                        "size": f"{(model['size'] / 1024 / 1024 / 1024):.1f}GB",
                        "modified": model.get("modified_at", "Unknown")
                    })

                return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

@router.get("/logs/system")
async def get_system_logs():
    """Get system logs for dashboard display"""
    try:
        # Get Echo Brain health for log entries
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:8309/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                health = await response.json()

        # Create log entries
        from datetime import datetime
        now = datetime.now()

        logs = [
            {
                "level": "info",
                "message": f"Echo Brain {health.get('status', 'unknown')} - Uptime: {health.get('uptime_seconds', 0)/60:.1f}m",
                "timestamp": now.isoformat()
            },
            {
                "level": "info",
                "message": f"Vector DB: {health.get('vector_count', 'N/A')} vectors",
                "timestamp": now.isoformat()
            },
            {
                "level": "info",
                "message": "MCP Server: Port 8312",
                "timestamp": now.isoformat()
            }
        ]

        return {"logs": logs}

    except Exception as e:
        return {
            "logs": [{
                "level": "error",
                "message": f"Error loading logs: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }]
        }

@router.get("/pipeline/status")
async def get_pipeline_status():
    """Get training pipeline status from database"""
    try:
        # Query actual pipeline data from database
        # For now, return empty list until real training pipelines are active
        pipelines = []

        # Could add database query here like:
        # SELECT name, status, progress FROM training_pipelines WHERE active = true

        return {"pipelines": pipelines}
    except Exception as e:
        logger.error(f"Pipeline status error: {e}")
        return {"pipelines": []}


@router.get("/system/metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    try:
        import psutil
        import subprocess
        import json
        from pathlib import Path

        # Get basic system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Get actual vector DB count from Qdrant
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:6333/collections/echo_memory/",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        qdrant_data = await response.json()
                        vector_count = f"{qdrant_data.get('result', {}).get('points_count', 0):,}"
                    else:
                        vector_count = "N/A"
        except:
            vector_count = "N/A"

        # Check ingestion status
        ingestion_status = "Idle"
        try:
            # Check if ingestion process is running
            result = subprocess.run(['pgrep', '-f', 'ingest_everything|fast_ingest'],
                                  capture_output=True, text=True, timeout=2)
            if result.stdout.strip():
                ingestion_status = f"RUNNING (PID: {result.stdout.strip().split()[0]})"

            # Also check status file if exists
            status_file = Path("/tmp/echo_ingestion_status.json")
            if status_file.exists():
                with open(status_file) as f:
                    status_data = json.load(f)
                    if 'total' in status_data:
                        ingestion_status = f"Last run: {status_data.get('total', 0)} vectors"
        except:
            pass

        # Get uptime
        try:
            uptime_result = subprocess.run(['uptime', '-p'], capture_output=True, text=True, timeout=5)
            uptime = uptime_result.stdout.strip()
        except:
            uptime = "N/A"

        # Get GPU metrics
        gpu_info = "N/A"
        try:
            gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,nounits,noheader'],
                                      capture_output=True, text=True, timeout=5)
            if gpu_result.returncode == 0:
                gpu_data = gpu_result.stdout.strip().split(',')
                if len(gpu_data) >= 3:
                    gpu_util, gpu_mem_used, gpu_mem_total = [x.strip() for x in gpu_data]
                    gpu_info = f"{gpu_util}% (Memory: {gpu_mem_used}/{gpu_mem_total}MB)"
        except:
            pass

        return {
            "cpu_percent": f"{cpu_percent:.1f}%",
            "memory_percent": f"{memory.percent:.1f}%",
            "memory_used_gb": f"{memory.used / (1024**3):.1f}GB",
            "memory_total_gb": f"{memory.total / (1024**3):.1f}GB",
            "disk_percent": f"{disk.percent:.1f}%",
            "disk_free_gb": f"{disk.free / (1024**3):.1f}GB",
            "gpu_info": gpu_info,
            "vector_count": vector_count,
            "ingestion_status": ingestion_status,
            "uptime": uptime,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "cpu_percent": "N/A",
            "memory_percent": "N/A",
            "disk_percent": "N/A",
            "gpu_percent": "N/A",
            "vector_count": "N/A",
            "uptime": "N/A"
        }

@router.post("/logs/service")
async def get_service_logs(request: dict):
    """Get systemd service logs"""
    try:
        import subprocess
        service = request.get("service", "tower-echo-brain")
        lines = request.get("lines", 20)

        result = subprocess.run([
            'sudo', 'journalctl', '-u', service, '-n', str(lines), '--no-pager'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            logs = result.stdout.strip().split('\n')[-lines:]
            return {"logs": logs}
        else:
            return {"logs": [f"Error getting logs for {service}"]}

    except Exception as e:
        return {"logs": [f"Service log error: {str(e)}"]}

@router.post("/actions/restart-echo-brain")
async def restart_echo_brain():
    """Restart Echo Brain service"""
    try:
        import subprocess
        result = subprocess.run(['sudo', 'systemctl', 'restart', 'tower-echo-brain'],
                              capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            return {"message": "Echo Brain restart initiated"}
        else:
            return {"message": f"Restart failed: {result.stderr}"}
    except Exception as e:
        return {"message": f"Restart error: {str(e)}"}

@router.post("/actions/clear-cache")
async def clear_cache():
    """Clear system caches"""
    try:
        import subprocess
        # Clear Python cache
        subprocess.run(['find', '/opt/tower-echo-brain', '-name', '*.pyc', '-delete'], timeout=10)
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        return {"message": f"Cache clear error: {str(e)}"}

@router.post("/actions/sync-models")
async def sync_models():
    """Sync model manifests"""
    try:
        import subprocess
        result = subprocess.run(['tower-models', 'sync-manifests'],
                              capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            return {"message": "Model sync completed"}
        else:
            return {"message": f"Sync failed: {result.stderr}"}
    except Exception as e:
        return {"message": f"Model sync error: {str(e)}"}

@router.get("/errors/recent")
async def get_recent_errors():
    """Get recent error logs from system services"""
    try:
        import subprocess

        # Get recent error logs from Echo Brain service
        result = subprocess.run([
            'sudo', 'journalctl', '-u', 'tower-echo-brain',
            '--since', '1 hour ago', '-p', 'err', '--no-pager'
        ], capture_output=True, text=True, timeout=10)

        error_logs = []
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:  # Last 10 errors
                if line.strip():
                    error_logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "service": "tower-echo-brain",
                        "message": line.strip()
                    })

        return {"errors": error_logs}
    except Exception as e:
        return {"errors": [{"message": f"Failed to get error logs: {str(e)}"}]}