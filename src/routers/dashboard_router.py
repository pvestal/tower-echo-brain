#!/usr/bin/env python3
"""
Dashboard API Router
Provides server-side endpoints for the Echo Brain dashboard
"""

import logging
import asyncio
import aiohttp
from typing import Dict, Any, List
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
    """Get training pipeline status (mock data for now)"""
    pipelines = [
        {"name": "Character LoRA Training", "status": "running", "progress": "45%"},
        {"name": "Voice Model Training", "status": "queued", "progress": "Waiting"},
        {"name": "Style Transfer", "status": "complete", "progress": "100%"}
    ]

    return {"pipelines": pipelines}

@router.get("/approvals/pending")
async def get_pending_approvals():
    """Get pending approvals (mock data for now)"""
    approvals = [
        {"task": "Dataset Review #42", "type": "urgent", "time": "2h ago"},
        {"task": "Model Quality Check", "type": "normal", "time": "4h ago"},
        {"task": "LoRA Validation", "type": "normal", "time": "1d ago"}
    ]

    return {"approvals": approvals}