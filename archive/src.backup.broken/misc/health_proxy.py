#!/usr/bin/env python3
"""
Universal Health Endpoint Proxy for Tower Services
Provides health endpoints for all services that don't have them
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import asyncio
import uvicorn
import psutil

app = FastAPI(title="Tower Health Proxy")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def check_port(port: int) -> bool:
    """Check if a service is listening on a port"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://127.0.0.1:{port}/", timeout=2)
            return True
    except:
        return False

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Tower Health Proxy"}

# Knowledge Base health
@app.get("/api/kb/health")
async def kb_health():
    try:
        async with httpx.AsyncClient() as client:
            # Try the actual KB health endpoint
            response = await client.get("http://127.0.0.1:8307/api/health", timeout=2)
            if response.status_code == 200:
                return response.json()
    except:
        pass

    # Fallback: check if port is open
    if await check_port(8307):
        return {"status": "healthy", "service": "Knowledge Base", "port": 8307}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "Knowledge Base"}
        )

# ComfyUI health
@app.get("/api/comfyui/health")
async def comfyui_health():
    try:
        async with httpx.AsyncClient() as client:
            # ComfyUI has a system_stats endpoint
            response = await client.get("http://127.0.0.1:8188/system_stats", timeout=2)
            if response.status_code == 200:
                stats = response.json()
                return {
                    "status": "healthy",
                    "service": "ComfyUI",
                    "system": stats.get("system", {}),
                    "devices": stats.get("devices", [])
                }
    except:
        pass

    # Check if port is open
    if await check_port(8188):
        return {"status": "healthy", "service": "ComfyUI", "port": 8188}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "ComfyUI"}
        )

# Anime service health
@app.get("/api/anime/health")
async def anime_health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:8328/api/health", timeout=2)
            if response.status_code == 200:
                return response.json()
    except:
        pass

    # Check if port is open
    if await check_port(8328):
        return {"status": "healthy", "service": "Anime Generator", "port": 8328}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "Anime Generator"}
        )

# Music service health
@app.get("/api/music/health")
async def music_health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:8315/api/health", timeout=2)
            if response.status_code == 200:
                return response.json()
    except:
        pass

    # Check if port is open
    if await check_port(8315):
        return {"status": "healthy", "service": "Music Production", "port": 8315}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "Music Production"}
        )

# Voice service health
@app.get("/api/voice/health")
async def voice_health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:8312/api/health", timeout=2)
            if response.status_code == 200:
                return response.json()
    except:
        pass

    # Check if port is open
    if await check_port(8312):
        return {"status": "healthy", "service": "Voice Service", "port": 8312}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "Voice Service"}
        )

# Vault health
@app.get("/api/vault/health")
async def vault_health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:8200/v1/sys/health", timeout=2)
            data = response.json()
            return {
                "status": "healthy" if data.get("initialized") else "uninitialized",
                "service": "HashiCorp Vault",
                "sealed": data.get("sealed", True),
                "version": data.get("version", "unknown")
            }
    except:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "HashiCorp Vault"}
        )

# Combined status endpoint
@app.get("/api/services/status")
async def all_services_status():
    """Check all services at once"""
    services = {}

    # Check each service
    for service_name, check_func in [
        ("echo", lambda: check_port(8309)),
        ("kb", kb_health),
        ("comfyui", comfyui_health),
        ("anime", anime_health),
        ("music", music_health),
        ("voice", voice_health),
        ("vault", vault_health),
        ("evolution", lambda: check_port(8311))
    ]:
        try:
            if callable(check_func):
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                    services[service_name] = result if isinstance(result, dict) else {"status": "healthy" if result else "unhealthy"}
                else:
                    result = check_func()
                    services[service_name] = {"status": "healthy" if result else "unhealthy"}
        except:
            services[service_name] = {"status": "error"}

    # System resources
    services["system"] = {
        "cpu": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent
    }

    return services

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8310, log_level="info")