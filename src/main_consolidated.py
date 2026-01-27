#!/usr/bin/env python3
"""
Echo Brain Consolidated API
Central entry point with domain-based routers
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Echo Brain API",
    description="Consolidated Echo Brain service with domain-based organization",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Import Routers =============
try:
    from src.routers.core_router import router as core_router
    from src.routers.production_router import router as production_router
    from src.routers.integration_router import router as integration_router
    from src.routers.system_router import router as system_router
    routers_loaded = True
    logger.info("All routers loaded successfully")
except Exception as e:
    logger.error(f"Failed to load routers: {e}")
    routers_loaded = False

# ============= Root Endpoints =============

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "service": "Echo Brain API",
        "version": "4.0.0",
        "status": "operational" if routers_loaded else "degraded",
        "documentation": "/docs",
        "health": "/health",
        "api_sections": {
            "core": {
                "description": "Core AI functionality - chat, models, agents",
                "endpoints": [
                    "/api/echo/chat",
                    "/api/echo/models",
                    "/api/echo/agents"
                ]
            },
            "production": {
                "description": "Content generation - anime, LoRA, video",
                "endpoints": [
                    "/api/anime/generate",
                    "/api/lora/train",
                    "/api/video/workflow"
                ]
            },
            "integration": {
                "description": "External services - Google, Vault, Knowledge",
                "endpoints": [
                    "/api/google/calendar",
                    "/api/vault/secret",
                    "/api/knowledge/search"
                ]
            },
            "system": {
                "description": "System operations - health, metrics, diagnostics",
                "endpoints": [
                    "/health",
                    "/metrics",
                    "/diagnostics"
                ]
            }
        },
        "timestamp": datetime.now().isoformat()
    }

# ============= Include Routers =============

if routers_loaded:
    # Core functionality (chat, models, agents)
    app.include_router(
        core_router,
        prefix="/api/echo",
        tags=["Core AI"]
    )

    # Production systems (anime, lora, video)
    app.include_router(
        production_router,
        prefix="/api",
        tags=["Production"]
    )

    # External integrations (google, vault, knowledge)
    app.include_router(
        integration_router,
        prefix="/api",
        tags=["Integration"]
    )

    # System operations (health, metrics, diagnostics)
    # No prefix for system endpoints - they should be at root level
    app.include_router(
        system_router,
        tags=["System"]
    )

# ============= Static Files =============

# Mount static files for the dashboard
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"✅ Static files mounted from {static_dir}")
else:
    logger.warning(f"⚠️ Static directory not found: {static_dir}")

# Add root dashboard endpoint
@app.get("/dashboard")
async def dashboard():
    """Redirect to dashboard"""
    return {"message": "Dashboard available at /static/dist/index.html"}

# ============= Error Handlers =============

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "path": str(request.url.path),
            "method": request.method,
            "documentation": "/docs",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Custom 500 handler"""
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Server error",
            "type": type(exc).__name__,
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ============= Middleware =============

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = datetime.now()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    try:
        response = await call_next(request)

        # Log response time
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Response: {response.status_code} in {duration:.3f}s")

        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)
        response.headers["X-Echo-Brain-Version"] = "4.0.0"

        return response

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Request failed after {duration:.3f}s: {e}")
        raise

# ============= Startup/Shutdown Events =============

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 60)
    logger.info("Echo Brain Consolidated API Starting...")
    logger.info("=" * 60)

    # Check critical services
    try:
        import httpx
        import asyncpg

        # Check database
        try:
            conn = await asyncpg.connect(
                host="localhost",
                database="echo_brain",
                user="patrick",
                password=os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"),
                timeout=5
            )
            await conn.close()
            logger.info("✅ Database: Connected")
        except Exception as e:
            logger.error(f"❌ Database: {e}")

        # Check Ollama
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    logger.info(f"✅ Ollama: {len(models)} models available")
            except:
                logger.warning("⚠️ Ollama: Not available")

        # Check Qdrant
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:6333/health", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ Qdrant: Connected")
            except:
                logger.warning("⚠️ Qdrant: Not available")

        # Check ComfyUI
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:8188/system_stats", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ ComfyUI: Connected")
            except:
                logger.warning("⚠️ ComfyUI: Not available")

    except Exception as e:
        logger.error(f"Startup checks failed: {e}")

    logger.info("=" * 60)
    logger.info(f"API Ready at http://0.0.0.0:8309")
    logger.info(f"Documentation at http://0.0.0.0:8309/docs")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Echo Brain API shutting down...")
    # Add any cleanup code here
    logger.info("Shutdown complete")

# ============= Main Entry Point =============

if __name__ == "__main__":
    uvicorn.run(
        "main_consolidated:app",
        host="0.0.0.0",
        port=8309,
        reload=True,
        log_level="info",
        access_log=True
    )