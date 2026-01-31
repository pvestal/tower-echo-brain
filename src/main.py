#!/usr/bin/env python3
"""
Tower Echo Brain - Main FastAPI Application
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add the src directory to the path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# Import routers
from routers.system_router import router as system_router
from routers.dashboard_router import router as dashboard_router
from routers.moltbook_router import router as moltbook_router

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Initialize FastAPI app
    app = FastAPI(
        title="Tower Echo Brain API",
        description="AI-powered document analysis and management system",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ============= API Routes =============
    
    # System monitoring endpoints
    # No prefix for system endpoints - they should be at root level
    app.include_router(
        system_router,
        tags=["System"]
    )
    
    # Dashboard API endpoints
    app.include_router(
        dashboard_router,
        tags=["Dashboard"]
    )
    
    # Moltbook API endpoints
    app.include_router(
        moltbook_router,
        prefix="/api/echo/moltbook",
        tags=["Moltbook"]
    )
    
    # ============= Static Files =============
    
    # Mount static files for the dashboard
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    return app

# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8309,
        reload=False,  # Set to True for development
        log_level="info"
    )
