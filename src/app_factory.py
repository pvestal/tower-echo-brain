#!/usr/bin/env python3
"""
Application factory for Echo Brain
"""
import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Core routers
from src.api.routes import router as main_router
from src.api.system_metrics import router as system_metrics_router
from src.api.enhanced_system_metrics import router as enhanced_system_metrics_router
from src.api.neural_metrics import router as neural_metrics_router
from src.api.learning_routes import router as learning_router
from src.api.autonomous_routes import router as autonomous_router
from src.api.coordination_routes import router as coordination_router
from src.api.integration_testing_routes import integration_router
from src.api.task_routes import router as task_router
from src.photo_comparison import router as photo_router

# External integrations
from agent_development_endpoints import agent_dev_router
from veteran_guardian_endpoints import veteran_router
from telegram_general_chat import general_telegram_router
from telegram_integration import telegram_router

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    # Load environment variables
    load_dotenv()

    # Create FastAPI app
    app = FastAPI(
        title="Echo Brain",
        description="Advanced AI orchestrator with modular architecture",
        version="2.0.0"
    )

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Include routers
    app.include_router(main_router, prefix="", tags=["main"])
    app.include_router(system_metrics_router, prefix="", tags=["metrics"])
    app.include_router(enhanced_system_metrics_router, prefix="", tags=["enhanced-metrics"])
    app.include_router(neural_metrics_router, prefix="", tags=["neural"])
    app.include_router(learning_router, prefix="", tags=["learning"])
    app.include_router(autonomous_router, prefix="", tags=["autonomous"])
    app.include_router(coordination_router, prefix="", tags=["coordination"])
    app.include_router(integration_router, prefix="", tags=["testing"])
    app.include_router(task_router, prefix="", tags=["tasks"])
    app.include_router(photo_router, prefix="", tags=["vision"])

    # External routers
    app.include_router(agent_dev_router, prefix="", tags=["agents"])
    app.include_router(veteran_router, prefix="", tags=["veteran"])
    app.include_router(general_telegram_router, prefix="", tags=["telegram"])
    app.include_router(telegram_router, prefix="", tags=["telegram"])

    # Static files
    static_dir = "/opt/tower-echo-brain/static"
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app