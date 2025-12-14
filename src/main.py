#!/usr/bin/env python3
"""
Echo Brain Main Entry Point - Refactored with Factory Pattern
"""
import uvicorn
import asyncio
import logging
from src.app_factory import create_app
from src.startup import startup

logger = logging.getLogger(__name__)

# Create application
app = create_app()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Echo Brain starting up...")
    # Run initialization in background to not block uvicorn
    asyncio.create_task(startup.initialize_services())
    logger.info("Echo Brain startup initiated")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Echo Brain shutting down...")
    await startup.shutdown()
    logger.info("Echo Brain shutdown complete")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8309,
        reload=False,
        log_level="info"
    )