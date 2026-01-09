#!/usr/bin/env python3
"""Main Refactored Echo Brain Service"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

# Import all components
from src.managers.configuration_manager import ConfigurationManager
from src.processors.input_processor import InputProcessor, InputType
from src.generators.output_generator import OutputGenerator
from src.components.dependency_container import DependencyContainer
from src.components.error_handler import ErrorHandler
from src.components.logging_system import StructuredLogger
from src.components.echo_orchestrator import EchoOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Echo Brain Refactored",
    description="Modular Echo Brain with clean architecture",
    version="2.0.0"
)

# Global components
config_manager = None
orchestrator = None

class ProcessRequest(BaseModel):
    content: Any
    input_type: str = "chat_message"

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global config_manager, orchestrator

    try:
        # Initialize configuration
        config_manager = ConfigurationManager()
        logger.info("Configuration manager initialized")

        # Initialize components
        input_processor = InputProcessor(config_manager)
        output_generator = OutputGenerator(config_manager)
        error_handler = ErrorHandler(config_manager)
        structured_logger = StructuredLogger('echo_brain', config_manager)

        # Create orchestrator
        orchestrator = EchoOrchestrator(
            input_processor=input_processor,
            output_generator=output_generator,
            config_manager=config_manager,
            error_handler=error_handler,
            logger=structured_logger
        )

        await orchestrator.initialize()
        logger.info("Echo Brain Refactored initialized successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Echo Brain Refactored",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "config_manager": config_manager is not None,
            "orchestrator": orchestrator is not None and orchestrator.initialized
        }
    }

@app.post("/process")
async def process(request: ProcessRequest):
    """Process input through the Echo Brain pipeline"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await orchestrator.process_request(request.content, request.input_type)

    if not result['success']:
        raise HTTPException(status_code=500, detail=result.get('error', 'Processing failed'))

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8309)
