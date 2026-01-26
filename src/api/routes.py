#!/usr/bin/env python3
"""
Refactored API routes for Echo Brain system - Main router
"""
import logging
from fastapi import APIRouter

# Import domain-specific routers
from src.api.health import router as health_router
from src.api.echo import router as echo_router
from src.api.models import router as models_router
from src.api.lora_training import router as lora_router
# TODO: Fix legacy imports after restructuring
# from src.api.legacy.multimedia import router as multimedia_router
from src.services.testing import router as testing_router
# from src.api.legacy.tower import router as tower_router

logger = logging.getLogger(__name__)

# Create main router
router = APIRouter()

# Include all domain routers
router.include_router(health_router, tags=["health"])
router.include_router(echo_router, tags=["echo"])
router.include_router(models_router, tags=["models"])
router.include_router(lora_router, tags=["lora-training"])
# router.include_router(multimedia_router, tags=["multimedia"])  # TODO: Fix after restructuring
router.include_router(testing_router, tags=["testing"])
# router.include_router(tower_router, tags=["tower"])  # TODO: Fix after restructuring

logger.info("✅ Routes refactored into domain modules: health, echo, models, multimedia, testing, tower")