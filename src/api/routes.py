#!/usr/bin/env python3
"""
Refactored API routes for Echo Brain system - Main router
"""
import logging
from fastapi import APIRouter

# Import domain-specific routers
from .health import router as health_router
from .echo import router as echo_router
from .models import router as models_router
from .multimedia import router as multimedia_router
from .testing import router as testing_router
from .tower import router as tower_router

logger = logging.getLogger(__name__)

# Create main router
router = APIRouter()

# Include all domain routers
router.include_router(health_router, tags=["health"])
router.include_router(echo_router, tags=["echo"])
router.include_router(models_router, tags=["models"])
router.include_router(multimedia_router, tags=["multimedia"])
router.include_router(testing_router, tags=["testing"])
router.include_router(tower_router, tags=["tower"])

logger.info("✅ Routes refactored into domain modules: health, echo, models, multimedia, testing, tower")