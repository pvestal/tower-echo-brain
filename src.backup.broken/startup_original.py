#!/usr/bin/env python3
"""
Startup initialization for Echo Brain
"""
import os
import asyncio
import logging
from src.db.database import database
from src.tasks.task_queue import TaskQueue
from tasks import BackgroundWorker
from tasks import AutonomousBehaviors
from src.board_integration import BoardIntegration
from src.routing.service_registry import ServiceRegistry
from src.routing.request_logger import RequestLogger
from src.routing.knowledge_manager import create_simple_knowledge_manager
from src.services.service_mesh import get_service_mesh, cleanup_service_mesh
from consciousness_integration import initialize_consciousness

logger = logging.getLogger(__name__)

class EchoBrainStartup:
    """Handles Echo Brain service initialization"""

    def __init__(self):
        self.task_queue = None
        self.background_worker = None
        self.autonomous_behaviors = None
        self.board_integration = None
        self.service_registry = None
        self.request_logger = None
        self.knowledge_manager = None
        self.service_mesh = None
        self.consciousness = None

    async def initialize_services(self):
        """Initialize all Echo Brain services"""
        try:
            # Initialize database
            await database.create_tables_if_needed()
            logger.info("✅ Database initialized")

            # Initialize service registry
            self.service_registry = ServiceRegistry()
            logger.info("✅ Service registry initialized")

            # Initialize request logger
            self.request_logger = RequestLogger()
            logger.info("✅ Request logger initialized")

            # Initialize knowledge manager
            self.knowledge_manager = create_simple_knowledge_manager(database.db_config)
            logger.info("✅ Knowledge manager initialized")

            # Initialize task queue
            self.task_queue = TaskQueue()
            logger.info("✅ Task queue initialized")

            # Initialize background worker
            # Initialize autonomous behaviors
            self.autonomous_behaviors = AutonomousBehaviors(self.task_queue)
            logger.info("✅ Autonomous behaviors initialized")
            self.background_worker = BackgroundWorker(self.task_queue)
            logger.info("✅ Background worker initialized")

            # Initialize board integration
            self.board_integration = BoardIntegration()
            logger.info("✅ Board integration initialized")

            # Initialize service mesh
            self.service_mesh = await get_service_mesh()
            logger.info("✅ Service mesh initialized")
            # Initialize consciousness
            if os.environ.get("ENABLE_INTERNAL_THOUGHTS", "false").lower() == "true":
                self.consciousness = await initialize_consciousness()
                logger.info("🧠 Consciousness activated - internal thoughts enabled")
            else:
                logger.info("💭 Consciousness dormant (set ENABLE_INTERNAL_THOUGHTS=true to awaken)")

            # Start background services
            await self._start_background_services()

        except Exception as e:
            logger.error(f"❌ Startup initialization failed: {e}")
            raise

    async def _start_background_services(self):
        """Start background services"""
        try:
            # Start background worker
            
            # Start autonomous behaviors
            if os.environ.get("ENABLE_AUTONOMOUS_BEHAVIORS", "true").lower() == "true":
                asyncio.create_task(self.autonomous_behaviors.start())
                logger.info("🤖 Autonomous behaviors started")
            else:
                logger.info("⚠️ Autonomous behaviors disabled")
            asyncio.create_task(self.background_worker.start())
            logger.info("🔄 Background worker started")

        except Exception as e:
            logger.error(f"❌ Background service startup failed: {e}")

    async def shutdown(self):
        """Graceful shutdown of services"""
        try:
            if self.autonomous_behaviors:
                await self.autonomous_behaviors.stop()
                logger.info("🛑 Autonomous behaviors stopped")

            if self.background_worker:
                await self.background_worker.stop()
                logger.info("🛑 Background worker stopped")


            if self.service_mesh:
                await cleanup_service_mesh()
                logger.info("🛑 Service mesh stopped")

            if self.database:
                await database.close()
                logger.info("🛑 Database connection closed")

        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

# Global startup instance
startup = EchoBrainStartup()