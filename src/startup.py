#!/usr/bin/env python3
"""
Startup initialization for Echo Brain
"""
import asyncio
import logging
from src.db.database import database
from src.tasks.task_queue import TaskQueue
from src.tasks import BackgroundWorker, AutonomousBehaviors
from src.board_integration import BoardIntegration
from src.routing.service_registry import ServiceRegistry
from src.routing.request_logger import RequestLogger
from src.routing.knowledge_manager import create_simple_knowledge_manager


from src.tasks.persona_trainer import PersonaTrainer
from src.integrations.vault_manager import get_vault_manager
from src.core.echo_identity import get_echo_identity
from src.core.user_context_manager import get_user_context_manager

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
        self.persona_trainer = None
        self.vault_manager = None
        self.echo_identity = None
        self.user_context_manager = None

    async def initialize_services(self):
        """Initialize all Echo Brain services"""
        try:
            # Install memory wrapper for Ollama FIRST
            try:
                from src.ollama_memory_wrapper import install_memory_wrapper
                install_memory_wrapper()
                logger.info("✅ Memory augmentation wrapper installed")
            except Exception as e:
                logger.warning(f"Could not install memory wrapper: {e}")

            # Initialize database
            await database.create_tables_if_needed()
            logger.info("✅ Database initialized")

            # Initialize Vault manager for credentials
            self.vault_manager = await get_vault_manager()
            if self.vault_manager.is_initialized:
                logger.info("🔐 Vault manager initialized with credentials")
            else:
                logger.warning("⚠️ Vault manager running without Vault connection")

            # Initialize Echo identity system
            self.echo_identity = get_echo_identity()
            logger.info("🧠 Echo identity system initialized")

            # Initialize user context manager
            self.user_context_manager = await get_user_context_manager()
            logger.info("👥 User context manager initialized")

            # Initialize service registry
            self.service_registry = ServiceRegistry()
            logger.info("✅ Service registry initialized")

            # Initialize request logger
            self.request_logger = RequestLogger()
            logger.info("✅ Request logger initialized")

            # Initialize knowledge manager
            self.knowledge_manager = create_simple_knowledge_manager(database.db_config)
            logger.info("✅ Knowledge manager initialized")

            # Initialize persona trainer
            self.persona_trainer = PersonaTrainer()
            await self.persona_trainer.initialize()
            logger.info("✅ Persona trainer initialized")

            # Initialize task queue
            self.task_queue = TaskQueue()
            logger.info("✅ Task queue initialized")

            # Initialize background worker
            self.background_worker = BackgroundWorker(self.task_queue)
            logger.info("✅ Background worker initialized")

            # Initialize autonomous behaviors
            self.autonomous_behaviors = AutonomousBehaviors(self.task_queue)
            logger.info("✅ Autonomous behaviors initialized")

            # Initialize board integration
            self.board_integration = BoardIntegration()
            logger.info("✅ Board integration initialized")

            # Start background services
            await self._start_background_services()

        except Exception as e:
            logger.error(f"❌ Startup initialization failed: {e}")
            raise

    async def _start_background_services(self):
        """Start background services"""
        try:
            # Start background worker
            asyncio.create_task(self.background_worker.start())
            logger.info("🔄 Background worker started")

            # Start autonomous behaviors - DISABLED due to blocking issues
            # asyncio.create_task(self.autonomous_behaviors.start())
            logger.info("🤖 Autonomous behaviors DISABLED - preventing startup blocking")

            # Start persona trainer learning loop
            asyncio.create_task(self.persona_trainer.autonomous_self_improvement())
            logger.info("🧠 Persona trainer learning loop STARTED")

        except Exception as e:
            logger.error(f"❌ Background service startup failed: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown of services"""
        try:
            if self.background_worker:
                await self.background_worker.stop()
                logger.info("🛑 Background worker stopped")

            if self.autonomous_behaviors:
                await self.autonomous_behaviors.stop()
                logger.info("🛑 Autonomous behaviors stopped")

            if self.database:
                await database.close()
                logger.info("🛑 Database connection closed")

        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

# Global startup instance
startup = EchoBrainStartup()