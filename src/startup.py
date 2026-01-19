#!/usr/bin/env python3
"""
Startup initialization for Echo Brain
"""
import asyncio
import logging
from src.db.database import database
from src.tasks.task_queue import TaskQueue
from src.tasks import BackgroundWorker, get_autonomous_behaviors
from src.board_integration import BoardIntegration
from src.routing.service_registry import ServiceRegistry
from src.routing.request_logger import RequestLogger
from src.routing.knowledge_manager import create_simple_knowledge_manager

# ADDED: Autonomous system imports
from src.capabilities.autonomous_loop import AutonomousEventLoop
from src.capabilities.capability_registry import CapabilityRegistry
from src.capabilities.echo_capability_coordinator import initialize_coordinator
from src.orchestrators.resilient_orchestrator import initialize_orchestrator

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

        # ADDED: Autonomous system components
        self.capability_registry = None
        self.autonomous_loop = None
        self.resilient_orchestrator = None
        self.capability_coordinator = None

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

            # Initialize database connection pool FIRST
            try:
                from src.db.connection_pool import initialize_database
                await initialize_database()
                logger.info("✅ Database connection pool initialized")
            except Exception as e:
                logger.error(f"❌ Database connection pool failed: {e}")
                # Continue anyway - some services may work without database

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

            # Initialize knowledge manager (non-critical - don't fail startup)
            try:
                self.knowledge_manager = create_simple_knowledge_manager(database.db_config)
                logger.info("✅ Knowledge manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Knowledge manager initialization failed (non-critical): {e}")
                self.knowledge_manager = None

            # Initialize persona trainer (non-critical)
            try:
                self.persona_trainer = PersonaTrainer()
                await self.persona_trainer.initialize()
                logger.info("✅ Persona trainer initialized")
            except Exception as e:
                logger.warning(f"⚠️ Persona trainer failed (non-critical): {e}")
                self.persona_trainer = None

            # Initialize task queue (non-critical)
            try:
                self.task_queue = TaskQueue()
                logger.info("✅ Task queue initialized")
            except Exception as e:
                logger.warning(f"⚠️ Task queue failed (non-critical): {e}")
                self.task_queue = None

            # Initialize background worker (non-critical)
            try:
                if self.task_queue:
                    self.background_worker = BackgroundWorker(self.task_queue)
                    logger.info("✅ Background worker initialized")
                else:
                    self.background_worker = None
            except Exception as e:
                logger.warning(f"⚠️ Background worker failed (non-critical): {e}")
                self.background_worker = None

            # Initialize autonomous behaviors (non-critical)
            try:
                if self.task_queue:
                    AutonomousBehaviors = get_autonomous_behaviors()
                    self.autonomous_behaviors = AutonomousBehaviors(self.task_queue)
                    logger.info("✅ Autonomous behaviors initialized")
                else:
                    self.autonomous_behaviors = None
            except Exception as e:
                logger.warning(f"⚠️ Autonomous behaviors failed (non-critical): {e}")
                self.autonomous_behaviors = None

            # Initialize board integration (non-critical)
            try:
                self.board_integration = BoardIntegration()
                logger.info("✅ Board integration initialized")
            except Exception as e:
                logger.warning(f"⚠️ Board integration failed (non-critical): {e}")
                self.board_integration = None

            # ADDED: Initialize autonomous systems
            await self._initialize_autonomous_systems()

            # Start background services
            await self._start_background_services()

        except Exception as e:
            logger.error(f"❌ Startup initialization failed: {e}")
            raise

    async def _initialize_autonomous_systems(self):
        """Initialize autonomous system components"""
        try:
            logger.info("🚀 STARTING AUTONOMOUS SYSTEMS INITIALIZATION...")

            # Initialize capability registry
            logger.info("📝 Creating capability registry...")
            self.capability_registry = CapabilityRegistry()
            logger.info(f"✅ Capability registry created: {self.capability_registry}")

            # Register basic capabilities
            logger.info("📝 Registering core capabilities...")
            await self._register_core_capabilities()
            capabilities = self.capability_registry.list_capabilities()
            logger.info(f"🧠 Capability registry initialized with {len(capabilities)} core capabilities")
            for cap in capabilities:
                logger.info(f"  - {cap.name}: {cap.status.value}")

            # Initialize resilient orchestrator
            logger.info("📝 Initializing resilient orchestrator...")
            self.resilient_orchestrator = initialize_orchestrator()
            await self.resilient_orchestrator.start_background_tasks()
            logger.info("🎭 Resilient orchestrator initialized and monitoring started")

            # Initialize autonomous event loop
            logger.info("📝 Creating autonomous event loop...")
            self.autonomous_loop = AutonomousEventLoop(self.capability_registry)
            logger.info("🔄 Autonomous event loop initialized")

            # Initialize capability coordinator to bridge Echo chat to capabilities
            logger.info("📝 Initializing capability coordinator...")
            self.capability_coordinator = initialize_coordinator(self.capability_registry)
            logger.info(f"🎯 Capability coordinator connected: {self.capability_coordinator}")

            # Log supported actions
            if self.capability_coordinator:
                actions = self.capability_coordinator.get_supported_actions()
                logger.info(f"📋 Supported action patterns: {actions}")

            logger.info("✅ AUTONOMOUS SYSTEMS INITIALIZATION COMPLETE!")

        except Exception as e:
            logger.error(f"❌ Autonomous system initialization failed: {e}", exc_info=True)
            # Don't fail startup, just disable autonomous features
            self.capability_registry = None
            self.autonomous_loop = None
            self.capability_coordinator = None

    async def _register_core_capabilities(self):
        """Register core Echo Brain capabilities"""
        from src.capabilities.capability_registry import CapabilityType

        # Register repair system capability
        self.capability_registry.register_capability(
            name="autonomous_repair",
            capability_type=CapabilityType.SELF_MODIFICATION,
            description="Autonomous system repair and service restart",
            handler=self._handle_autonomous_repair,
            requirements=["module:subprocess", "service:systemctl"],
            permissions=["system_restart", "service_control"]
        )

        # Register code analysis capability
        self.capability_registry.register_capability(
            name="code_analysis",
            capability_type=CapabilityType.ANALYSIS,
            description="Code analysis and improvement suggestions",
            handler=self._handle_code_analysis,
            requirements=["module:ast", "module:subprocess"],
            permissions=["file_read", "code_execution"]
        )

        # Register service monitoring capability
        self.capability_registry.register_capability(
            name="service_monitoring",
            capability_type=CapabilityType.ANALYSIS,
            description="Monitor Tower services health",
            handler=self._handle_service_monitoring,
            requirements=["module:aiohttp"],
            permissions=["network_access"]
        )

        # Register notification capability
        self.capability_registry.register_capability(
            name="send_notification",
            capability_type=CapabilityType.COMMUNICATION,
            description="Send notifications via various channels",
            handler=self._handle_send_notification,
            requirements=["module:aiohttp"],
            permissions=["network_access"]
        )

    async def _handle_autonomous_repair(self, **kwargs):
        """Handle autonomous repair requests"""
        try:
            import subprocess
            service_name = kwargs.get("service_name")
            issue_type = kwargs.get("issue_type", "restart")

            if service_name:
                # Execute service restart
                logger.info(f"🔧 Executing repair: {issue_type} for {service_name}")

                if issue_type == "restart":
                    result = subprocess.run(
                        ["sudo", "systemctl", "restart", service_name],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode == 0:
                        logger.info(f"✅ Successfully restarted {service_name}")
                        return {
                            "success": True,
                            "service": service_name,
                            "action": "restart",
                            "message": f"Service {service_name} has been restarted successfully"
                        }
                    else:
                        logger.error(f"❌ Failed to restart {service_name}: {result.stderr}")
                        return {
                            "success": False,
                            "service": service_name,
                            "error": result.stderr or "Unknown error"
                        }

                return {"success": False, "error": f"Unknown issue type: {issue_type}"}
            else:
                # Diagnose system
                services = ["tower-dashboard", "tower-auth", "tower-echo-brain", "tower-anime-production"]
                status = {}

                for service in services:
                    result = subprocess.run(
                        ["systemctl", "is-active", service],
                        capture_output=True,
                        text=True
                    )
                    status[service] = result.stdout.strip()

                return {
                    "success": True,
                    "system_status": status,
                    "message": "System diagnosis complete"
                }

        except Exception as e:
            logger.error(f"❌ Repair execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_code_analysis(self, **kwargs):
        """Handle code analysis requests"""
        try:
            from src.agents.deepseek_coding_agent import DeepSeekCodingAgent
            agent = DeepSeekCodingAgent()

            file_path = kwargs.get("file_path")
            if file_path:
                from pathlib import Path
                return await agent.analyze_codebase(Path(file_path))
            else:
                return await agent.analyze_codebase()

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_service_monitoring(self, **kwargs):
        """Handle service monitoring requests"""
        try:
            if self.resilient_orchestrator:
                return await self.resilient_orchestrator.check_all_services()
            else:
                return {"success": False, "error": "Orchestrator not available"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_send_notification(self, **kwargs):
        """Handle notification sending requests"""
        try:
            import aiohttp

            notification_data = {
                "message": kwargs.get("message", "Echo Brain notification"),
                "title": kwargs.get("title", "System Alert"),
                "channel": kwargs.get("channel", "ntfy"),
                "priority": kwargs.get("priority", 3)
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8309/api/notifications/send",
                    json=notification_data
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _start_background_services(self):
        """Start background services"""
        try:
            # Start background worker
            asyncio.create_task(self.background_worker.start())
            logger.info("🔄 Background worker started")

            # Start autonomous behaviors - DISABLED due to blocking issues
            # asyncio.create_task(self.autonomous_behaviors.start())
            logger.info("🤖 Autonomous behaviors DISABLED - preventing startup blocking")

            # Start persona trainer learning loop (if available)
            if self.persona_trainer:
                asyncio.create_task(self.persona_trainer.autonomous_self_improvement())
                logger.info("🧠 Persona trainer learning loop STARTED")
            else:
                logger.info("⚠️ Persona trainer not available - skipping learning loop")

            # DISABLED: Autonomous loop was stuck trying to restart non-existent services
            # if self.autonomous_loop:
            #     asyncio.create_task(self.autonomous_loop.start())
            #     logger.info("🔄 Autonomous event loop STARTED")

        except Exception as e:
            logger.error(f"❌ Background service startup failed: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown of services"""
        try:
            # ADDED: Shutdown autonomous systems first
            if self.autonomous_loop:
                await self.autonomous_loop.stop()
                logger.info("🛑 Autonomous event loop stopped")

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