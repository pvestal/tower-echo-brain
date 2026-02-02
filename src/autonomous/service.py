#!/usr/bin/env python3
"""
Autonomous Core Service Script for Echo Brain

Standalone service that runs the AutonomousCore in a production environment.
Handles graceful startup/shutdown, signal handling, and comprehensive logging.
"""

import sys
import os
import signal
import asyncio
import logging
import traceback
from typing import Optional
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, '/opt/tower-echo-brain')

from src.autonomous import AutonomousCore, AutonomousState


class AutonomousService:
    """
    Service wrapper for the AutonomousCore that handles system lifecycle,
    signal handling, and service management functionality.
    """

    def __init__(self):
        """Initialize the autonomous service."""
        self.core: Optional[AutonomousCore] = None
        self.running = False
        self.shutdown_requested = False

        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Register signal handlers
        self.setup_signal_handlers()

        # Service configuration
        self.config = {
            'cycle_interval': int(os.environ.get('AUTONOMOUS_CYCLE_INTERVAL', '30')),
            'max_concurrent_tasks': int(os.environ.get('MAX_CONCURRENT_TASKS', '3')),
            'max_tasks_per_minute': int(os.environ.get('MAX_TASKS_PER_MINUTE', '10')),
            'max_tasks_per_hour': int(os.environ.get('MAX_TASKS_PER_HOUR', '100')),
            'check_kill_switch': os.environ.get('CHECK_KILL_SWITCH', 'true').lower() == 'true',
            'kill_switch_file': os.environ.get('KILL_SWITCH_FILE', '/tmp/echo_brain_kill_switch')
        }

    def setup_logging(self):
        """Setup comprehensive logging for the service."""
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Set specific log levels for different components
        logging.getLogger('asyncpg').setLevel(logging.WARNING)
        logging.getLogger('qdrant_client').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

        # Autonomous components should be more verbose
        logging.getLogger('src.autonomous').setLevel(logging.DEBUG)

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            self.logger.info(f"Received signal {signal_name} ({signum})")

            if signum in (signal.SIGTERM, signal.SIGINT):
                self.logger.info("Initiating graceful shutdown...")
                self.shutdown_requested = True
            elif signum == signal.SIGHUP:
                self.logger.info("Received SIGHUP - reloading configuration...")
                # Could implement config reload here
            elif signum == signal.SIGUSR1:
                self.logger.info("Received SIGUSR1 - dumping status...")
                asyncio.create_task(self.dump_status())

        # Register handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)

    async def dump_status(self):
        """Dump current system status to logs."""
        if self.core:
            try:
                status = await self.core.get_status()
                self.logger.info(f"System Status Dump:")
                self.logger.info(f"  State: {status.state.value}")
                self.logger.info(f"  Uptime: {status.uptime_seconds:.1f}s")
                self.logger.info(f"  Cycles: {status.cycles_completed}")
                self.logger.info(f"  Tasks Executed: {status.tasks_executed}")
                self.logger.info(f"  Last Cycle: {status.last_cycle_time}")
                self.logger.info(f"  Components: {status.components_status}")
                if status.last_error:
                    self.logger.info(f"  Last Error: {status.last_error}")
            except Exception as e:
                self.logger.error(f"Failed to dump status: {e}")

    async def initialize(self) -> bool:
        """Initialize the autonomous core and all components."""
        try:
            self.logger.info("Initializing Autonomous Core Service...")
            self.logger.info(f"Configuration: {self.config}")

            # Create autonomous core with configuration
            self.core = AutonomousCore(config=self.config)

            # Initialize the core
            self.logger.info("Initializing autonomous core components...")
            await self.core.initialize()

            self.logger.info("✅ Autonomous Core Service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Autonomous Core: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    async def run(self):
        """Main service run loop."""
        self.running = True
        self.logger.info("🚀 Starting Autonomous Core Service...")

        try:
            # Initialize the service
            if not await self.initialize():
                self.logger.error("Failed to initialize - shutting down")
                return 1

            # Start the autonomous core
            self.logger.info("Starting autonomous operations...")
            await self.core.start()

            # Main service loop
            self.logger.info("🔄 Entering main service loop...")
            while self.running and not self.shutdown_requested:
                try:
                    # Check core status
                    if self.core.state == AutonomousState.ERROR:
                        self.logger.error("Core is in error state - attempting restart...")
                        await self.core.stop()
                        await asyncio.sleep(5)
                        await self.core.start()

                    # Sleep between checks
                    await asyncio.sleep(5)

                except asyncio.CancelledError:
                    self.logger.info("Main loop cancelled")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    await asyncio.sleep(10)  # Wait before retrying

        except Exception as e:
            self.logger.error(f"Critical error in service run: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return 1

        finally:
            await self.shutdown()

        self.logger.info("🛑 Autonomous Core Service stopped")
        return 0

    async def shutdown(self):
        """Graceful shutdown of the service."""
        self.logger.info("🔄 Shutting down Autonomous Core Service...")
        self.running = False

        if self.core:
            try:
                # Stop autonomous operations
                self.logger.info("Stopping autonomous operations...")
                await self.core.stop()

                # Close database connections and cleanup
                self.logger.info("Cleaning up resources...")
                await self.core.cleanup()

            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")

        self.logger.info("✅ Shutdown complete")

    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import asyncpg
            import qdrant_client
            self.logger.info("✅ Required dependencies available")
            return True
        except ImportError as e:
            self.logger.error(f"❌ Missing required dependency: {e}")
            return False

    def validate_environment(self) -> bool:
        """Validate required environment variables and configuration."""
        required_vars = [
            'ECHO_BRAIN_DB_PASSWORD'
        ]

        missing_vars = []
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)

        if missing_vars:
            self.logger.error(f"❌ Missing required environment variables: {missing_vars}")
            return False

        # Check database password specifically
        db_password = os.environ.get('ECHO_BRAIN_DB_PASSWORD')
        if not db_password or len(db_password) < 8:
            self.logger.error("❌ Invalid database password")
            return False

        self.logger.info("✅ Environment validation passed")
        return True


async def main():
    """Main entry point for the autonomous service."""
    service = AutonomousService()

    # Pre-flight checks
    if not service.check_dependencies():
        print("❌ Dependency check failed", file=sys.stderr)
        return 1

    if not service.validate_environment():
        print("❌ Environment validation failed", file=sys.stderr)
        return 1

    # Run the service
    try:
        return await service.run()
    except KeyboardInterrupt:
        service.logger.info("Service interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Service failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Set process title for easier identification
    try:
        import setproctitle
        setproctitle.setproctitle("tower-autonomous")
    except ImportError:
        pass

    # Run the service
    exit_code = asyncio.run(main())
    sys.exit(exit_code)