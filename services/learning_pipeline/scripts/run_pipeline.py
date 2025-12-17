#!/usr/bin/env python3
"""
Main execution script for Echo Brain Learning Pipeline.

This script replaces the broken cron job and provides the proper
entry point for scheduled learning pipeline execution.
"""

import asyncio
import sys
import logging
import signal
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.pipeline import LearningPipeline
from src.config.settings import load_config
from src.utils.health_check import perform_health_check
from src.utils.metrics import export_metrics


# Global pipeline instance for graceful shutdown
pipeline_instance: Optional[LearningPipeline] = None


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/opt/tower-echo-brain/logs/learning_pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce noise from external libraries
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")

    if pipeline_instance:
        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run cleanup
        loop.run_until_complete(pipeline_instance.cleanup())

    logger.info("Graceful shutdown completed")
    sys.exit(0)


async def run_health_check() -> bool:
    """
    Perform comprehensive health check before running pipeline.

    Returns:
        True if all systems healthy, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info("Performing pre-execution health check")

    try:
        health_status = await perform_health_check()

        if health_status['healthy']:
            logger.info("All systems healthy, proceeding with pipeline execution")
            return True
        else:
            logger.error("Health check failed:")
            for component, status in health_status['components'].items():
                if not status.get('healthy', True):
                    logger.error(f"  {component}: {status.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        logger.error(f"Health check encountered error: {e}")
        return False


async def run_pipeline(config_path: str, dry_run: bool = False) -> int:
    """
    Execute the learning pipeline.

    Args:
        config_path: Path to configuration file
        dry_run: If True, validate setup but don't process content

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    global pipeline_instance
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)

        # Perform health check
        if not await run_health_check():
            logger.error("Health check failed, aborting pipeline execution")
            return 1

        # Initialize pipeline
        logger.info("Initializing learning pipeline")
        pipeline_instance = LearningPipeline(config)
        await pipeline_instance.initialize()

        if dry_run:
            logger.info("Dry run mode - validating configuration and connectivity")
            status = await pipeline_instance.get_pipeline_status()
            logger.info(f"Pipeline validation successful: {status}")
            return 0

        # Execute learning cycle
        logger.info("Starting learning cycle execution")
        start_time = datetime.now()

        pipeline_run = await pipeline_instance.run_learning_cycle()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Log results
        logger.info(
            f"Learning cycle completed successfully in {duration:.2f} seconds. "
            f"Results: {pipeline_run.conversations_processed} conversations processed, "
            f"{pipeline_run.learning_items_extracted} learning items extracted, "
            f"{pipeline_run.vectors_updated} vectors updated, "
            f"{pipeline_run.errors_encountered} errors encountered"
        )

        # Export metrics for monitoring
        await export_metrics(pipeline_run, config)

        return 0

    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup
        if pipeline_instance:
            logger.info("Performing cleanup")
            await pipeline_instance.cleanup()


async def check_dependencies() -> bool:
    """
    Check if all required services and dependencies are available.

    Returns:
        True if dependencies are met, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info("Checking dependencies...")

    dependencies = {
        'PostgreSQL': {'host': 'localhost', 'port': 5432},
        'Qdrant': {'host': 'localhost', 'port': 6333},
        'Claude conversations directory': {'path': '/home/patrick/.claude/conversations'},
    }

    all_good = True

    for service, config in dependencies.items():
        try:
            if 'path' in config:
                # Check file system dependency
                if not Path(config['path']).exists():
                    logger.error(f"{service}: Path {config['path']} does not exist")
                    all_good = False
                else:
                    logger.info(f"{service}: ✓")
            else:
                # Check network service
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((config['host'], config['port']))
                sock.close()

                if result == 0:
                    logger.info(f"{service}: ✓")
                else:
                    logger.error(f"{service}: Connection failed to {config['host']}:{config['port']}")
                    all_good = False

        except Exception as e:
            logger.error(f"{service}: Error checking dependency - {e}")
            all_good = False

    return all_good


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Echo Brain Learning Pipeline")

    parser.add_argument(
        '--config', '-c',
        default='/opt/tower-echo-brain/services/learning_pipeline/config/production.yaml',
        help='Configuration file path'
    )

    parser.add_argument(
        '--log-level', '-l',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Validate configuration without processing content'
    )

    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )

    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Perform health check and exit'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Echo Brain Learning Pipeline starting...")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Log level: {args.log_level}")

    # Handle special modes
    if args.check_deps:
        logger.info("Checking dependencies...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        deps_ok = loop.run_until_complete(check_dependencies())
        return 0 if deps_ok else 1

    if args.health_check:
        logger.info("Performing health check...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        healthy = loop.run_until_complete(run_health_check())
        return 0 if healthy else 1

    # Verify configuration file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1

    # Run main pipeline
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        exit_code = loop.run_until_complete(run_pipeline(args.config, args.dry_run))
        return exit_code

    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())