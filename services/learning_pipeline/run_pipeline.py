#!/usr/bin/env python3
"""
Learning pipeline runner script.
Executes the complete learning cycle with configuration and error handling.
"""

import asyncio
import sys
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import PipelineConfig
from src.core.pipeline import LearningPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_default_config() -> dict:
    """Create default configuration for the learning pipeline."""
    return {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'echo_brain',
            'user': 'patrick',
            'password_env': 'ECHO_BRAIN_DB_PASSWORD'
        },
        'vector_database': {
            'host': 'localhost',
            'port': 6333,
            'collection_name': 'claude_conversations',
            'embedding_dimension': 384,
            'batch_size': 100
        },
        'sources': {
            'claude_conversations': {
                'path': os.path.expanduser('~/.claude/conversations'),
                'file_pattern': '*.md',
                'watch_for_changes': True,
                'exclude_patterns': ['**/test_*', '**/.tmp_*'],
                'max_file_age_days': 365
            }
        },
        'pipeline': {
            'batch_size': 20,
            'max_concurrent_processors': 3,
            'processing_timeout': 300,
            'min_content_length': 50
        },
        'circuit_breaker': {
            'failure_threshold': 5,
            'reset_timeout': 60,
            'half_open_max_calls': 3
        },
        'content_processing': {
            'extract_code_blocks': True,
            'extract_commands': True,
            'extract_insights': True,
            'extract_solutions': True,
            'min_insight_length': 100,
            'code_block_languages': ['python', 'bash', 'sql', 'javascript', 'typescript', 'yaml', 'json']
        },
        'knowledge_base': {
            'enabled': True,
            'api_url': 'http://localhost:8307/api',
            'timeout': 30,
            'batch_size': 20,
            'include_categories': ['technical', 'development', 'troubleshooting'],
            'exclude_categories': ['personal', 'draft'],
            'min_article_length': 200
        }
    }


async def run_pipeline(config_dict: dict, dry_run: bool = False) -> bool:
    """
    Run the learning pipeline.

    Returns:
        True if successful, False if failed
    """
    try:
        logger.info(f"Starting learning pipeline (dry_run={dry_run})")

        # Create configuration
        config = PipelineConfig(config_dict)

        # Validate configuration
        validation_errors = config.validate()
        if validation_errors:
            logger.error(f"Configuration validation failed: {validation_errors}")
            return False

        # Initialize pipeline
        pipeline = LearningPipeline(config)

        if dry_run:
            logger.info("Dry run mode - performing initialization and health checks only")

            # Test pipeline status
            status = await pipeline.get_pipeline_status()
            logger.info(f"Pipeline status: {status}")

            # Run health checks without actual processing
            health = await pipeline.health_check()
            logger.info(f"Health check result: {health}")

            logger.info("Dry run completed successfully")
            return health['healthy']
        else:
            # Initialize async components
            await pipeline.initialize()

            try:
                # Run learning cycle
                result = await pipeline.run_learning_cycle()

                # Log results
                logger.info(f"Learning cycle completed successfully")
                logger.info(f"Conversations processed: {result.conversations_processed}")
                logger.info(f"Learning items extracted: {result.learning_items_extracted}")
                logger.info(f"Vectors updated: {result.vectors_updated}")
                logger.info(f"Errors encountered: {result.errors_encountered}")

                return result.is_successful

            finally:
                # Clean up
                await pipeline.cleanup()

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Run the Echo Brain learning pipeline')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform a dry run without actual processing')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (optional)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set log level')
    parser.add_argument('--health-check', action='store_true',
                       help='Run health check only')

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.health_check:
        # Run health check only
        logger.info("Running health check only")
        os.system("python health_check.py")
        return

    # Create configuration
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        # Would load from file here
        config_dict = create_default_config()
    else:
        logger.info("Using default configuration")
        config_dict = create_default_config()

    # Run pipeline
    try:
        success = asyncio.run(run_pipeline(config_dict, args.dry_run))

        if success:
            logger.info("Pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error("Pipeline failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()