#!/usr/bin/env python3
"""
One-time sync script to process all unprocessed conversations.
Fixes the PostgreSQL serialization issue and catches up on the backlog.
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.learning_pipeline.src.config.settings import PipelineConfig
from services.learning_pipeline.src.core.pipeline import LearningPipeline

# Setup logging
log_dir = Path('/var/log/tower')
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'conversation_sync.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConversationSyncer:
    def __init__(self):
        # Create configuration matching the working pipeline setup
        config_dict = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'echo_brain',  # Fixed database name
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
                    'file_pattern': '*.json',  # Fixed to look for JSON files
                    'watch_for_changes': True,
                    'exclude_patterns': ['**/test_*', '**/.tmp_*'],
                    'max_file_age_days': 0  # Process all files regardless of age
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
            }
        }
        self.config = PipelineConfig(config_dict)
        self.pipeline = None

    async def sync_all_conversations(self):
        """Sync all conversations from ~/.claude/conversations/ to PostgreSQL and Qdrant."""
        logger.info("Starting conversation sync...")

        try:
            # Initialize pipeline
            self.pipeline = LearningPipeline(self.config)
            await self.pipeline.initialize()

            # Get conversation file statistics
            stats = await self.pipeline.claude_connector.get_file_stats()
            logger.info(f"Found {stats['total_files']} conversation files")
            logger.info(f"Total size: {stats['total_size_mb']} MB")
            logger.info(f"Date range: {stats['oldest_file']} to {stats['newest_file']}")

            # Check current status
            health = await self.pipeline.health_check()
            if not health['healthy']:
                logger.error("Pipeline health check failed")
                for component, status in health['components'].items():
                    if not status['healthy']:
                        logger.error(f"  {component}: {status.get('error', 'Unknown error')}")
                return

            # Run full sync (processes all files since last run time is None)
            logger.info("Running complete sync cycle...")
            result = await self.pipeline.run_learning_cycle()

            logger.info(f"""
===== SYNC RESULTS =====
Run ID: {result.run_id}
Status: {result.status.value}
Conversations Processed: {result.conversations_processed}
Learning Items Extracted: {result.learning_items_extracted}
Vectors Updated: {result.vectors_updated}
Errors: {result.errors_encountered}
Duration: {result.completed_at - result.started_at if result.completed_at else 'N/A'}
=========================""")

            if result.error_message:
                logger.error(f"Pipeline failed: {result.error_message}")
                return False

            return True

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return False

        finally:
            if self.pipeline:
                await self.pipeline.cleanup()

    async def get_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        try:
            # Count conversation files
            conv_path = Path("/home/patrick/.claude/conversations")
            file_count = len(list(conv_path.glob("*.json")))

            # Check Qdrant collections
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:6333/collections/claude_conversations")
                qdrant_count = response.json()["result"]["points_count"] if response.status_code == 200 else 0

            # Check PostgreSQL
            import asyncpg
            conn = await asyncpg.connect(
                host="localhost",
                port=5432,
                database="echo_brain",
                user="patrick",
                password=os.environ.get("ECHO_BRAIN_DB_PASSWORD", os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"))
            )

            pg_conversations = await conn.fetchval("SELECT COUNT(*) FROM echo_conversations")
            pg_messages = await conn.fetchval("SELECT COUNT(*) FROM echo_messages")

            await conn.close()

            return {
                "conversation_files": file_count,
                "qdrant_vectors": qdrant_count,
                "postgresql_conversations": pg_conversations,
                "postgresql_messages": pg_messages,
                "gap_conversations": file_count - pg_conversations,
                "gap_vectors": file_count - qdrant_count
            }

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='Sync Claude conversations')
    parser.add_argument('--status', action='store_true', help='Show current status only')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without doing it')
    args = parser.parse_args()

    syncer = ConversationSyncer()

    if args.status:
        status = await syncer.get_status()
        print(json.dumps(status, indent=2))
    elif args.dry_run:
        logger.info("DRY RUN - would process conversations but not actually doing it")
        # TODO: Implement dry run mode
        print("Dry run mode not yet implemented")
    else:
        success = await syncer.sync_all_conversations()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())