#!/usr/bin/env python3
"""
Autonomous Goals Seeding Script

Seeds the autonomous system with initial safe goals and tasks to bootstrap
autonomous operations. This script creates foundational goals that are safe
to execute automatically and help establish the autonomous workflow.

Goals Created:
1. "Complete fact extraction" - Extract facts from remaining vectors (AUTO safety)
2. "Index new conversations" - Watch for new Claude conversations (AUTO safety)
3. "Knowledge maintenance" - Deduplicate facts, clean orphaned vectors (NOTIFY safety)

Each goal includes initial tasks to begin execution immediately.
"""

import asyncio
import asyncpg
import logging
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoalSeeder:
    """
    Seeds initial autonomous goals and tasks for the Echo Brain system.

    This class handles the creation of foundational autonomous goals that are
    safe to execute automatically and help bootstrap the autonomous workflow.
    """

    def __init__(self):
        """Initialize the GoalSeeder with database configuration."""
        self.db_config = {
            'host': os.environ.get('ECHO_BRAIN_DB_HOST', 'localhost'),
            'port': int(os.environ.get('ECHO_BRAIN_DB_PORT', '5432')),
            'database': os.environ.get('ECHO_BRAIN_DB_NAME', 'echo_brain'),
            'user': os.environ.get('ECHO_BRAIN_DB_USER', 'patrick'),
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        }

        self.connection = None
        self.created_goals = []
        self.created_tasks = []

        logger.info("GoalSeeder initialized with database configuration")

    async def connect(self) -> bool:
        """
        Connect to the PostgreSQL database.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = await asyncpg.connect(**self.db_config)
            logger.info("Connected to database successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the database."""
        if self.connection:
            await self.connection.close()
            logger.info("Disconnected from database")

    async def check_schema_exists(self) -> bool:
        """
        Check if the autonomous schema tables exist.

        Returns:
            bool: True if schema exists, False otherwise
        """
        try:
            # Check if autonomous_goals table exists
            result = await self.connection.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'autonomous_goals'
                );
            """)

            if not result:
                logger.error("Autonomous schema tables do not exist. Please run schema setup first.")
                return False

            logger.info("Autonomous schema tables found")
            return True

        except Exception as e:
            logger.error(f"Error checking schema: {e}")
            return False

    async def goal_exists(self, name: str) -> Optional[int]:
        """
        Check if a goal with the given name already exists.

        Args:
            name: The goal name to check

        Returns:
            Optional[int]: Goal ID if exists, None otherwise
        """
        try:
            goal_id = await self.connection.fetchval(
                "SELECT id FROM autonomous_goals WHERE name = $1",
                name
            )
            return goal_id
        except Exception as e:
            logger.error(f"Error checking if goal exists: {e}")
            return None

    async def create_goal(self, goal_data: Dict[str, Any]) -> Optional[int]:
        """
        Create a new autonomous goal.

        Args:
            goal_data: Dictionary containing goal information

        Returns:
            Optional[int]: Created goal ID or None if failed
        """
        try:
            goal_id = await self.connection.fetchval("""
                INSERT INTO autonomous_goals (
                    name, description, goal_type, status, priority, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """,
                goal_data['name'],
                goal_data['description'],
                goal_data['goal_type'],
                goal_data['status'],
                goal_data['priority'],
                json.dumps(goal_data['metadata'])
            )

            logger.info(f"Created goal '{goal_data['name']}' with ID {goal_id}")
            self.created_goals.append({
                'id': goal_id,
                'name': goal_data['name'],
                'type': goal_data['goal_type']
            })

            return goal_id

        except Exception as e:
            logger.error(f"Failed to create goal '{goal_data['name']}': {e}")
            return None

    async def create_task(self, task_data: Dict[str, Any]) -> Optional[int]:
        """
        Create a new autonomous task.

        Args:
            task_data: Dictionary containing task information

        Returns:
            Optional[int]: Created task ID or None if failed
        """
        try:
            task_id = await self.connection.fetchval("""
                INSERT INTO autonomous_tasks (
                    goal_id, name, task_type, status, safety_level,
                    priority, scheduled_at, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """,
                task_data['goal_id'],
                task_data['name'],
                task_data['task_type'],
                task_data['status'],
                task_data['safety_level'],
                task_data['priority'],
                task_data.get('scheduled_at'),
                json.dumps(task_data.get('metadata', {}))
            )

            logger.info(f"Created task '{task_data['name']}' with ID {task_id}")
            self.created_tasks.append({
                'id': task_id,
                'name': task_data['name'],
                'goal_id': task_data['goal_id'],
                'type': task_data['task_type']
            })

            return task_id

        except Exception as e:
            logger.error(f"Failed to create task '{task_data['name']}': {e}")
            return None

    def get_seed_goals(self) -> List[Dict[str, Any]]:
        """
        Get the list of seed goals to create.

        Returns:
            List of goal dictionaries with their configuration
        """
        return [
            {
                'name': 'Complete fact extraction',
                'description': 'Extract facts from remaining vectors in the knowledge base. '
                              'This goal processes unanalyzed conversation vectors to extract '
                              'structured facts and relationships for the knowledge graph.',
                'goal_type': 'knowledge_processing',
                'status': 'active',
                'priority': 3,
                'metadata': {
                    'safety_level': 'auto',
                    'max_vectors_per_batch': 50,
                    'extraction_model': 'claude-3-haiku-20240307',
                    'target_completion_date': (datetime.now() + timedelta(days=7)).isoformat(),
                    'estimated_vectors': 1000,
                    'recurring': False,
                    'created_by': 'autonomous_seeder',
                    'category': 'data_processing'
                }
            },
            {
                'name': 'Index new conversations',
                'description': 'Monitor and index new Claude conversations as they are added. '
                              'This goal watches for new conversation files and automatically '
                              'processes them into the vector database for searchability.',
                'goal_type': 'monitoring',
                'status': 'active',
                'priority': 4,
                'metadata': {
                    'safety_level': 'auto',
                    'watch_directories': [
                        '/home/patrick/.claude/conversations',
                        '/opt/tower-echo-brain/conversations'
                    ],
                    'file_patterns': ['*.md', '*.txt', '*.json'],
                    'processing_model': 'claude-3-haiku-20240307',
                    'check_interval_minutes': 15,
                    'recurring': True,
                    'created_by': 'autonomous_seeder',
                    'category': 'file_monitoring'
                }
            },
            {
                'name': 'Knowledge maintenance',
                'description': 'Perform maintenance on the knowledge base including deduplicating '
                              'facts, cleaning orphaned vectors, and optimizing data structures. '
                              'This goal ensures knowledge base integrity and performance.',
                'goal_type': 'maintenance',
                'status': 'active',
                'priority': 5,
                'metadata': {
                    'safety_level': 'notify',
                    'maintenance_types': [
                        'deduplicate_facts',
                        'clean_orphaned_vectors',
                        'optimize_indices',
                        'compress_old_data'
                    ],
                    'schedule': 'weekly',
                    'next_run': (datetime.now() + timedelta(days=1)).isoformat(),
                    'max_cleanup_percent': 10,
                    'backup_before_cleanup': True,
                    'recurring': True,
                    'created_by': 'autonomous_seeder',
                    'category': 'system_maintenance'
                }
            }
        ]

    def get_tasks_for_goal(self, goal_name: str, goal_id: int) -> List[Dict[str, Any]]:
        """
        Get initial tasks for a specific goal.

        Args:
            goal_name: Name of the goal
            goal_id: Database ID of the goal

        Returns:
            List of task dictionaries
        """
        now = datetime.now()

        if goal_name == 'Complete fact extraction':
            return [
                {
                    'goal_id': goal_id,
                    'name': 'Scan for unprocessed vectors',
                    'task_type': 'database_query',
                    'status': 'pending',
                    'safety_level': 'auto',
                    'priority': 3,
                    'scheduled_at': now + timedelta(minutes=5),
                    'metadata': {
                        'query_type': 'count_unprocessed',
                        'table': 'conversation_vectors',
                        'condition': 'facts_extracted = false',
                        'timeout_seconds': 30,
                        'description': 'Count vectors that need fact extraction'
                    }
                },
                {
                    'goal_id': goal_id,
                    'name': 'Process first batch of vectors',
                    'task_type': 'fact_extraction',
                    'status': 'pending',
                    'safety_level': 'auto',
                    'priority': 3,
                    'scheduled_at': now + timedelta(minutes=10),
                    'metadata': {
                        'batch_size': 10,
                        'extraction_prompt': 'Extract key facts and relationships from this text',
                        'model': 'claude-3-haiku-20240307',
                        'timeout_seconds': 300,
                        'description': 'Extract facts from first 10 unprocessed vectors'
                    }
                }
            ]

        elif goal_name == 'Index new conversations':
            return [
                {
                    'goal_id': goal_id,
                    'name': 'Initialize file system watchers',
                    'task_type': 'file_monitoring',
                    'status': 'pending',
                    'safety_level': 'auto',
                    'priority': 4,
                    'scheduled_at': now + timedelta(minutes=2),
                    'metadata': {
                        'action': 'start_watchers',
                        'directories': ['/home/patrick/.claude/conversations'],
                        'patterns': ['*.md', '*.txt'],
                        'recursive': True,
                        'description': 'Set up file system monitoring for new conversations'
                    }
                },
                {
                    'goal_id': goal_id,
                    'name': 'Scan for recent files',
                    'task_type': 'file_scan',
                    'status': 'pending',
                    'safety_level': 'auto',
                    'priority': 4,
                    'scheduled_at': now + timedelta(minutes=5),
                    'metadata': {
                        'scan_directories': ['/home/patrick/.claude/conversations'],
                        'modified_since_hours': 24,
                        'file_extensions': ['.md', '.txt'],
                        'max_files': 100,
                        'description': 'Scan for conversation files modified in last 24 hours'
                    }
                }
            ]

        elif goal_name == 'Knowledge maintenance':
            return [
                {
                    'goal_id': goal_id,
                    'name': 'Analyze knowledge base health',
                    'task_type': 'analysis',
                    'status': 'pending',
                    'safety_level': 'notify',
                    'priority': 5,
                    'scheduled_at': now + timedelta(hours=1),
                    'metadata': {
                        'analysis_types': ['duplicate_detection', 'orphan_detection', 'index_health'],
                        'tables': ['facts', 'conversation_vectors', 'embeddings'],
                        'generate_report': True,
                        'report_format': 'json',
                        'description': 'Comprehensive analysis of knowledge base health metrics'
                    }
                },
                {
                    'goal_id': goal_id,
                    'name': 'Create maintenance plan',
                    'task_type': 'planning',
                    'status': 'pending',
                    'safety_level': 'notify',
                    'priority': 5,
                    'scheduled_at': now + timedelta(hours=2),
                    'metadata': {
                        'plan_type': 'maintenance_schedule',
                        'based_on_analysis': True,
                        'require_approval': True,
                        'max_impact': 'low',
                        'description': 'Create detailed plan for knowledge base maintenance tasks'
                    }
                }
            ]

        return []

    async def seed_goals_and_tasks(self) -> bool:
        """
        Seed all goals and their initial tasks.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            goals_data = self.get_seed_goals()

            for goal_data in goals_data:
                # Check if goal already exists
                existing_goal_id = await self.goal_exists(goal_data['name'])

                if existing_goal_id:
                    logger.info(f"Goal '{goal_data['name']}' already exists with ID {existing_goal_id}")
                    continue

                # Create the goal
                goal_id = await self.create_goal(goal_data)
                if not goal_id:
                    logger.error(f"Failed to create goal: {goal_data['name']}")
                    continue

                # Create initial tasks for this goal
                tasks_data = self.get_tasks_for_goal(goal_data['name'], goal_id)

                for task_data in tasks_data:
                    task_id = await self.create_task(task_data)
                    if not task_id:
                        logger.warning(f"Failed to create task: {task_data['name']}")

            return True

        except Exception as e:
            logger.error(f"Error seeding goals and tasks: {e}")
            return False

    async def create_audit_log_entry(self):
        """Create an audit log entry for the seeding operation."""
        try:
            await self.connection.execute("""
                INSERT INTO autonomous_audit_log (
                    event_type, action, details, outcome
                ) VALUES ($1, $2, $3, $4)
            """,
                'system_seeded',
                'Initial goals and tasks seeded',
                json.dumps({
                    'goals_created': len(self.created_goals),
                    'tasks_created': len(self.created_tasks),
                    'goals': [g['name'] for g in self.created_goals],
                    'timestamp': datetime.now().isoformat()
                }),
                'success'
            )
            logger.info("Created audit log entry for seeding operation")

        except Exception as e:
            logger.error(f"Failed to create audit log entry: {e}")

    async def print_summary(self):
        """Print a summary of what was created."""
        print("\n" + "="*60)
        print("AUTONOMOUS GOALS SEEDING SUMMARY")
        print("="*60)

        if self.created_goals:
            print(f"\n✅ Created {len(self.created_goals)} goals:")
            for goal in self.created_goals:
                print(f"   • {goal['name']} (ID: {goal['id']}, Type: {goal['type']})")
        else:
            print("\n✅ No new goals created (all already exist)")

        if self.created_tasks:
            print(f"\n📋 Created {len(self.created_tasks)} tasks:")
            for task in self.created_tasks:
                print(f"   • {task['name']} (ID: {task['id']}, Goal: {task['goal_id']})")
        else:
            print("\n📋 No new tasks created")

        print(f"\n🔍 Database: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
        print(f"👤 User: {self.db_config['user']}")
        print(f"⏰ Seeded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "="*60)

    async def run(self) -> bool:
        """
        Run the complete seeding process.

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting autonomous goals seeding process")

        try:
            # Connect to database
            if not await self.connect():
                return False

            # Check schema exists
            if not await self.check_schema_exists():
                return False

            # Seed goals and tasks
            if not await self.seed_goals_and_tasks():
                return False

            # Create audit log entry
            await self.create_audit_log_entry()

            # Print summary
            await self.print_summary()

            logger.info("Autonomous goals seeding completed successfully")
            return True

        except Exception as e:
            logger.error(f"Seeding process failed: {e}")
            return False

        finally:
            await self.disconnect()


async def main():
    """Main entry point for the seeding script."""
    print("Echo Brain Autonomous Goals Seeder")
    print("=" * 40)

    seeder = GoalSeeder()
    success = await seeder.run()

    if success:
        print("\n✅ Seeding completed successfully!")
        return 0
    else:
        print("\n❌ Seeding failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    # Run the seeder
    exit_code = asyncio.run(main())
    sys.exit(exit_code)