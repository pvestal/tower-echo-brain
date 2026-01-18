"""
Task Orchestrator Module
Manages autonomous task execution and prioritization
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import asyncpg
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents an autonomous task"""
    id: str
    description: str
    priority: int
    scheduled_time: Optional[datetime]
    recurring: Optional[str]  # 'daily', 'weekly', 'hourly'
    status: str  # 'pending', 'running', 'completed', 'failed'
    created_at: datetime
    metadata: Dict[str, Any]

class TaskOrchestrator:
    """Orchestrates autonomous task execution"""

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.conn = None
        self.running_tasks = {}
        self.task_queue = asyncio.Queue()

    async def initialize(self):
        """Initialize database connection and tables"""
        try:
            self.conn = await asyncpg.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                user=self.db_config.get('user', 'patrick'),
                password=self.db_config.get('password', 'RP78eIrW7cI2jYvL5akt1yurE'),
                database=self.db_config.get('database', 'echo_brain')
            )

            # Drop existing table if it doesn't have all columns
            try:
                await self.conn.execute('SELECT scheduled_time FROM task_queue LIMIT 1')
            except:
                await self.conn.execute('DROP TABLE IF EXISTS task_queue')

            # Create task queue table
            await self.conn.execute('''
                CREATE TABLE IF NOT EXISTS task_queue (
                    id VARCHAR(255) PRIMARY KEY,
                    description TEXT NOT NULL,
                    priority INTEGER DEFAULT 0,
                    scheduled_time TIMESTAMP,
                    recurring VARCHAR(50),
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    result JSONB,
                    error TEXT
                )
            ''')

            # Create indexes
            await self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_task_status ON task_queue(status);
            ''')
            await self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_task_scheduled ON task_queue(scheduled_time);
            ''')
            await self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_task_priority ON task_queue(priority DESC);
            ''')

            logger.info("Task orchestrator initialized")

        except Exception as e:
            logger.error(f"Failed to initialize task orchestrator: {e}")
            raise

    async def add_task(
        self,
        description: str,
        priority: int = 0,
        scheduled_time: Optional[datetime] = None,
        recurring: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add a new task to the queue"""

        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(description) % 10000}"

        try:
            await self.conn.execute('''
                INSERT INTO task_queue
                (id, description, priority, scheduled_time, recurring, metadata, status)
                VALUES ($1, $2, $3, $4, $5, $6, 'pending')
            ''', task_id, description, priority, scheduled_time, recurring,
                json.dumps(metadata or {}))

            logger.info(f"Added task: {task_id} - {description}")
            return task_id

        except Exception as e:
            logger.error(f"Failed to add task: {e}")
            raise

    async def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute"""

        try:
            # Get highest priority task that's ready to run
            row = await self.conn.fetchrow('''
                SELECT * FROM task_queue
                WHERE status = 'pending'
                AND (scheduled_time IS NULL OR scheduled_time <= $1)
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            ''', datetime.now())

            if row:
                # Update status to running
                await self.conn.execute('''
                    UPDATE task_queue
                    SET status = 'running', updated_at = $1
                    WHERE id = $2
                ''', datetime.now(), row['id'])

                return Task(
                    id=row['id'],
                    description=row['description'],
                    priority=row['priority'],
                    scheduled_time=row['scheduled_time'],
                    recurring=row['recurring'],
                    status='running',
                    created_at=row['created_at'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get next task: {e}")
            return None

    async def complete_task(self, task_id: str, result: Dict[str, Any] = None):
        """Mark a task as completed"""

        try:
            # Update task status
            await self.conn.execute('''
                UPDATE task_queue
                SET status = 'completed', updated_at = $1, result = $2
                WHERE id = $3
            ''', datetime.now(), json.dumps(result or {}), task_id)

            # Check if it's a recurring task
            row = await self.conn.fetchrow(
                'SELECT recurring, scheduled_time FROM task_queue WHERE id = $1',
                task_id
            )

            if row and row['recurring']:
                # Schedule next occurrence
                next_time = self._calculate_next_occurrence(
                    row['scheduled_time'] or datetime.now(),
                    row['recurring']
                )

                # Create new task for next occurrence
                await self.conn.execute('''
                    INSERT INTO task_queue
                    (id, description, priority, scheduled_time, recurring, metadata, status)
                    SELECT
                        $1 || '_next',
                        description,
                        priority,
                        $2,
                        recurring,
                        metadata,
                        'pending'
                    FROM task_queue WHERE id = $1
                ''', task_id, next_time)

            logger.info(f"Completed task: {task_id}")

        except Exception as e:
            logger.error(f"Failed to complete task: {e}")

    async def fail_task(self, task_id: str, error: str):
        """Mark a task as failed"""

        try:
            await self.conn.execute('''
                UPDATE task_queue
                SET status = 'failed', updated_at = $1, error = $2
                WHERE id = $3
            ''', datetime.now(), error, task_id)

            logger.error(f"Task failed: {task_id} - {error}")

        except Exception as e:
            logger.error(f"Failed to update task status: {e}")

    async def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tasks with optional status filter"""

        try:
            if status:
                rows = await self.conn.fetch(
                    'SELECT * FROM task_queue WHERE status = $1 ORDER BY priority DESC, created_at DESC',
                    status
                )
            else:
                rows = await self.conn.fetch(
                    'SELECT * FROM task_queue ORDER BY priority DESC, created_at DESC LIMIT 100'
                )

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return []

    def _calculate_next_occurrence(self, base_time: datetime, recurring: str) -> datetime:
        """Calculate next occurrence for recurring task"""

        if recurring == 'hourly':
            return base_time + timedelta(hours=1)
        elif recurring == 'daily':
            return base_time + timedelta(days=1)
        elif recurring == 'weekly':
            return base_time + timedelta(weeks=1)
        else:
            return base_time + timedelta(days=1)  # Default to daily

    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            logger.info("Task orchestrator closed")


# Export for compatibility
TaskOrchestratorInterface = TaskOrchestrator