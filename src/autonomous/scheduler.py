"""
Task Scheduler for Echo Brain Autonomous Operations

The Scheduler class manages task scheduling, prioritization, and rate limiting
for autonomous operations, ensuring efficient and controlled execution.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncpg
from contextlib import asynccontextmanager
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """Represents a scheduled task with timing and priority information"""
    id: int
    goal_id: int
    name: str
    task_type: str
    priority: int
    scheduled_at: Optional[datetime]
    safety_level: str
    status: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ScheduleConfig:
    """Configuration for task scheduling"""
    max_concurrent_tasks: int = 3
    max_tasks_per_minute: int = 10
    max_tasks_per_hour: int = 100
    high_priority_threshold: int = 3
    rate_limit_window: int = 60  # seconds


class Scheduler:
    """
    Manages task scheduling and prioritization for autonomous operations.

    Provides capabilities for task queuing, priority management, recurring tasks,
    and rate limiting to ensure controlled and efficient execution.
    """

    def __init__(self, config: Optional[ScheduleConfig] = None):
        """Initialize the Scheduler with database configuration and limits."""
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        }
        self._pool = None

        # Configuration
        self.config = config or ScheduleConfig()

        # Rate limiting
        self.task_execution_times = []  # List of execution timestamps
        self.last_cleanup = time.time()

        # Recurring task management
        self.recurring_tasks = {}  # task_id -> next_run_time

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=2, max_size=10)

        async with self._pool.acquire() as connection:
            yield connection

    async def get_next_task(self) -> Optional[ScheduledTask]:
        """
        Get the next task to execute based on priority and scheduling.

        Returns:
            ScheduledTask object or None if no tasks available
        """
        try:
            # Clean rate limiting data periodically
            self._cleanup_rate_limiting_data()

            # Check if we're at rate limits
            if not self._can_execute_task():
                logger.debug("Rate limit reached, no tasks available")
                return None

            async with self.get_connection() as conn:
                # Check current running tasks
                running_count = await conn.fetchval("""
                    SELECT COUNT(*)
                    FROM autonomous_tasks
                    WHERE status = 'in_progress'
                """)

                if running_count >= self.config.max_concurrent_tasks:
                    logger.debug(f"Max concurrent tasks reached ({running_count}/{self.config.max_concurrent_tasks})")
                    return None

                # Get next task with priority ordering
                # Priority: 1 = highest, 10 = lowest
                # Also consider safety level and scheduling
                task_row = await conn.fetchrow("""
                    SELECT id, goal_id, name, task_type, priority, scheduled_at,
                           safety_level, status, created_at, metadata
                    FROM autonomous_tasks
                    WHERE status IN ('pending', 'approved')  -- Include approved tasks
                      AND safety_level IN ('auto', 'notify')  -- Only auto-executable tasks
                      AND (scheduled_at IS NULL OR scheduled_at <= NOW())
                    ORDER BY
                        priority ASC,  -- Lower number = higher priority
                        created_at ASC  -- FIFO for same priority
                    LIMIT 1
                """)

                if not task_row:
                    return None

                # Convert to ScheduledTask object
                return ScheduledTask(
                    id=task_row['id'],
                    goal_id=task_row['goal_id'],
                    name=task_row['name'],
                    task_type=task_row['task_type'],
                    priority=task_row['priority'],
                    scheduled_at=task_row['scheduled_at'],
                    safety_level=task_row['safety_level'],
                    status=task_row['status'],
                    created_at=task_row['created_at'],
                    metadata=task_row['metadata']
                )

        except Exception as e:
            logger.error(f"Failed to get next task: {e}")
            return None

    async def schedule_task(self, goal_id: int, name: str, task_type: str,
                          priority: int = 5, scheduled_at: Optional[datetime] = None,
                          safety_level: str = 'auto', metadata: Optional[Dict] = None) -> Optional[int]:
        """
        Schedule a new task for execution.

        Args:
            goal_id: ID of the parent goal
            name: Task name/description
            task_type: Type of task (used for agent routing)
            priority: Priority level (1=highest, 10=lowest)
            scheduled_at: When to execute (None = immediate)
            safety_level: Safety classification
            metadata: Additional task metadata

        Returns:
            Task ID if created successfully, None otherwise
        """
        try:
            async with self.get_connection() as conn:
                task_id = await conn.fetchval("""
                    INSERT INTO autonomous_tasks (
                        goal_id, name, task_type, priority, scheduled_at, safety_level, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                """, goal_id, name, task_type, priority, scheduled_at, safety_level, metadata or {})

                logger.info(f"Scheduled task {task_id}: {name} (priority={priority}, type={task_type})")
                return task_id

        except Exception as e:
            logger.error(f"Failed to schedule task: {e}")
            return None

    async def cancel_task(self, task_id: int) -> bool:
        """
        Cancel a pending task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                result = await conn.execute("""
                    UPDATE autonomous_tasks
                    SET status = 'cancelled'
                    WHERE id = $1 AND status = 'pending'
                """, task_id)

                # Check if any rows were affected
                rows_affected = int(result.split()[-1])
                if rows_affected > 0:
                    logger.info(f"Cancelled task {task_id}")
                    return True
                else:
                    logger.warning(f"Task {task_id} could not be cancelled (may not exist or not pending)")
                    return False

        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    async def get_schedule(self, hours: int = 24) -> List[ScheduledTask]:
        """
        Get scheduled tasks for the next specified hours.

        Args:
            hours: How many hours ahead to look

        Returns:
            List of scheduled tasks
        """
        try:
            cutoff_time = datetime.now() + timedelta(hours=hours)

            async with self.get_connection() as conn:
                tasks = await conn.fetch("""
                    SELECT id, goal_id, name, task_type, priority, scheduled_at,
                           safety_level, status, created_at, metadata
                    FROM autonomous_tasks
                    WHERE status = 'pending'
                      AND (scheduled_at IS NULL OR scheduled_at <= $1)
                    ORDER BY
                        COALESCE(scheduled_at, NOW()) ASC,
                        priority ASC
                """, cutoff_time)

                return [
                    ScheduledTask(
                        id=task['id'],
                        goal_id=task['goal_id'],
                        name=task['name'],
                        task_type=task['task_type'],
                        priority=task['priority'],
                        scheduled_at=task['scheduled_at'],
                        safety_level=task['safety_level'],
                        status=task['status'],
                        created_at=task['created_at'],
                        metadata=task['metadata']
                    )
                    for task in tasks
                ]

        except Exception as e:
            logger.error(f"Failed to get schedule: {e}")
            return []

    async def reschedule_task(self, task_id: int, new_time: datetime, new_priority: Optional[int] = None) -> bool:
        """
        Reschedule an existing task.

        Args:
            task_id: ID of task to reschedule
            new_time: New execution time
            new_priority: New priority (optional)

        Returns:
            True if rescheduled successfully, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                if new_priority is not None:
                    result = await conn.execute("""
                        UPDATE autonomous_tasks
                        SET scheduled_at = $1, priority = $2
                        WHERE id = $3 AND status = 'pending'
                    """, new_time, new_priority, task_id)
                else:
                    result = await conn.execute("""
                        UPDATE autonomous_tasks
                        SET scheduled_at = $1
                        WHERE id = $2 AND status = 'pending'
                    """, new_time, task_id)

                rows_affected = int(result.split()[-1])
                if rows_affected > 0:
                    logger.info(f"Rescheduled task {task_id} to {new_time}")
                    return True
                else:
                    logger.warning(f"Task {task_id} could not be rescheduled")
                    return False

        except Exception as e:
            logger.error(f"Failed to reschedule task {task_id}: {e}")
            return False

    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status and statistics.

        Returns:
            Dictionary with queue status information
        """
        try:
            async with self.get_connection() as conn:
                # Get task counts by status
                status_counts = await conn.fetch("""
                    SELECT status, COUNT(*) as count
                    FROM autonomous_tasks
                    GROUP BY status
                """)

                # Get priority distribution for pending tasks
                priority_counts = await conn.fetch("""
                    SELECT priority, COUNT(*) as count
                    FROM autonomous_tasks
                    WHERE status = 'pending'
                    GROUP BY priority
                    ORDER BY priority
                """)

                # Get safety level distribution
                safety_counts = await conn.fetch("""
                    SELECT safety_level, COUNT(*) as count
                    FROM autonomous_tasks
                    WHERE status = 'pending'
                    GROUP BY safety_level
                """)

                # Calculate rate limit status
                current_rate = self._get_current_execution_rate()

                return {
                    'status_counts': {row['status']: row['count'] for row in status_counts},
                    'priority_distribution': {row['priority']: row['count'] for row in priority_counts},
                    'safety_level_distribution': {row['safety_level']: row['count'] for row in safety_counts},
                    'rate_limit_status': {
                        'current_rate_per_minute': current_rate,
                        'max_rate_per_minute': self.config.max_tasks_per_minute,
                        'rate_limit_percentage': (current_rate / self.config.max_tasks_per_minute) * 100,
                        'can_execute': self._can_execute_task()
                    },
                    'concurrent_limit': {
                        'max_concurrent': self.config.max_concurrent_tasks
                    }
                }

        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {}

    def _can_execute_task(self) -> bool:
        """
        Check if we can execute another task based on rate limits.

        Returns:
            True if task can be executed, False if rate limited
        """
        current_time = time.time()

        # Count tasks in the last minute
        minute_ago = current_time - 60
        tasks_last_minute = len([t for t in self.task_execution_times if t > minute_ago])

        # Count tasks in the last hour
        hour_ago = current_time - 3600
        tasks_last_hour = len([t for t in self.task_execution_times if t > hour_ago])

        return (tasks_last_minute < self.config.max_tasks_per_minute and
                tasks_last_hour < self.config.max_tasks_per_hour)

    def _get_current_execution_rate(self) -> int:
        """Get current execution rate (tasks per minute)."""
        current_time = time.time()
        minute_ago = current_time - 60
        return len([t for t in self.task_execution_times if t > minute_ago])

    def _cleanup_rate_limiting_data(self):
        """Clean up old execution times to prevent memory growth."""
        current_time = time.time()

        # Only cleanup periodically
        if current_time - self.last_cleanup < 300:  # 5 minutes
            return

        # Remove entries older than 1 hour
        hour_ago = current_time - 3600
        self.task_execution_times = [t for t in self.task_execution_times if t > hour_ago]
        self.last_cleanup = current_time

    def record_task_execution(self):
        """Record that a task was executed (for rate limiting)."""
        self.task_execution_times.append(time.time())

    async def create_recurring_task(self, goal_id: int, name: str, task_type: str,
                                  interval_minutes: int, priority: int = 5,
                                  safety_level: str = 'auto', metadata: Optional[Dict] = None) -> Optional[int]:
        """
        Create a recurring task that runs at specified intervals.

        Args:
            goal_id: ID of the parent goal
            name: Task name/description
            task_type: Type of task
            interval_minutes: How often to run (in minutes)
            priority: Priority level
            safety_level: Safety classification
            metadata: Additional metadata

        Returns:
            Task ID if created successfully, None otherwise
        """
        try:
            # Add recurring info to metadata
            recurring_metadata = metadata or {}
            recurring_metadata.update({
                'recurring': True,
                'interval_minutes': interval_minutes,
                'next_run_offset': interval_minutes
            })

            # Schedule first execution
            first_run = datetime.now() + timedelta(minutes=interval_minutes)
            task_id = await self.schedule_task(
                goal_id=goal_id,
                name=f"[RECURRING] {name}",
                task_type=task_type,
                priority=priority,
                scheduled_at=first_run,
                safety_level=safety_level,
                metadata=recurring_metadata
            )

            if task_id:
                # Track as recurring task
                self.recurring_tasks[task_id] = first_run
                logger.info(f"Created recurring task {task_id} with {interval_minutes}min interval")

            return task_id

        except Exception as e:
            logger.error(f"Failed to create recurring task: {e}")
            return None

    async def handle_recurring_task_completion(self, task_id: int):
        """
        Handle completion of a recurring task by scheduling the next instance.

        Args:
            task_id: ID of the completed recurring task
        """
        try:
            async with self.get_connection() as conn:
                # Get task details
                task = await conn.fetchrow("""
                    SELECT goal_id, name, task_type, priority, safety_level, metadata
                    FROM autonomous_tasks
                    WHERE id = $1
                """, task_id)

                if not task or not task['metadata'].get('recurring'):
                    return

                interval_minutes = task['metadata'].get('interval_minutes', 60)
                next_run = datetime.now() + timedelta(minutes=interval_minutes)

                # Create next instance
                new_task_id = await self.schedule_task(
                    goal_id=task['goal_id'],
                    name=task['name'],
                    task_type=task['task_type'],
                    priority=task['priority'],
                    scheduled_at=next_run,
                    safety_level=task['safety_level'],
                    metadata=task['metadata']
                )

                if new_task_id:
                    self.recurring_tasks[new_task_id] = next_run
                    logger.info(f"Scheduled next instance of recurring task: {new_task_id}")

        except Exception as e:
            logger.error(f"Failed to handle recurring task completion for {task_id}: {e}")

    async def cleanup(self):
        """Cleanup resources."""
        if self._pool:
            await self._pool.close()
        logger.info("Scheduler cleaned up")