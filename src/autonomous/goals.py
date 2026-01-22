"""
Goal Management System for Echo Brain Autonomous Operations

The GoalManager class provides comprehensive goal and task management capabilities
for Echo Brain's autonomous operations, including goal creation, task orchestration,
progress tracking, and completion management.
"""

import logging
import asyncio
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncpg
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class GoalManager:
    """
    Manages autonomous goals and their associated tasks.

    Provides methods for creating, tracking, and managing high-level goals
    and their constituent tasks within Echo Brain's autonomous system.
    """

    def __init__(self):
        """Initialize the GoalManager with database configuration."""
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'tower_consolidated',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        }
        self._pool = None

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=2, max_size=10)

        async with self._pool.acquire() as connection:
            yield connection

    async def create_goal(self, name: str, description: str, goal_type: str,
                         priority: int = 5, metadata: Optional[Dict] = None) -> int:
        """
        Create a new autonomous goal.

        Args:
            name: Human-readable goal name
            description: Detailed description of the goal
            goal_type: Type of goal (e.g., 'research', 'coding', 'analysis', 'maintenance')
            priority: Priority level from 1 (highest) to 10 (lowest), default 5
            metadata: Optional metadata dictionary for goal-specific data

        Returns:
            int: The ID of the created goal

        Raises:
            ValueError: If invalid parameters are provided
            asyncpg.PostgresError: If database operation fails
        """
        if not name or not description or not goal_type:
            raise ValueError("Name, description, and goal_type are required")

        if not 1 <= priority <= 10:
            raise ValueError("Priority must be between 1 and 10")

        metadata = metadata or {}

        try:
            async with self.get_connection() as conn:
                goal_id = await conn.fetchval(
                    """
                    INSERT INTO autonomous_goals (name, description, goal_type, priority, metadata)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    RETURNING id
                    """,
                    name, description, goal_type, priority, json.dumps(metadata)
                )

                logger.info(f"Created goal '{name}' with ID {goal_id}")
                return goal_id

        except Exception as e:
            logger.error(f"Failed to create goal '{name}': {e}")
            raise

    async def get_active_goals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve all active goals ordered by priority and creation time.

        Args:
            limit: Maximum number of goals to return, default 50

        Returns:
            List[Dict]: List of active goal dictionaries with all fields
        """
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, name, description, goal_type, status, priority,
                           progress_percent, created_at, updated_at, completed_at, metadata
                    FROM autonomous_goals
                    WHERE status = 'active'
                    ORDER BY priority ASC, created_at ASC
                    LIMIT $1
                    """,
                    limit
                )

                goals = [dict(row) for row in rows]
                logger.debug(f"Retrieved {len(goals)} active goals")
                return goals

        except Exception as e:
            logger.error(f"Failed to retrieve active goals: {e}")
            raise

    async def update_progress(self, goal_id: int, progress_percent: float) -> bool:
        """
        Update the progress percentage for a goal.

        Args:
            goal_id: ID of the goal to update
            progress_percent: Progress percentage (0.00 to 100.00)

        Returns:
            bool: True if update was successful, False otherwise

        Raises:
            ValueError: If progress_percent is not between 0 and 100
        """
        if not 0.0 <= progress_percent <= 100.0:
            raise ValueError("Progress percent must be between 0.00 and 100.00")

        try:
            async with self.get_connection() as conn:
                result = await conn.execute(
                    """
                    UPDATE autonomous_goals
                    SET progress_percent = $1, updated_at = NOW()
                    WHERE id = $2 AND status = 'active'
                    """,
                    progress_percent, goal_id
                )

                if result == "UPDATE 1":
                    logger.debug(f"Updated goal {goal_id} progress to {progress_percent}%")
                    return True
                else:
                    logger.warning(f"Goal {goal_id} not found or not active")
                    return False

        except Exception as e:
            logger.error(f"Failed to update progress for goal {goal_id}: {e}")
            raise

    async def complete_goal(self, goal_id: int) -> bool:
        """
        Mark a goal as completed.

        Args:
            goal_id: ID of the goal to complete

        Returns:
            bool: True if goal was successfully completed, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                result = await conn.execute(
                    """
                    UPDATE autonomous_goals
                    SET status = 'completed', progress_percent = 100.00,
                        completed_at = NOW(), updated_at = NOW()
                    WHERE id = $1 AND status = 'active'
                    """,
                    goal_id
                )

                if result == "UPDATE 1":
                    logger.info(f"Completed goal {goal_id}")
                    return True
                else:
                    logger.warning(f"Goal {goal_id} not found or already completed")
                    return False

        except Exception as e:
            logger.error(f"Failed to complete goal {goal_id}: {e}")
            raise

    async def fail_goal(self, goal_id: int, error_reason: str = None) -> bool:
        """
        Mark a goal as failed.

        Args:
            goal_id: ID of the goal to mark as failed
            error_reason: Optional reason for failure

        Returns:
            bool: True if goal was successfully marked as failed, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                # Update metadata with failure reason if provided
                metadata_update = {}
                if error_reason:
                    # Get current metadata and add failure reason
                    current_meta = await conn.fetchval(
                        "SELECT metadata FROM autonomous_goals WHERE id = $1", goal_id
                    ) or {}
                    metadata_update = {**current_meta, 'failure_reason': error_reason}

                if metadata_update:
                    result = await conn.execute(
                        """
                        UPDATE autonomous_goals
                        SET status = 'failed', updated_at = NOW(), metadata = $1
                        WHERE id = $2 AND status = 'active'
                        """,
                        metadata_update, goal_id
                    )
                else:
                    result = await conn.execute(
                        """
                        UPDATE autonomous_goals
                        SET status = 'failed', updated_at = NOW()
                        WHERE id = $1 AND status = 'active'
                        """,
                        goal_id
                    )

                if result == "UPDATE 1":
                    logger.info(f"Failed goal {goal_id}: {error_reason or 'No reason provided'}")
                    return True
                else:
                    logger.warning(f"Goal {goal_id} not found or already completed")
                    return False

        except Exception as e:
            logger.error(f"Failed to mark goal {goal_id} as failed: {e}")
            raise

    async def create_task(self, goal_id: int, name: str, task_type: str,
                         safety_level: str = 'auto', priority: int = 5,
                         scheduled_at: Optional[datetime] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a new task under a goal.

        Args:
            goal_id: ID of the parent goal
            name: Human-readable task name
            task_type: Type of task (e.g., 'api_call', 'file_operation', 'analysis')
            safety_level: Safety level ('auto', 'notify', 'review', 'forbidden')
            priority: Priority level from 1 (highest) to 10 (lowest), default 5
            scheduled_at: Optional scheduled execution time

        Returns:
            int: The ID of the created task

        Raises:
            ValueError: If invalid parameters are provided
        """
        if not name or not task_type:
            raise ValueError("Name and task_type are required")

        if safety_level not in ['auto', 'notify', 'review', 'forbidden']:
            raise ValueError("Safety level must be one of: auto, notify, review, forbidden")

        if not 1 <= priority <= 10:
            raise ValueError("Priority must be between 1 and 10")

        # If safety_level is in metadata, extract and use it
        if metadata and 'safety_level' in metadata:
            safety_level = metadata['safety_level']

        # Ensure metadata includes safety_level for consistency
        if metadata is None:
            metadata = {}
        metadata['safety_level'] = safety_level

        try:
            async with self.get_connection() as conn:
                # Verify goal exists
                goal_exists = await conn.fetchval(
                    "SELECT 1 FROM autonomous_goals WHERE id = $1", goal_id
                )
                if not goal_exists:
                    raise ValueError(f"Goal {goal_id} does not exist")

                task_id = await conn.fetchval(
                    """
                    INSERT INTO autonomous_tasks (goal_id, name, task_type, safety_level, priority, scheduled_at, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    RETURNING id
                    """,
                    goal_id, name, task_type, safety_level, priority, scheduled_at, json.dumps(metadata)
                )

                logger.info(f"Created task '{name}' with ID {task_id} for goal {goal_id}")
                return task_id

        except Exception as e:
            logger.error(f"Failed to create task '{name}' for goal {goal_id}: {e}")
            raise

    async def get_pending_tasks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all pending tasks ordered by priority and creation time.

        Args:
            limit: Maximum number of tasks to return, default 100

        Returns:
            List[Dict]: List of pending task dictionaries with all fields
        """
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT t.id, t.goal_id, t.name, t.task_type, t.status, t.safety_level,
                           t.priority, t.scheduled_at, t.started_at, t.completed_at,
                           t.result, t.error, t.created_at, g.name as goal_name
                    FROM autonomous_tasks t
                    JOIN autonomous_goals g ON t.goal_id = g.id
                    WHERE t.status = 'pending'
                    AND (t.scheduled_at IS NULL OR t.scheduled_at <= NOW())
                    ORDER BY t.priority ASC, t.created_at ASC
                    LIMIT $1
                    """,
                    limit
                )

                tasks = [dict(row) for row in rows]
                logger.debug(f"Retrieved {len(tasks)} pending tasks")
                return tasks

        except Exception as e:
            logger.error(f"Failed to retrieve pending tasks: {e}")
            raise

    async def get_tasks_needing_approval(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get all tasks that require approval ordered by priority and creation time.

        Args:
            limit: Maximum number of tasks to return, default 50

        Returns:
            List[Dict]: List of task dictionaries requiring approval
        """
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT t.id, t.goal_id, t.name, t.task_type, t.status, t.safety_level,
                           t.priority, t.scheduled_at, t.created_at, g.name as goal_name,
                           a.id as approval_id, a.action_description, a.risk_assessment,
                           a.status as approval_status
                    FROM autonomous_tasks t
                    JOIN autonomous_goals g ON t.goal_id = g.id
                    LEFT JOIN autonomous_approvals a ON t.id = a.task_id
                    WHERE t.status = 'requires_approval'
                    ORDER BY t.priority ASC, t.created_at ASC
                    LIMIT $1
                    """,
                    limit
                )

                tasks = [dict(row) for row in rows]
                logger.debug(f"Retrieved {len(tasks)} tasks needing approval")
                return tasks

        except Exception as e:
            logger.error(f"Failed to retrieve tasks needing approval: {e}")
            raise

    async def get_goal_by_id(self, goal_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific goal by ID.

        Args:
            goal_id: ID of the goal to retrieve

        Returns:
            Optional[Dict]: Goal dictionary if found, None otherwise
        """
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, name, description, goal_type, status, priority,
                           progress_percent, created_at, updated_at, completed_at, metadata
                    FROM autonomous_goals
                    WHERE id = $1
                    """,
                    goal_id
                )

                if row:
                    return dict(row)
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve goal {goal_id}: {e}")
            raise

    async def get_tasks_for_goal(self, goal_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve all tasks for a specific goal.

        Args:
            goal_id: ID of the goal

        Returns:
            List[Dict]: List of task dictionaries for the goal
        """
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, goal_id, name, task_type, status, safety_level,
                           priority, scheduled_at, started_at, completed_at,
                           result, error, created_at
                    FROM autonomous_tasks
                    WHERE goal_id = $1
                    ORDER BY priority ASC, created_at ASC
                    """,
                    goal_id
                )

                tasks = [dict(row) for row in rows]
                logger.debug(f"Retrieved {len(tasks)} tasks for goal {goal_id}")
                return tasks

        except Exception as e:
            logger.error(f"Failed to retrieve tasks for goal {goal_id}: {e}")
            raise

    async def close(self):
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()
            logger.debug("Closed GoalManager database pool")