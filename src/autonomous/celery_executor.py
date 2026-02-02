"""
Celery-based executor for distributed task execution
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
import asyncpg
from contextlib import asynccontextmanager

from .celery_tasks import get_celery_task_for_type
from .models import TaskResult

logger = logging.getLogger(__name__)


class CeleryExecutor:
    """
    Executor that delegates tasks to Celery workers
    """

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self._pool = None

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=2, max_size=10)

        async with self._pool.acquire() as connection:
            yield connection

    async def initialize_agents(self):
        """Initialize agents - not needed for Celery executor as agents run in workers"""
        logger.info("CeleryExecutor: Agents will be initialized in Celery workers")
        pass

    async def execute(self, task_id: int) -> TaskResult:
        """
        Execute a task by delegating to Celery workers

        Args:
            task_id: Database ID of the task to execute

        Returns:
            TaskResult with execution outcome
        """
        logger.info(f"CeleryExecutor.execute called for task {task_id}")
        start_time = datetime.now()

        try:
            # Get task details from database
            async with self.get_connection() as conn:
                task = await conn.fetchrow("""
                    SELECT id, name, task_type, goal_id, metadata, safety_level, status
                    FROM autonomous_tasks
                    WHERE id = $1
                """, task_id)

                if not task:
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        error=f"Task {task_id} not found in database"
                    )

            task_type = task['task_type']
            task_status = task['status']
            safety_level = task.get('safety_level', 'auto')

            # Check if task is approved or auto-executable
            if task_status != 'approved' and safety_level != 'auto':
                logger.info(f"Task {task_id} requires approval (status={task_status}, safety={safety_level})")
                return TaskResult(
                    task_id=task_id,
                    success=True,
                    result="Task queued for approval",
                    metadata={'requires_approval': True}
                )

            # Get the appropriate Celery task
            celery_task = get_celery_task_for_type(task_type)

            if not celery_task:
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=f"No Celery task handler for type: {task_type}"
                )

            # Update status to in_progress
            await self._update_task_status(task_id, 'in_progress', start_time)

            # Prepare task data
            task_data = {
                'id': task['id'],
                'name': task['name'],
                'task_type': task['task_type'],
                'goal_id': task['goal_id'],
                'metadata': task['metadata']
            }

            # Submit to Celery queue
            # Determine priority queue based on task type
            if task_type in ['testing', 'validation']:
                queue = 'high_priority'
            elif task_type in ['coding', 'code_generation']:
                queue = 'coding'
            elif task_type in ['reasoning', 'analysis']:
                queue = 'reasoning'
            else:
                queue = 'default'

            logger.info(f"Submitting task {task_id} ({task_type}) to Celery queue '{queue}'")

            # Apply async - returns AsyncResult
            try:
                result = celery_task.apply_async(
                    args=[task_id, task_data],
                    queue=queue,
                    task_id=f"echo_task_{task_id}"
                )
                logger.info(f"Celery apply_async returned result: {result}, ID: {result.id if result else 'None'}")
            except Exception as ce:
                logger.error(f"Failed to submit task to Celery: {ce}")
                raise

            # Store Celery task ID in database for tracking
            logger.info(f"Storing Celery task ID {result.id} for task {task_id}")
            async with self.get_connection() as conn:
                await conn.execute("""
                    UPDATE autonomous_tasks
                    SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('celery_task_id', $1::text)
                    WHERE id = $2::integer
                """, str(result.id), task_id)

            logger.info(f"Task {task_id} submitted to Celery with ID {result.id}")

            # Return immediately - Celery will handle execution
            return TaskResult(
                task_id=task_id,
                success=True,
                result=f"Task submitted to worker queue (Celery ID: {result.id})",
                metadata={
                    'celery_task_id': result.id,
                    'queue': queue,
                    'submitted_at': datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Failed to execute task {task_id}: {e}")
            await self._update_task_status(task_id, 'failed', None, str(e))
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e)
            )

    async def _update_task_status(self, task_id: int, status: str,
                                 started_at: Optional[datetime] = None,
                                 error: Optional[str] = None):
        """Update task status in database"""
        try:
            async with self.get_connection() as conn:
                if status == 'in_progress' and started_at:
                    await conn.execute("""
                        UPDATE autonomous_tasks
                        SET status = $1, started_at = $2
                        WHERE id = $3
                    """, status, started_at, task_id)
                elif status == 'failed' and error:
                    await conn.execute("""
                        UPDATE autonomous_tasks
                        SET status = $1, error = $2, completed_at = NOW()
                        WHERE id = $3
                    """, status, error, task_id)
                else:
                    await conn.execute("""
                        UPDATE autonomous_tasks
                        SET status = $1
                        WHERE id = $2
                    """, status, task_id)

                logger.info(f"Updated task {task_id} status to {status}")

        except Exception as e:
            logger.error(f"Failed to update task {task_id} status: {e}")

    async def get_task_result(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the result of a Celery task

        Args:
            task_id: Database task ID

        Returns:
            Task result if available
        """
        try:
            async with self.get_connection() as conn:
                task = await conn.fetchrow("""
                    SELECT metadata, status, result, error
                    FROM autonomous_tasks
                    WHERE id = $1
                """, task_id)

                if not task:
                    return None

                # Check if task has Celery ID
                metadata = task['metadata'] or {}
                celery_task_id = metadata.get('celery_task_id')

                if not celery_task_id:
                    return {
                        'status': task['status'],
                        'result': task['result'],
                        'error': task['error']
                    }

                # Check Celery task status
                from celery.result import AsyncResult
                from .celery_app import app

                celery_result = AsyncResult(celery_task_id, app=app)

                return {
                    'status': task['status'],
                    'celery_status': celery_result.status,
                    'result': task['result'],
                    'error': task['error'],
                    'celery_result': celery_result.result if celery_result.ready() else None
                }

        except Exception as e:
            logger.error(f"Failed to get task {task_id} result: {e}")
            return None

    async def cleanup(self):
        """Clean up resources"""
        if self._pool:
            await self._pool.close()
            logger.debug("Closed CeleryExecutor database pool")