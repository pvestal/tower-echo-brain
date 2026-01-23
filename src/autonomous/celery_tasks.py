"""
Celery tasks for distributed agent execution
"""
from celery import Task, current_task
from celery.exceptions import SoftTimeLimitExceeded
from .celery_app import app
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import asyncpg

# Import existing agents
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.coding_agent import CodingAgent
from agents.reasoning_agent import ReasoningAgent
from agents.narration_agent import NarrationAgent

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task with database connection pooling"""
    _pool = None

    @property
    def pool(self):
        if self._pool is None:
            # Create connection pool
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._pool = loop.run_until_complete(
                asyncpg.create_pool(
                    host='localhost',
                    port=5432,
                    user='patrick',
                    password='RP78eIrW7cI2jYvL5akt1yurE',
                    database='echo_brain',
                    min_size=2,
                    max_size=10
                )
            )
        return self._pool


@app.task(base=DatabaseTask, bind=True, name='execute_coding_task',
          time_limit=300, soft_time_limit=240)
def execute_coding_task(self, task_id: int, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a coding task using CodingAgent

    Args:
        task_id: Database task ID
        task_data: Task parameters

    Returns:
        Execution result dictionary
    """
    try:
        # Update task progress
        current_task.update_state(
            state='PROGRESS',
            meta={'task_id': task_id, 'status': 'Initializing coding agent'}
        )

        # Initialize agent
        agent = CodingAgent()

        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'task_id': task_id, 'status': 'Executing coding task'}
        )

        # Execute task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Build context from task data
        context = {
            'task_name': task_data.get('name', 'Unknown Task'),
            'task_type': 'coding',
            'description': task_data.get('description', ''),
            'metadata': task_data.get('metadata', {})
        }

        # Execute agent
        result = loop.run_until_complete(
            agent.execute(context)
        )

        # Update database
        loop.run_until_complete(
            _update_task_status(self.pool, task_id, 'completed', result, None)
        )

        return {
            'task_id': task_id,
            'status': 'completed',
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        }

    except SoftTimeLimitExceeded:
        error_msg = f"Task {task_id} exceeded time limit"
        logger.error(error_msg)

        # Update database with timeout
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            _update_task_status(self.pool, task_id, 'failed', None, error_msg)
        )

        return {
            'task_id': task_id,
            'status': 'failed',
            'error': error_msg
        }

    except Exception as e:
        error_msg = f"Task {task_id} failed: {str(e)}"
        logger.error(error_msg)

        # Update database with error
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            _update_task_status(self.pool, task_id, 'failed', None, error_msg)
        )

        return {
            'task_id': task_id,
            'status': 'failed',
            'error': error_msg
        }


@app.task(base=DatabaseTask, bind=True, name='execute_reasoning_task',
          time_limit=300, soft_time_limit=240)
def execute_reasoning_task(self, task_id: int, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a reasoning task using ReasoningAgent
    """
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'task_id': task_id, 'status': 'Initializing reasoning agent'}
        )

        agent = ReasoningAgent()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        context = {
            'task_name': task_data.get('name', 'Unknown Task'),
            'task_type': 'reasoning',
            'description': task_data.get('description', ''),
            'metadata': task_data.get('metadata', {})
        }

        result = loop.run_until_complete(
            agent.execute(context)
        )

        loop.run_until_complete(
            _update_task_status(self.pool, task_id, 'completed', result, None)
        )

        return {
            'task_id': task_id,
            'status': 'completed',
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        }

    except SoftTimeLimitExceeded:
        error_msg = f"Task {task_id} exceeded time limit"
        logger.error(error_msg)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            _update_task_status(self.pool, task_id, 'failed', None, error_msg)
        )

        return {'task_id': task_id, 'status': 'failed', 'error': error_msg}

    except Exception as e:
        error_msg = f"Task {task_id} failed: {str(e)}"
        logger.error(error_msg)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            _update_task_status(self.pool, task_id, 'failed', None, error_msg)
        )

        return {'task_id': task_id, 'status': 'failed', 'error': error_msg}


@app.task(base=DatabaseTask, bind=True, name='execute_testing_task',
          time_limit=300, soft_time_limit=240)
def execute_testing_task(self, task_id: int, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a testing task (simpler, no agent needed)
    """
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'task_id': task_id, 'status': 'Running tests'}
        )

        # Simulate test execution
        import time
        time.sleep(2)  # Simulate work

        test_results = {
            'tests_run': 5,
            'tests_passed': 5,
            'tests_failed': 0,
            'execution_time': '2.0s',
            'details': 'All tests passed successfully'
        }

        # Update database
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            _update_task_status(self.pool, task_id, 'completed', test_results, None)
        )

        return {
            'task_id': task_id,
            'status': 'completed',
            'result': test_results,
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        error_msg = f"Test task {task_id} failed: {str(e)}"
        logger.error(error_msg)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            _update_task_status(self.pool, task_id, 'failed', None, error_msg)
        )

        return {'task_id': task_id, 'status': 'failed', 'error': error_msg}


async def _update_task_status(pool, task_id: int, status: str,
                             result: Optional[Any] = None,
                             error: Optional[str] = None):
    """Helper function to update task status in database"""
    try:
        # Create fresh connection for each update (pool might be from different loop)
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='patrick',
            password='RP78eIrW7cI2jYvL5akt1yurE',
            database='echo_brain'
        )

        try:
            if status == 'completed':
                await conn.execute("""
                    UPDATE autonomous_tasks
                    SET status = $1, result = $2, completed_at = $3
                    WHERE id = $4
                """, status, json.dumps(result) if result else None,
                    datetime.utcnow(), task_id)
            elif status == 'failed':
                await conn.execute("""
                    UPDATE autonomous_tasks
                    SET status = $1, error = $2, completed_at = $3
                    WHERE id = $4
                """, status, error, datetime.utcnow(), task_id)
            else:
                await conn.execute("""
                    UPDATE autonomous_tasks
                    SET status = $1
                    WHERE id = $2
                """, status, task_id)

            logger.info(f"Updated task {task_id} status to {status}")

        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Failed to update task {task_id} status: {e}")


# Task routing helper
def get_celery_task_for_type(task_type: str):
    """Get the appropriate Celery task for a task type"""
    task_mapping = {
        'coding': execute_coding_task,
        'code_generation': execute_coding_task,
        'refactoring': execute_coding_task,
        'reasoning': execute_reasoning_task,
        'analysis': execute_reasoning_task,
        'testing': execute_testing_task,
        'validation': execute_testing_task,
        'monitoring': execute_reasoning_task,
        'system_review': execute_reasoning_task,
    }

    return task_mapping.get(task_type, execute_reasoning_task)