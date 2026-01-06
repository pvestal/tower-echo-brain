#!/usr/bin/env python3
"""
Task Orchestrator Interface Protocol
Defines the contract for task management and orchestration systems
"""

from typing import Protocol, runtime_checkable, List, Dict, Optional, Any, Callable, AsyncIterator
from datetime import datetime
from enum import Enum

# Import existing enums from task_queue.py to maintain compatibility
from ..tasks.task_queue import Task, TaskPriority, TaskStatus, TaskType

@runtime_checkable
class TaskOrchestratorInterface(Protocol):
    """
    Protocol for task orchestration and management systems

    Defines standardized methods for creating, managing, and executing
    tasks in the Echo Brain autonomous system.
    """

    async def add_task(self, name: str, task_type: TaskType, priority: TaskPriority,
                      payload: Dict[str, Any], scheduled_for: Optional[datetime] = None,
                      dependencies: Optional[List[str]] = None,
                      timeout: int = 300, max_retries: int = 3) -> str:
        """
        Add a new task to the queue

        Args:
            name: Human-readable task name
            task_type: Type of task from TaskType enum
            priority: Task priority from TaskPriority enum
            payload: Task payload data
            scheduled_for: Optional scheduled execution time
            dependencies: Optional list of task IDs that must complete first
            timeout: Task timeout in seconds
            max_retries: Maximum retry attempts

        Returns:
            str: Unique task ID
        """
        ...

    async def get_next_task(self) -> Optional[Task]:
        """
        Get the next available task for execution

        Returns:
            Optional[Task]: Next task to execute or None if queue is empty
        """
        ...

    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """
        Mark a task as completed with results

        Args:
            task_id: Unique task identifier
            result: Task execution results

        Returns:
            bool: True if task was successfully completed
        """
        ...

    async def fail_task(self, task_id: str, error: str, retry: bool = True) -> bool:
        """
        Mark a task as failed

        Args:
            task_id: Unique task identifier
            error: Error message describing the failure
            retry: Whether to retry the task automatically

        Returns:
            bool: True if task failure was recorded
        """
        ...

    async def cancel_task(self, task_id: str, reason: Optional[str] = None) -> bool:
        """
        Cancel a pending or running task

        Args:
            task_id: Unique task identifier
            reason: Optional cancellation reason

        Returns:
            bool: True if task was successfully cancelled
        """
        ...

    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Retrieve a specific task by ID

        Args:
            task_id: Unique task identifier

        Returns:
            Optional[Task]: Task object or None if not found
        """
        ...

    async def list_tasks(self, status: Optional[TaskStatus] = None,
                        task_type: Optional[TaskType] = None,
                        limit: int = 100, offset: int = 0) -> List[Task]:
        """
        List tasks with optional filtering

        Args:
            status: Optional status filter
            task_type: Optional type filter
            limit: Maximum number of tasks to return
            offset: Number of tasks to skip

        Returns:
            List[Task]: List of matching tasks
        """
        ...

    async def get_task_statistics(self) -> Dict[str, Any]:
        """
        Get task queue statistics

        Returns:
            Dict: Statistics including counts by status, type, etc.
        """
        ...

    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update task metadata

        Args:
            task_id: Unique task identifier
            updates: Dictionary of fields to update

        Returns:
            bool: True if task was successfully updated
        """
        ...

    async def retry_task(self, task_id: str) -> bool:
        """
        Manually retry a failed task

        Args:
            task_id: Unique task identifier

        Returns:
            bool: True if task was queued for retry
        """
        ...

    async def schedule_recurring_task(self, name: str, task_type: TaskType,
                                    priority: TaskPriority, payload: Dict[str, Any],
                                    cron_expression: str) -> str:
        """
        Schedule a recurring task using cron syntax

        Args:
            name: Task name
            task_type: Task type
            priority: Task priority
            payload: Task payload
            cron_expression: Cron expression for scheduling

        Returns:
            str: Recurring task schedule ID
        """
        ...

    async def pause_task_processing(self) -> bool:
        """
        Pause task processing temporarily

        Returns:
            bool: True if processing was paused
        """
        ...

    async def resume_task_processing(self) -> bool:
        """
        Resume task processing

        Returns:
            bool: True if processing was resumed
        """
        ...

    async def clear_completed_tasks(self, older_than_hours: int = 24) -> int:
        """
        Clear completed tasks older than specified hours

        Args:
            older_than_hours: Age threshold in hours

        Returns:
            int: Number of tasks cleared
        """
        ...


@runtime_checkable
class TaskExecutorInterface(Protocol):
    """
    Protocol for task execution implementations
    """

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a specific task

        Args:
            task: Task to execute

        Returns:
            Dict[str, Any]: Task execution results
        """
        ...

    def can_handle_task_type(self, task_type: TaskType) -> bool:
        """
        Check if this executor can handle a task type

        Args:
            task_type: Task type to check

        Returns:
            bool: True if can handle this task type
        """
        ...

    def get_supported_task_types(self) -> List[TaskType]:
        """
        Get list of supported task types

        Returns:
            List[TaskType]: Supported task types
        """
        ...


@runtime_checkable
class TaskQueueInterface(Protocol):
    """
    Protocol for task queue storage backends
    """

    async def enqueue(self, task: Task) -> bool:
        """
        Add task to queue

        Args:
            task: Task to enqueue

        Returns:
            bool: True if task was successfully enqueued
        """
        ...

    async def dequeue(self) -> Optional[Task]:
        """
        Remove and return next task from queue

        Returns:
            Optional[Task]: Next task or None if queue is empty
        """
        ...

    async def peek(self) -> Optional[Task]:
        """
        View next task without removing from queue

        Returns:
            Optional[Task]: Next task or None if queue is empty
        """
        ...

    async def size(self) -> int:
        """
        Get current queue size

        Returns:
            int: Number of tasks in queue
        """
        ...

    async def clear(self) -> int:
        """
        Clear all tasks from queue

        Returns:
            int: Number of tasks cleared
        """
        ...


@runtime_checkable
class TaskSchedulerInterface(Protocol):
    """
    Protocol for task scheduling systems
    """

    async def schedule_task(self, task: Task, run_at: datetime) -> bool:
        """
        Schedule a task for future execution

        Args:
            task: Task to schedule
            run_at: When to execute the task

        Returns:
            bool: True if task was successfully scheduled
        """
        ...

    async def schedule_recurring(self, task_template: Task, cron_expression: str) -> str:
        """
        Schedule a recurring task

        Args:
            task_template: Template task for recurring execution
            cron_expression: Cron expression for timing

        Returns:
            str: Schedule identifier
        """
        ...

    async def cancel_schedule(self, schedule_id: str) -> bool:
        """
        Cancel a scheduled task

        Args:
            schedule_id: Schedule identifier

        Returns:
            bool: True if schedule was cancelled
        """
        ...

    async def list_scheduled(self) -> List[Dict[str, Any]]:
        """
        List all scheduled tasks

        Returns:
            List[Dict]: Scheduled task information
        """
        ...

    async def get_due_tasks(self) -> List[Task]:
        """
        Get tasks that are due for execution

        Returns:
            List[Task]: Tasks ready to execute
        """
        ...