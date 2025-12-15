#!/usr/bin/env python3
"""
Task Management API Routes for Echo Brain
Provides task creation, execution, and queue management endpoints
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Import task system components
from src.tasks.task_queue import TaskQueue, Task, TaskType, TaskPriority, TaskStatus
from src.tasks.task_implementation_executor import get_task_implementation_executor
from src.tasks.autonomous_behaviors import AutonomousBehaviors

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances
task_queue: Optional[TaskQueue] = None
autonomous_behaviors: Optional[AutonomousBehaviors] = None

class TaskCreateRequest(BaseModel):
    name: str
    task_type: str
    priority: str = "NORMAL"
    payload: Dict[str, Any] = {}
    scheduled_for: Optional[str] = None
    max_retries: int = 3
    timeout: int = 300

class TaskResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    result: Optional[Dict] = None
    error: Optional[str] = None

class ImplementTaskRequest(BaseModel):
    task: str
    service: str
    test: bool = True

async def get_task_queue() -> TaskQueue:
    """Get or initialize task queue"""
    global task_queue
    if task_queue is None:
        task_queue = TaskQueue()
        await task_queue.initialize()
        logger.info("✅ Task queue initialized")
    return task_queue

async def get_autonomous_behaviors() -> AutonomousBehaviors:
    """Get or initialize autonomous behaviors"""
    global autonomous_behaviors
    if autonomous_behaviors is None:
        queue = await get_task_queue()
        autonomous_behaviors = AutonomousBehaviors(queue)
        logger.info("✅ Autonomous behaviors initialized")
    return autonomous_behaviors

@router.post("/api/echo/tasks/create", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest):
    """Create a new task in the queue"""
    try:
        queue = await get_task_queue()

        # Parse enums
        try:
            task_type = TaskType(request.task_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid task_type: {request.task_type}")

        try:
            priority = TaskPriority[request.priority.upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {request.priority}")

        # Parse scheduled_for if provided
        scheduled_for = None
        if request.scheduled_for:
            try:
                scheduled_for = datetime.fromisoformat(request.scheduled_for)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid scheduled_for format (use ISO format)")

        # Create task
        task = Task(
            id=str(uuid.uuid4()),
            name=request.name,
            task_type=task_type,
            priority=priority,
            status=TaskStatus.PENDING,
            payload=request.payload,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            scheduled_for=scheduled_for,
            max_retries=request.max_retries,
            timeout=request.timeout
        )

        # Add to queue
        success = await queue.add_task(task)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add task to queue")

        return TaskResponse(
            id=task.id,
            name=task.name,
            status=task.status.value,
            created_at=task.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/api/echo/tasks/queue")
async def get_task_queue_status():
    """Get task queue status and statistics"""
    try:
        queue = await get_task_queue()
        stats = await queue.get_task_stats()

        return {
            "status": "active",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail=f"Queue status error: {str(e)}")

@router.get("/api/echo/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task by ID"""
    try:
        queue = await get_task_queue()
        task = await queue._load_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return TaskResponse(
            id=task.id,
            name=task.name,
            status=task.status.value,
            created_at=task.created_at.isoformat(),
            result=task.result,
            error=task.error
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Task retrieval error: {str(e)}")

@router.post("/api/echo/tasks/implement")
async def implement_task(request: ImplementTaskRequest):
    """Implement a task using the autonomous execution engine"""
    try:
        executor = get_task_implementation_executor()

        # Execute task implementation
        result = await executor.implement_task(
            task=request.task,
            service=request.service,
            test=request.test
        )

        return {
            "task_id": result.get("task_id"),
            "status": result.get("status"),
            "success": result.get("success", False),
            "message": result.get("message", "Task execution completed"),
            "steps_completed": result.get("steps_completed", []),
            "code_changes": result.get("code_changes", {}),
            "test_results": result.get("test_results"),
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error implementing task: {e}")
        raise HTTPException(status_code=500, detail=f"Task implementation error: {str(e)}")

@router.post("/api/echo/tasks/autonomous/start")
async def start_autonomous_behaviors():
    """Start autonomous behavior loops"""
    try:
        behaviors = await get_autonomous_behaviors()

        if behaviors.running:
            return {
                "status": "already_running",
                "message": "Autonomous behaviors are already active",
                "timestamp": datetime.now().isoformat()
            }

        # Start in background
        asyncio.create_task(behaviors.start())

        return {
            "status": "starting",
            "message": "Autonomous behaviors started successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error starting autonomous behaviors: {e}")
        raise HTTPException(status_code=500, detail=f"Autonomous start error: {str(e)}")

@router.post("/api/echo/tasks/autonomous/stop")
async def stop_autonomous_behaviors():
    """Stop autonomous behavior loops"""
    try:
        behaviors = await get_autonomous_behaviors()

        if not behaviors.running:
            return {
                "status": "not_running",
                "message": "Autonomous behaviors are not active",
                "timestamp": datetime.now().isoformat()
            }

        await behaviors.stop()

        return {
            "status": "stopped",
            "message": "Autonomous behaviors stopped successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error stopping autonomous behaviors: {e}")
        raise HTTPException(status_code=500, detail=f"Autonomous stop error: {str(e)}")

@router.get("/api/echo/tasks/autonomous/status")
async def get_autonomous_status():
    """Get autonomous behaviors status"""
    try:
        behaviors = await get_autonomous_behaviors()

        return {
            "running": behaviors.running,
            "intervals": behaviors.intervals,
            "modules": {
                "service_monitor": "active" if behaviors.running else "inactive",
                "system_monitor": "active" if behaviors.running else "inactive",
                "code_quality_monitor": "active" if behaviors.running else "inactive",
                "scheduler": "active" if behaviors.running else "inactive"
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting autonomous status: {e}")
        raise HTTPException(status_code=500, detail=f"Autonomous status error: {str(e)}")

@router.delete("/api/echo/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending task"""
    try:
        queue = await get_task_queue()

        # Update task status to cancelled
        await queue.update_task_status(task_id, TaskStatus.CANCELLED, error="Task cancelled by user")

        return {
            "status": "cancelled",
            "task_id": task_id,
            "message": "Task cancelled successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Task cancellation error: {str(e)}")

@router.post("/api/echo/tasks/cleanup")
async def cleanup_old_tasks(days: int = 7):
    """Clean up old completed/failed tasks"""
    try:
        queue = await get_task_queue()
        deleted_count = await queue.cleanup_old_tasks(days)

        return {
            "status": "completed",
            "deleted_count": deleted_count,
            "days": days,
            "message": f"Cleaned up {deleted_count} tasks older than {days} days",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error cleaning up tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Task cleanup error: {str(e)}")

logger.info("✅ Task routes module loaded with endpoints: create, queue, implement, autonomous control")