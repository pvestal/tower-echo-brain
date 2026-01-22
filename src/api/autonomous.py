"""
Autonomous Core API endpoints for Echo Brain

Provides REST API endpoints for managing autonomous operations including
goals, tasks, approvals, system control, and audit logging.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field
import asyncio

from src.autonomous import (
    AutonomousCore, AutonomousState, SystemStatus,
    GoalManager, Scheduler, Executor, SafetyController, AuditLogger
)
from src.autonomous.notifications import get_notification_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/autonomous", tags=["autonomous"])

# Global autonomous core instance
autonomous_core: Optional[AutonomousCore] = None

# Pydantic models for API requests/responses
class GoalCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Goal name")
    description: str = Field(..., min_length=1, max_length=1000, description="Goal description")
    goal_type: str = Field(..., description="Type of goal (e.g., 'research', 'optimization', 'maintenance')")
    priority: int = Field(default=5, ge=1, le=10, description="Priority level (1-10, higher is more important)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional goal metadata")

class GoalUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1, max_length=1000)
    priority: Optional[int] = Field(None, ge=1, le=10)
    status: Optional[str] = Field(None, description="Goal status")
    metadata: Optional[Dict[str, Any]] = None

class GoalResponse(BaseModel):
    id: int
    name: str
    description: str
    goal_type: str
    priority: int
    status: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    id: int
    goal_id: int
    name: str
    description: str
    task_type: str
    status: str
    priority: int
    scheduled_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]

class ApprovalResponse(BaseModel):
    task_id: int
    task_name: str
    task_description: str
    task_type: str
    requires_approval: bool
    approval_status: Optional[str]
    requested_at: datetime
    metadata: Optional[Dict[str, Any]]

class ApprovalDecisionRequest(BaseModel):
    reason: Optional[str] = Field(None, max_length=500, description="Reason for approval/rejection")

class SystemStatusResponse(BaseModel):
    state: str
    uptime_seconds: float
    cycles_completed: int
    tasks_executed: int
    last_cycle_time: Optional[datetime]
    last_error: Optional[str]
    components_status: Dict[str, bool]

class AuditLogResponse(BaseModel):
    id: int
    timestamp: datetime
    event_type: str
    component: str
    message: str
    level: str
    metadata: Optional[Dict[str, Any]]

class ControlRequest(BaseModel):
    reason: Optional[str] = Field(None, max_length=500, description="Reason for the action")

class NotificationResponse(BaseModel):
    id: int
    notification_type: str
    title: str
    message: str
    task_id: Optional[int]
    read: bool
    created_at: datetime

class NotificationCountResponse(BaseModel):
    unread_count: int
    total_count: Optional[int] = None

# Utility functions
async def get_autonomous_core() -> AutonomousCore:
    """Get or create the autonomous core instance."""
    global autonomous_core
    if autonomous_core is None:
        autonomous_core = AutonomousCore()
        # Core initializes its components in __init__, no separate initialize() needed
    return autonomous_core

# Status endpoints
@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get current system status and health information."""
    try:
        core = await get_autonomous_core()
        status = await core.get_status()

        return SystemStatusResponse(
            state=status.state.value,
            uptime_seconds=status.uptime_seconds,
            cycles_completed=status.cycles_completed,
            tasks_executed=status.tasks_executed,
            last_cycle_time=status.last_cycle_time,
            last_error=status.last_error,
            components_status=status.components_status
        )
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

# Goal management endpoints
@router.get("/goals", response_model=List[GoalResponse])
async def list_goals(
    status: Optional[str] = Query(None, description="Filter by goal status"),
    goal_type: Optional[str] = Query(None, description="Filter by goal type"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of goals to return")
):
    """List all goals with optional filtering."""
    try:
        core = await get_autonomous_core()
        goals = await core.goal_manager.get_goals(
            status=status,
            goal_type=goal_type,
            limit=limit
        )

        return [
            GoalResponse(
                id=goal['id'],
                name=goal['name'],
                description=goal['description'],
                goal_type=goal['goal_type'],
                priority=goal['priority'],
                status=goal['status'],
                created_at=goal['created_at'],
                updated_at=goal['updated_at'],
                completed_at=goal.get('completed_at'),
                metadata=goal.get('metadata')
            )
            for goal in goals
        ]
    except Exception as e:
        logger.error(f"Failed to list goals: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list goals: {str(e)}")

@router.post("/goals", response_model=GoalResponse)
async def create_goal(request: GoalCreateRequest):
    """Create a new autonomous goal."""
    try:
        core = await get_autonomous_core()
        goal_id = await core.goal_manager.create_goal(
            name=request.name,
            description=request.description,
            goal_type=request.goal_type,
            priority=request.priority,
            metadata=request.metadata
        )

        # Get the created goal to return full details
        goal = await core.goal_manager.get_goal_by_id(goal_id)
        if not goal:
            raise HTTPException(status_code=404, detail="Created goal not found")

        return GoalResponse(
            id=goal['id'],
            name=goal['name'],
            description=goal['description'],
            goal_type=goal['goal_type'],
            priority=goal['priority'],
            status=goal['status'],
            created_at=goal['created_at'],
            updated_at=goal['updated_at'],
            completed_at=goal.get('completed_at'),
            metadata=goal.get('metadata')
        )
    except Exception as e:
        logger.error(f"Failed to create goal: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create goal: {str(e)}")

@router.patch("/goals/{goal_id}", response_model=GoalResponse)
async def update_goal(goal_id: int = Path(..., description="Goal ID"), request: GoalUpdateRequest = ...):
    """Update an existing goal."""
    try:
        core = await get_autonomous_core()

        # Build update data from non-None fields
        update_data = {}
        if request.name is not None:
            update_data['name'] = request.name
        if request.description is not None:
            update_data['description'] = request.description
        if request.priority is not None:
            update_data['priority'] = request.priority
        if request.status is not None:
            update_data['status'] = request.status
        if request.metadata is not None:
            update_data['metadata'] = request.metadata

        if not update_data:
            raise HTTPException(status_code=400, detail="No update data provided")

        success = await core.goal_manager.update_goal(goal_id, update_data)
        if not success:
            raise HTTPException(status_code=404, detail="Goal not found")

        # Get updated goal
        goal = await core.goal_manager.get_goal_by_id(goal_id)
        return GoalResponse(
            id=goal['id'],
            name=goal['name'],
            description=goal['description'],
            goal_type=goal['goal_type'],
            priority=goal['priority'],
            status=goal['status'],
            created_at=goal['created_at'],
            updated_at=goal['updated_at'],
            completed_at=goal.get('completed_at'),
            metadata=goal.get('metadata')
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update goal {goal_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update goal: {str(e)}")

@router.delete("/goals/{goal_id}")
async def cancel_goal(goal_id: int = Path(..., description="Goal ID")):
    """Cancel/delete a goal and its associated tasks."""
    try:
        core = await get_autonomous_core()
        success = await core.goal_manager.cancel_goal(goal_id)
        if not success:
            raise HTTPException(status_code=404, detail="Goal not found")

        return {"message": f"Goal {goal_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel goal {goal_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel goal: {str(e)}")

# Task management endpoints
@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    goal_id: Optional[int] = Query(None, description="Filter by goal ID"),
    status: Optional[str] = Query(None, description="Filter by task status"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of tasks to return")
):
    """List tasks with optional filtering."""
    try:
        core = await get_autonomous_core()
        tasks = await core.goal_manager.get_tasks(
            goal_id=goal_id,
            status=status,
            task_type=task_type,
            limit=limit
        )

        return [
            TaskResponse(
                id=task['id'],
                goal_id=task['goal_id'],
                name=task['name'],
                description=task['description'],
                task_type=task['task_type'],
                status=task['status'],
                priority=task['priority'],
                scheduled_at=task.get('scheduled_at'),
                started_at=task.get('started_at'),
                completed_at=task.get('completed_at'),
                result=task.get('result'),
                metadata=task.get('metadata')
            )
            for task in tasks
        ]
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

# Approval workflow endpoints
@router.get("/approvals", response_model=List[ApprovalResponse])
async def get_pending_approvals():
    """Get all tasks pending approval."""
    try:
        core = await get_autonomous_core()
        approvals = await core.safety_controller.get_pending_approvals()

        return [
            ApprovalResponse(
                task_id=approval['task_id'],
                task_name=approval['task_name'],
                task_description=approval['task_description'],
                task_type=approval['task_type'],
                requires_approval=approval['requires_approval'],
                approval_status=approval.get('approval_status'),
                requested_at=approval['requested_at'],
                metadata=approval.get('metadata')
            )
            for approval in approvals
        ]
    except Exception as e:
        logger.error(f"Failed to get pending approvals: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get approvals: {str(e)}")

@router.post("/approvals/{task_id}/approve")
async def approve_task(
    task_id: int = Path(..., description="Task ID to approve"),
    request: ApprovalDecisionRequest = ApprovalDecisionRequest()
):
    """Approve a pending task."""
    try:
        core = await get_autonomous_core()
        success = await core.safety_controller.approve_task(task_id, reason=request.reason)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or not pending approval")

        return {"message": f"Task {task_id} approved successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to approve task: {str(e)}")

@router.post("/approvals/{task_id}/reject")
async def reject_task(
    task_id: int = Path(..., description="Task ID to reject"),
    request: ApprovalDecisionRequest = ApprovalDecisionRequest()
):
    """Reject a pending task."""
    try:
        core = await get_autonomous_core()
        success = await core.safety_controller.reject_task(task_id, reason=request.reason)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or not pending approval")

        return {"message": f"Task {task_id} rejected successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reject task: {str(e)}")

# System control endpoints
@router.post("/start")
async def start_autonomous_operations(request: ControlRequest = ControlRequest()):
    """Start autonomous operations."""
    try:
        core = await get_autonomous_core()
        success = await core.start()
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start autonomous operations")

        return {"message": "Autonomous operations started successfully", "reason": request.reason}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start autonomous operations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start: {str(e)}")

@router.post("/stop")
async def stop_autonomous_operations(request: ControlRequest = ControlRequest()):
    """Stop autonomous operations."""
    try:
        core = await get_autonomous_core()
        success = await core.stop()
        if not success:
            raise HTTPException(status_code=400, detail="Failed to stop autonomous operations")

        return {"message": "Autonomous operations stopped successfully", "reason": request.reason}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop autonomous operations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop: {str(e)}")

@router.post("/pause")
async def pause_autonomous_operations(request: ControlRequest = ControlRequest()):
    """Pause autonomous operations."""
    try:
        core = await get_autonomous_core()
        success = await core.pause()
        if not success:
            raise HTTPException(status_code=400, detail="Failed to pause autonomous operations")

        return {"message": "Autonomous operations paused successfully", "reason": request.reason}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause autonomous operations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause: {str(e)}")

@router.post("/resume")
async def resume_autonomous_operations(request: ControlRequest = ControlRequest()):
    """Resume paused autonomous operations."""
    try:
        core = await get_autonomous_core()
        success = await core.resume()
        if not success:
            raise HTTPException(status_code=400, detail="Failed to resume autonomous operations")

        return {"message": "Autonomous operations resumed successfully", "reason": request.reason}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume autonomous operations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume: {str(e)}")

@router.post("/kill")
async def activate_kill_switch(request: ControlRequest = ControlRequest()):
    """Activate emergency kill switch to immediately stop all autonomous operations."""
    try:
        core = await get_autonomous_core()
        success = await core.kill_switch()
        if not success:
            raise HTTPException(status_code=400, detail="Failed to activate kill switch")

        return {"message": "Kill switch activated - all autonomous operations stopped", "reason": request.reason}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate kill switch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to activate kill switch: {str(e)}")

# Audit and logging endpoints
@router.get("/audit", response_model=List[AuditLogResponse])
async def get_audit_logs(
    component: Optional[str] = Query(None, description="Filter by component"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    start_time: Optional[datetime] = Query(None, description="Filter from this time"),
    end_time: Optional[datetime] = Query(None, description="Filter until this time"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of logs to return")
):
    """Get audit logs with optional filtering."""
    try:
        core = await get_autonomous_core()
        logs = await core.audit_logger.get_audit_logs(
            component=component,
            event_type=event_type,
            level=level,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        return [
            AuditLogResponse(
                id=log['id'],
                timestamp=log['timestamp'],
                event_type=log['event_type'],
                component=log['component'],
                message=log['message'],
                level=log['level'],
                metadata=log.get('metadata')
            )
            for log in logs
        ]
    except Exception as e:
        logger.error(f"Failed to get audit logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get audit logs: {str(e)}")

# Notification endpoints
@router.get("/notifications", response_model=List[NotificationResponse])
async def get_notifications(
    unread_only: bool = Query(True, description="Only return unread notifications"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of notifications to return")
):
    """Get notifications, defaulting to unread only."""
    try:
        notification_manager = get_notification_manager()

        if unread_only:
            notifications = await notification_manager.get_unread_notifications(limit=limit)
        else:
            notifications = await notification_manager.get_all_notifications(limit=limit)

        return [
            NotificationResponse(
                id=notif['id'],
                notification_type=notif['notification_type'],
                title=notif['title'],
                message=notif['message'],
                task_id=notif.get('task_id'),
                read=notif['read'],
                created_at=notif['created_at']
            )
            for notif in notifications
        ]
    except Exception as e:
        logger.error(f"Failed to get notifications: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get notifications: {str(e)}")

@router.get("/notifications/count", response_model=NotificationCountResponse)
async def get_notification_count(
    notification_type: Optional[str] = Query(None, description="Filter by notification type")
):
    """Get count of unread notifications. Patrick can poll this endpoint."""
    try:
        notification_manager = get_notification_manager()
        unread_count = await notification_manager.get_unread_count(notification_type=notification_type)

        return NotificationCountResponse(unread_count=unread_count)
    except Exception as e:
        logger.error(f"Failed to get notification count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get count: {str(e)}")

@router.post("/notifications/{notification_id}/read")
async def mark_notification_as_read(
    notification_id: int = Path(..., description="Notification ID to mark as read")
):
    """Mark a specific notification as read."""
    try:
        notification_manager = get_notification_manager()
        success = await notification_manager.mark_as_read(notification_id)

        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")

        return {"message": f"Notification {notification_id} marked as read"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark notification as read: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to mark as read: {str(e)}")

@router.post("/notifications/mark-all-read")
async def mark_all_notifications_as_read(
    notification_type: Optional[str] = Query(None, description="Only mark this type as read")
):
    """Mark all notifications as read."""
    try:
        notification_manager = get_notification_manager()
        count = await notification_manager.mark_all_as_read(notification_type=notification_type)

        return {"message": f"Marked {count} notifications as read"}
    except Exception as e:
        logger.error(f"Failed to mark all as read: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to mark all as read: {str(e)}")