"""
Data models for autonomous system
"""
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
from datetime import datetime


@dataclass
class TaskResult:
    """Represents the result of a task execution"""
    task_id: int
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    agent_used: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            'task_id': self.task_id,
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'agent_used': self.agent_used,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    id: int
    goal_id: int
    name: str
    task_type: str
    priority: int = 5
    scheduled_at: Optional[datetime] = None
    safety_level: str = 'auto'
    status: str = 'pending'
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)