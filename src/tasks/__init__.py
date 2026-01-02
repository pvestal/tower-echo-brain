"""
Echo Brain Task System
Autonomous task execution and management for Echo
"""

from src.tasks.task_queue import TaskQueue, Task, TaskPriority, TaskStatus
from src.tasks.background_worker import BackgroundWorker

__all__ = [
    'TaskQueue', 'Task', 'TaskPriority', 'TaskStatus',
    'BackgroundWorker'
]

# Import AutonomousBehaviors lazily to avoid circular imports
def get_autonomous_behaviors():
    from src.tasks.autonomous_behaviors import AutonomousBehaviors
    return AutonomousBehaviors
