"""
Autonomous Core Foundation for Echo Brain Phase 1

This module provides the foundational components for Echo Brain's autonomous operations:
- Goal management and task orchestration
- Task scheduling and prioritization
- Task execution with agent routing
- Event monitoring and triggers
- Safety controls and approval workflows
- Comprehensive audit logging
- Database persistence layer

Key Components:
- GoalManager: Creates, tracks, and manages autonomous goals
- Scheduler: Manages task scheduling and prioritization
- Executor: Routes and executes tasks using specialized agents
- EventWatcher: Monitors for triggering events
- SafetyController: Enforces safety limits and approval requirements
- AuditLogger: Maintains comprehensive audit trails
- AutonomousCore: Main orchestrator for all autonomous operations
"""

from .goals import GoalManager
from .scheduler import Scheduler, ScheduleConfig, ScheduledTask
from .executor import Executor, TaskResult
from .events import EventWatcher, EventTrigger, DetectedEvent
from .safety import SafetyController
from .audit import AuditLogger
from .core import AutonomousCore, AutonomousState, SystemStatus

__all__ = [
    "GoalManager",
    "Scheduler",
    "ScheduleConfig",
    "ScheduledTask",
    "Executor",
    "TaskResult",
    "EventWatcher",
    "EventTrigger",
    "DetectedEvent",
    "SafetyController",
    "AuditLogger",
    "AutonomousCore",
    "AutonomousState",
    "SystemStatus"
]