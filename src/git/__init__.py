"""
Git Operations Module for Echo Brain
Comprehensive git control system for autonomous operations
"""

from .autonomous_git_controller import (
    AutonomousGitController,
    AutonomousMode,
    OperationPriority,
    GitRepository,
    GitOperation,
    GitWorkflowConfig,
    autonomous_git_controller
)

__all__ = [
    'AutonomousGitController',
    'AutonomousMode',
    'OperationPriority',
    'GitRepository',
    'GitOperation',
    'GitWorkflowConfig',
    'autonomous_git_controller'
]