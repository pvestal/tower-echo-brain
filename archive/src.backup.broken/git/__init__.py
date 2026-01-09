"""
Git Operations Module for Echo Brain
Comprehensive git control system for autonomous operations
"""

# Core git control system
from .autonomous_git_controller import (
    AutonomousGitController,
    AutonomousMode,
    OperationPriority,
    GitRepository,
    GitOperation,
    GitWorkflowConfig,
    autonomous_git_controller
)

# Cross-repository workflow coordination
from .workflow_coordinator import (
    WorkflowCoordinator,
    WorkflowType,
    TriggerType,
    WorkflowRule,
    WorkflowExecution,
    DependencyMap,
    get_workflow_coordinator
)

# Security and credential management
from .security_manager import (
    GitSecurityManager,
    CredentialType,
    SecurityLevel,
    Credential,
    SSHKey,
    SecurityAuditEntry,
    git_security_manager
)

# Intelligent git operations
from .intelligent_git_assistant import (
    IntelligentGitAssistant,
    ChangeType,
    ConflictType,
    Priority,
    CodeChange,
    CommitAnalysis,
    ConflictInfo,
    ConflictResolution,
    intelligent_git_assistant
)

# Testing framework
from .git_test_framework import (
    GitTestFramework,
    TestType,
    TestStatus,
    TestResult,
    TestSuite,
    SandboxEnvironment,
    git_test_framework
)

# Echo Brain integration
from .echo_git_integration import (
    EchoGitIntegration,
    GitOperationPriority as EchoGitPriority,
    AutonomousGitAction,
    GitAutonomousTask,
    GitIntelligenceReport,
    echo_git_integration
)

__all__ = [
    # Core controller
    'AutonomousGitController',
    'AutonomousMode',
    'OperationPriority',
    'GitRepository',
    'GitOperation',
    'GitWorkflowConfig',
    'autonomous_git_controller',

    # Workflow coordination
    'WorkflowCoordinator',
    'WorkflowType',
    'TriggerType',
    'WorkflowRule',
    'WorkflowExecution',
    'DependencyMap',
    'get_workflow_coordinator',

    # Security management
    'GitSecurityManager',
    'CredentialType',
    'SecurityLevel',
    'Credential',
    'SSHKey',
    'SecurityAuditEntry',
    'git_security_manager',

    # Intelligent operations
    'IntelligentGitAssistant',
    'ChangeType',
    'ConflictType',
    'Priority',
    'CodeChange',
    'CommitAnalysis',
    'ConflictInfo',
    'ConflictResolution',
    'intelligent_git_assistant',

    # Testing framework
    'GitTestFramework',
    'TestType',
    'TestStatus',
    'TestResult',
    'TestSuite',
    'SandboxEnvironment',
    'git_test_framework',

    # Echo Brain integration
    'EchoGitIntegration',
    'EchoGitPriority',
    'AutonomousGitAction',
    'GitAutonomousTask',
    'GitIntelligenceReport',
    'echo_git_integration'
]