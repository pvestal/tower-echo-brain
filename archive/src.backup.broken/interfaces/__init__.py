#!/usr/bin/env python3
"""
Echo Brain Interface Protocols
Comprehensive protocol definitions for type safety and architectural consistency
"""

from typing import Protocol, runtime_checkable

from .vector_memory import VectorMemoryInterface
from .task_orchestrator import TaskOrchestratorInterface
from .model_manager import ModelInterface, ModelManagerInterface
from .database import DatabaseInterface, AsyncDatabaseInterface
from .conversation import ConversationManagerInterface
from .security import (
    AuthenticationInterface,
    AuthorizationInterface,
    SecurityAuditInterface,
    EncryptionInterface
)
from .conversation import (
    ConversationManagerInterface,
    ContextManagerInterface,
    MessageProcessorInterface,
    ConversationAnalyzerInterface
)

__all__ = [
    'VectorMemoryInterface',
    'TaskOrchestratorInterface',
    'ModelInterface',
    'ModelManagerInterface',
    'DatabaseInterface',
    'AsyncDatabaseInterface',
    'ConversationManagerInterface',
    'ContextManagerInterface',
    'MessageProcessorInterface',
    'ConversationAnalyzerInterface',
    'AuthenticationInterface',
    'AuthorizationInterface',
    'SecurityAuditInterface',
    'EncryptionInterface'
]