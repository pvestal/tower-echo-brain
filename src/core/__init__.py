#!/usr/bin/env python3
"""
Echo Brain Core Components

This package contains core infrastructure for the Echo Brain system,
including dependency injection, configuration management, and
component lifecycle management.
"""

from .container import (
    Container,
    Environment,
    LifecycleType,
    ComponentConfig,
    ScopeManager,
    ServiceLocator,
    get_container,
    set_container,
    reset_container,
    get_llm,
    get_embedding,
    get_vision,
    get_vector_store,
    create_test_container,
    create_mock_container
)

__all__ = [
    # Dependency injection
    'Container',
    'Environment',
    'LifecycleType',
    'ComponentConfig',
    'ScopeManager',
    'ServiceLocator',

    # Container management
    'get_container',
    'set_container',
    'reset_container',

    # Convenience functions
    'get_llm',
    'get_embedding',
    'get_vision',
    'get_vector_store',

    # Testing utilities
    'create_test_container',
    'create_mock_container',
]