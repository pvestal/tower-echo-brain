#!/usr/bin/env python3
"""Dependency Container - IoC container for Echo Brain"""

import logging
from typing import Any, Dict, Callable, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class Lifecycle(Enum):
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

class DependencyContainer:
    """Dependency injection container"""

    def __init__(self):
        self.services = {}
        self.instances = {}

    def register_singleton(self, name: str, factory: Callable) -> None:
        """Register a singleton service"""
        self.services[name] = (factory, Lifecycle.SINGLETON)

    def register_transient(self, name: str, factory: Callable) -> None:
        """Register a transient service"""
        self.services[name] = (factory, Lifecycle.TRANSIENT)

    async def resolve(self, name: str) -> Any:
        """Resolve a service"""
        if name not in self.services:
            raise ValueError(f"Service {name} not registered")

        factory, lifecycle = self.services[name]

        if lifecycle == Lifecycle.SINGLETON:
            if name not in self.instances:
                self.instances[name] = factory()
            return self.instances[name]
        elif lifecycle == Lifecycle.TRANSIENT:
            return factory()

        return factory()

    def resolve(self, name: str) -> Any:
        """Sync resolve for compatibility"""
        if name not in self.services:
            raise ValueError(f"Service {name} not registered")

        factory, lifecycle = self.services[name]

        if lifecycle == Lifecycle.SINGLETON:
            if name not in self.instances:
                self.instances[name] = factory()
            return self.instances[name]

        return factory()
