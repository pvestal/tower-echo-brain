#!/usr/bin/env python3
"""
Dependency Injection Container for Echo Brain ML components.
Manages creation and lifecycle of ML interfaces and implementations.
Patrick Vestal - December 9, 2025
"""

import os
import logging
from typing import Dict, Any, Optional, Type, TypeVar, Union, Protocol
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

# Import interfaces
from ..interfaces.ml_model_interface import MLModelInterface
from ..interfaces.llm_interface import (
    LLMInterface, OllamaLLMInterface, CodeLLMInterface, ConversationalLLMInterface
)
from ..interfaces.embedding_interface import (
    EmbeddingInterface, SemanticEmbeddingInterface, MultilingualEmbeddingInterface
)
from ..interfaces.vision_interface import (
    VisionInterface, ImageClassificationInterface, ObjectDetectionInterface,
    ImageQualityInterface, ImageGenerationInterface, ImageEnhancementInterface
)
from ..interfaces.vector_store_interface import (
    VectorStoreInterface, AdvancedVectorStoreInterface, SemanticVectorStoreInterface
)

# Import mock implementations
from ..mocks.mock_llm import MockLLM, MockOllamaLLM, MockCodeLLM, MockConversationalLLM
from ..mocks.mock_embedding import MockEmbedding, MockSemanticEmbedding, MockMultilingualEmbedding
from ..mocks.mock_vision import (
    MockVision, MockImageClassification, MockObjectDetection,
    MockImageQuality, MockImageGeneration, MockImageEnhancement
)
from ..mocks.mock_vector_store import MockVectorStore, MockAdvancedVectorStore, MockSemanticVectorStore

T = TypeVar('T')

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Runtime environment types."""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TESTING = "testing"
    CI = "ci"


class LifecycleType(Enum):
    """Component lifecycle types."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ComponentConfig:
    """Configuration for a dependency injection component."""
    interface_type: Type
    implementation_type: Type
    lifecycle: LifecycleType = LifecycleType.SINGLETON
    config: Optional[Dict[str, Any]] = None
    lazy_load: bool = True


class ServiceLocator(Protocol):
    """Protocol for service location."""

    def get[T](self, interface_type: Type[T]) -> T:
        """Get service instance by interface type."""
        ...

    def has(self, interface_type: Type) -> bool:
        """Check if service is registered."""
        ...


class Container:
    """Dependency injection container for Echo Brain ML components."""

    def __init__(self, environment: Optional[Environment] = None):
        """Initialize dependency injection container."""
        self._environment = environment or self._detect_environment()
        self._components: Dict[Type, ComponentConfig] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None

        # Register default components based on environment
        self._register_default_components()

        logger.info(f"Container initialized for environment: {self._environment.value}")

    @property
    def environment(self) -> Environment:
        """Get current environment."""
        return self._environment

    def register[T](
        self,
        interface_type: Type[T],
        implementation_type: Type[T],
        lifecycle: LifecycleType = LifecycleType.SINGLETON,
        config: Optional[Dict[str, Any]] = None,
        lazy_load: bool = True
    ) -> None:
        """Register a component in the container."""
        if not issubclass(implementation_type, interface_type):
            raise ValueError(f"Implementation {implementation_type} must implement {interface_type}")

        self._components[interface_type] = ComponentConfig(
            interface_type=interface_type,
            implementation_type=implementation_type,
            lifecycle=lifecycle,
            config=config,
            lazy_load=lazy_load
        )

        # Clear existing instance if re-registering
        if interface_type in self._instances:
            del self._instances[interface_type]

        logger.debug(f"Registered {implementation_type.__name__} for {interface_type.__name__}")

    def get[T](self, interface_type: Type[T], config_override: Optional[Dict[str, Any]] = None) -> T:
        """Get service instance by interface type."""
        if interface_type not in self._components:
            raise KeyError(f"No component registered for {interface_type}")

        component_config = self._components[interface_type]

        # Handle lifecycle
        if component_config.lifecycle == LifecycleType.SINGLETON:
            return self._get_singleton(interface_type, component_config, config_override)
        elif component_config.lifecycle == LifecycleType.SCOPED:
            return self._get_scoped(interface_type, component_config, config_override)
        else:  # TRANSIENT
            return self._create_instance(component_config, config_override)

    def has(self, interface_type: Type) -> bool:
        """Check if interface type is registered."""
        return interface_type in self._components

    def create_scope(self, scope_name: str) -> 'ScopeManager':
        """Create a new scope for scoped dependencies."""
        return ScopeManager(self, scope_name)

    def dispose(self) -> None:
        """Dispose of all instances and clean up resources."""
        logger.info("Disposing container resources...")

        # Dispose singletons
        for instance in self._instances.values():
            self._dispose_instance(instance)

        # Dispose scoped instances
        for scope_instances in self._scoped_instances.values():
            for instance in scope_instances.values():
                self._dispose_instance(instance)

        # Clear all references
        self._instances.clear()
        self._scoped_instances.clear()

        logger.info("Container disposed successfully")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get container diagnostics information."""
        return {
            "environment": self._environment.value,
            "registered_components": len(self._components),
            "active_singletons": len(self._instances),
            "active_scopes": len(self._scoped_instances),
            "current_scope": self._current_scope,
            "components": {
                interface.__name__: {
                    "implementation": config.implementation_type.__name__,
                    "lifecycle": config.lifecycle.value,
                    "lazy_load": config.lazy_load
                }
                for interface, config in self._components.items()
            }
        }

    def _detect_environment(self) -> Environment:
        """Detect runtime environment from environment variables."""
        env_var = os.getenv('ECHO_ENVIRONMENT', '').lower()

        if env_var == 'production':
            return Environment.PRODUCTION
        elif env_var == 'development':
            return Environment.DEVELOPMENT
        elif env_var == 'testing' or os.getenv('CI') or os.getenv('PYTEST_CURRENT_TEST'):
            return Environment.TESTING
        elif env_var == 'ci':
            return Environment.CI
        else:
            # Default based on other indicators
            if os.getenv('CI'):
                return Environment.CI
            elif os.getenv('DEBUG'):
                return Environment.DEVELOPMENT
            else:
                return Environment.PRODUCTION

    def _register_default_components(self) -> None:
        """Register default components based on environment."""
        if self._environment in [Environment.TESTING, Environment.CI]:
            # Register mock implementations for testing
            self._register_mock_components()
        else:
            # Register real implementations for production/development
            self._register_production_components()

    def _register_mock_components(self) -> None:
        """Register mock implementations for testing."""
        logger.info("Registering mock components for testing environment")

        # LLM components
        self.register(LLMInterface, MockLLM, LifecycleType.SINGLETON)
        self.register(OllamaLLMInterface, MockOllamaLLM, LifecycleType.SINGLETON)
        self.register(CodeLLMInterface, MockCodeLLM, LifecycleType.SINGLETON)
        self.register(ConversationalLLMInterface, MockConversationalLLM, LifecycleType.SINGLETON)

        # Embedding components
        self.register(EmbeddingInterface, MockEmbedding, LifecycleType.SINGLETON)
        self.register(SemanticEmbeddingInterface, MockSemanticEmbedding, LifecycleType.SINGLETON)
        self.register(MultilingualEmbeddingInterface, MockMultilingualEmbedding, LifecycleType.SINGLETON)

        # Vision components
        self.register(VisionInterface, MockVision, LifecycleType.SINGLETON)
        self.register(ImageClassificationInterface, MockImageClassification, LifecycleType.SINGLETON)
        self.register(ObjectDetectionInterface, MockObjectDetection, LifecycleType.SINGLETON)
        self.register(ImageQualityInterface, MockImageQuality, LifecycleType.SINGLETON)
        self.register(ImageGenerationInterface, MockImageGeneration, LifecycleType.SINGLETON)
        self.register(ImageEnhancementInterface, MockImageEnhancement, LifecycleType.SINGLETON)

        # Vector store components
        self.register(VectorStoreInterface, MockVectorStore, LifecycleType.SINGLETON)
        self.register(AdvancedVectorStoreInterface, MockAdvancedVectorStore, LifecycleType.SINGLETON)
        self.register(SemanticVectorStoreInterface, MockSemanticVectorStore, LifecycleType.SINGLETON)

    def _register_production_components(self) -> None:
        """Register real implementations for production."""
        logger.info("Registering production components")

        # In a real implementation, these would import the actual ML implementations
        # For now, we'll use mocks with a warning that real implementations need to be created

        logger.warning("Production components not yet implemented - using mocks as fallback")
        self._register_mock_components()

        # TODO: When real implementations are created, register them here like:
        # from ..implementations.ollama_llm import OllamaLLM
        # from ..implementations.qdrant_vector_store import QdrantVectorStore
        # self.register(LLMInterface, OllamaLLM, LifecycleType.SINGLETON)
        # self.register(VectorStoreInterface, QdrantVectorStore, LifecycleType.SINGLETON)

    def _get_singleton(self, interface_type: Type[T], config: ComponentConfig, config_override: Optional[Dict[str, Any]]) -> T:
        """Get singleton instance."""
        if interface_type not in self._instances:
            self._instances[interface_type] = self._create_instance(config, config_override)
        return self._instances[interface_type]

    def _get_scoped(self, interface_type: Type[T], config: ComponentConfig, config_override: Optional[Dict[str, Any]]) -> T:
        """Get scoped instance."""
        if self._current_scope is None:
            raise RuntimeError("No active scope for scoped dependency")

        if self._current_scope not in self._scoped_instances:
            self._scoped_instances[self._current_scope] = {}

        scope_instances = self._scoped_instances[self._current_scope]
        if interface_type not in scope_instances:
            scope_instances[interface_type] = self._create_instance(config, config_override)

        return scope_instances[interface_type]

    def _create_instance(self, config: ComponentConfig, config_override: Optional[Dict[str, Any]]) -> Any:
        """Create new instance of component."""
        try:
            # Merge configuration
            instance_config = {}
            if config.config:
                instance_config.update(config.config)
            if config_override:
                instance_config.update(config_override)

            # Create instance
            if instance_config:
                instance = config.implementation_type(**instance_config)
            else:
                instance = config.implementation_type()

            logger.debug(f"Created instance of {config.implementation_type.__name__}")
            return instance

        except Exception as e:
            logger.error(f"Failed to create instance of {config.implementation_type.__name__}: {e}")
            raise

    def _dispose_instance(self, instance: Any) -> None:
        """Dispose of an instance if it has cleanup methods."""
        try:
            if hasattr(instance, 'dispose'):
                instance.dispose()
            elif hasattr(instance, 'close'):
                instance.close()
            elif hasattr(instance, 'unload'):
                # For ML models
                if asyncio.iscoroutinefunction(instance.unload):
                    import asyncio
                    asyncio.create_task(instance.unload())
                else:
                    instance.unload()
        except Exception as e:
            logger.warning(f"Error disposing instance {type(instance).__name__}: {e}")

    def _enter_scope(self, scope_name: str) -> None:
        """Enter a dependency scope."""
        self._current_scope = scope_name
        if scope_name not in self._scoped_instances:
            self._scoped_instances[scope_name] = {}

    def _exit_scope(self, scope_name: str) -> None:
        """Exit a dependency scope and dispose scoped instances."""
        if scope_name in self._scoped_instances:
            scope_instances = self._scoped_instances[scope_name]
            for instance in scope_instances.values():
                self._dispose_instance(instance)
            del self._scoped_instances[scope_name]

        if self._current_scope == scope_name:
            self._current_scope = None

    def get_llm(self) -> LLMInterface:
        """Get LLM interface instance."""
        return self.get(LLMInterface)

    def get_embedding(self) -> EmbeddingInterface:
        """Get embedding interface instance."""
        return self.get(EmbeddingInterface)

    def get_vision(self) -> VisionInterface:
        """Get vision interface instance."""
        return self.get(VisionInterface)

    def get_vector_store(self) -> VectorStoreInterface:
        """Get vector store interface instance."""
        return self.get(VectorStoreInterface)


class ScopeManager:
    """Context manager for dependency scopes."""

    def __init__(self, container: Container, scope_name: str):
        """Initialize scope manager."""
        self._container = container
        self._scope_name = scope_name

    def __enter__(self) -> Container:
        """Enter scope."""
        self._container._enter_scope(self._scope_name)
        return self._container

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit scope."""
        self._container._exit_scope(self._scope_name)


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get global container instance."""
    global _container
    if _container is None:
        _container = Container()
    return _container


def set_container(container: Container) -> None:
    """Set global container instance."""
    global _container
    if _container is not None:
        _container.dispose()
    _container = container


def reset_container() -> None:
    """Reset global container instance."""
    global _container
    if _container is not None:
        _container.dispose()
        _container = None


# Container convenience methods are defined within the Container class


# Convenience functions
def get_llm() -> LLMInterface:
    """Get LLM interface instance."""
    return get_container().get(LLMInterface)


def get_embedding() -> EmbeddingInterface:
    """Get embedding interface instance."""
    return get_container().get(EmbeddingInterface)


def get_vision() -> VisionInterface:
    """Get vision interface instance."""
    return get_container().get(VisionInterface)


def get_vector_store() -> VectorStoreInterface:
    """Get vector store interface instance."""
    return get_container().get(VectorStoreInterface)


# Testing utilities
def create_test_container() -> Container:
    """Create container configured for testing."""
    test_container = Container(Environment.TESTING)
    return test_container


def create_mock_container() -> Container:
    """Create container with all mock implementations."""
    container = Container(Environment.TESTING)
    return container