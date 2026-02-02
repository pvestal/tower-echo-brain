# 🔴 DEPRECATED: Use unified_router.py instead
# This file is being phased out in favor of single source of truth
# Import from: from src.routing.unified_router import unified_router

#!/usr/bin/env python3
"""
Model Management Interface Protocols
Defines contracts for AI model management and routing systems
"""

from typing import Protocol, runtime_checkable, List, Dict, Optional, Any, AsyncGenerator
from datetime import datetime
from enum import Enum

class ModelStatus(Enum):
    """Model status enumeration"""
    AVAILABLE = "available"
    LOADING = "loading"
    ERROR = "error"
    DOWNLOADING = "downloading"
    NOT_FOUND = "not_found"

class ModelCapability(Enum):
    """Model capability enumeration"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    REASONING = "reasoning"
    MATH = "math"
    FUNCTION_CALLING = "function_calling"

@runtime_checkable
class ModelInterface(Protocol):
    """
    Protocol for individual AI model implementations

    Defines the contract for interacting with AI models regardless
    of their backend (Ollama, OpenAI, etc.)
    """

    async def query(self, prompt: str, context: Optional[Dict[str, Any]] = None,
                   stream: bool = False, **kwargs) -> Any:
        """
        Query the model with a prompt

        Args:
            prompt: Input prompt for the model
            context: Optional context dictionary
            stream: Whether to stream the response
            **kwargs: Additional model-specific parameters

        Returns:
            Model response (string for non-streaming, AsyncGenerator for streaming)
        """
        ...

    async def query_with_system(self, system_prompt: str, user_prompt: str,
                              context: Optional[Dict[str, Any]] = None,
                              **kwargs) -> str:
        """
        Query model with system and user prompts

        Args:
            system_prompt: System instruction prompt
            user_prompt: User input prompt
            context: Optional context dictionary
            **kwargs: Additional parameters

        Returns:
            str: Model response
        """
        ...

    async def generate_streaming(self, prompt: str,
                               context: Optional[Dict[str, Any]] = None,
                               **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate streaming response

        Args:
            prompt: Input prompt
            context: Optional context
            **kwargs: Additional parameters

        Yields:
            str: Response chunks
        """
        ...

    def get_model_name(self) -> str:
        """
        Get the model name/identifier

        Returns:
            str: Model name
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information

        Returns:
            Dict: Model metadata including capabilities, parameters, etc.
        """
        ...

    async def get_model_status(self) -> ModelStatus:
        """
        Get current model status

        Returns:
            ModelStatus: Current status of the model
        """
        ...

    def get_capabilities(self) -> List[ModelCapability]:
        """
        Get model capabilities

        Returns:
            List[ModelCapability]: List of supported capabilities
        """
        ...

    async def health_check(self) -> bool:
        """
        Perform model health check

        Returns:
            bool: True if model is healthy and responsive
        """
        ...

    def get_parameter_count(self) -> Optional[int]:
        """
        Get model parameter count

        Returns:
            Optional[int]: Number of parameters or None if unknown
        """
        ...

    def get_context_length(self) -> Optional[int]:
        """
        Get maximum context length

        Returns:
            Optional[int]: Maximum context length or None if unknown
        """
        ...


@runtime_checkable
class ModelManagerInterface(Protocol):
    """
    Protocol for AI model management systems

    Handles model lifecycle, routing, and resource management
    """

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models

        Returns:
            List[Dict]: Model information dictionaries
        """
        ...

    async def get_model(self, model_name: str) -> Optional[ModelInterface]:
        """
        Get a specific model by name

        Args:
            model_name: Name of the model to retrieve

        Returns:
            Optional[ModelInterface]: Model instance or None if not found
        """
        ...

    async def load_model(self, model_name: str, **kwargs) -> bool:
        """
        Load a model into memory

        Args:
            model_name: Name of the model to load
            **kwargs: Model-specific loading parameters

        Returns:
            bool: True if model was successfully loaded
        """
        ...

    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory

        Args:
            model_name: Name of the model to unload

        Returns:
            bool: True if model was successfully unloaded
        """
        ...

    async def download_model(self, model_name: str,
                           progress_callback: Optional[callable] = None) -> bool:
        """
        Download a model from remote repository

        Args:
            model_name: Name of the model to download
            progress_callback: Optional callback for download progress

        Returns:
            bool: True if model was successfully downloaded
        """
        ...

    async def remove_model(self, model_name: str) -> bool:
        """
        Remove a model from local storage

        Args:
            model_name: Name of the model to remove

        Returns:
            bool: True if model was successfully removed
        """
        ...

    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed model information

        Args:
            model_name: Name of the model

        Returns:
            Optional[Dict]: Model information or None if not found
        """
        ...

    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health check on all loaded models

        Returns:
            Dict[str, bool]: Model name to health status mapping
        """
        ...

    async def get_model_usage_stats(self, model_name: str) -> Dict[str, Any]:
        """
        Get usage statistics for a model

        Args:
            model_name: Name of the model

        Returns:
            Dict: Usage statistics including request count, avg response time, etc.
        """
        ...

    def get_recommended_model(self, task_type: str,
                            complexity: str = "medium") -> Optional[str]:
        """
        Get recommended model for a task

        Args:
            task_type: Type of task (e.g., "code", "reasoning", "creative")
            complexity: Task complexity level

        Returns:
            Optional[str]: Recommended model name or None
        """
        ...


@runtime_checkable
class ModelRouterInterface(Protocol):
    """
    Protocol for intelligent model routing systems
    """

    async def route_query(self, query: str, context: Optional[Dict[str, Any]] = None,
                         preferred_models: Optional[List[str]] = None) -> str:
        """
        Route query to appropriate model and return response

        Args:
            query: User query
            context: Optional context information
            preferred_models: Optional list of preferred model names

        Returns:
            str: Model response
        """
        ...

    async def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine complexity and requirements

        Args:
            query: User query to analyze

        Returns:
            Dict: Analysis including complexity, task type, recommended models
        """
        ...

    async def select_optimal_model(self, query: str,
                                 requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        Select optimal model for a query

        Args:
            query: User query
            requirements: Optional requirements dict (speed, accuracy, etc.)

        Returns:
            str: Selected model name
        """
        ...

    async def escalate_to_larger_model(self, query: str, current_model: str,
                                     context: Dict[str, Any]) -> str:
        """
        Escalate query to larger/more capable model

        Args:
            query: Original query
            current_model: Model that couldn't handle the query
            context: Context from previous attempt

        Returns:
            str: Response from more capable model
        """
        ...


@runtime_checkable
class OllamaManagerInterface(Protocol):
    """
    Protocol for Ollama-specific model management
    """

    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry

        Args:
            model_name: Name of model to pull

        Returns:
            bool: True if successfully pulled
        """
        ...

    async def create_model(self, model_name: str, modelfile: str) -> bool:
        """
        Create custom model from Modelfile

        Args:
            model_name: Name for the new model
            modelfile: Modelfile content

        Returns:
            bool: True if model was created successfully
        """
        ...

    async def copy_model(self, source: str, destination: str) -> bool:
        """
        Copy an existing model

        Args:
            source: Source model name
            destination: Destination model name

        Returns:
            bool: True if model was copied successfully
        """
        ...

    async def delete_model(self, model_name: str) -> bool:
        """
        Delete a model

        Args:
            model_name: Name of model to delete

        Returns:
            bool: True if model was deleted successfully
        """
        ...

    async def show_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed model information

        Args:
            model_name: Name of the model

        Returns:
            Optional[Dict]: Model details or None if not found
        """
        ...

    async def list_running_models(self) -> List[Dict[str, Any]]:
        """
        List currently running models

        Returns:
            List[Dict]: Running model information
        """
        ...

    async def generate(self, model: str, prompt: str, stream: bool = False,
                     **options) -> Any:
        """
        Generate response using Ollama API

        Args:
            model: Model name
            prompt: Input prompt
            stream: Whether to stream response
            **options: Additional Ollama options

        Returns:
            Response (string or stream)
        """
        ...

    async def embed(self, model: str, input_text: str) -> Optional[List[float]]:
        """
        Generate embeddings using Ollama

        Args:
            model: Embedding model name
            input_text: Text to embed

        Returns:
            Optional[List[float]]: Embedding vector or None if failed
        """
        ...