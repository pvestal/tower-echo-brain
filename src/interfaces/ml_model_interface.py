#!/usr/bin/env python3
"""
Abstract base interface for all ML models in Echo Brain.
Provides type safety and dependency injection abstraction.
Patrick Vestal - December 9, 2025
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, AsyncIterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ModelType(Enum):
    """Enumeration of ML model types."""
    LLM = "llm"
    EMBEDDING = "embedding"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    CLASSIFIER = "classifier"
    REGRESSION = "regression"
    GENERATIVE = "generative"


class ModelStatus(Enum):
    """Enumeration of model status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class ModelMetadata:
    """Metadata for ML models."""
    name: str
    version: str
    model_type: ModelType
    parameters: int
    memory_usage: int  # in MB
    compute_requirements: Dict[str, Any]
    capabilities: List[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    accuracy: Optional[float] = None
    confidence_threshold: float = 0.7


@dataclass
class ModelPrediction:
    """Standardized prediction result."""
    prediction: Any
    confidence: float
    metadata: Dict[str, Any]
    processing_time: float
    model_name: str
    timestamp: datetime


class MLModelInterface(ABC):
    """Abstract base class for all ML models in Echo Brain."""

    def __init__(self, model_name: str, model_type: ModelType):
        """Initialize ML model interface."""
        self._model_name = model_name
        self._model_type = model_type
        self._status = ModelStatus.UNLOADED
        self._metadata: Optional[ModelMetadata] = None

    @property
    def name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def model_type(self) -> ModelType:
        """Get model type."""
        return self._model_type

    @property
    def status(self) -> ModelStatus:
        """Get current model status."""
        return self._status

    @property
    def metadata(self) -> Optional[ModelMetadata]:
        """Get model metadata."""
        return self._metadata

    @abstractmethod
    async def load(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load the model into memory.

        Args:
            config: Optional configuration parameters

        Returns:
            True if model loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    async def unload(self) -> bool:
        """Unload the model from memory.

        Returns:
            True if model unloaded successfully, False otherwise
        """
        pass

    @abstractmethod
    async def predict(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelPrediction:
        """Make a prediction with the model.

        Args:
            input_data: Input data for prediction
            context: Optional context information

        Returns:
            ModelPrediction object with result and metadata
        """
        pass

    @abstractmethod
    async def batch_predict(
        self,
        inputs: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ModelPrediction]:
        """Make batch predictions.

        Args:
            inputs: List of input data for predictions
            context: Optional context information

        Returns:
            List of ModelPrediction objects
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data format.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check model health and status.

        Returns:
            Dictionary with health status information
        """
        pass

    async def get_capabilities(self) -> List[str]:
        """Get model capabilities.

        Returns:
            List of capability strings
        """
        if self._metadata:
            return self._metadata.capabilities.copy()
        return []

    async def get_memory_usage(self) -> int:
        """Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        if self._metadata:
            return self._metadata.memory_usage
        return 0

    def _update_status(self, new_status: ModelStatus) -> None:
        """Update model status (protected method)."""
        self._status = new_status

    def _set_metadata(self, metadata: ModelMetadata) -> None:
        """Set model metadata (protected method)."""
        self._metadata = metadata


class StreamingMLModelInterface(MLModelInterface):
    """Interface for streaming ML models (like LLMs with token streaming)."""

    @abstractmethod
    async def predict_stream(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterable[Dict[str, Any]]:
        """Make a streaming prediction.

        Args:
            input_data: Input data for prediction
            context: Optional context information

        Yields:
            Dictionary with partial results and metadata
        """
        pass


class TrainableMLModelInterface(MLModelInterface):
    """Interface for trainable ML models."""

    @abstractmethod
    async def train(
        self,
        training_data: List[Any],
        validation_data: Optional[List[Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            training_data: Training dataset
            validation_data: Optional validation dataset
            config: Training configuration

        Returns:
            Training results and metrics
        """
        pass

    @abstractmethod
    async def save_checkpoint(self, checkpoint_path: str) -> bool:
        """Save model checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    async def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to load checkpoint from

        Returns:
            True if loaded successfully, False otherwise
        """
        pass


class MultiModalMLModelInterface(MLModelInterface):
    """Interface for multimodal ML models (text, image, audio, etc.)."""

    @abstractmethod
    async def predict_multimodal(
        self,
        text_input: Optional[str] = None,
        image_input: Optional[Any] = None,
        audio_input: Optional[Any] = None,
        video_input: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelPrediction:
        """Make multimodal prediction.

        Args:
            text_input: Optional text input
            image_input: Optional image input
            audio_input: Optional audio input
            video_input: Optional video input
            context: Optional context information

        Returns:
            ModelPrediction with multimodal result
        """
        pass

    @abstractmethod
    def get_supported_modalities(self) -> List[str]:
        """Get list of supported input modalities.

        Returns:
            List of supported modality strings (text, image, audio, video)
        """
        pass